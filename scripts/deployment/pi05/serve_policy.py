#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pi0.5 WebSocket 策略服务：对齐 openpi ``scripts/serve_policy.py``，并支持 PyTorch / TensorRT
# 推理及可替换 engine（与 ``standalone_inference_script.py`` 一致）。
#
# 示例：
#
#   # PyTorch（默认）
#   python scripts/deployment/pi05/serve_policy.py \\
#     --config-name pi05_libero \\
#     --checkpoint-dir /path/to/checkpoint \\
#     --inference-mode pytorch --port 8000
#
#   # TensorRT（目录下放 vit/llm/expert/denoise/embed_prefix 等 engine）
#   python scripts/deployment/pi05/serve_policy.py \\
#     --config-name pi05_libero \\
#     --checkpoint-dir /path/to/checkpoint \\
#     --inference-mode tensorrt \\
#     --trt-engine-path /path/to/trt_engines/ \\
#     --vit-engine vit.engine --llm-engine llm.engine --expert-engine expert.engine \\
#     --denoise-engine denoise.engine --embed-prefix-engine embed_prefix.engine \\
#     --port 8000
#
# 依赖：已安装 openpi（含 ``openpi.policies``、``openpi.serving.websocket_policy_server``）、
# ``tyro``、``addict``、GPU 上 TensorRT 相关环境与 ``model_optimizer`` 包。

from __future__ import annotations

import dataclasses
import importlib.util
import logging
import socket
from pathlib import Path
from typing import Any, Literal

import torch
import tyro

try:
    import addict
except ImportError as e:  # pragma: no cover
    raise ImportError("serve_policy requires the `addict` package") from e

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


def load_train_config(config_ref: str) -> _config.TrainConfig:
    """与 ``standalone_inference_script.load_train_config`` 相同：注册名或 ``TrainConfig`` 的 ``.py`` 路径。"""
    path = Path(config_ref).expanduser()
    if path.is_file() and path.suffix == ".py":
        mod_name = f"_serve_policy_cfg_{path.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, str(path.resolve()))
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load config module from {config_ref!r}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tc = _config.TrainConfig
        for attr in ("cfg", "config", "train_config"):
            if hasattr(mod, attr):
                obj = getattr(mod, attr)
                if isinstance(obj, tc):
                    logging.info("Loaded TrainConfig from %s (attribute %r)", path, attr)
                    return obj
        raise ValueError(
            f"File {path} does not define a TrainConfig instance. "
            "Define one of: cfg, config, or train_config = TrainConfig(...)."
        )
    return _config.get_config(config_ref)


@dataclasses.dataclass
class Args:
    """Pi0.5 ``serve_policy``：WebSocket 上提供与 openpi 兼容的 ``infer`` / ``score``。"""

    checkpoint_dir: str
    """训练 checkpoint 目录（含权重等）。"""

    # --- 与 openpi serve_policy 对齐 ---
    config_name: str = "pi05_libero"
    """``get_config`` 注册名，或指向 ``cfg = TrainConfig(...)`` 的 ``.py`` 路径。"""

    default_prompt: str | None = None
    """观测中无 ``prompt``/``task`` 时注入的默认文本。"""

    robot_type: str = "bi_piper_follower"
    """与 ``create_trained_policy(..., robot_type=...)`` 一致。"""

    host: str = "0.0.0.0"
    port: int = 8000
    record: bool = False
    """若为 True，用 ``PolicyRecorder`` 落盘请求/响应。"""

    enable_value_endpoint: bool = False
    """是否暴露 ``score``；仅当策略实现 ``score_observation`` 时有效。"""

    value_temperature: float = 1.0

    # --- 推理后端（对齐 standalone_inference_script）---
    inference_mode: Literal["pytorch", "tensorrt"] = "pytorch"

    precision: Literal["fp16", "bf16", "fp32"] = "bf16"
    """仅 ``tensorrt`` 模式下用于 ``Pi05TensorRTExecutor`` 的激活精度标签。"""

    trt_engine_path: str = ""
    """TRT engine 所在**目录**；各 ``*_engine`` 可为该目录下的文件名或相对路径。"""

    vit_engine: str = ""
    llm_engine: str = ""
    expert_engine: str = ""
    denoise_engine: str = ""
    embed_prefix_engine: str = ""


def _build_trt_engine_config(args: Args) -> addict.Dict | None:
    if not args.trt_engine_path.strip():
        logging.warning("tensorrt mode but trt_engine_path empty; no engines will be loaded")
        return None
    cfg: dict[str, Any] = {"engine_path": args.trt_engine_path}
    if args.vit_engine:
        cfg["vit_engine"] = args.vit_engine
    if args.llm_engine:
        cfg["llm_engine"] = args.llm_engine
    if args.expert_engine:
        cfg["expert_engine"] = args.expert_engine
    if args.denoise_engine:
        cfg["denoise_engine"] = args.denoise_engine
    if args.embed_prefix_engine:
        cfg["embed_prefix_engine"] = args.embed_prefix_engine
    return addict.Dict(cfg)


def _apply_inference_backend(policy: _policy.Policy, args: Args) -> None:
    """就地挂载 PyTorch / TensorRT 执行路径（与 ``standalone_inference_script`` 一致）。"""
    if args.inference_mode == "tensorrt":
        from model_optimizer.infer.tensorrt.pi05_executor import Pi05TensorRTExecutor

        if args.precision == "fp16":
            precision = torch.float16
        elif args.precision == "bf16":
            precision = torch.bfloat16
        else:
            precision = torch.float32
        logging.info("TensorRT mode, precision=%s", precision)
        executor = Pi05TensorRTExecutor(policy, precision)
        trt_cfg = _build_trt_engine_config(args)
        executor.load_model(trt_cfg)
        logging.info("Pi05TensorRTExecutor.load_model done (engines under %r)", args.trt_engine_path)
        return

    if args.inference_mode == "pytorch":
        from model_optimizer.infer.pytorch.pi05_executor import Pi05PyTorchExecutor

        executor = Pi05PyTorchExecutor(policy)
        executor.load_model()
        logging.info("PyTorch mode (Pi05PyTorchExecutor; load_model may torch.compile action head)")
        return

    raise ValueError(f"Unknown inference_mode: {args.inference_mode!r}")


def main(args: Args) -> None:
    train_cfg = load_train_config(args.config_name)
    policy = _policy_config.create_trained_policy(
        train_cfg,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
        unify_action_mode=getattr(train_cfg.data, "unify_action_space", False),
        robot_type=args.robot_type,
    )

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    _apply_inference_backend(policy, args)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    policy_metadata = policy.metadata
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "(unknown)"
    logging.info("Creating Pi0.5 policy server (host=%s ip=%s port=%s mode=%s)", hostname, local_ip, args.port, args.inference_mode)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy_metadata,
        enable_score_endpoint=args.enable_value_endpoint,
        value_temperature=args.value_temperature,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
