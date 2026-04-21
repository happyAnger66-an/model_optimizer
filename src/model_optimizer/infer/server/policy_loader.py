"""策略加载器：根据 ServerConfig.mode 加载 PyTorch / TRT / PTQ 策略。"""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from termcolor import colored

from .config import ServerConfig

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str], None]


def _noop_progress(stage: str, message: str) -> None:
    pass


def _resolve_precision(precision: str):
    import torch

    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _load_pytorch_policy(config: ServerConfig, on_progress: ProgressCallback):
    """加载一份 PyTorch 浮点策略。"""
    from openpi.policies import policy_config
    from openpi.training import config as _config

    on_progress("config", "读取训练配置 …")
    train_cfg = _config.get_config(config.config_name)

    on_progress("policy_pt", "加载 PyTorch 策略（checkpoint → 内存/显存）…")
    policy = policy_config.create_trained_policy(
        train_cfg,
        config.checkpoint,
        pytorch_device=config.device,
    )
    on_progress("policy_pt", "PyTorch 策略已就绪")
    return policy, train_cfg


def _mount_tensorrt_engines(policy: Any, config: ServerConfig, on_progress: ProgressCallback) -> None:
    """在已有策略上挂载 TensorRT 引擎。"""
    from model_optimizer.infer.tensorrt.pi05_executor import Pi05TensorRTExecutor

    import addict

    prec = _resolve_precision(config.precision)
    on_progress("tensorrt", "加载 TensorRT 引擎 …")
    executor = Pi05TensorRTExecutor(policy, prec)

    trt_cfg: dict[str, str] = {"engine_path": config.tensorrt.engine_path}
    for attr in ("vit_engine", "llm_engine", "expert_engine", "denoise_engine", "embed_prefix_engine"):
        val = getattr(config.tensorrt, attr, "")
        if val:
            trt_cfg[attr] = val

    executor.load_model(addict.Dict(trt_cfg))
    on_progress("tensorrt", "TensorRT 引擎已就绪")


def _load_ptq_quant_cfg(path: str) -> dict[str, Any]:
    """加载量化配置：.json 或 .py（定义 QUANT_CFG）。"""
    from model_optimizer.utils.utils import normalize_quant_cfg

    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"ptq quant_cfg not found: {p}")

    suffix = p.suffix.lower()
    if suffix in (".py", ".pyw"):
        mod_name = f"_ptq_quant_cfg_{p.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, p)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot create module from: {p}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "QUANT_CFG"):
            raise AttributeError(f"{p} does not define QUANT_CFG")
        raw = getattr(mod, "QUANT_CFG")
        if not isinstance(raw, dict):
            raise TypeError(f"QUANT_CFG must be dict, got {type(raw).__name__}")
        cfg = copy.deepcopy(raw)
    else:
        with open(p, encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise TypeError(f"JSON root must be dict in {p}")
        cfg = copy.deepcopy(loaded)

    if "quant_mode" in cfg:
        _, cfg = normalize_quant_cfg(cfg)
    return cfg


def _mount_onnxrt_engines(policy: Any, config: ServerConfig, on_progress: ProgressCallback) -> None:
    """在已有策略上挂载 ONNX Runtime 引擎。"""
    from model_optimizer.infer.onnxrt.pi05_executor import Pi05OnnxRTExecutor

    import addict

    prec = _resolve_precision(config.precision)
    on_progress("onnxrt", "加载 ONNX Runtime 引擎 …")
    executor = Pi05OnnxRTExecutor(policy, prec)

    ort_cfg: dict[str, str] = {"engine_path": config.onnxrt.engine_path}
    for attr in ("vit_engine", "llm_engine", "expert_engine", "denoise_engine", "embed_prefix_engine"):
        val = getattr(config.onnxrt, attr, "")
        if val:
            ort_cfg[attr] = val

    executor.load_model(addict.Dict(ort_cfg))
    on_progress("onnxrt", "ONNX Runtime 引擎已就绪")


def _apply_selective_ptq(
    policy: Any,
    config: ServerConfig,
    on_progress: ProgressCallback,
) -> None:
    """对策略的子模块就地应用选择性 PTQ。"""
    from model_optimizer.quantization.quantization_utils import quantize_model
    from model_optimizer.utils.utils import set_dynamic_quant

    from model_optimizer.models.pi05.vit import Vit
    from model_optimizer.models.pi05.llm import LLM
    from model_optimizer.models.pi05.expert import Expert
    from model_optimizer.models.pi05.dit import Pi05DenoiseStep

    m = _unwrap_pi05_model(policy)
    if m is None:
        raise RuntimeError("Cannot unwrap Pi0.5 model from policy for PTQ")

    quant_cfg = _load_ptq_quant_cfg(config.ptq.quant_cfg)
    calib_dir = str(Path(config.ptq.calib_dir).expanduser().resolve())

    parts_s = ",".join(config.ptq.parts)
    on_progress("ptq_apply", f"对 [{parts_s}] 应用选择性 PTQ …")

    for part in config.ptq.parts:
        sub_cfg = copy.deepcopy(quant_cfg)
        if part == "vit":
            wrap = Vit(
                m.paligemma_with_expert.paligemma.config,
                m.paligemma_with_expert.paligemma.model.vision_tower,
                m.paligemma_with_expert.paligemma.model.multi_modal_projector,
            )
            dl = wrap.get_calibrate_dataset(calib_dir)
            print(colored("[ptq] quantize vit …", "cyan"), flush=True)
            quantize_model(
                wrap,
                sub_cfg,
                dl,
                measure_quant_error=config.ptq.measure_quant_error,
            )
            set_dynamic_quant(wrap, "bf16")
        elif part == "llm":
            pal = m.paligemma_with_expert.paligemma
            wrap = LLM(pal.config.text_config, pal.get_decoder())
            dl = wrap.get_calibrate_dataset(calib_dir)
            print(colored("[ptq] quantize llm …", "cyan"), flush=True)
            quantize_model(
                wrap,
                sub_cfg,
                dl,
                measure_quant_error=config.ptq.measure_quant_error,
            )
            set_dynamic_quant(wrap, "bf16")
        elif part == "expert":
            ge = m.paligemma_with_expert.gemma_expert
            wrap = Expert(ge.config, ge.model)
            dl = wrap.get_calibrate_dataset(calib_dir)
            print(colored("[ptq] quantize expert …", "cyan"), flush=True)
            quantize_model(
                wrap,
                sub_cfg,
                dl,
                measure_quant_error=config.ptq.measure_quant_error,
            )
            set_dynamic_quant(wrap, "bf16")
        elif part == "denoise":
            wrap = Pi05DenoiseStep.construct_model(m)
            dl = wrap.get_calibrate_dataset(calib_dir)
            print(colored("[ptq] quantize denoise …", "cyan"), flush=True)
            quantize_model(
                wrap,
                sub_cfg,
                dl,
                measure_quant_error=config.ptq.measure_quant_error,
            )
            set_dynamic_quant(wrap, "fp16")
        else:
            raise ValueError(f"Unknown ptq part: {part!r}")

    print(colored("[ptq] 选择性量化完成", "green"), flush=True)
    on_progress("ptq_apply", "选择性 PTQ 已应用")


def _unwrap_pi05_model(policy: Any) -> Any | None:
    """取底层 Pi0.5 torch 模块。"""
    inner = getattr(policy, "_policy", None)
    if inner is not None:
        m = getattr(inner, "_model", None)
        if m is not None:
            return m
    return getattr(policy, "_model", None)


def load_policy_for_serve(
    config: ServerConfig,
    on_progress: ProgressCallback | None = None,
) -> Any:
    """加载策略用于在线 serve（传递 default_prompt / robot_type / unify_action_mode）。

    Returns:
        openpi Policy 对象（支持 ``.infer(obs)``）。
    """
    if on_progress is None:
        on_progress = _noop_progress

    from openpi.policies import policy_config
    from openpi.training import config as _config

    serve_cfg = config.serve

    on_progress("config", "读取训练配置 …")
    train_cfg = _config.get_config(config.config_name)

    on_progress("policy", "加载策略（checkpoint → 内存/显存）…")
    policy = policy_config.create_trained_policy(
        train_cfg,
        config.checkpoint,
        default_prompt=serve_cfg.default_prompt,
        unify_action_mode=serve_cfg.unify_action_mode,
        robot_type=serve_cfg.robot_type,
        pytorch_device=config.device,
    )

    # TensorRT 引擎挂载（tensorrt / pt_trt_compare / ptq_trt_compare 模式）
    if config.mode in ("tensorrt", "pt_trt_compare", "ptq_trt_compare"):
        _mount_tensorrt_engines(policy, config, on_progress)

    # ONNX Runtime 引擎挂载（onnxrt / pt_ort_compare 模式）
    if config.mode in ("onnxrt", "pt_ort_compare"):
        _mount_onnxrt_engines(policy, config, on_progress)

    # PTQ 量化（pt_ptq_compare / ptq_trt_compare 模式）
    if config.mode in ("pt_ptq_compare", "ptq_trt_compare"):
        _apply_selective_ptq(policy, config, on_progress)

    on_progress("ready", "策略加载完成")
    return policy


def load_policies(
    config: ServerConfig,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """根据 config.mode 加载所需策略，返回 dict 包含 policy / policy_trt / policy_ptq。

    Args:
        config: 服务配置。
        on_progress: 可选进度回调 (stage, message)。

    Returns:
        {"policy": ..., "policy_trt": ... | None, "policy_ptq": ... | None, "train_cfg": ...}
    """
    if on_progress is None:
        on_progress = _noop_progress

    from openpi.policies import policy_config
    from openpi.training import config as _config

    policy, train_cfg = _load_pytorch_policy(config, on_progress)
    policy_trt = None
    policy_ptq = None

    mode = config.mode

    if mode == "tensorrt":
        _mount_tensorrt_engines(policy, config, on_progress)

    elif mode == "onnxrt":
        _mount_onnxrt_engines(policy, config, on_progress)

    elif mode == "pt_trt_compare":
        on_progress("policy_trt", "对比模式：加载第二套策略并挂载 TensorRT …")
        policy_trt = policy_config.create_trained_policy(
            train_cfg,
            config.checkpoint,
            pytorch_device=config.device,
        )
        _mount_tensorrt_engines(policy_trt, config, on_progress)
        on_progress("policy_trt", "PyTorch + TensorRT 双路就绪")

    elif mode == "pt_ort_compare":
        on_progress("policy_ort", "对比模式：加载第二套策略并挂载 ONNX Runtime …")
        policy_trt = policy_config.create_trained_policy(
            train_cfg,
            config.checkpoint,
            pytorch_device=config.device,
        )
        _mount_onnxrt_engines(policy_trt, config, on_progress)
        on_progress("policy_ort", "PyTorch + ONNX Runtime 双路就绪")

    elif mode == "pt_ptq_compare":
        on_progress("ptq_policy", "对比模式：加载第二套策略并应用 PTQ …")
        policy_ptq = policy_config.create_trained_policy(
            train_cfg,
            config.checkpoint,
            pytorch_device=config.device,
        )
        _apply_selective_ptq(policy_ptq, config, on_progress)
        on_progress("ptq_policy", "PyTorch + PTQ 双路就绪")

    elif mode == "ptq_trt_compare":
        on_progress("ptq_apply", "PTQ+TRT 对比：对主策略应用 PTQ …")
        _apply_selective_ptq(policy, config, on_progress)

        on_progress("policy_trt", "PTQ+TRT 对比：加载第二套策略并挂载 TensorRT …")
        policy_trt = policy_config.create_trained_policy(
            train_cfg,
            config.checkpoint,
            pytorch_device=config.device,
        )
        _mount_tensorrt_engines(policy_trt, config, on_progress)
        on_progress("policy_trt", "PTQ + TensorRT 双路就绪")

    return {
        "policy": policy,
        "policy_trt": policy_trt,
        "policy_ptq": policy_ptq,
        "train_cfg": train_cfg,
    }
