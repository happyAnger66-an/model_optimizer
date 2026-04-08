"""加载数据集、策略与 meta 消息（推理线程内调用）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from openpi.policies import policy_config
from openpi.training import config as _config
from termcolor import colored

from .calib import start_pi05_calib_collectors
from .config import Args
from .dataset import (
    build_repack_only,
    global_episode_id_per_frame,
    make_lerobot_dataset,
    unwrap_lerobot_base,
)
from .gpu_stats import effective_gpu_index
from .protocol import event_to_json
from .running_stats import RunningErrorStats, RunningPerDimMsePctStats, RunningPerDimRelP99Stats
from .tensorrt_backend import load_tensorrt_engines


def load_infer_bundle(args: Args, run_id: str) -> dict[str, Any]:
    """在专用推理线程中执行：数据集 + policy + TRT，避免阻塞 asyncio 事件循环。"""
    print(colored("[infer] get_config + data_config ...", "cyan"), flush=True)
    train_cfg = _config.get_config(args.config)
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if not data_config.repo_id:
        raise ValueError("当前配置未设置 repo_id，无法加载 LeRobot 数据。")
    action_horizon = train_cfg.model.action_horizon
    action_keys = tuple(data_config.action_sequence_keys)

    print(colored(f"[infer] LeRobotDataset(repo={data_config.repo_id!r}) ...", "cyan"), flush=True)
    dataset = make_lerobot_dataset(
        repo_id=data_config.repo_id,
        action_horizon=action_horizon,
        action_sequence_keys=action_keys,
        prompt_from_task=data_config.prompt_from_task,
        dataset_root=args.dataset_root,
    )
    repack_fn = build_repack_only(data_config)

    print(colored("[infer] create_trained_policy（可能较慢，磁盘/显存占用会上升）...", "cyan"), flush=True)
    policy = policy_config.create_trained_policy(
        train_cfg,
        args.checkpoint,
        pytorch_device=args.device,
    )
    try:
        pd = getattr(policy, "_pytorch_device", None)
        is_pt = getattr(policy, "_is_pytorch_model", None)
        print(
            colored(f"[infer] policy 就绪 is_pytorch={is_pt} device={pd!r}", "cyan"),
            flush=True,
        )
    except Exception:
        pass
    if args.inference_mode == "tensorrt":
        if not args.engine_path:
            raise ValueError("inference_mode=tensorrt 时必须设置 --engine-path（引擎目录）。")
        print(colored("[infer] 加载 TensorRT 引擎 ...", "cyan"), flush=True)
        load_tensorrt_engines(
            policy,
            engine_path=args.engine_path,
            precision=args.precision,
            vit_engine=args.vit_engine,
            llm_engine=args.llm_engine,
            expert_engine=args.expert_engine,
            denoise_engine=args.denoise_engine,
            embed_prefix_engine=args.embed_prefix_engine,
        )
        print(colored("[infer] TensorRT 引擎已就绪", "cyan"), flush=True)

    n = len(dataset)
    end = min(args.start_index + args.num_samples, n)
    if args.start_index >= n:
        raise ValueError(f"start_index={args.start_index} >= dataset len={n}")

    base_ds = unwrap_lerobot_base(dataset)
    ep_per_frame = global_episode_id_per_frame(base_ds, n)

    calib_collectors: list[Any] | None = None
    if args.calib_save_path is not None:
        if args.inference_mode != "pytorch":
            logging.warning(
                "已忽略 --calib-save-path：Pi0.5 calib 仅支持 inference_mode=pytorch（当前为 %s）。",
                args.inference_mode,
            )
        else:
            try:
                calib_collectors = start_pi05_calib_collectors(policy, Path(args.calib_save_path))
            except Exception as exc:  # pragma: no cover
                logging.warning("启动 calib 收集失败，将继续评估但不保存 calib: %s", exc, exc_info=True)

    meta_payload: dict[str, Any] = {
        "type": "meta",
        "run_id": run_id,
        "repo_id": data_config.repo_id,
        "backend": args.inference_mode,
        "action_horizon": int(action_horizon),
        "start_index": int(args.start_index),
        "end_index_exclusive": int(end),
        "send_wrist": bool(args.send_wrist),
        "jpeg_quality": int(args.jpeg_quality),
    }
    if args.calib_save_path is not None and calib_collectors is not None:
        meta_payload["calib_save_path"] = str(Path(args.calib_save_path).expanduser().resolve())

    if args.gpu_stats_interval_sec and args.gpu_stats_interval_sec > 0:
        meta_payload["gpu_stats_interval_sec"] = float(args.gpu_stats_interval_sec)
        meta_payload["gpu_device_index"] = int(effective_gpu_index(args))
    meta_payload["rel_err_denominator"] = "max_abs_gt_eps"
    meta_payload["rel_eps"] = float(args.rel_eps)

    meta_msg = event_to_json(meta_payload)

    return {
        "meta_msg": meta_msg,
        "dataset": dataset,
        "repack_fn": repack_fn,
        "policy": policy,
        "n": n,
        "end": end,
        "start_index": int(args.start_index),
        "action_horizon": int(action_horizon),
        "ep_per_frame": ep_per_frame,
        "run_id": run_id,
        "args": args,
        "calib_collectors": calib_collectors,
        "running_err_stats": RunningErrorStats(),
        "running_per_dim_mse_pct": RunningPerDimMsePctStats(),
        "running_per_dim_rel_p99": RunningPerDimRelP99Stats(),
    }
