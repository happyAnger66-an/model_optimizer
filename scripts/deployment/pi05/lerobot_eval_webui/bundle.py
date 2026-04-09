"""加载数据集、策略与 meta 消息（推理线程内调用）。"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
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
from .running_stats import (
    RunningErrorStats,
    RunningPerDimMsePctStats,
    RunningPerDimPairMseStats,
    RunningPerDimRelP99Stats,
)
from .tensorrt_backend import load_tensorrt_engines


def load_infer_bundle(
    args: Args,
    run_id: str,
    *,
    on_progress: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """在专用推理线程中执行：数据集 + policy + TRT，避免阻塞 asyncio 事件循环。

    on_progress(stage_id, message)：可选，供 WebUI 推送加载步骤（stage_id 稳定、便于 client 去重/排序）。
    """

    def _p(stage: str, msg: str) -> None:
        if on_progress is not None:
            on_progress(stage, msg)

    if args.compare_mode and args.ptq_compare:
        raise ValueError("compare_mode 与 ptq_compare 互斥，请勿同时开启。")
    if args.ptq_compare and args.inference_mode != "pytorch":
        raise ValueError("ptq_compare 仅支持 inference_mode=pytorch。")

    print(colored("[infer] get_config + data_config ...", "cyan"), flush=True)
    _p("config", "读取训练配置与 data_config …")
    train_cfg = _config.get_config(args.config)
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if not data_config.repo_id:
        raise ValueError("当前配置未设置 repo_id，无法加载 LeRobot 数据。")
    action_horizon = train_cfg.model.action_horizon
    action_keys = tuple(data_config.action_sequence_keys)

    print(colored(f"[infer] LeRobotDataset(repo={data_config.repo_id!r}) ...", "cyan"), flush=True)
    _p("dataset", f"加载 LeRobot 数据集（repo_id={data_config.repo_id}）…")
    dataset = make_lerobot_dataset(
        repo_id=data_config.repo_id,
        action_horizon=action_horizon,
        action_sequence_keys=action_keys,
        prompt_from_task=data_config.prompt_from_task,
        dataset_root=args.dataset_root,
    )
    repack_fn = build_repack_only(data_config)
    _p("dataset", f"数据集就绪（共 {len(dataset)} 条）")

    print(colored("[infer] create_trained_policy（可能较慢，磁盘/显存占用会上升）...", "cyan"), flush=True)
    _p("policy_pt", "加载 PyTorch 策略（checkpoint → 内存/显存，可能较慢）…")
    policy = policy_config.create_trained_policy(
        train_cfg,
        args.checkpoint,
        pytorch_device=args.device,
    )
    policy_trt: Any | None = None
    policy_ptq: Any | None = None
    try:
        pd = getattr(policy, "_pytorch_device", None)
        is_pt = getattr(policy, "_is_pytorch_model", None)
        print(
            colored(f"[infer] policy 就绪 is_pytorch={is_pt} device={pd!r}", "cyan"),
            flush=True,
        )
    except Exception:
        pass
    _p("policy_pt", "PyTorch 策略已就绪")

    if args.compare_mode:
        if not args.engine_path:
            raise ValueError("compare_mode=True 时必须设置 --engine-path（TensorRT 引擎目录）。")
        print(colored("[infer] compare_mode：加载第二套 policy 并挂 TensorRT …", "cyan"), flush=True)
        _p("policy_trt", "compare：加载 TensorRT 路 PyTorch 封装并挂载引擎 …")
        policy_trt = policy_config.create_trained_policy(
            train_cfg,
            args.checkpoint,
            pytorch_device=args.device,
        )
        load_tensorrt_engines(
            policy_trt,
            engine_path=args.engine_path,
            precision=args.precision,
            vit_engine=args.vit_engine,
            llm_engine=args.llm_engine,
            expert_engine=args.expert_engine,
            denoise_engine=args.denoise_engine,
            embed_prefix_engine=args.embed_prefix_engine,
        )
        print(colored("[infer] compare_mode：PyTorch + TensorRT 双策略已就绪", "cyan"), flush=True)
        _p("policy_trt", "TensorRT 引擎已挂载（compare 双路就绪）")
    elif args.ptq_compare:
        if args.ptq_quant_cfg is None or not Path(args.ptq_quant_cfg).is_file():
            raise ValueError("ptq_compare 需要有效的 --ptq-quant-cfg（存在的 .json 或定义 QUANT_CFG 的 .py）。")
        if args.ptq_calib_dir is None or not Path(args.ptq_calib_dir).expanduser().is_dir():
            raise ValueError("ptq_compare 需要 --ptq-calib-dir 指向含 Pi0.5 calib 的目录。")
        if not args.ptq_parts:
            raise ValueError("ptq_compare 需要非空 --ptq-parts，例如 vit、llm、expert。")
        bad = [p for p in args.ptq_parts if p not in ("vit", "llm", "expert")]
        if bad:
            raise ValueError(f"非法 --ptq-parts: {bad}（仅允许 vit / llm / expert）。")

        print(colored("[infer] ptq_compare：第二份 PyTorch policy + 选择性 PTQ …", "cyan"), flush=True)
        _p("ptq_policy", "ptq_compare：加载第二套 PyTorch 策略 …")
        policy_ptq = policy_config.create_trained_policy(
            train_cfg,
            args.checkpoint,
            pytorch_device=args.device,
        )
        _p("ptq_policy", "PTQ 路 PyTorch 策略已加载")
        from .ptq_compare import apply_selective_ptq, load_ptq_quant_cfg

        qcfg = load_ptq_quant_cfg(Path(args.ptq_quant_cfg))
        parts_s = ",".join(args.ptq_parts)
        _p("ptq_apply", f"ptq_compare：读取 calib 并对 [{parts_s}] 应用量化（quantize + dynamic）…")
        apply_selective_ptq(policy_ptq, Path(args.ptq_calib_dir), qcfg, tuple(args.ptq_parts))

        print(colored("[infer] ptq_compare：浮点 policy + PTQ policy 已就绪", "cyan"), flush=True)
        _p("ptq_apply", "选择性 PTQ 已应用（浮点 + PTQ 双路就绪）")
    elif args.inference_mode == "tensorrt":
        if not args.engine_path:
            raise ValueError("inference_mode=tensorrt 时必须设置 --engine-path（引擎目录）。")
        print(colored("[infer] 加载 TensorRT 引擎 ...", "cyan"), flush=True)
        _p("tensorrt", "加载 TensorRT 引擎（vit/llm/expert 等）…")
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
        _p("tensorrt", "TensorRT 引擎已就绪")

    n = len(dataset)
    end = min(args.start_index + args.num_samples, n)
    if args.start_index >= n:
        raise ValueError(f"start_index={args.start_index} >= dataset len={n}")

    base_ds = unwrap_lerobot_base(dataset)
    ep_per_frame = global_episode_id_per_frame(base_ds, n)

    ptq_layer_report_path_resolved: Path | None = None
    ptq_layer_report_data: dict[str, Any] | None = None
    if args.ptq_compare and args.ptq_layer_report_path is not None:
        from .ptq_compare import write_ptq_layer_report

        ptq_layer_report_path_resolved = Path(args.ptq_layer_report_path).expanduser().resolve()
        _p("ptq_report", "生成分层 PTQ 报告（hook 对比 FP/PTQ，可能较慢）…")
        write_ptq_layer_report(
            policy,
            policy_ptq,
            tuple(args.ptq_parts),
            dataset=dataset,
            repack_fn=repack_fn,
            start_index=int(args.start_index),
            num_samples=int(args.ptq_layer_report_samples),
            report_path=ptq_layer_report_path_resolved,
        )
        try:
            with open(ptq_layer_report_path_resolved, encoding="utf-8") as rf:
                ptq_layer_report_data = json.load(rf)
        except Exception as exc:  # pragma: no cover
            logging.warning("读取 ptq layer report 嵌入 meta 失败: %s", exc)
            ptq_layer_report_data = {
                "error": str(exc),
                "path": str(ptq_layer_report_path_resolved),
            }
        _p("ptq_report", "分层 PTQ 报告已写入并完成读取")

    calib_collectors: list[Any] | None = None
    if args.calib_save_path is not None:
        calib_ok = args.inference_mode == "pytorch" or args.compare_mode or args.ptq_compare
        if not calib_ok:
            logging.warning(
                "已忽略 --calib-save-path：Pi0.5 calib 仅支持 pytorch / compare_mode / ptq_compare（当前为 %s）。",
                args.inference_mode,
            )
        else:
            try:
                _p("calib", "启动 Pi0.5 calib 收集器 …")
                calib_collectors = start_pi05_calib_collectors(policy, Path(args.calib_save_path))
                _p("calib", f"calib 收集器已启动 → {args.calib_save_path}")
            except Exception as exc:  # pragma: no cover
                logging.warning("启动 calib 收集失败，将继续评估但不保存 calib: %s", exc, exc_info=True)

    if args.compare_mode:
        backend = "pytorch+tensorrt"
    elif args.ptq_compare:
        backend = "pytorch+ptq"
    else:
        backend = args.inference_mode
    meta_payload: dict[str, Any] = {
        "type": "meta",
        "run_id": run_id,
        "repo_id": data_config.repo_id,
        "backend": backend,
        "compare_mode": bool(args.compare_mode),
        "ptq_compare": bool(args.ptq_compare),
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

    if args.ptq_compare:
        meta_payload["ptq_parts"] = list(args.ptq_parts)
        meta_payload["ptq_quant_cfg"] = str(Path(args.ptq_quant_cfg).expanduser().resolve())
        meta_payload["ptq_calib_dir"] = str(Path(args.ptq_calib_dir).expanduser().resolve())
        if args.ptq_layer_report_path is not None and ptq_layer_report_path_resolved is not None:
            meta_payload["ptq_layer_report_path"] = str(ptq_layer_report_path_resolved)
            if ptq_layer_report_data is not None:
                meta_payload["ptq_layer_report"] = ptq_layer_report_data

    if args.inference_mode == "tensorrt" or args.compare_mode:
        meta_payload["tensorrt"] = {
            "precision": args.precision,
            "engine_path": args.engine_path or "",
            "vit_engine": args.vit_engine or "",
            "llm_engine": args.llm_engine or "",
            "expert_engine": args.expert_engine or "",
            "denoise_engine": args.denoise_engine or "",
            "embed_prefix_engine": args.embed_prefix_engine or "",
        }

    _p("ready", "组装 meta、运行态统计器 …")
    meta_msg = event_to_json(meta_payload)
    _p("ready", "加载阶段完成，即将推送 meta 与 step 流")

    out: dict[str, Any] = {
        "meta_msg": meta_msg,
        "dataset": dataset,
        "repack_fn": repack_fn,
        "policy": policy,
        "policy_trt": policy_trt,
        "policy_ptq": policy_ptq,
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
    if args.compare_mode:
        out["running_per_dim_mse_pct_trt"] = RunningPerDimMsePctStats()
        out["running_per_dim_rel_p99_trt"] = RunningPerDimRelP99Stats()
        out["running_pt_trt_mse_per_dim"] = RunningPerDimPairMseStats()
    if args.ptq_compare:
        out["running_per_dim_mse_pct_ptq"] = RunningPerDimMsePctStats()
        out["running_per_dim_rel_p99_ptq"] = RunningPerDimRelP99Stats()
        out["running_pt_ptq_mse_per_dim"] = RunningPerDimPairMseStats()
    return out
