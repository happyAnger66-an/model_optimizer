"""meta 消息与推理 bundle 字典组装。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .bundle_common import BundleProgress, trt_trt_second_engine_filenames
from .bundle_dataset import DatasetBundle
from .bundle_policies import PolicyBundle
from .config import Args
from .dataset import global_episode_id_per_frame, unwrap_lerobot_base
from .gpu_stats import effective_gpu_index
from .protocol import event_to_json
from .running_stats import (
    RunningErrorStats,
    RunningPerDimMsePctStats,
    RunningPerDimPairMseStats,
    RunningPerDimRelP99Stats,
    RunningVitCompareStats,
)


def compute_eval_range(args: Args, dataset: Any) -> tuple[int, int, Any]:
    n = len(dataset)
    end = min(args.start_index + args.num_samples, n)
    if args.start_index >= n:
        raise ValueError(f"start_index={args.start_index} >= dataset len={n}")
    base_ds = unwrap_lerobot_base(dataset)
    ep_per_frame = global_episode_id_per_frame(base_ds, n)
    return n, end, ep_per_frame


def _resolve_backend_label(args: Args) -> str:
    if args.compare_mode:
        return "pytorch+tensorrt"
    if args.ptq_compare:
        return "pytorch+ptq"
    if getattr(args, "ptq_trt_compare", False):
        return "pytorch_ptq+tensorrt"
    if getattr(args, "trt_ort_compare", False):
        return "tensorrt+onnxrt"
    if getattr(args, "trt_trt_compare", False):
        return "tensorrt+tensorrt"
    if getattr(args, "ort_compare", False):
        return "pytorch+onnxrt"
    return args.inference_mode


def _apply_pred_name_labels(args: Args, meta: dict[str, Any]) -> None:
    if getattr(args, "ptq_trt_compare", False):
        meta["pred1_name"] = "PTQ"
        meta["pred2_name"] = "TRT"
        meta["pair_name"] = "PTQ−TRT"
    elif args.ptq_compare:
        meta["pred1_name"] = "PT"
        meta["pred2_name"] = "PTQ"
        meta["pair_name"] = "PT−PTQ"
    elif args.compare_mode:
        meta["pred1_name"] = "PT"
        meta["pred2_name"] = "TRT"
        meta["pair_name"] = "PT−TRT"
    elif getattr(args, "ort_compare", False):
        meta["pred1_name"] = "PT"
        meta["pred2_name"] = "ORT"
        meta["pair_name"] = "PT−ORT"
    elif getattr(args, "trt_ort_compare", False):
        meta["pred1_name"] = "TRT"
        meta["pred2_name"] = "ORT"
        meta["pair_name"] = "TRT−ORT"
    elif getattr(args, "trt_trt_compare", False):
        meta["pred1_name"] = "TRT_ref"
        meta["pred2_name"] = "TRT_tgt"
        meta["pair_name"] = "TRT_ref−TRT_tgt"
    elif args.inference_mode == "tensorrt":
        meta["pred1_name"] = (
            "TRT+FlashRT"
            if bool(getattr(args, "native_flashrt_decoder", False))
            else "TRT"
        )


def _apply_tensorrt_meta(args: Args, meta: dict[str, Any]) -> None:
    if not (
        args.inference_mode == "tensorrt"
        or args.compare_mode
        or getattr(args, "ptq_trt_compare", False)
        or getattr(args, "trt_ort_compare", False)
        or getattr(args, "trt_trt_compare", False)
    ):
        return
    meta["tensorrt"] = {
        "precision": args.precision,
        "engine_path": args.engine_path or "",
        "vit_engine": args.vit_engine or "",
        "llm_engine": args.llm_engine or "",
        "expert_engine": args.expert_engine or "",
        "denoise_engine": args.denoise_engine or "",
        "embed_prefix_engine": args.embed_prefix_engine or "",
    }
    if getattr(args, "trt_trt_compare", False):
        sec_nm = trt_trt_second_engine_filenames(args)
        meta["tensorrt"]["trt_trt_second_engine_path"] = str(
            getattr(args, "trt_trt_second_engine_path", "") or "",
        )
        meta["tensorrt"]["trt_trt_second_vit_engine"] = sec_nm["vit_engine"] or ""
        meta["tensorrt"]["trt_trt_second_llm_engine"] = sec_nm["llm_engine"] or ""
        meta["tensorrt"]["trt_trt_second_expert_engine"] = sec_nm["expert_engine"] or ""
        meta["tensorrt"]["trt_trt_second_denoise_engine"] = sec_nm["denoise_engine"] or ""
        meta["tensorrt"]["trt_trt_second_embed_prefix_engine"] = sec_nm["embed_prefix_engine"] or ""


def _apply_onnxrt_meta(args: Args, meta: dict[str, Any]) -> None:
    if not (
        args.inference_mode == "onnxrt"
        or getattr(args, "ort_compare", False)
        or getattr(args, "trt_ort_compare", False)
    ):
        return
    meta["onnxrt"] = {
        "engine_path": getattr(args, "ort_engine_path", "") or "",
        "vit_engine": getattr(args, "ort_vit_engine", "") or "",
        "llm_engine": getattr(args, "ort_llm_engine", "") or "",
        "expert_engine": getattr(args, "ort_expert_engine", "") or "",
        "denoise_engine": getattr(args, "ort_denoise_engine", "") or "",
        "embed_prefix_engine": getattr(args, "ort_embed_prefix_engine", "") or "",
    }


def _native_meta_block(args: Args) -> dict[str, Any]:
    return {
        "use_cuda_graph": bool(getattr(args, "native_use_cuda_graph", True)),
        "full_loop_graph": bool(getattr(args, "native_full_loop_graph", False)),
        "graph_warmup": int(getattr(args, "native_graph_warmup", 3)),
        "compile_expert": bool(getattr(args, "native_compile_expert", False)),
        "enable_expert": bool(getattr(args, "native_enable_expert", True)),
        "enable_denoise": bool(getattr(args, "native_enable_denoise", True)),
        "quant_spec_path": str(getattr(args, "native_quant_spec_path", "") or ""),
        "recalib_enable": bool(getattr(args, "native_recalib_enable", False)),
        "recalib_max_samples": int(getattr(args, "native_recalib_max_samples", 0)),
        "recalib_percentile": float(getattr(args, "native_recalib_percentile", 99.9)),
        "flashrt_decoder": bool(getattr(args, "native_flashrt_decoder", False)),
        "flashrt_use_fp8": bool(getattr(args, "native_flashrt_use_fp8", True)),
        "flashrt_calibrate": bool(getattr(args, "native_flashrt_calibrate", False)),
    }


def _apply_native_meta(args: Args, meta: dict[str, Any]) -> None:
    if args.inference_mode == "native":
        meta["native"] = _native_meta_block(args)
    elif args.inference_mode == "tensorrt" and bool(getattr(args, "native_overlay_on_tensorrt", False)):
        block = _native_meta_block(args)
        block["overlay_on_tensorrt"] = True
        meta["native"] = block


def maybe_ptq_layer_report(
    args: Args,
    *,
    policy: Any,
    policy_ptq: Any | None,
    dataset: Any,
    repack_fn: Any,
    progress: BundleProgress,
) -> tuple[Path | None, dict[str, Any] | None]:
    if not args.ptq_compare or args.ptq_layer_report_path is None:
        return None, None
    from .ptq_compare import write_ptq_layer_report

    report_path = Path(args.ptq_layer_report_path).expanduser().resolve()
    progress.emit("ptq_report", "生成分层 PTQ 报告（hook 对比 FP/PTQ，可能较慢）…")
    write_ptq_layer_report(
        policy,
        policy_ptq,
        tuple(args.ptq_parts),
        dataset=dataset,
        repack_fn=repack_fn,
        start_index=int(args.start_index),
        num_samples=int(args.ptq_layer_report_samples),
        report_path=report_path,
        include_activation_histogram=bool(args.ptq_layer_report_histogram),
        hist_bins=int(args.ptq_layer_report_hist_bins),
        hist_max_elems=int(args.ptq_layer_report_hist_max_elems),
    )
    try:
        with open(report_path, encoding="utf-8") as rf:
            data = json.load(rf)
    except Exception as exc:  # pragma: no cover
        logging.warning("读取 ptq layer report 嵌入 meta 失败: %s", exc)
        data = {"error": str(exc), "path": str(report_path)}
    progress.emit("ptq_report", "分层 PTQ 报告已写入并完成读取")
    return report_path, data


def maybe_calib_collectors(args: Args, policy: Any, progress: BundleProgress) -> list[Any] | None:
    if args.calib_save_path is None:
        return None
    calib_ok = args.inference_mode == "pytorch" or args.compare_mode or args.ptq_compare
    if not calib_ok:
        logging.warning(
            "已忽略 --calib-save-path：Pi0.5 calib 仅支持 pytorch / compare_mode / ptq_compare（当前为 %s）。",
            args.inference_mode,
        )
        return None
    try:
        from .calib import start_pi05_calib_collectors

        progress.emit("calib", "启动 Pi0.5 calib 收集器 …")
        collectors = start_pi05_calib_collectors(
            policy,
            Path(args.calib_save_path),
            max_samples=int(getattr(args, "calib_max_samples", 0)),
            calib_item=str(getattr(args, "calib_item", "all")),
        )
        progress.emit("calib", f"calib 收集器已启动 → {args.calib_save_path}")
        return collectors
    except Exception as exc:  # pragma: no cover
        logging.warning("启动 calib 收集失败，将继续评估但不保存 calib: %s", exc, exc_info=True)
        return None


def maybe_polygraphy_report(args: Args, progress: BundleProgress) -> dict[str, Any] | None:
    if not getattr(args, "trt_ort_compare", False) or not getattr(args, "trt_ort_polygraphy_compare", False):
        return None
    from .trt_ort_polygraphy_compare import build_trt_ort_polygraphy_report

    progress.emit("trt_ort_polygraphy", "Polygraphy：子图 TRT vs ORT 对比（可能较慢）…")
    report = build_trt_ort_polygraphy_report(args)
    ok_pg = bool(report.get("ok")) if isinstance(report, dict) else False
    progress.emit(
        "trt_ort_polygraphy",
        "Polygraphy：对比完成（"
        + ("全部子图通过" if ok_pg else "存在失败、缺依赖或未配置子图；见 meta.trt_ort_polygraphy")
        + "）",
    )
    return report


def build_meta_payload(
    args: Args,
    run_id: str,
    ds: DatasetBundle,
    *,
    end: int,
    calib_collectors: list[Any] | None,
    ptq_layer_report_path: Path | None,
    ptq_layer_report_data: dict[str, Any] | None,
    polygraphy_report: dict[str, Any] | None,
) -> dict[str, Any]:
    data_config = ds.data_config
    meta: dict[str, Any] = {
        "type": "meta",
        "run_id": run_id,
        "repo_id": data_config.repo_id,
        "backend": _resolve_backend_label(args),
        "compare_mode": bool(args.compare_mode),
        "vit_pt_trt_compare": bool(getattr(args, "vit_pt_trt_compare", False)),
        "ptq_compare": bool(args.ptq_compare),
        "ptq_trt_compare": bool(getattr(args, "ptq_trt_compare", False)),
        "ort_compare": bool(getattr(args, "ort_compare", False)),
        "trt_ort_compare": bool(getattr(args, "trt_ort_compare", False)),
        "trt_trt_compare": bool(getattr(args, "trt_trt_compare", False)),
        "action_horizon": int(ds.action_horizon),
        "action_dim": int(ds.action_dim),
        "start_index": int(args.start_index),
        "end_index_exclusive": int(end),
        "send_wrist": bool(args.send_wrist),
        "jpeg_quality": int(args.jpeg_quality),
    }
    if args.calib_save_path is not None and calib_collectors is not None:
        meta["calib_save_path"] = str(Path(args.calib_save_path).expanduser().resolve())
    if args.gpu_stats_interval_sec and args.gpu_stats_interval_sec > 0:
        meta["gpu_stats_interval_sec"] = float(args.gpu_stats_interval_sec)
        meta["gpu_device_index"] = int(effective_gpu_index(args))
    meta["rel_err_denominator"] = "max_abs_gt_eps"
    meta["rel_eps"] = float(args.rel_eps)
    meta["flow_match_noise"] = str(args.noise)
    if args.noise == "fixed":
        meta["flow_match_noise_seed"] = int(args.noise_seed)
    if args.ptq_compare:
        meta["ptq_parts"] = list(args.ptq_parts)
        meta["ptq_quant_cfg"] = str(Path(args.ptq_quant_cfg).expanduser().resolve())
        meta["ptq_calib_dir"] = str(Path(args.ptq_calib_dir).expanduser().resolve())
        if ptq_layer_report_path is not None:
            meta["ptq_layer_report_path"] = str(ptq_layer_report_path)
            if ptq_layer_report_data is not None:
                meta["ptq_layer_report"] = ptq_layer_report_data
    if getattr(args, "ptq_trt_compare", False):
        meta["ptq_parts"] = list(args.ptq_parts)
        meta["ptq_quant_cfg"] = str(Path(args.ptq_quant_cfg).expanduser().resolve())
        meta["ptq_calib_dir"] = str(Path(args.ptq_calib_dir).expanduser().resolve())
    _apply_pred_name_labels(args, meta)
    _apply_tensorrt_meta(args, meta)
    _apply_onnxrt_meta(args, meta)
    _apply_native_meta(args, meta)
    if polygraphy_report is not None:
        meta["trt_ort_polygraphy"] = polygraphy_report
    return meta


def assemble_infer_bundle(
    args: Args,
    run_id: str,
    ds: DatasetBundle,
    policies: PolicyBundle,
    *,
    n: int,
    end: int,
    ep_per_frame: Any,
    meta_msg: str,
    calib_collectors: list[Any] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "meta_msg": meta_msg,
        "dataset": ds.dataset,
        "repack_fn": ds.repack_fn,
        "policy": policies.policy,
        "native_executor": policies.native_executor,
        "policy_trt": policies.policy_trt,
        "policy_ptq": policies.policy_ptq,
        "n": n,
        "end": end,
        "start_index": int(args.start_index),
        "action_horizon": int(ds.action_horizon),
        "action_dim": int(ds.action_dim),
        "ep_per_frame": ep_per_frame,
        "run_id": run_id,
        "args": args,
        "calib_collectors": calib_collectors,
        "running_err_stats": RunningErrorStats(),
        "running_per_dim_mse_pct": RunningPerDimMsePctStats(),
        "running_per_dim_rel_p99": RunningPerDimRelP99Stats(),
        "running_vit_pt_trt": RunningVitCompareStats(),
        "trt_ort_compare": bool(getattr(args, "trt_ort_compare", False)),
        "trt_trt_compare": bool(getattr(args, "trt_trt_compare", False)),
    }
    second_path = (
        bool(args.compare_mode)
        or bool(getattr(args, "ort_compare", False))
        or bool(getattr(args, "trt_ort_compare", False))
        or bool(getattr(args, "trt_trt_compare", False))
        or bool(getattr(args, "ptq_trt_compare", False))
    )
    if second_path:
        out["running_per_dim_mse_pct_trt"] = RunningPerDimMsePctStats()
        out["running_per_dim_rel_p99_trt"] = RunningPerDimRelP99Stats()
        out["running_pt_trt_mse_per_dim"] = RunningPerDimPairMseStats()
    if args.ptq_compare:
        out["running_per_dim_mse_pct_ptq"] = RunningPerDimMsePctStats()
        out["running_per_dim_rel_p99_ptq"] = RunningPerDimRelP99Stats()
        out["running_pt_ptq_mse_per_dim"] = RunningPerDimPairMseStats()
    return out


def finalize_meta_and_bundle(
    args: Args,
    run_id: str,
    ds: DatasetBundle,
    policies: PolicyBundle,
    progress: BundleProgress,
) -> dict[str, Any]:
    n, end, ep_per_frame = compute_eval_range(args, ds.dataset)
    ptq_path, ptq_data = maybe_ptq_layer_report(
        args,
        policy=policies.policy,
        policy_ptq=policies.policy_ptq,
        dataset=ds.dataset,
        repack_fn=ds.repack_fn,
        progress=progress,
    )
    calib_collectors = maybe_calib_collectors(args, policies.policy, progress)
    polygraphy = maybe_polygraphy_report(args, progress)
    meta_payload = build_meta_payload(
        args,
        run_id,
        ds,
        end=end,
        calib_collectors=calib_collectors,
        ptq_layer_report_path=ptq_path,
        ptq_layer_report_data=ptq_data,
        polygraphy_report=polygraphy,
    )
    progress.emit("ready", "组装 meta、运行态统计器 …")
    meta_msg = event_to_json(meta_payload)
    progress.emit("ready", "加载阶段完成，即将推送 meta 与 step 流")
    return assemble_infer_bundle(
        args,
        run_id,
        ds,
        policies,
        n=n,
        end=end,
        ep_per_frame=ep_per_frame,
        meta_msg=meta_msg,
        calib_collectors=calib_collectors,
    )
