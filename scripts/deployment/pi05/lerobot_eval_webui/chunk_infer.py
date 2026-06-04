"""单段 chunk：取样、推理、编码图像、生成 step 事件 JSON；推理结束 perf 汇总。"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from termcolor import colored

from .chunk_pipeline import process_infer_chunk
from .chunk_policy import policy_torch_model as _policy_torch_model
from .chunk_predict import flow_match_noise_for_chunk
from .chunk_profile import chunk_prof_snapshot
from .chunk_stage_perf import print_compare_stage_summary as _print_compare_stage_summary
from .chunk_stage_perf import to_ms_list as _to_ms_list

# 与 standalone / perf 一致：复用同一底层 model 引用打印 time_results
_perf_model: Any | None = None


def _stats_line_ms(values: list[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return "n=0"
    return (
        f"n={int(arr.size)} mean={float(np.mean(arr)):.2f} "
        f"p50={float(np.percentile(arr, 50)):.2f} p90={float(np.percentile(arr, 90)):.2f} "
        f"p99={float(np.percentile(arr, 99)):.2f} ms"
    )


def _perf_lines_from_holder(obj: Any) -> list[str]:
    """从挂载了 stage_perf 的对象取汇总行，仅用公开 API（兼容旧版 perf 包无私有符号）。"""
    if obj is None:
        return []
    try:
        from model_optimizer.infer.perf.stage_perf import _lines_from_perf_holder

        return list(_lines_from_perf_holder(obj))
    except (ImportError, AttributeError):
        pass
    fn = getattr(obj, "format_perf_summary_lines", None)
    if callable(fn):
        try:
            return list(fn())
        except Exception:
            return []
    sp = getattr(obj, "_stage_perf", None) or getattr(obj, "stage_perf", None)
    fmt = getattr(sp, "format_summary_lines", None) if sp is not None else None
    if callable(fmt):
        try:
            return list(fmt())
        except Exception:
            return []
    return []


def _dump_policy_time_results(model: Any, *, tag: str) -> None:
    tr = getattr(model, "time_results", None) if model is not None else None
    if not isinstance(tr, dict):
        return
    for key, label in (
        ("vit", "vit"),
        ("lang_emb", "lang_emb"),
        ("suffix", "suffix"),
        ("action", "action"),
        ("llm", "llm"),
    ):
        vals = tr.get(key, None)
        if vals:
            ms_vals = _to_ms_list(list(vals))
            print(
                colored(
                    f"[summary:model:{tag}] {label:<11} {_stats_line_ms(ms_vals)}",
                    "yellow",
                )
            )


def _dump_trt_engine_summary(policy: Any, *, tag: str) -> None:
    ex = getattr(policy, "_trt_executor", None) if policy is not None else None
    engs = getattr(ex, "_trt_engines", None) if ex is not None else None
    if not isinstance(engs, dict) or not engs:
        return
    for name in sorted(engs.keys()):
        eng = engs[name]
        tr_e = getattr(eng, "time_results", None)
        if not isinstance(tr_e, dict):
            continue
        total_s = tr_e.get("total", [])
        if not total_s:
            continue
        total_ms_vals = [float(x) * 1000.0 for x in total_s]
        print(
            colored(
                f"[summary:engine:{tag}] {name}.total   {_stats_line_ms(total_ms_vals)}",
                "yellow",
            )
        )
        for k in ("prepare", "execute", "post"):
            vv = tr_e.get(k, [])
            if vv:
                ms_vals = [float(x) * 1000.0 for x in vv]
                print(
                    colored(
                        f"[summary:engine:{tag}] {name}.{k:<7} {_stats_line_ms(ms_vals)}",
                        "yellow",
                    )
                )


def dump_perf_final_summary(bundle: dict[str, Any] | None) -> None:
    """推理结束时输出一次最终性能汇总。"""
    if not bundle:
        return
    args = bundle.get("args")
    if args is None:
        return

    _chunk_prof = chunk_prof_snapshot()
    policy = bundle.get("policy")
    policy_trt = bundle.get("policy_trt")
    model = _policy_torch_model(policy) if policy is not None else None
    model_trt = _policy_torch_model(policy_trt) if policy_trt is not None else None

    print(colored("========== FINAL PERF SUMMARY ==========", "yellow"))
    if policy_trt is not None:
        print(
            colored(
                "[summary] compare_mode：下列 [summary:model:pt] 为第一路 PyTorch；"
                "[summary:model:trt] / [summary:engine:trt] 为第二路 TensorRT。",
                "yellow",
            )
        )
    if _chunk_prof["total_ms"]:
        print(colored(f"[summary] e2e/chunk   {_stats_line_ms(_chunk_prof['total_ms'])}", "yellow"))
        if _chunk_prof["predict_ms"]:
            print(colored(f"[summary] predict_ms  {_stats_line_ms(_chunk_prof['predict_ms'])}", "yellow"))
        if _chunk_prof["load_ms"]:
            print(colored(f"[summary] load_ms     {_stats_line_ms(_chunk_prof['load_ms'])}", "yellow"))
        if _chunk_prof["repack_ms"]:
            print(colored(f"[summary] repack_ms   {_stats_line_ms(_chunk_prof['repack_ms'])}", "yellow"))
        if _chunk_prof["post_ms"]:
            print(colored(f"[summary] post_ms     {_stats_line_ms(_chunk_prof['post_ms'])}", "yellow"))
        data_ms = (
            float(np.mean(np.asarray(_chunk_prof["load_ms"], dtype=np.float64)))
            + float(np.mean(np.asarray(_chunk_prof["repack_ms"], dtype=np.float64)))
            if _chunk_prof["load_ms"] and _chunk_prof["repack_ms"]
            else 0.0
        )
        total_ms = float(np.mean(np.asarray(_chunk_prof["total_ms"], dtype=np.float64)))
        pred_ms = (
            float(np.mean(np.asarray(_chunk_prof["predict_ms"], dtype=np.float64)))
            if _chunk_prof["predict_ms"]
            else 0.0
        )
        py_overhead_ms = max(total_ms - pred_ms, 0.0)
        print(colored(f"[summary] data_processing ~= {data_ms:.2f} ms", "yellow"))
        print(
            colored(
                f"[summary] python_overhead ~= {py_overhead_ms:.2f} ms "
                f"({(py_overhead_ms/total_ms*100.0) if total_ms > 0 else 0.0:.1f}%)",
                "yellow",
            )
        )
        if _chunk_prof.get("infer_ms_pt"):
            print(
                colored(
                    f"[summary] infer_ms_pt    {_stats_line_ms(_chunk_prof['infer_ms_pt'])}",
                    "yellow",
                )
            )
        if _chunk_prof.get("infer_ms_second"):
            print(
                colored(
                    f"[summary] infer_ms_trt   {_stats_line_ms(_chunk_prof['infer_ms_second'])}",
                    "yellow",
                )
            )

    if policy_trt is not None:
        _print_compare_stage_summary(policy=policy, model=model, tag="pt", is_trt=False)
        _print_compare_stage_summary(policy=policy_trt, model=model_trt, tag="trt", is_trt=True)
        _dump_policy_time_results(model, tag="pt")
        _dump_policy_time_results(model_trt, tag="trt")
        _dump_trt_engine_summary(policy_trt, tag="trt")
        try:
            from model_optimizer.infer.tensorrt.trt_hook_timer import format_trt_hook_summary_lines

            for line in format_trt_hook_summary_lines(tag="trt"):
                print(colored(line, "yellow"))
        except ImportError:
            pass
    else:
        _single_is_trt = bool(getattr(policy, "_trt_executor", None) is not None)
        _print_compare_stage_summary(
            policy=policy, model=model, tag="pt", is_trt=_single_is_trt
        )
        _dump_policy_time_results(model, tag="pt")
        _dump_trt_engine_summary(policy, tag="pt")

    try:
        from model_optimizer.infer.perf import (
            KEY_POLICY_ALIGN,
            KEY_POLICY_INFER,
            format_collector_from_policy,
            stage_perf_from_policy,
        )

        perf_lines: list[str] = []
        if policy is not None:
            perf_lines.extend(format_collector_from_policy(policy))
        if policy_trt is not None:
            perf_lines.extend(format_collector_from_policy(policy_trt))
        native_ex = bundle.get("native_executor") if bundle else None
        if native_ex is not None:
            perf_lines.extend(_perf_lines_from_holder(native_ex))
        if perf_lines:
            for line in perf_lines:
                print(colored(line, "yellow"))
        sp = stage_perf_from_policy(policy) if policy is not None else None
        if sp is not None and _chunk_prof.get("predict_ms"):
            pred_mean = float(np.mean(np.asarray(_chunk_prof["predict_ms"], dtype=np.float64)))

            def _mean_key(key: str) -> float:
                vals = sp.values_for_key(key)
                if not vals:
                    return 0.0
                return float(np.mean(np.asarray(vals, dtype=np.float64)))

            infer_mean = _mean_key(KEY_POLICY_INFER)
            align_mean = _mean_key(KEY_POLICY_ALIGN)
            if infer_mean > 0.0:
                gap = pred_mean - infer_mean - align_mean
                print(
                    colored(
                        f"[summary] predict_reconcile  predict_ms={pred_mean:.2f} "
                        f"≈ policy.infer({infer_mean:.2f}) + policy.align({align_mean:.2f}) "
                        f"+ gap({gap:.2f})",
                        "yellow",
                    )
                )
        if not perf_lines:
            native_ex = bundle.get("native_executor") if bundle else None
            meta_native = {}
            if args is not None:
                meta_native = {
                    "overlay": bool(getattr(args, "native_overlay_on_tensorrt", False)),
                    "flashrt": bool(getattr(args, "native_flashrt_decoder", False)),
                    "denoise": bool(getattr(args, "native_enable_denoise", True)),
                }
            if meta_native.get("flashrt") or (
                meta_native.get("overlay") and meta_native.get("denoise")
            ):
                enabled = getattr(native_ex, "_stage_perf", None) if native_ex else None
                en = getattr(enabled, "enabled", None) if enabled is not None else None
                logging.warning(
                    "[summary] native/FlashRT perf 无样本（enabled=%s, native_executor=%s）。"
                    " 若 denoise 已走 FlashRT，请确认已部署含 infer.perf 的 model_optimizer 且"
                    " load_native_runtime 在 TRT 之后执行。",
                    en,
                    native_ex is not None,
                )
    except ImportError as exc:
        logging.warning("[summary] model_optimizer.infer.perf 不可用: %s", exc)

    print(colored("========================================", "yellow"))


__all__ = [
    "dump_perf_final_summary",
    "flow_match_noise_for_chunk",
    "process_infer_chunk",
]
