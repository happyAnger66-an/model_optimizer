"""按 stage 采样推理耗时（PT / TRT 分路）。"""

from __future__ import annotations

from typing import Any

import numpy as np
from termcolor import colored

# 统一阶段顺序（两路对比时左侧名）。
COMPARE_STAGES: tuple[str, ...] = ("embed_prefix", "vit", "llm", "denoise", "action")


def _stats_line_ms(values: list[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return "n=0"
    return (
        f"n={int(arr.size)} mean={float(np.mean(arr)):.2f} "
        f"p50={float(np.percentile(arr, 50)):.2f} p90={float(np.percentile(arr, 90)):.2f} "
        f"p99={float(np.percentile(arr, 99)):.2f} ms"
    )


def to_ms_list(values: list[float]) -> list[float]:
    """``time_results`` 多为秒；钩子 / stage_perf 多为毫秒。自动按量级换算。"""
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    if arr.size and float(np.max(arr)) < 20.0:
        return [float(x) * 1000.0 for x in arr]
    return [float(x) for x in arr]


def time_results_ms(model: Any, key: str) -> list[float]:
    tr = getattr(model, "time_results", None) if model is not None else None
    if isinstance(tr, dict):
        vals = tr.get(key, None)
        if vals:
            return to_ms_list(list(vals))
    return []


def engine_total_ms(policy: Any, name: str) -> list[float]:
    ex = getattr(policy, "_trt_executor", None) if policy is not None else None
    engs = getattr(ex, "_trt_engines", None) if ex is not None else None
    if isinstance(engs, dict):
        eng = engs.get(name)
        tr_e = getattr(eng, "time_results", None) if eng is not None else None
        if isinstance(tr_e, dict) and tr_e.get("total"):
            return to_ms_list(list(tr_e["total"]))
    return []


def hook_ms(name: str) -> list[float]:
    try:
        from model_optimizer.infer.tensorrt.trt_hook_timer import get_trt_hook_stats_snapshot
    except ImportError:
        return []
    snap = get_trt_hook_stats_snapshot()
    vals = snap.get(name)
    return list(vals) if vals else []


def stage_perf_ms(policy: Any, key: str) -> list[float]:
    try:
        from model_optimizer.infer.perf import stage_perf_from_policy
    except ImportError:
        return []
    sp = stage_perf_from_policy(policy) if policy is not None else None
    if sp is None:
        return []
    vals = sp.values_for_key(key)
    return to_ms_list(list(vals)) if vals else []


def denoise_chunk_ms_from_profiler(model: Any, *, num_steps: int = 10) -> list[float]:
    """将 ``denoise_step`` 逐次样本折算为单次 ``sample_actions`` 的 denoise 总耗时（ms）。"""
    prof = getattr(model, "_pi05_stage_profiler", None) if model is not None else None
    if prof is None:
        return []
    steps = getattr(prof, "latencies", {}).get("denoise_step", [])
    if not steps:
        return []
    arr = np.asarray(steps, dtype=np.float64)
    n = max(int(num_steps), 1)
    if arr.size >= n and arr.size % n == 0:
        return [float(x) for x in (arr.reshape(-1, n).sum(axis=1))]
    return [float(np.mean(arr) * n)]


def _first_nonempty(*candidates: list[float]) -> list[float]:
    for c in candidates:
        if c:
            return c
    return []


def stage_samples_ms(
    *,
    policy: Any | None,
    model: Any | None,
    stage: str,
    is_trt: bool,
) -> list[float]:
    """按阶段、按路（pt / trt）取毫秒样本。"""
    if is_trt:
        if stage == "embed_prefix":
            return _first_nonempty(
                engine_total_ms(policy, "embed_prefix"),
                hook_ms("trt.embed_prefix"),
                hook_ms("trt.embed_prefix_vit_batched"),
                stage_perf_ms(policy, "embed_prefix"),
            )
        if stage == "vit":
            return _first_nonempty(
                engine_total_ms(policy, "vit"),
                hook_ms("trt.vit.get_image_features"),
            )
        if stage == "llm":
            return _first_nonempty(
                engine_total_ms(policy, "llm"),
                hook_ms("trt.llm.forward"),
            )
        if stage == "denoise":
            return _first_nonempty(
                engine_total_ms(policy, "denoise"),
                hook_ms("trt.denoise_step"),
                denoise_chunk_ms_from_profiler(model),
                hook_ms("trt.expert.forward"),
            )
        if stage == "action":
            return _first_nonempty(
                time_results_ms(model, "action"),
                stage_perf_ms(policy, "action"),
            )
        return []

    if stage == "embed_prefix":
        return _first_nonempty(
            stage_perf_ms(policy, "embed_prefix"),
            time_results_ms(model, "vit"),
        )
    if stage == "vit":
        return stage_perf_ms(policy, "vit")
    if stage == "llm":
        return _first_nonempty(
            time_results_ms(model, "llm"),
            stage_perf_ms(policy, "prefix_llm"),
        )
    if stage == "denoise":
        return _first_nonempty(
            denoise_chunk_ms_from_profiler(model),
            stage_perf_ms(policy, "denoise.total"),
            time_results_ms(model, "suffix"),
        )
    if stage == "action":
        return time_results_ms(model, "action")
    return []


def print_compare_stage_summary(
    *,
    policy: Any | None,
    model: Any | None,
    tag: str,
    is_trt: bool,
) -> None:
    """对比模式：打印一路（pt / trt）各阶段 mean/p50/p90/p99。"""
    print(colored(f"[summary:path:{tag}] --- stage breakdown ---", "yellow"))
    for stage in COMPARE_STAGES:
        ms_vals = stage_samples_ms(policy=policy, model=model, stage=stage, is_trt=is_trt)
        if not ms_vals:
            print(colored(f"[summary:path:{tag}] {stage:<12} n=0 (no samples)", "yellow"))
            continue
        print(colored(f"[summary:path:{tag}] {stage:<12} {_stats_line_ms(ms_vals)}", "yellow"))
