"""chunk 级 wall-time 累计与周期性打印。"""

from __future__ import annotations

from typing import Any

import numpy as np
from termcolor import colored

from .chunk_context import InferChunkContext
from .chunk_predict import ChunkPrediction
from .chunk_sample import ChunkSample
_chunk_prof: dict[str, Any] = {
    "seen": 0,
    "load_ms": [],
    "repack_ms": [],
    "predict_ms": [],
    "infer_ms_pt": [],
    "infer_ms_second": [],
    "post_ms": [],
    "total_ms": [],
}


def chunk_prof_snapshot() -> dict[str, Any]:
    return _chunk_prof


def _chunk_stats_ms(values: list[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return "n=0"
    return (
        f"n={int(arr.size)} mean={float(np.mean(arr)):.2f} "
        f"p50={float(np.percentile(arr, 50)):.2f} p90={float(np.percentile(arr, 90)):.2f} "
        f"p99={float(np.percentile(arr, 99)):.2f} ms"
    )


def _maybe_print_chunk_profile(args: Any, idx: int) -> None:
    if not bool(getattr(args, "perf_profile_chunk", True)):
        return
    seen = int(_chunk_prof["seen"])
    warmup = max(int(getattr(args, "perf_profile_warmup_chunks", 10)), 0)
    interval = max(int(getattr(args, "perf_profile_print_interval", 20)), 1)
    eff = seen - warmup
    if eff <= 0 or eff % interval != 0:
        return

    print(colored(f"[chunk-prof] idx={idx}, samples(after warmup)={eff}", "magenta"))
    for key in ("load_ms", "repack_ms", "predict_ms", "post_ms", "total_ms"):
        vals = _chunk_prof[key]
        if vals:
            print(colored(f"[chunk-prof] {key:<9} {_chunk_stats_ms(vals)}", "magenta"))

    if _chunk_prof["predict_ms"] and _chunk_prof["total_ms"]:
        p = float(np.mean(np.asarray(_chunk_prof["predict_ms"], dtype=np.float64)))
        t = float(np.mean(np.asarray(_chunk_prof["total_ms"], dtype=np.float64)))
        if t > 0:
            overhead = max(t - p, 0.0)
            print(
                colored(
                    f"[chunk-prof] python/other overhead ~= {overhead:.2f} ms ({(overhead/t)*100.0:.1f}%)",
                    "magenta",
                )
            )
    top_n = max(int(getattr(args, "perf_profile_top_n", 3)), 1)
    means: list[tuple[str, float]] = []
    for key in ("predict_ms", "post_ms", "load_ms", "repack_ms"):
        vals = _chunk_prof[key]
        if vals:
            means.append((key, float(np.mean(np.asarray(vals, dtype=np.float64)))))
    if means:
        means.sort(key=lambda x: x[1], reverse=True)
        total_mean = (
            float(np.mean(np.asarray(_chunk_prof["total_ms"], dtype=np.float64)))
            if _chunk_prof["total_ms"]
            else 0.0
        )
        for rank, (name, mean_ms) in enumerate(means[:top_n], start=1):
            ratio = (mean_ms / total_mean * 100.0) if total_mean > 0.0 else 0.0
            print(
                colored(
                    f"[chunk-prof][rank#{rank}] {name}: {mean_ms:.2f} ms ({ratio:.1f}% of total)",
                    "magenta",
                )
            )


def record_chunk_profile(
    ctx: InferChunkContext,
    sample: ChunkSample,
    pred: ChunkPrediction,
    idx: int,
) -> None:
    import time

    args = ctx.args
    if not bool(getattr(args, "perf_profile_chunk", True)):
        return

    pack = pred.pack
    tm = sample.timings
    total_ms = (time.perf_counter() - tm.chunk_t0) * 1000.0
    load_ms = (tm.t_after_load - tm.chunk_t0) * 1000.0
    repack_ms = (tm.t_after_repack - tm.t_after_load) * 1000.0
    predict_ms = (pred.t_after_predict - pred.t_predict_0) * 1000.0
    post_ms = max(total_ms - load_ms - repack_ms - predict_ms, 0.0)

    _chunk_prof["seen"] = int(_chunk_prof["seen"]) + 1
    warmup = max(int(getattr(args, "perf_profile_warmup_chunks", 10)), 0)
    if int(_chunk_prof["seen"]) > warmup:
        _chunk_prof["load_ms"].append(float(load_ms))
        _chunk_prof["repack_ms"].append(float(repack_ms))
        _chunk_prof["predict_ms"].append(float(predict_ms))
        if pack.infer_ms_pt is not None:
            _chunk_prof["infer_ms_pt"].append(float(pack.infer_ms_pt))
        if pack.infer_ms_second is not None:
            _chunk_prof["infer_ms_second"].append(float(pack.infer_ms_second))
        _chunk_prof["post_ms"].append(float(post_ms))
        _chunk_prof["total_ms"].append(float(total_ms))
    _maybe_print_chunk_profile(args, idx)
