"""单 chunk 推理时的 stage / e2e 终端 perf 打印。"""

from __future__ import annotations

from typing import Any

import numpy as np
from termcolor import colored

from .chunk_context import InferChunkContext
from .chunk_policy import policy_torch_model
from .chunk_predict import ChunkPrediction
from .chunk_stage_perf import COMPARE_STAGES, stage_samples_ms
_perf_model: Any | None = None


def _print_chunk_stages(
    pol: Any | None,
    path_tag: str,
    is_trt: bool,
    *,
    perf_on: bool,
) -> None:
    if not perf_on or pol is None:
        return
    m = policy_torch_model(pol)
    for stage in COMPARE_STAGES:
        ms_vals = stage_samples_ms(policy=pol, model=m, stage=stage, is_trt=is_trt)
        if not ms_vals:
            continue
        mean_ms = float(np.mean(np.asarray(ms_vals, dtype=np.float64)))
        std_ms = float(np.std(np.asarray(ms_vals, dtype=np.float64)))
        print(
            colored(
                f"[{path_tag}] {stage} {mean_ms:.2f} ± {std_ms:.2f} ms",
                "green",
            )
        )


def maybe_log_chunk_stage_perf(ctx: InferChunkContext, pred: ChunkPrediction, idx: int) -> None:
    global _perf_model
    pack = pred.pack
    args = ctx.args
    policy = ctx.policy
    policy_trt = ctx.policy_trt

    if _perf_model is None:
        _perf_model = policy_torch_model(policy)

    perf_on = bool(getattr(args, "trt_perf", False))
    if perf_on:
        for pol in (policy, policy_trt):
            if pol is None:
                continue
            m = policy_torch_model(pol)
            if m is not None and not getattr(m, "perf", False):
                try:
                    m.perf = True
                except Exception:
                    pass

    if policy_trt is not None:
        _print_chunk_stages(policy, "pt", is_trt=False, perf_on=perf_on)
        _print_chunk_stages(policy_trt, "trt", is_trt=True, perf_on=perf_on)
    else:
        _print_chunk_stages(
            policy,
            "pt",
            is_trt=bool(getattr(policy, "_trt_executor", None) is not None),
            perf_on=perf_on,
        )

    if perf_on and pack.infer_ms_pt:
        e2e = float(pack.infer_ms_pt) + (float(pack.infer_ms_second) if pack.infer_ms_second else 0.0)
        print(colored(f"e2e {e2e:.2f} ms (chunk idx={idx})", "green"))
