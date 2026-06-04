"""单 chunk 推理管线编排。"""

from __future__ import annotations

from typing import Any

from .chunk_context import InferChunkContext
from .chunk_eligibility import chunk_eligibility
from .chunk_media import encode_chunk_images
from .chunk_perf_log import maybe_log_chunk_stage_perf
from .chunk_predict import run_chunk_prediction
from .chunk_profile import record_chunk_profile
from .chunk_sample import load_chunk_sample
from .chunk_steps import emit_chunk_steps


def process_infer_chunk(bundle: dict[str, Any], idx: int) -> list[str]:
    """在专用推理线程中执行单段 chunk：dataset 取样 + infer + 图像编码；返回 JSON 字符串列表。"""
    ctx = InferChunkContext.from_bundle(bundle)
    ep0 = chunk_eligibility(ctx, idx)
    if ep0 is None:
        return []

    sample = load_chunk_sample(ctx, idx)
    pred = run_chunk_prediction(ctx, sample, idx)
    if pred is None:
        return []

    maybe_log_chunk_stage_perf(ctx, pred, idx)
    images = encode_chunk_images(sample.packed, ctx.args, idx)
    msgs = emit_chunk_steps(
        ctx,
        sample,
        pred.pack,
        idx=idx,
        ep0=ep0,
        images=images,
    )
    record_chunk_profile(ctx, sample, pred, idx)
    return msgs
