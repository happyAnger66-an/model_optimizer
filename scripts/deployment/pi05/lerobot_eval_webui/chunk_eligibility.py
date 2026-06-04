"""chunk 是否应在当前 index 上执行推理。"""

from __future__ import annotations

from .chunk_context import InferChunkContext


def chunk_eligibility(ctx: InferChunkContext, idx: int) -> int | None:
    """可处理则返回 episode_id，否则 ``None``。"""
    ah = ctx.action_horizon
    stride_ok = (idx - ctx.start_index) % ah == 0
    chunk_fits = idx + ah <= ctx.n and idx + ah <= ctx.end
    if not (stride_ok and chunk_fits):
        return None

    ep0 = int(ctx.ep_per_frame[idx])
    ep_last = int(ctx.ep_per_frame[idx + ah - 1])
    if ep0 != ep_last:
        return None
    return ep0
