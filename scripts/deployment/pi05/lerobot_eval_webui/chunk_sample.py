"""dataset 取样与 repack。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .chunk_context import InferChunkContext
from .dataset import tree_to_numpy


@dataclass
class ChunkTimings:
    chunk_t0: float
    t_after_load: float
    t_after_repack: float


@dataclass
class ChunkSample:
    packed: dict[str, Any]
    gt_h: np.ndarray
    obs: dict[str, Any]
    prompt: str | None
    timings: ChunkTimings


def load_chunk_sample(ctx: InferChunkContext, idx: int) -> ChunkSample:
    chunk_t0 = time.perf_counter()
    raw = tree_to_numpy(ctx.dataset[idx])
    t_after_load = time.perf_counter()
    packed = ctx.repack_fn(dict(raw))
    t_after_repack = time.perf_counter()
    if "actions" not in packed:
        raise KeyError("repack 后缺少 actions，请检查数据配置与数据集列名是否一致。")

    gt = np.asarray(packed["actions"])
    obs = {k: v for k, v in packed.items() if k != "actions"}

    prompt: str | None = None
    if "prompt" in packed:
        try:
            prompt = str(packed["prompt"])
        except Exception:
            prompt = None

    return ChunkSample(
        packed=packed,
        gt_h=gt,
        obs=obs,
        prompt=prompt,
        timings=ChunkTimings(
            chunk_t0=chunk_t0,
            t_after_load=t_after_load,
            t_after_repack=t_after_repack,
        ),
    )
