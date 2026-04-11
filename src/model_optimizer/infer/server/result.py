"""推理结果数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .backends.base import PredictionPack


@dataclass
class ChunkPayload:
    """推理原始产物 — 仅包含 numpy 拷贝，不做 metrics/图像编码。

    这是 ``infer_chunk`` 的直接输出，设计为尽可能轻量，
    以便推理路径零额外开销。后处理（metrics 计算、图像 JPEG 编码、
    StepResult 构建）由可选的 ``ResultWorker`` 在独立线程中异步完成。
    """

    idx: int
    episode_id: int
    action_horizon: int
    pack: PredictionPack
    prompt: str | None = None
    raw_images: dict[str, np.ndarray] | None = None
    """观测图像原始 numpy（HWC uint8 / CHW float 均可），不在推理路径做 JPEG 编码。"""


@dataclass
class StepResult:
    """单个 step（action_horizon 中的一行）的后处理结果。"""

    episode_id: int
    global_index: int
    k_in_chunk: int
    is_chunk_start: bool
    action_horizon: int
    gt_action: np.ndarray
    pred_action: np.ndarray
    metrics: dict[str, Any] = field(default_factory=dict)
    pred_action_trt: np.ndarray | None = None
    pred_action_ptq: np.ndarray | None = None
    prompt: str | None = None
    images: dict[str, str] | None = None
    timing: dict[str, float] | None = None


@dataclass
class InferResult:
    """完整推理运行的结果集合。"""

    steps: list[StepResult] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    start_index: int = 0
    end_index_exclusive: int = 0
