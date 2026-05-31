"""单路 Native decoder 推理后端。"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import InferBackend, PredictionPack, align_action_dim


class SingleNativeBackend(InferBackend):
    """使用 Native executor 挂载后的策略进行单路推理。"""

    def __init__(self, policy: Any) -> None:
        self._policy = policy

    def predict(
        self,
        obs: dict[str, Any],
        gt: np.ndarray,
        action_horizon: int,
    ) -> PredictionPack:
        t0 = time.monotonic()
        out = self._policy.infer(obs)
        infer_ms = (time.monotonic() - t0) * 1000.0

        pred = np.asarray(out["actions"])
        pred_a, gt_a = align_action_dim(pred, gt)
        return PredictionPack(
            pred_h=pred_a[:action_horizon],
            gt_h=gt_a[:action_horizon],
            infer_ms_pt=infer_ms,
        )

