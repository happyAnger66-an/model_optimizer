"""单路 FlashRT 推理后端。"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import InferBackend, PredictionPack, align_action_dim


class SingleFlashRtBackend(InferBackend):
    """使用 FlashRT policy adapter 进行单路推理。"""

    def __init__(self, policy_flashrt: Any) -> None:
        self._policy_flashrt = policy_flashrt

    def predict(
        self,
        obs: dict[str, Any],
        gt: np.ndarray,
        action_horizon: int,
    ) -> PredictionPack:
        t0 = time.monotonic()
        out = self._policy_flashrt.infer(obs)
        infer_ms = (time.monotonic() - t0) * 1000.0
        pred = np.asarray(out["actions"])
        pred_a, gt_a = align_action_dim(pred, gt)
        return PredictionPack(
            pred_h=pred_a[:action_horizon],
            gt_h=gt_a[:action_horizon],
            infer_ms_pt=infer_ms,
        )

