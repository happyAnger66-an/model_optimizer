"""PyTorch 浮点 + PTQ (fake quant) 双路对比后端。"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import InferBackend, PredictionPack, align_action_dim


class PtPtqCompareBackend(InferBackend):
    """双路推理：PyTorch 浮点 vs PTQ fake quant，评估量化精度损失。"""

    def __init__(self, policy: Any, policy_ptq: Any) -> None:
        self._policy = policy
        self._policy_ptq = policy_ptq

    def predict(
        self,
        obs: dict[str, Any],
        gt: np.ndarray,
        action_horizon: int,
    ) -> PredictionPack:
        t0 = time.monotonic()
        out_pt = self._policy.infer(obs)
        infer_ms_pt = (time.monotonic() - t0) * 1000.0

        t0 = time.monotonic()
        out_ptq = self._policy_ptq.infer(obs)
        infer_ms_ptq = (time.monotonic() - t0) * 1000.0

        pred_pt = np.asarray(out_pt["actions"])
        pred_ptq = np.asarray(out_ptq["actions"])

        pred_a_pt, gt_a = align_action_dim(pred_pt, gt)
        pred_a_ptq, _ = align_action_dim(pred_ptq, gt)

        return PredictionPack(
            pred_h=pred_a_pt[:action_horizon],
            gt_h=gt_a[:action_horizon],
            pred_h_ptq=pred_a_ptq[:action_horizon],
            infer_ms_pt=infer_ms_pt,
            infer_ms_second=infer_ms_ptq,
        )
