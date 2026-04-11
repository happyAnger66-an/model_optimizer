"""PTQ (fake quant) + TensorRT 双路对比后端。"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import InferBackend, PredictionPack, align_action_dim


class PtqTrtCompareBackend(InferBackend):
    """双路推理：PTQ fake quant vs TensorRT 引擎。

    pred1 = PTQ（主路），pred2 = TRT（第二路）。
    """

    def __init__(self, policy_ptq: Any, policy_trt: Any) -> None:
        self._policy_ptq = policy_ptq
        self._policy_trt = policy_trt

    def predict(
        self,
        obs: dict[str, Any],
        gt: np.ndarray,
        action_horizon: int,
    ) -> PredictionPack:
        t0 = time.monotonic()
        out_ptq = self._policy_ptq.infer(obs)
        infer_ms_ptq = (time.monotonic() - t0) * 1000.0

        t0 = time.monotonic()
        out_trt = self._policy_trt.infer(obs)
        infer_ms_trt = (time.monotonic() - t0) * 1000.0

        pred_ptq = np.asarray(out_ptq["actions"])
        pred_trt = np.asarray(out_trt["actions"])

        pred_a_ptq, gt_a = align_action_dim(pred_ptq, gt)
        pred_a_trt, _ = align_action_dim(pred_trt, gt)

        return PredictionPack(
            pred_h=pred_a_ptq[:action_horizon],
            gt_h=gt_a[:action_horizon],
            pred_h_trt=pred_a_trt[:action_horizon],
            infer_ms_pt=infer_ms_ptq,
            infer_ms_second=infer_ms_trt,
        )
