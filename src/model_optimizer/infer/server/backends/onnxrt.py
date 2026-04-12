"""ONNX Runtime 推理后端（单路 + PyTorch 对比）。"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import InferBackend, PredictionPack, align_action_dim


class SingleOnnxRTBackend(InferBackend):
    """使用 ONNX Runtime 引擎挂载后的策略进行单路推理。"""

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


class PtOrtCompareBackend(InferBackend):
    """双路推理：PyTorch 浮点 vs ONNX Runtime 引擎，对比两者与 GT 的误差。

    pred1 = PyTorch（主路），pred2 = ORT（第二路）。
    ORT 结果放在 ``pred_h_trt`` 字段中（复用 PredictionPack 的 trt 槽位）。
    """

    def __init__(self, policy: Any, policy_ort: Any) -> None:
        self._policy = policy
        self._policy_ort = policy_ort

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
        out_ort = self._policy_ort.infer(obs)
        infer_ms_ort = (time.monotonic() - t0) * 1000.0

        pred_pt = np.asarray(out_pt["actions"])
        pred_ort = np.asarray(out_ort["actions"])

        pred_a_pt, gt_a = align_action_dim(pred_pt, gt)
        pred_a_ort, _ = align_action_dim(pred_ort, gt)

        return PredictionPack(
            pred_h=pred_a_pt[:action_horizon],
            gt_h=gt_a[:action_horizon],
            pred_h_trt=pred_a_ort[:action_horizon],
            infer_ms_pt=infer_ms_pt,
            infer_ms_second=infer_ms_ort,
        )
