"""多路推理策略（Strategy）：PyTorch 单路 / PyTorch+TensorRT / PyTorch+PTQ / TensorRT+ONNX。"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from .action_align import align_action_dim


@dataclass
class PredictionPack:
    """单 chunk 对齐后的主预测、可选第二路预测与耗时（毫秒）。"""

    pred_h: np.ndarray
    gt_h: np.ndarray
    pred_h_trt: np.ndarray | None
    pred_h_ptq: np.ndarray | None
    infer_ms_pt: float
    infer_ms_second: float | None


class InferBackend(ABC):
    @abstractmethod
    def predict(
        self,
        policy: Any,
        policy_trt: Any | None,
        policy_ptq: Any | None,
        obs: dict[str, Any],
        gt: np.ndarray,
        action_horizon: int,
    ) -> PredictionPack:
        ...


class SingleTorchBackend(InferBackend):
    def predict(
        self,
        policy: Any,
        policy_trt: Any | None,
        policy_ptq: Any | None,
        obs: dict[str, Any],
        gt: np.ndarray,
        action_horizon: int,
    ) -> PredictionPack:
        del policy_trt, policy_ptq
        t0 = time.monotonic()
        out = policy.infer(obs)
        infer_ms_pt = (time.monotonic() - t0) * 1000.0
        pred = np.asarray(out["actions"])
        pred_a, gt_a = align_action_dim(pred, gt)
        return PredictionPack(
            pred_h=pred_a[:action_horizon],
            gt_h=gt_a[:action_horizon],
            pred_h_trt=None,
            pred_h_ptq=None,
            infer_ms_pt=infer_ms_pt,
            infer_ms_second=None,
        )


class PtTrtCompareBackend(InferBackend):
    def predict(
        self,
        policy: Any,
        policy_trt: Any | None,
        policy_ptq: Any | None,
        obs: dict[str, Any],
        gt: np.ndarray,
        action_horizon: int,
    ) -> PredictionPack:
        del policy_ptq
        if policy_trt is None:
            raise RuntimeError("PtTrtCompareBackend 需要 policy_trt")
        t0 = time.monotonic()
        out_pt = policy.infer(obs)
        infer_ms_pt = (time.monotonic() - t0) * 1000.0
        t0 = time.monotonic()
        out_trt = policy_trt.infer(obs)
        infer_ms_second = (time.monotonic() - t0) * 1000.0
        pred_pt = np.asarray(out_pt["actions"])
        pred_trt_raw = np.asarray(out_trt["actions"])
        pred_a_pt, gt_a = align_action_dim(pred_pt, gt)
        pred_a_trt, _ = align_action_dim(pred_trt_raw, gt)
        return PredictionPack(
            pred_h=pred_a_pt[:action_horizon],
            gt_h=gt_a[:action_horizon],
            pred_h_trt=pred_a_trt[:action_horizon],
            pred_h_ptq=None,
            infer_ms_pt=infer_ms_pt,
            infer_ms_second=infer_ms_second,
        )


class PtPtqCompareBackend(InferBackend):
    def predict(
        self,
        policy: Any,
        policy_trt: Any | None,
        policy_ptq: Any | None,
        obs: dict[str, Any],
        gt: np.ndarray,
        action_horizon: int,
    ) -> PredictionPack:
        del policy_trt
        if policy_ptq is None:
            raise RuntimeError("PtPtqCompareBackend 需要 policy_ptq")
        t0 = time.monotonic()
        out_pt = policy.infer(obs)
        infer_ms_pt = (time.monotonic() - t0) * 1000.0
        t0 = time.monotonic()
        out_ptq = policy_ptq.infer(obs)
        infer_ms_second = (time.monotonic() - t0) * 1000.0
        pred_pt = np.asarray(out_pt["actions"])
        pred_ptq_raw = np.asarray(out_ptq["actions"])
        pred_a_pt, gt_a = align_action_dim(pred_pt, gt)
        pred_a_ptq, _ = align_action_dim(pred_ptq_raw, gt)
        return PredictionPack(
            pred_h=pred_a_pt[:action_horizon],
            gt_h=gt_a[:action_horizon],
            pred_h_trt=None,
            pred_h_ptq=pred_a_ptq[:action_horizon],
            infer_ms_pt=infer_ms_pt,
            infer_ms_second=infer_ms_second,
        )


class TrtOrtCompareBackend(InferBackend):
    """policy 先 ``infer``，policy_trt 后 ``infer``；输出与 ``PtTrtCompareBackend`` 同一套 pred 槽位。

    用于 **TensorRT vs ONNX Runtime**（``trt_ort_compare``）或 **双 TensorRT**（``trt_trt_compare``，如 FP16 vs NVFP4）。
    """

    def predict(
        self,
        policy: Any,
        policy_trt: Any | None,
        policy_ptq: Any | None,
        obs: dict[str, Any],
        gt: np.ndarray,
        action_horizon: int,
    ) -> PredictionPack:
        del policy_ptq
        if policy_trt is None:
            raise RuntimeError("TrtOrtCompareBackend 需要 policy_trt（ORT 路）")
        t0 = time.monotonic()
        out_trt = policy.infer(obs)
        infer_ms_pt = (time.monotonic() - t0) * 1000.0
        t0 = time.monotonic()
        out_ort = policy_trt.infer(obs)
        infer_ms_second = (time.monotonic() - t0) * 1000.0
        pred_trt = np.asarray(out_trt["actions"])
        pred_ort_raw = np.asarray(out_ort["actions"])
        pred_a_trt, gt_a = align_action_dim(pred_trt, gt)
        pred_a_ort, _ = align_action_dim(pred_ort_raw, gt)
        return PredictionPack(
            pred_h=pred_a_trt[:action_horizon],
            gt_h=gt_a[:action_horizon],
            pred_h_trt=pred_a_ort[:action_horizon],
            pred_h_ptq=None,
            infer_ms_pt=infer_ms_pt,
            infer_ms_second=infer_ms_second,
        )


def select_infer_backend(bundle: dict[str, Any]) -> InferBackend:
    if bundle.get("trt_ort_compare") or bundle.get("trt_trt_compare"):
        return TrtOrtCompareBackend()
    if bundle.get("policy_trt") is not None:
        return PtTrtCompareBackend()
    if bundle.get("policy_ptq") is not None:
        return PtPtqCompareBackend()
    return SingleTorchBackend()
