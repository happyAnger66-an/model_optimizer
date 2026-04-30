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
    vit_pt_trt: dict[str, Any] | None = None


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
        *,
        flow_noise: np.ndarray | None = None,
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
        *,
        flow_noise: np.ndarray | None = None,
    ) -> PredictionPack:
        del policy_trt, policy_ptq
        t0 = time.monotonic()
        out = policy.infer(obs, noise=flow_noise)
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
        *,
        flow_noise: np.ndarray | None = None,
    ) -> PredictionPack:
        del policy_ptq
        if policy_trt is None:
            raise RuntimeError("PtTrtCompareBackend 需要 policy_trt")
        t0 = time.monotonic()
        out_pt = policy.infer(obs, noise=flow_noise)
        infer_ms_pt = (time.monotonic() - t0) * 1000.0
        t0 = time.monotonic()
        out_trt = policy_trt.infer(obs, noise=flow_noise)
        infer_ms_second = (time.monotonic() - t0) * 1000.0
        pred_pt = np.asarray(out_pt["actions"])
        pred_trt_raw = np.asarray(out_trt["actions"])
        pred_a_pt, gt_a = align_action_dim(pred_pt, gt)
        pred_a_trt, _ = align_action_dim(pred_trt_raw, gt)
        vit_pack: dict[str, Any] | None = None
        # Optional: chunk-level ViT output compare (PyTorch get_image_features vs TRT engine).
        fetch_pt = getattr(policy, "_webui_fetch_vit_io_pt", None)
        fetch_trt = getattr(policy_trt, "_webui_fetch_vit_io_trt", None)
        if callable(fetch_pt) and callable(fetch_trt):
            try:
                x_pt, y_pt, s_pt = fetch_pt()
                x_trt, y_trt, s_trt = fetch_trt()
                # We expect both paths to call get_image_features once per chunk.
                if y_pt is not None and y_trt is not None and hasattr(y_pt, "detach") and hasattr(y_trt, "detach"):
                    import torch

                    a = y_pt.detach()
                    b = y_trt.detach()
                    if a.shape == b.shape:
                        diff = (a - b).to(torch.float32)
                        absd = diff.abs()
                        denom = a.abs().to(torch.float32).mean().clamp_min(1e-8)
                        vit_pack = {
                            "shape": list(a.shape),
                            "dtype_pt": str(a.dtype),
                            "dtype_trt": str(b.dtype),
                            "max_abs": float(absd.max().item()),
                            "mean_abs": float(absd.mean().item()),
                            "rmse": float(torch.sqrt((diff * diff).mean()).item()),
                            "mean_abs_rel_to_pt_mean_abs": float((absd.mean() / denom).item()),
                        }
                    else:
                        vit_pack = {
                            "shape_pt": list(getattr(a, "shape", ())),
                            "shape_trt": list(getattr(b, "shape", ())),
                            "error": "shape_mismatch",
                        }
                if vit_pack is None:
                    vit_pack = {}
                # Attach input pixel_values stats for sanity check.
                if isinstance(s_pt, dict):
                    vit_pack["input_pt"] = s_pt
                if isinstance(s_trt, dict):
                    vit_pack["input_trt"] = s_trt
            except Exception:
                vit_pack = {"error": "exception"}

        return PredictionPack(
            pred_h=pred_a_pt[:action_horizon],
            gt_h=gt_a[:action_horizon],
            pred_h_trt=pred_a_trt[:action_horizon],
            pred_h_ptq=None,
            infer_ms_pt=infer_ms_pt,
            infer_ms_second=infer_ms_second,
            vit_pt_trt=vit_pack,
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
        *,
        flow_noise: np.ndarray | None = None,
    ) -> PredictionPack:
        del policy_trt
        if policy_ptq is None:
            raise RuntimeError("PtPtqCompareBackend 需要 policy_ptq")
        t0 = time.monotonic()
        out_pt = policy.infer(obs, noise=flow_noise)
        infer_ms_pt = (time.monotonic() - t0) * 1000.0
        t0 = time.monotonic()
        out_ptq = policy_ptq.infer(obs, noise=flow_noise)
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
        *,
        flow_noise: np.ndarray | None = None,
    ) -> PredictionPack:
        del policy_ptq
        if policy_trt is None:
            raise RuntimeError("TrtOrtCompareBackend 需要 policy_trt（ORT 路）")
        t0 = time.monotonic()
        out_trt = policy.infer(obs, noise=flow_noise)
        infer_ms_pt = (time.monotonic() - t0) * 1000.0
        t0 = time.monotonic()
        out_ort = policy_trt.infer(obs, noise=flow_noise)
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
