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
                calls_pt = fetch_pt()
                calls_trt = fetch_trt()
                if not isinstance(calls_pt, list) or not isinstance(calls_trt, list):
                    raise TypeError("vit tap must return list[call]")
                n = min(len(calls_pt), len(calls_trt))
                if n <= 0:
                    vit_pack = {"error": "no_calls"}
                else:
                    import torch

                    def _tstats(t):
                        if t is None or not hasattr(t, "detach"):
                            return None
                        try:
                            tt = t.detach()
                            tf = tt.to(torch.float32)
                            return {
                                "shape": list(tt.shape),
                                "dtype": str(tt.dtype),
                                "min": float(tf.amin().item()),
                                "max": float(tf.amax().item()),
                                "mean": float(tf.mean().item()),
                                "std": float(tf.std(unbiased=False).item()),
                            }
                        except Exception:
                            return None

                    per_call = []
                    max_abs_overall = 0.0
                    mean_abs_sum = 0.0
                    rmse_sum = 0.0
                    rel_sum = 0.0
                    rel_n = 0
                    shape_ref = None
                    dtype_pt = None
                    dtype_trt = None
                    for i in range(n):
                        a = calls_pt[i].get("out", None)
                        b = calls_trt[i].get("out", None)
                        if a is None or b is None:
                            per_call.append({"i": i, "error": "missing_out"})
                            continue
                        if hasattr(a, "detach"):
                            a = a.detach()
                        if hasattr(b, "detach"):
                            b = b.detach()
                        if getattr(a, "shape", None) != getattr(b, "shape", None):
                            per_call.append(
                                {
                                    "i": i,
                                    "error": "shape_mismatch",
                                    "shape_pt": list(getattr(a, "shape", ())),
                                    "shape_trt": list(getattr(b, "shape", ())),
                                }
                            )
                            continue
                        diff = (a - b).to(torch.float32)
                        absd = diff.abs()
                        denom = a.abs().to(torch.float32).mean().clamp_min(1e-8)
                        max_abs_i = float(absd.max().item())
                        mean_abs_i = float(absd.mean().item())
                        rmse_i = float(torch.sqrt((diff * diff).mean()).item())
                        rel_i = float((absd.mean() / denom).item())
                        # Heuristic: check whether scale mismatch equals sqrt(hidden_size)=sqrt(D).
                        d_last = int(a.shape[-1]) if hasattr(a, "shape") and len(a.shape) > 0 else 0
                        scale = float(d_last**0.5) if d_last > 0 else 1.0
                        a_scaled = a / scale if scale and scale > 0 else a
                        diff_s = (a_scaled - b).to(torch.float32)
                        absd_s = diff_s.abs()
                        denom_s = a_scaled.abs().to(torch.float32).mean().clamp_min(1e-8)
                        mean_abs_s = float(absd_s.mean().item())
                        rmse_s = float(torch.sqrt((diff_s * diff_s).mean()).item())
                        rel_s = float((absd_s.mean() / denom_s).item())
                        per_call.append(
                            {
                                "i": i,
                                "max_abs": max_abs_i,
                                "mean_abs": mean_abs_i,
                                "rmse": rmse_i,
                                "rel": rel_i,
                                "scale_sqrt_d": float(scale),
                                "mean_abs_scaled": mean_abs_s,
                                "rmse_scaled": rmse_s,
                                "rel_scaled": rel_s,
                                "input_pt": calls_pt[i].get("in_stats", None),
                                "input_trt": calls_trt[i].get("in_stats", None),
                                "out_pt": _tstats(a),
                                "out_trt": _tstats(b),
                            }
                        )
                        max_abs_overall = max(max_abs_overall, max_abs_i)
                        mean_abs_sum += mean_abs_i
                        rmse_sum += rmse_i
                        rel_sum += rel_i
                        rel_n += 1
                        if shape_ref is None:
                            shape_ref = list(getattr(a, "shape", ()))
                            dtype_pt = str(getattr(a, "dtype", ""))
                            dtype_trt = str(getattr(b, "dtype", ""))
                    vit_pack = {
                        "calls": per_call,
                        "num_calls": n,
                    }
                    if rel_n > 0:
                        vit_pack.update(
                            {
                                "shape": shape_ref,
                                "dtype_pt": dtype_pt,
                                "dtype_trt": dtype_trt,
                                "max_abs": float(max_abs_overall),
                                "mean_abs": float(mean_abs_sum / rel_n),
                                "rmse": float(rmse_sum / rel_n),
                                "mean_abs_rel_to_pt_mean_abs": float(rel_sum / rel_n),
                            }
                        )
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
