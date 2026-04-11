"""误差指标计算：MSE / MAE / 相对误差 / 逐维指标。"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_step_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    rel_eps: float = 1e-8,
    *,
    pred_trt: np.ndarray | None = None,
    pred_ptq: np.ndarray | None = None,
) -> dict[str, Any]:
    """计算单 step 的误差指标。

    Args:
        pred: 主路预测（PyTorch 浮点 / PTQ，取决于模式）。
        gt: 标注动作。
        rel_eps: 相对误差分母 eps。
        pred_trt: 可选 TensorRT 路预测。
        pred_ptq: 可选 PTQ 路预测。

    Returns:
        指标字典（与 eval_webui 协议兼容）。
    """
    diff = (pred - gt).astype(np.float64)
    diff_flat = np.ravel(diff)
    abs_diff = np.abs(diff).astype(np.float64, copy=False)
    abs_gt = np.abs(gt).astype(np.float64, copy=False)
    denom = np.maximum(abs_gt, float(rel_eps))
    rel_err = (abs_diff / denom).astype(np.float64, copy=False)

    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(abs_diff))

    metrics: dict[str, Any] = {
        "mse": mse,
        "mae": mae,
        "mse_pt": mse,
        "mae_pt": mae,
        "mae_per_dim": [float(np.abs(diff_flat[i])) for i in range(diff_flat.size)],
        "mse_per_dim": [float(diff_flat[i] ** 2) for i in range(diff_flat.size)],
    }

    if pred_trt is not None:
        diff_trt = (pred_trt - gt).astype(np.float64)
        diff_pair = (pred - pred_trt).astype(np.float64)
        metrics["mse_trt"] = float(np.mean(diff_trt ** 2))
        metrics["mae_trt"] = float(np.mean(np.abs(diff_trt)))
        metrics["mse_pt_trt"] = float(np.mean(diff_pair ** 2))
        metrics["mae_pt_trt"] = float(np.mean(np.abs(diff_pair)))
        dpair_flat = np.ravel(diff_pair)
        metrics["mae_pt_trt_per_dim"] = [
            float(np.abs(dpair_flat[i])) for i in range(dpair_flat.size)
        ]
        metrics["mse_pt_trt_per_dim"] = [
            float(dpair_flat[i] ** 2) for i in range(dpair_flat.size)
        ]

    if pred_ptq is not None:
        diff_ptq = (pred_ptq - gt).astype(np.float64)
        diff_pair_q = (pred - pred_ptq).astype(np.float64)
        metrics["mse_ptq"] = float(np.mean(diff_ptq ** 2))
        metrics["mae_ptq"] = float(np.mean(np.abs(diff_ptq)))
        metrics["mse_pt_ptq"] = float(np.mean(diff_pair_q ** 2))
        metrics["mae_pt_ptq"] = float(np.mean(np.abs(diff_pair_q)))
        dpair_q = np.ravel(diff_pair_q)
        metrics["mae_pt_ptq_per_dim"] = [
            float(np.abs(dpair_q[i])) for i in range(dpair_q.size)
        ]
        metrics["mse_pt_ptq_per_dim"] = [
            float(dpair_q[i] ** 2) for i in range(dpair_q.size)
        ]

    return metrics


def compute_timing(
    infer_ms_pt: float,
    infer_ms_second: float | None,
    *,
    has_trt: bool = False,
    has_ptq: bool = False,
) -> dict[str, float]:
    """计算推理耗时字典（仅在 chunk 首 step 使用）。"""
    timing: dict[str, float] = {}
    if infer_ms_second is not None:
        timing["infer_ms_pt"] = infer_ms_pt
        timing["infer_ms"] = infer_ms_pt + infer_ms_second
        if has_trt:
            timing["infer_ms_trt"] = infer_ms_second
        if has_ptq:
            timing["infer_ms_ptq"] = infer_ms_second
    else:
        timing["infer_ms"] = infer_ms_pt
    return timing
