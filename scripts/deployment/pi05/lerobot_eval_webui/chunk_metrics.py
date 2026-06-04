"""单 step 误差 metrics 构建与 running 统计更新。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .chunk_context import InferChunkContext, SecondaryPathStats
from .infer_backends import PredictionPack
from .running_stats import RunningPerDimPairMseStats


@dataclass
class StepMetricsState:
    """单步 TRT/PTQ 中间量（供 running 更新复用）。"""

    diff_trt: np.ndarray | None = None
    diff_ptq: np.ndarray | None = None


def _per_dim_lists(flat: np.ndarray) -> tuple[list[float], list[float]]:
    n = int(flat.size)
    return (
        [float(np.abs(flat[i])) for i in range(n)],
        [float(flat[i] * flat[i]) for i in range(n)],
    )


def _mse_mae(diff: np.ndarray) -> tuple[float, float]:
    return float(np.mean(diff**2)), float(np.mean(np.abs(diff)))


def metrics_pt_vs_gt(diff_pt: np.ndarray) -> dict[str, Any]:
    dpt_flat = np.ravel(diff_pt.astype(np.float64))
    mse_pt, mae_pt = _mse_mae(diff_pt)
    mae_per_dim, mse_per_dim = _per_dim_lists(dpt_flat)
    return {
        "mse": mse_pt,
        "mae": mae_pt,
        "mse_pt": mse_pt,
        "mae_pt": mae_pt,
        "mae_per_dim": mae_per_dim,
        "mse_per_dim": mse_per_dim,
    }


def attach_secondary_path_metrics(
    metrics: dict[str, Any],
    *,
    pred_pt_row: np.ndarray,
    pred_sec_row: np.ndarray,
    gt_row: np.ndarray,
    path: str,
    pair_mse: RunningPerDimPairMseStats | None,
) -> tuple[dict[str, Any], list[float]]:
    """``path`` 为 ``trt`` 或 ``ptq``；返回更新后的 metrics 与 secondary 预测列表。"""
    diff_sec = pred_sec_row - gt_row
    diff_pair = pred_pt_row - pred_sec_row
    mse_sec, mae_sec = _mse_mae(diff_sec)
    mse_pair, mae_pair = _mse_mae(diff_pair)
    dpair_flat = np.ravel(diff_pair.astype(np.float64))
    mae_pair_per_dim, mse_pair_per_dim = _per_dim_lists(dpair_flat)

    if path == "trt":
        metrics["mse_trt"] = mse_sec
        metrics["mae_trt"] = mae_sec
        metrics["mse_pt_trt"] = mse_pair
        metrics["mae_pt_trt"] = mae_pair
        metrics["mae_pt_trt_per_dim"] = mae_pair_per_dim
        metrics["mse_pt_trt_per_dim"] = mse_pair_per_dim
        dim_mean_key = "mse_pt_trt_dim_mean"
    else:
        metrics["mse_ptq"] = mse_sec
        metrics["mae_ptq"] = mae_sec
        metrics["mse_pt_ptq"] = mse_pair
        metrics["mae_pt_ptq"] = mae_pair
        metrics["mae_pt_ptq_per_dim"] = mae_pair_per_dim
        metrics["mse_pt_ptq_per_dim"] = mse_pair_per_dim
        dim_mean_key = "mse_pt_ptq_dim_mean"

    pred_list = [float(x) for x in pred_sec_row.astype(np.float64).tolist()]
    if pair_mse is not None:
        pair_mse.update(np.ravel(diff_pair).astype(np.float64))
        metrics[dim_mean_key] = pair_mse.mean_mse_per_dim()
    return metrics, pred_list


def attach_vit_metrics(
    metrics: dict[str, Any],
    *,
    vit_pt_trt: dict[str, Any] | None,
    running_vit: Any | None,
    k: int,
) -> dict[str, Any]:
    if not isinstance(vit_pt_trt, dict) or not vit_pt_trt or not isinstance(running_vit, object):
        return metrics
    if k == 0 and hasattr(running_vit, "update_from_pack"):
        try:
            running_vit.update_from_pack(vit_pt_trt)
        except Exception:
            pass
    cur_mean_abs = vit_pt_trt.get("mean_abs")
    cur_rmse = vit_pt_trt.get("rmse")
    if isinstance(cur_mean_abs, (int, float)):
        metrics["vit_mean_abs"] = float(cur_mean_abs)
    if isinstance(cur_rmse, (int, float)):
        metrics["vit_rmse"] = float(cur_rmse)
    if hasattr(running_vit, "mean_abs_mean"):
        try:
            v = running_vit.mean_abs_mean()
            if v is not None:
                metrics["vit_mean_abs_cum"] = float(v)
        except Exception:
            pass
    if hasattr(running_vit, "rmse_mean"):
        try:
            v = running_vit.rmse_mean()
            if v is not None:
                metrics["vit_rmse_cum"] = float(v)
        except Exception:
            pass
    if k == 0:
        metrics["vit_pt_trt"] = vit_pt_trt
    return metrics


def update_running_pt(
    ctx: InferChunkContext,
    metrics: dict[str, Any],
    *,
    diff_pt: np.ndarray,
    gt_row: np.ndarray,
    rel_err: np.ndarray,
) -> dict[str, Any]:
    abs_diff = np.abs(diff_pt).astype(np.float64, copy=False)
    ctx.running_err.update_abs_and_rel(
        abs_err_values=np.ravel(abs_diff),
        rel_err_values=np.ravel(rel_err),
    )
    ctx.per_dim_mse_pct.update(
        np.ravel(diff_pt).astype(np.float64),
        np.ravel(gt_row).astype(np.float64),
    )
    ctx.per_dim_rel_p99.update(np.ravel(rel_err))
    metrics["mse_pct_dim_mean"] = ctx.per_dim_mse_pct.mse_pct_mean()
    metrics["rel_p99_dim"] = ctx.per_dim_rel_p99.rel_p99()
    return metrics


def update_running_secondary(
    metrics: dict[str, Any],
    *,
    diff: np.ndarray,
    gt_row: np.ndarray,
    denom: np.ndarray,
    stats: SecondaryPathStats,
    suffix: str,
) -> dict[str, Any]:
    abs_diff = np.abs(diff).astype(np.float64, copy=False)
    rel_err = (abs_diff / denom).astype(np.float64, copy=False)
    stats.per_dim_mse_pct.update(
        np.ravel(diff).astype(np.float64),
        np.ravel(gt_row).astype(np.float64),
    )
    stats.per_dim_rel_p99.update(np.ravel(rel_err))
    metrics[f"mse_pct_dim_mean{suffix}"] = stats.per_dim_mse_pct.mse_pct_mean()
    metrics[f"rel_p99_dim{suffix}"] = stats.per_dim_rel_p99.rel_p99()
    return metrics


@dataclass
class StepMetricsResult:
    metrics: dict[str, Any]
    pred_trt_list: list[float] | None = None
    pred_ptq_list: list[float] | None = None


@dataclass
class StepMetricsAssembler:
    ctx: InferChunkContext
    pack: PredictionPack
    vit_pt_trt: dict[str, Any] | None = None
    _state: StepMetricsState = field(default_factory=StepMetricsState)

    def build(self, k: int, pred_row: np.ndarray, gt_row: np.ndarray) -> StepMetricsResult:
        diff_pt = pred_row - gt_row
        abs_diff = np.abs(diff_pt).astype(np.float64, copy=False)
        abs_gt = np.abs(gt_row).astype(np.float64, copy=False)
        denom = np.maximum(abs_gt, float(self.ctx.args.rel_eps))
        rel_err = (abs_diff / denom).astype(np.float64, copy=False)

        metrics = metrics_pt_vs_gt(diff_pt)
        metrics = attach_vit_metrics(
            metrics,
            vit_pt_trt=self.vit_pt_trt,
            running_vit=self.ctx.running_vit,
            k=k,
        )

        pred_trt_list: list[float] | None = None
        pred_ptq_list: list[float] | None = None

        if self.pack.pred_h_trt is not None:
            row_trt = self.pack.pred_h_trt[k]
            diff_trt = row_trt - gt_row
            self._state.diff_trt = diff_trt
            metrics, pred_trt_list = attach_secondary_path_metrics(
                metrics,
                pred_pt_row=pred_row,
                pred_sec_row=row_trt,
                gt_row=gt_row,
                path="trt",
                pair_mse=self.ctx.trt.pair_mse if self.ctx.trt else None,
            )

        if self.pack.pred_h_ptq is not None:
            row_ptq = self.pack.pred_h_ptq[k]
            diff_ptq = row_ptq - gt_row
            self._state.diff_ptq = diff_ptq
            metrics, pred_ptq_list = attach_secondary_path_metrics(
                metrics,
                pred_pt_row=pred_row,
                pred_sec_row=row_ptq,
                gt_row=gt_row,
                path="ptq",
                pair_mse=self.ctx.ptq.pair_mse if self.ctx.ptq else None,
            )

        metrics = update_running_pt(
            self.ctx,
            metrics,
            diff_pt=diff_pt,
            gt_row=gt_row,
            rel_err=rel_err,
        )

        if self.pack.pred_h_trt is not None and self.ctx.trt is not None and self._state.diff_trt is not None:
            metrics = update_running_secondary(
                metrics,
                diff=self._state.diff_trt,
                gt_row=gt_row,
                denom=denom,
                stats=self.ctx.trt,
                suffix="_trt",
            )

        if self.pack.pred_h_ptq is not None and self.ctx.ptq is not None and self._state.diff_ptq is not None:
            metrics = update_running_secondary(
                metrics,
                diff=self._state.diff_ptq,
                gt_row=gt_row,
                denom=denom,
                stats=self.ctx.ptq,
                suffix="_ptq",
            )

        return StepMetricsResult(
            metrics=metrics,
            pred_trt_list=pred_trt_list,
            pred_ptq_list=pred_ptq_list,
        )
