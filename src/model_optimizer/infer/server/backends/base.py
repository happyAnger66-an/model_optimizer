"""推理后端抽象基类与公共数据结构。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PredictionPack:
    """单 chunk 推理结果：主预测、可选第二路预测与耗时（毫秒）。"""

    pred_h: np.ndarray
    gt_h: np.ndarray
    pred_h_trt: np.ndarray | None = None
    pred_h_ptq: np.ndarray | None = None
    infer_ms_pt: float = 0.0
    infer_ms_second: float | None = None


class InferBackend(ABC):
    """推理后端策略接口。

    每种运行模式（PyTorch / TensorRT / 各种对比模式）对应一个具体子类。
    """

    @abstractmethod
    def predict(
        self,
        obs: dict[str, Any],
        gt: np.ndarray,
        action_horizon: int,
    ) -> PredictionPack:
        """执行推理并返回对齐后的预测结果。

        Args:
            obs: 观测数据（repack 后，不含 actions）。
            gt: 标注动作数组。
            action_horizon: 动作时间步数。

        Returns:
            PredictionPack：对齐后的 pred/gt 以及可选第二路预测和耗时。
        """
        ...


def align_action_dim(
    pred: np.ndarray, gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """对齐 pred 与 gt 的动作维度（截断到较小维度）。"""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred[np.newaxis, :]
    if gt.ndim == 1:
        gt = gt[np.newaxis, :]
    d = min(pred.shape[-1], gt.shape[-1])
    return pred[..., :d], gt[..., :d]
