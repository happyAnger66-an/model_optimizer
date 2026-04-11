"""ActionBuffer — 异步推理的动作缓冲区与融合。

管理从策略推理线程返回的 action_chunk，处理：

1. **延迟对齐**：推理有延迟，返回结果时已过期的动作被丢弃。
2. **加权融合**：新旧动作在重叠区间平滑过渡。
3. **RTC 支持**：为下一次推理提供已执行动作前缀和掩码。
"""

from __future__ import annotations

import math
from collections import deque
from typing import Literal

import numpy as np


class ActionBuffer:
    """动作缓冲区，实现时间对齐和动作融合。

    Args:
        action_horizon: 每次推理返回的动作步数。
        action_dim: 单步动作维度。
        fusion_type: 融合策略 ("none"、"linear"、"exponential")。
        fusion_start_weight: linear 融合起始权重（旧值权重）。
        fusion_end_weight: linear 融合结束权重。
        fusion_exp_decay: exponential 融合衰减常数。
    """

    def __init__(
        self,
        action_horizon: int,
        action_dim: int,
        fusion_type: Literal["none", "linear", "exponential"] = "linear",
        fusion_start_weight: float = 0.9,
        fusion_end_weight: float = 0.1,
        fusion_exp_decay: float = 2.0,
    ) -> None:
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._fusion_type = fusion_type
        self._fusion_start_weight = fusion_start_weight
        self._fusion_end_weight = fusion_end_weight
        self._fusion_exp_decay = fusion_exp_decay

        # 核心缓冲区：deque of np.ndarray(action_dim,)
        self._buffer: deque[np.ndarray] = deque()
        self._available_count: int = 0

        # 记录最近一次推理写入的 action_chunk，用于 RTC
        self._last_chunk: np.ndarray | None = None
        self._last_chunk_start_step: int = 0

    @property
    def is_empty(self) -> bool:
        return self._available_count <= 0

    @property
    def available_count(self) -> int:
        return self._available_count

    def reset(self) -> None:
        """清空缓冲区。"""
        self._buffer.clear()
        self._available_count = 0
        self._last_chunk = None
        self._last_chunk_start_step = 0

    def update_from_inference(
        self,
        chunk: np.ndarray,
        start_step: int,
        current_step: int,
        enable_fusion: bool = True,
    ) -> None:
        """接收推理结果并更新缓冲区。

        Args:
            chunk: 推理返回的动作序列，shape ``(action_horizon, action_dim)``。
            start_step: 该推理请求投递时的全局步数。
            current_step: 当前全局步数。
            enable_fusion: 是否对重叠部分做加权融合。
        """
        latency_steps = max(0, current_step - start_step)
        if latency_steps >= len(chunk):
            # 全部过期，丢弃
            return

        valid = chunk[latency_steps:]  # (remaining, action_dim)

        if not enable_fusion or self._fusion_type == "none" or self.is_empty:
            # 直接替换
            self._buffer.clear()
            for action in valid:
                self._buffer.append(action.copy())
            self._available_count = len(valid)
        else:
            # 重叠部分融合，超出部分追加
            overlap = min(self._available_count, len(valid))
            buf_list = list(self._buffer)

            for i in range(overlap):
                w = self._compute_fusion_weight(i, overlap)
                buf_list[i] = buf_list[i] * w + valid[i] * (1.0 - w)

            # 超出部分追加
            for j in range(overlap, len(valid)):
                buf_list.append(valid[j].copy())

            self._buffer = deque(buf_list)
            self._available_count = len(buf_list)

        self._last_chunk = chunk.copy()
        self._last_chunk_start_step = start_step

    def get_next_action(self) -> np.ndarray:
        """弹出并返回 buffer[0]。

        Returns:
            单步动作 ``np.ndarray(action_dim,)``。

        Raises:
            IndexError: 缓冲区为空。
        """
        if self.is_empty:
            raise IndexError("ActionBuffer is empty")
        action = self._buffer.popleft()
        self._available_count -= 1
        return action

    def get_future_actions(self, delay: int) -> tuple[np.ndarray | None, np.ndarray | None, int]:
        """为 RTC 返回未来动作前缀。

        Args:
            delay: 期望的延迟步数。

        Returns:
            ``(action_prefix, action_mask, actual_delay)``：
            - action_prefix: ``(delay, action_dim)`` 或 ``None``（无可用数据）
            - action_mask: ``(delay,)`` bool 数组
            - actual_delay: 实际可提供的延迟步数
        """
        if self.is_empty or delay <= 0:
            return None, None, 0

        actual_delay = min(delay, self._available_count)
        prefix = np.zeros((delay, self._action_dim), dtype=np.float32)
        mask = np.zeros(delay, dtype=bool)

        buf_list = list(self._buffer)
        for i in range(actual_delay):
            prefix[i] = buf_list[i]
            mask[i] = True

        return prefix, mask, actual_delay

    def _compute_fusion_weight(self, i: int, overlap: int) -> float:
        """计算第 i 个重叠位置的旧值权重。

        Args:
            i: 重叠区间内的索引 (0-based)。
            overlap: 总重叠长度。

        Returns:
            旧值的权重 w，新值权重为 (1 - w)。
        """
        if self._fusion_type == "linear":
            if overlap <= 1:
                return self._fusion_start_weight
            t = i / (overlap - 1)
            return self._fusion_start_weight + t * (self._fusion_end_weight - self._fusion_start_weight)
        elif self._fusion_type == "exponential":
            return math.exp(-(i + 0.5) / self._fusion_exp_decay)
        else:
            return 0.0
