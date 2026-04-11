"""InferenceState — 异步推理会话管理。

封装 ActionBuffer + 推理工作线程 + 请求/结果队列，
实现推理与控制循环解耦的异步执行模式。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

import numpy as np

from .action_buffer import ActionBuffer
from .config import InferTaskConfig
from .policy_client import PolicyInferenceClient

logger = logging.getLogger(__name__)


class InferenceState:
    """异步推理会话状态。

    管理 ActionBuffer、推理工作线程和通信队列。

    Args:
        config: 推理任务配置。
        policy_client: 已连接的策略客户端。
        action_dim: 动作维度。
    """

    def __init__(
        self,
        config: InferTaskConfig,
        policy_client: PolicyInferenceClient,
        action_dim: int,
    ) -> None:
        self._config = config
        self._policy_client = policy_client
        self._action_dim = action_dim

        self.buffer = ActionBuffer(
            action_horizon=config.action_horizon,
            action_dim=action_dim,
            fusion_type=config.fusion_type,
            fusion_start_weight=config.fusion_start_weight,
            fusion_end_weight=config.fusion_end_weight,
            fusion_exp_decay=config.fusion_exp_decay,
        )

        # 推理请求队列（maxsize=1：推理跟不上时丢弃旧请求）
        self.input_queue: queue.Queue[tuple[dict[str, Any], int]] = queue.Queue(maxsize=1)
        # 推理结果队列
        self.output_queue: queue.Queue[tuple[np.ndarray, int]] = queue.Queue()

        self.global_step_id: int = 0

        # 记录最近一次请求是否携带了 action_prefix（用于决定融合策略）
        self._last_request_had_prefix: bool = False

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self, warmup_obs: dict[str, Any] | None = None) -> None:
        """启动推理工作线程。

        Args:
            warmup_obs: 可选的 warmup 观测，用于预填充 ActionBuffer。
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Inference thread already running")

        self._stop_event.clear()

        if warmup_obs is not None and self._config.warmup:
            self._do_warmup(warmup_obs)

        self._thread = threading.Thread(
            target=self._inference_worker,
            daemon=True,
            name="inference_worker",
        )
        self._thread.start()
        logger.info("Inference worker thread started")

    def reset(self, warmup_obs: dict[str, Any] | None = None) -> None:
        """重置会话状态（新 prompt 到达时调用），不重启推理线程。

        清空 ActionBuffer、通信队列和步数计数器，可选执行 warmup。
        推理工作线程保持运行——只是清掉旧数据，等待新请求。

        Args:
            warmup_obs: 可选的 warmup 观测，用于预填充 ActionBuffer。
        """
        # 清空队列中的残留请求和结果
        _drain_queue(self.input_queue)
        _drain_queue(self.output_queue)

        # 重置 buffer
        self.buffer.reset()

        # 重置计数器
        self.global_step_id = 0
        self._last_request_had_prefix = False

        if warmup_obs is not None and self._config.warmup:
            self._do_warmup(warmup_obs)

        logger.info("InferenceState reset for new prompt")

    def stop(self) -> None:
        """停止推理工作线程。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Inference worker thread stopped")

    def submit_request(
        self, obs_dict: dict[str, Any], step_id: int, *, has_action_prefix: bool = False,
    ) -> None:
        """投递推理请求，队列满时丢弃最旧请求。

        Args:
            obs_dict: 观测字典。
            step_id: 当前全局步数。
            has_action_prefix: 本次请求是否携带了 RTC action_prefix。
                用于 ``check_result`` 决定是否启用融合：
                有 prefix 时不融合（与原版 ``enable_fusion=without_action_prefix`` 一致）。
        """
        self._last_request_had_prefix = has_action_prefix
        queue_put_replace_oldest(self.input_queue, (obs_dict, step_id))

    def check_result(self) -> bool:
        """非阻塞检查推理结果并更新 ActionBuffer。

        融合策略与原版 ``record_unified.py`` 一致：
        - 请求携带了 action_prefix（RTC 模式）→ **不融合**（锁定前缀已保证连续性）
        - 请求未携带 action_prefix → 按配置的 fusion_type 融合

        Returns:
            是否有新结果。
        """
        try:
            chunk, start_step = self.output_queue.get_nowait()
        except queue.Empty:
            return False

        # 与原版逻辑一致：without_action_prefix = infer_delay == 0
        enable_fusion = not self._last_request_had_prefix
        self.buffer.update_from_inference(
            chunk,
            start_step,
            self.global_step_id,
            enable_fusion=enable_fusion,
        )
        return True

    def _do_warmup(self, obs_dict: dict[str, Any]) -> None:
        """同步执行一次 warmup 推理，填充 ActionBuffer 初始值。"""
        logger.info("Running warmup inference …")
        t0 = time.monotonic()
        actions = self._policy_client.get_action(obs_dict)
        dt = (time.monotonic() - t0) * 1000
        logger.info("Warmup done in %.1f ms, actions shape: %s", dt, actions.shape)
        self.buffer.update_from_inference(
            actions, start_step=0, current_step=0, enable_fusion=False,
        )

    def _inference_worker(self) -> None:
        """推理工作线程主循环。"""
        while not self._stop_event.is_set():
            try:
                obs_dict, step_id = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                t0 = time.monotonic()
                actions = self._policy_client.get_action(obs_dict)
                dt = (time.monotonic() - t0) * 1000
                logger.debug("Inference step %d done in %.1f ms", step_id, dt)
                self.output_queue.put((actions, step_id))
            except Exception:
                logger.exception("Inference failed at step %d", step_id)


def _drain_queue(q: queue.Queue) -> None:
    """清空队列中的所有元素。"""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            break


def queue_put_replace_oldest(q: queue.Queue, item: Any) -> None:
    """向队列放入 item，队列满时丢弃最旧的元素。"""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass
