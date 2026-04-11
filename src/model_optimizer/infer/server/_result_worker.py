"""后处理消费者：独立线程从队列消费 ChunkPayload → 构建 StepResult。

当 ``enable_result=False`` 时不会被创建，推理路径零开销。
"""

from __future__ import annotations

import base64
import logging
import queue
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from .config import ServerConfig
from .metrics import compute_step_metrics, compute_timing
from .result import ChunkPayload, StepResult

logger = logging.getLogger(__name__)

_SENTINEL = object()

StepCallback = Callable[[StepResult], None]


class ResultWorker:
    """后处理线程：从队列消费 ChunkPayload → metrics → 图像编码 → StepResult → 回调。

    生命周期::

        worker = ResultWorker(config, on_step=my_callback)
        worker.start()
        # ... 推理线程调 worker.submit(payload) ...
        worker.stop()            # 发送 sentinel，阻塞直到队列排空
        results = worker.results  # 所有已构建的 StepResult
    """

    def __init__(
        self,
        config: ServerConfig,
        *,
        on_step: StepCallback | None = None,
        maxsize: int = 0,
    ) -> None:
        self._config = config
        self._on_step = on_step
        self._queue: queue.Queue[ChunkPayload | object] = queue.Queue(maxsize=maxsize)
        self._results: list[StepResult] = []
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def results(self) -> list[StepResult]:
        """已构建的全部 StepResult（线程安全快照）。"""
        with self._lock:
            return list(self._results)

    def start(self) -> None:
        """启动后处理线程。"""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("ResultWorker is already running")
        self._thread = threading.Thread(
            target=self._consume_loop,
            daemon=True,
            name="result_worker",
        )
        self._thread.start()

    def submit(self, payload: ChunkPayload) -> None:
        """非阻塞投递（推理线程调用）。"""
        self._queue.put(payload)

    def stop(self) -> None:
        """发送结束标记并等待后处理线程排空退出。"""
        self._queue.put(_SENTINEL)
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _consume_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            if not isinstance(item, ChunkPayload):
                continue
            try:
                steps = self._process_payload(item)
                with self._lock:
                    self._results.extend(steps)
                if self._on_step is not None:
                    for step in steps:
                        try:
                            self._on_step(step)
                        except Exception:
                            logger.exception("on_step callback error")
            except Exception:
                logger.exception("ResultWorker: failed to process chunk idx=%s", item.idx)

    def _process_payload(self, payload: ChunkPayload) -> list[StepResult]:
        """从 ChunkPayload 构建 StepResult 列表（metrics + 图像编码）。"""
        ah = payload.action_horizon
        pack = payload.pack
        cfg = self._config

        # 图像编码（仅首 step）
        images: dict[str, str] | None = None
        if payload.raw_images is not None:
            images = self._encode_images(payload.raw_images, cfg.websocket.jpeg_quality)

        steps: list[StepResult] = []
        for k in range(ah):
            g = payload.idx + k
            pred_k = pack.pred_h[k]
            gt_k = pack.gt_h[k]
            pred_trt_k = pack.pred_h_trt[k] if pack.pred_h_trt is not None else None
            pred_ptq_k = pack.pred_h_ptq[k] if pack.pred_h_ptq is not None else None

            metrics = compute_step_metrics(
                pred_k,
                gt_k,
                cfg.rel_eps,
                pred_trt=pred_trt_k,
                pred_ptq=pred_ptq_k,
            )

            timing = None
            if k == 0:
                timing = compute_timing(
                    pack.infer_ms_pt,
                    pack.infer_ms_second,
                    has_trt=pack.pred_h_trt is not None,
                    has_ptq=pack.pred_h_ptq is not None,
                )

            step = StepResult(
                episode_id=payload.episode_id,
                global_index=g,
                k_in_chunk=k,
                is_chunk_start=(k == 0),
                action_horizon=ah,
                gt_action=gt_k,
                pred_action=pred_k,
                metrics=metrics,
                pred_action_trt=pred_trt_k,
                pred_action_ptq=pred_ptq_k,
                prompt=payload.prompt if k == 0 else None,
                images=images if k == 0 else None,
                timing=timing,
            )
            steps.append(step)
        return steps

    @staticmethod
    def _encode_images(
        raw_images: dict[str, np.ndarray], jpeg_quality: int
    ) -> dict[str, str] | None:
        try:
            import cv2
        except ImportError:
            logger.warning("cv2 not available, skipping image encoding")
            return None

        result: dict[str, str] = {}
        for key, img_np in raw_images.items():
            try:
                img = np.asarray(img_np)
                # HWC uint8 normalize
                if np.issubdtype(img.dtype, np.floating):
                    img = (255.0 * img).clip(0.0, 255.0).astype(np.uint8)
                elif img.dtype != np.uint8:
                    img = img.astype(np.uint8, copy=False)
                if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
                    img = np.transpose(img, (1, 2, 0))
                bgr = img[..., ::-1]
                ok, buf = cv2.imencode(
                    ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
                )
                if ok:
                    result[key + "_jpeg_b64"] = base64.b64encode(buf.tobytes()).decode("ascii")
            except Exception as exc:
                logger.warning("Image encode failed for %s: %s", key, exc)
        return result if result else None
