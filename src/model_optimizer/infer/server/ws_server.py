"""WebSocket 流式推理服务 — 在 InferServer 之上包装 WebSocket 推送。

推理与后处理解耦：

- 推理线程调 ``infer_chunk`` 拿轻量 ``ChunkPayload``
- 后处理（metrics/图像编码/JSON 序列化）在 ws 推送侧完成，不阻塞推理
- 通过 Janus 队列桥接推理线程（sync）与 asyncio 事件循环（async）

用法::

    from model_optimizer.infer.server import WebSocketInferServer, load_config

    config = load_config("my_config.json")
    ws = WebSocketInferServer(config)
    ws.run()  # 阻塞，直到推理完毕
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import uuid
from collections import deque
from typing import Any, Deque

import numpy as np

from .config import ServerConfig
from .metrics import compute_step_metrics, compute_timing
from .result import ChunkPayload
from .server import InferServer

logger = logging.getLogger(__name__)


# ── Protocol helpers ──


def _event_to_json(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False, separators=(",", ":"))


_LOADING_META_MSG = _event_to_json(
    {
        "type": "meta",
        "phase": "loading",
        "message": "服务端正在初始化，完成后下发完整 meta 与 step 流。",
    }
)


def _encode_image_b64(img_np: np.ndarray, jpeg_quality: int) -> str | None:
    """将原始 numpy 图像编码为 JPEG base64。"""
    try:
        import cv2
    except ImportError:
        return None
    img = np.asarray(img_np)
    if np.issubdtype(img.dtype, np.floating):
        img = (255.0 * img).clip(0.0, 255.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8, copy=False)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    bgr = img[..., ::-1]
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _payload_to_step_events(payload: ChunkPayload, run_id: str, config: ServerConfig) -> list[str]:
    """将 ChunkPayload 后处理为协议 step 事件 JSON 列表。

    在 ws 推送线程中调用，不影响推理线程。
    """
    ah = payload.action_horizon
    pack = payload.pack

    # 图像编码（仅首 step）
    images: dict[str, str] | None = None
    if payload.raw_images is not None:
        images = {}
        for key, img_np in payload.raw_images.items():
            b64 = _encode_image_b64(img_np, config.websocket.jpeg_quality)
            if b64 is not None:
                images[key + "_jpeg_b64"] = b64
        if not images:
            images = None

    msgs: list[str] = []
    for k in range(ah):
        g = payload.idx + k
        pred_k = pack.pred_h[k]
        gt_k = pack.gt_h[k]
        pred_trt_k = pack.pred_h_trt[k] if pack.pred_h_trt is not None else None
        pred_ptq_k = pack.pred_h_ptq[k] if pack.pred_h_ptq is not None else None

        metrics = compute_step_metrics(
            pred_k, gt_k, config.rel_eps,
            pred_trt=pred_trt_k, pred_ptq=pred_ptq_k,
        )

        timing = None
        if k == 0:
            timing = compute_timing(
                pack.infer_ms_pt, pack.infer_ms_second,
                has_trt=pack.pred_h_trt is not None,
                has_ptq=pack.pred_h_ptq is not None,
            )

        event: dict[str, Any] = {
            "type": "step",
            "run_id": run_id,
            "episode_id": payload.episode_id,
            "global_index": g,
            "k_in_chunk": k,
            "is_chunk_start": (k == 0),
            "action_horizon": ah,
            "prompt": payload.prompt if k == 0 else None,
            "gt_action": [float(x) for x in gt_k.astype(np.float64).ravel()],
            "pred_action": [float(x) for x in pred_k.astype(np.float64).ravel()],
            "metrics": metrics,
            "images": images if k == 0 else None,
            "server_timing": timing,
        }
        if pred_trt_k is not None:
            event["pred_action_trt"] = [float(x) for x in pred_trt_k.astype(np.float64).ravel()]
        if pred_ptq_k is not None:
            event["pred_action_ptq"] = [float(x) for x in pred_ptq_k.astype(np.float64).ravel()]
        msgs.append(_event_to_json(event))
    return msgs


# ── Broadcaster ──


class _Broadcaster:
    """WebSocket 多客户端广播 + 历史回放。"""

    def __init__(self, *, history_size: int) -> None:
        self._clients: set[Any] = set()
        self._history: Deque[str] = deque(maxlen=max(history_size, 0))

    def add_history(self, msg: str) -> None:
        if self._history.maxlen and self._history.maxlen > 0:
            self._history.append(msg)

    async def register(self, ws: Any) -> None:
        self._clients.add(ws)

    async def unregister(self, ws: Any) -> None:
        self._clients.discard(ws)

    async def send_history(self, ws: Any) -> None:
        for msg in list(self._history):
            await ws.send(msg)

    async def broadcast(self, msg: str) -> None:
        if not self._clients:
            return
        dead: list[Any] = []
        for ws in list(self._clients):
            try:
                await ws.send(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.unregister(ws)

    def snapshot_clients(self) -> list[Any]:
        return list(self._clients)


# ── Outbound bridge (sync thread → async loop) ──


class _OutboundMessage:
    __slots__ = ("text", "add_history")

    def __init__(self, text: str, add_history: bool = True) -> None:
        self.text = text
        self.add_history = add_history


class _OutboundStop:
    pass


_STOP = _OutboundStop()


class _JanusBridge:
    """推理线程 sync_emit → asyncio drain → broadcaster。"""

    def __init__(self, broadcaster: _Broadcaster) -> None:
        import janus

        self._queue: janus.Queue[Any] = janus.Queue(0)
        self._broadcaster = broadcaster

    def sync_emit(self, text: str, *, add_history: bool = True) -> None:
        self._queue.sync_q.put(_OutboundMessage(text, add_history))

    def sync_close(self) -> None:
        self._queue.sync_q.put(_STOP)

    async def drain(self) -> None:
        try:
            while True:
                item = await self._queue.async_q.get()
                if isinstance(item, _OutboundStop):
                    break
                if not isinstance(item, _OutboundMessage):
                    continue
                if item.add_history:
                    self._broadcaster.add_history(item.text)
                await self._broadcaster.broadcast(item.text)
        finally:
            self._queue.close()
            await self._queue.wait_closed()


# ── Control commands ──


def _parse_control(msg: Any) -> str | None:
    if not isinstance(msg, dict) or msg.get("type") != "control":
        return None
    act = msg.get("action")
    if act in ("pause", "resume"):
        return act
    return None


# ── GPU stats ──


def _sample_gpu_util(device_index: int = 0) -> dict[str, Any] | None:
    import subprocess

    try:
        proc = subprocess.run(
            [
                "nvidia-smi", "-i", str(device_index),
                "--query-gpu=utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=3.0, check=False,
        )
        if proc.returncode != 0:
            return None
        line = proc.stdout.strip().splitlines()[0] if proc.stdout else ""
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            return None
        return {"gpu_util_pct": float(parts[0]), "mem_util_pct": float(parts[1])}
    except (FileNotFoundError, Exception):
        return None


def _parse_gpu_index(device: str | None) -> int:
    if device is None:
        return 0
    import re

    s = str(device).strip().lower()
    m = re.match(r"cuda\s*:\s*(\d+)\s*$", s)
    return int(m.group(1)) if m else 0


# ── Main server ──


class WebSocketInferServer:
    """基于 WebSocket 的流式推理服务。

    推理线程只调 ``infer_chunk`` 拿 ``ChunkPayload``；
    后处理（metrics + 图像编码 + JSON 序列化）在推送线程中完成，不阻塞推理。
    """

    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._infer_server = InferServer(config)

    def run(self) -> None:
        """同步入口：启动 asyncio 事件循环（阻塞直到推理完毕）。"""
        asyncio.run(self.start())

    async def start(self) -> None:
        """异步入口：启动 WebSocket 服务、推理线程与 GPU 统计。"""
        import websockets.asyncio.server as _server
        import websockets.exceptions as _wsex

        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

        cfg = self._config
        ws_cfg = cfg.websocket
        run_id = uuid.uuid4().hex[:12]
        broadcaster = _Broadcaster(history_size=ws_cfg.history_size)
        meta_ready: dict[str, Any] = {"msg": None}
        infer_paused = threading.Event()
        bridge = _JanusBridge(broadcaster)

        async def handler(ws: Any) -> None:
            path = getattr(ws, "path", None)
            if path not in (None, ws_cfg.path):
                await ws.close(code=1008, reason="invalid path")
                return
            await broadcaster.register(ws)
            try:
                if meta_ready["msg"] is None:
                    await ws.send(_LOADING_META_MSG)
                else:
                    await ws.send(meta_ready["msg"])
                    if ws_cfg.history_size > 0:
                        await broadcaster.send_history(ws)
                await ws.send(
                    _event_to_json(
                        {"type": "control_ack", "action": "sync", "paused": infer_paused.is_set()}
                    )
                )
                try:
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        cmd = _parse_control(msg)
                        if cmd == "pause":
                            infer_paused.set()
                            await ws.send(
                                _event_to_json({"type": "control_ack", "action": "pause", "paused": True})
                            )
                        elif cmd == "resume":
                            infer_paused.clear()
                            await ws.send(
                                _event_to_json({"type": "control_ack", "action": "resume", "paused": False})
                            )
                except (_wsex.ConnectionClosedOK, _wsex.ConnectionClosedError):
                    pass
            finally:
                await broadcaster.unregister(ws)

        loop = asyncio.get_running_loop()
        pump_task = asyncio.create_task(bridge.drain(), name="outbound_pump")

        # GPU stats coroutine
        async def gpu_stats_loop() -> None:
            interval = float(ws_cfg.gpu_stats_interval)
            if interval <= 0:
                return
            interval = max(0.2, interval)
            dev_idx = _parse_gpu_index(cfg.device)
            while True:
                try:
                    await asyncio.wait_for(pump_task, timeout=interval)
                    return
                except asyncio.TimeoutError:
                    pass
                try:
                    stats = await loop.run_in_executor(None, _sample_gpu_util, dev_idx)
                except Exception:
                    continue
                if stats is None:
                    continue
                msg = _event_to_json({
                    "type": "gpu_stats",
                    "run_id": run_id,
                    "device_index": dev_idx,
                    **stats,
                })
                broadcaster.add_history(msg)
                await broadcaster.broadcast(msg)

        # Infer + post-process worker thread
        def infer_worker() -> None:
            try:
                import time

                def on_progress(stage: str, message: str) -> None:
                    bridge.sync_emit(
                        _event_to_json({
                            "type": "server_progress",
                            "run_id": run_id,
                            "stage": stage,
                            "message": message,
                        }),
                        add_history=False,
                    )

                self._infer_server.load(on_progress=on_progress)
                meta = dict(self._infer_server.meta)
                meta["type"] = "meta"
                meta["run_id"] = run_id
                meta_msg = _event_to_json(meta)
                meta_ready["msg"] = meta_msg
                bridge.sync_emit(meta_msg)

                min_period = (1.0 / ws_cfg.max_fps) if ws_cfg.max_fps > 0 else 0.0
                last_t = 0.0

                for idx in range(self._infer_server.start_index, self._infer_server.end):
                    while infer_paused.is_set():
                        time.sleep(0.05)

                    # 纯推理 → 轻量 ChunkPayload
                    payload = self._infer_server.infer_chunk(idx)
                    if payload is None:
                        continue

                    # 后处理（metrics + 图像编码 + JSON 序列化）在此线程完成，
                    # 与 GPU 推理不竞争（推理已在 predict() 内完成）
                    step_msgs = _payload_to_step_events(payload, run_id, cfg)
                    for msg in step_msgs:
                        bridge.sync_emit(msg)
                        if min_period > 0:
                            now = time.monotonic()
                            dt = now - last_t
                            if dt < min_period:
                                time.sleep(min_period - dt)
                            last_t = time.monotonic()

                bridge.sync_emit(_event_to_json({
                    "type": "done",
                    "phase": "finished",
                    "run_id": run_id,
                    "message": "推理完毕；server 即将关闭。",
                    "start_index": self._infer_server.start_index,
                    "end_index_exclusive": self._infer_server.end,
                }))
            except Exception as exc:
                logger.exception("Infer worker failed: %s", exc)
                try:
                    bridge.sync_emit(_event_to_json({"type": "error", "run_id": run_id, "message": str(exc)}))
                except Exception:
                    pass
            finally:
                self._infer_server.close()
                bridge.sync_close()

        logger.info(
            "WebSocket listening: %s:%s%s  run_id=%s",
            ws_cfg.host, ws_cfg.port, ws_cfg.path, run_id,
        )

        async with _server.serve(
            handler, ws_cfg.host, ws_cfg.port,
            compression=None, max_size=None,
        ) as server:
            gpu_task = None
            if ws_cfg.gpu_stats_interval > 0:
                gpu_task = asyncio.create_task(gpu_stats_loop(), name="gpu_stats")

            threading.Thread(target=infer_worker, daemon=True, name="infer_worker").start()

            try:
                await pump_task
            finally:
                clients = broadcaster.snapshot_clients()
                if clients:
                    await asyncio.gather(
                        *[c.close(code=1001, reason="server finished") for c in clients],
                        return_exceptions=True,
                    )
                if gpu_task is not None:
                    gpu_task.cancel()
                    try:
                        await gpu_task
                    except asyncio.CancelledError:
                        pass
                await asyncio.sleep(0.25)
                server.close()
                await server.wait_closed()
                logger.info("WebSocket server closed, run_id=%s", run_id)
