"""WebSocket 服务：Janus 出站泵、GPU 旁路、推理线程（与 eval_session 编排分离）。"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from typing import Any

import janus
from termcolor import colored

from .broadcaster import WebsocketBroadcaster
from .config import Args
from .control_commands import ControlPause, ControlResume, parse_control_message
from .eval_session import run_infer_worker
from .gpu_stats import effective_gpu_index, sample_gpu_util
from .hints import write_webui_server_hint
from .outbound_bridge import JanusOutboundBridge
from .protocol import LOADING_META_MSG, event_to_json


def _handshake_request_path(ws: Any) -> str | None:
    """读取 WebSocket 握手路径。

    ``websockets`` 旧版在连接对象上暴露 ``path``；13+ asyncio 服务端多为 ``request.path``。
    """
    p = getattr(ws, "path", None)
    if isinstance(p, str):
        return p
    req = getattr(ws, "request", None)
    if req is not None:
        rp = getattr(req, "path", None)
        if isinstance(rp, str):
            return rp
    return None


def _paths_equivalent(a: str, b: str) -> bool:
    """``/ws`` 与 ``/ws/`` 视为同一路径。"""
    aa = a.rstrip("/") or "/"
    bb = b.rstrip("/") or "/"
    return aa == bb


async def run_server(args: Args) -> None:
    import websockets.asyncio.server as _server
    import websockets.exceptions as _wsex

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    class _SuppressWsHandshakeNoise(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if "opening handshake failed" in record.getMessage():
                return False
            return True

    for _name in ("websockets.server", "websockets.asyncio.server", "websockets"):
        logging.getLogger(_name).addFilter(_SuppressWsHandshakeNoise())

    run_id = uuid.uuid4().hex[:12]
    broadcaster = WebsocketBroadcaster(history_size=args.history_size)
    meta_ready: dict[str, Any] = {"msg": None}
    infer_paused = threading.Event()

    qmax = int(args.outbound_queue_maxsize)
    outbound_queue: janus.Queue[Any] = janus.Queue(qmax if qmax > 0 else 0)
    bridge = JanusOutboundBridge(outbound_queue, broadcaster)

    async def handler(ws: Any) -> None:
        req_path = _handshake_request_path(ws)
        if req_path is None or not _paths_equivalent(req_path, args.path):
            await ws.close(code=1008, reason="invalid path")
            return
        await broadcaster.register(ws)
        try:
            if meta_ready["msg"] is None:
                await ws.send(LOADING_META_MSG)
            else:
                await ws.send(meta_ready["msg"])
                if args.history_size > 0:
                    await broadcaster.send_history(ws)
            await ws.send(
                event_to_json({"type": "control_ack", "action": "sync", "paused": infer_paused.is_set()})
            )
            try:
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    cmd = parse_control_message(msg)
                    if isinstance(cmd, ControlPause):
                        infer_paused.set()
                        print(colored("[infer] 收到 pause：下一 chunk 前将阻塞推理", "yellow"), flush=True)
                        await ws.send(
                            event_to_json({"type": "control_ack", "action": "pause", "paused": True})
                        )
                    elif isinstance(cmd, ControlResume):
                        infer_paused.clear()
                        print(colored("[infer] 收到 resume：继续推理", "green"), flush=True)
                        await ws.send(
                            event_to_json({"type": "control_ack", "action": "resume", "paused": False})
                        )
            except (_wsex.ConnectionClosedOK, _wsex.ConnectionClosedError):
                pass
        finally:
            await broadcaster.unregister(ws)

    loop = asyncio.get_running_loop()

    async def publish_direct(msg: str, *, add_history: bool = True) -> None:
        """GPU 统计等 asyncio 侧消息：不经 Janus，避免从协程向 sync_q 投递。"""
        if add_history:
            broadcaster.add_history(msg)
        await broadcaster.broadcast(msg)

    pump_task = asyncio.create_task(bridge.drain(), name="outbound_pump")

    async def gpu_stats_loop() -> None:
        interval = float(args.gpu_stats_interval_sec)
        if interval <= 0:
            return
        interval = max(0.2, interval)
        dev_idx = int(effective_gpu_index(args))
        while True:
            # 不能用 wait_for(pump_task, timeout=…)：超时会取消 pump_task，导致 drain 在 get() 上 CancelledError。
            done, _pending = await asyncio.wait((pump_task,), timeout=interval, return_when=asyncio.FIRST_COMPLETED)
            if pump_task in done:
                return
            try:
                stats = await loop.run_in_executor(None, sample_gpu_util, dev_idx)
            except Exception:  # pragma: no cover
                logging.exception("gpu sample run_in_executor failed")
                continue
            if stats is None:
                continue
            await publish_direct(
                event_to_json(
                    {
                        "type": "gpu_stats",
                        "run_id": run_id,
                        "device_index": dev_idx,
                        "gpu_util_pct": stats["gpu_util_pct"],
                        "mem_util_pct": stats["mem_util_pct"],
                    }
                ),
                add_history=False,
            )

    hint_url = write_webui_server_hint(args)
    print(
        colored(
            f"WebUI WebSocket 监听: {args.host}:{args.port}{args.path} · "
            f"client 默认地址（已写入 webui_client/server_hint.json）: {hint_url}",
            "green",
        ),
        flush=True,
    )

    async with _server.serve(
        handler,
        args.host,
        args.port,
        compression=None,
        max_size=None,
    ) as server:
        print(colored(f"run_id={run_id}（加载完成后会广播完整 meta）", "green"), flush=True)
        gpu_task = None
        if args.gpu_stats_interval_sec and args.gpu_stats_interval_sec > 0:
            gpu_task = asyncio.create_task(gpu_stats_loop(), name="gpu_stats")
        threading.Thread(
            target=run_infer_worker,
            kwargs={
                "args": args,
                "run_id": run_id,
                "meta_ready": meta_ready,
                "infer_paused": infer_paused,
                "bridge": bridge,
            },
            daemon=True,
            name="pi05_infer",
        ).start()
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
            print(colored("[main] 推理管线已结束，关闭 WebSocket 并退出进程", "green"), flush=True)
            server.close()
            await server.wait_closed()


def main() -> None:
    from .cli_config import parse_args_with_optional_config_file

    args = parse_args_with_optional_config_file(Args)
    asyncio.run(run_server(args))
