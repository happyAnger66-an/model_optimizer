"""WebSocket 服务、推理线程与 pause/resume 控制。"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from typing import Any

import tyro
from termcolor import colored

from .broadcaster import WebsocketBroadcaster
from .bundle import load_infer_bundle
from .calib import stop_pi05_calib_collectors
from .chunk_infer import process_infer_chunk
from .config import Args
from .gpu_stats import effective_gpu_index, sample_gpu_util
from .hints import write_webui_server_hint
from .protocol import LOADING_META_MSG, event_to_json


async def run_server(args: Args) -> None:
    import websockets.asyncio.server as _server
    import websockets.exceptions as _wsex

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # 对端在握手完成前断开时，websockets 会打 ERROR + 堆栈（ConnectionClosedError: no close frame…）。
    # 常见于：浏览器/工具误连、端口扫描、HTTP 探活、错误 path。与推理逻辑无关，故压低噪声。
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

    async def handler(ws: Any) -> None:
        if getattr(ws, "path", None) not in (None, args.path):
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
                    if not isinstance(msg, dict) or msg.get("type") != "control":
                        continue
                    act = msg.get("action")
                    if act == "pause":
                        infer_paused.set()
                        print(colored("[infer] 收到 pause：下一 chunk 前将阻塞推理", "yellow"), flush=True)
                        await ws.send(
                            event_to_json({"type": "control_ack", "action": "pause", "paused": True})
                        )
                    elif act == "resume":
                        infer_paused.clear()
                        print(colored("[infer] 收到 resume：继续推理", "green"), flush=True)
                        await ws.send(
                            event_to_json({"type": "control_ack", "action": "resume", "paused": False})
                        )
            except (_wsex.ConnectionClosedOK, _wsex.ConnectionClosedError):
                # 正常：server 即将退出或 client 断开时可能出现 close handshake 不完整
                pass
        finally:
            await broadcaster.unregister(ws)

    loop = asyncio.get_running_loop()
    infer_done = asyncio.Event()

    def signal_infer_complete() -> None:
        loop.call_soon_threadsafe(infer_done.set)

    async def publish(msg: str, *, add_history: bool = True) -> None:
        if add_history:
            broadcaster.add_history(msg)
        await broadcaster.broadcast(msg)

    async def gpu_stats_loop() -> None:
        """主循环：周期向 client 推送 ``type=gpu_stats``（不写入 history，避免回放刷屏）。"""
        interval = float(args.gpu_stats_interval_sec)
        if interval <= 0:
            return
        interval = max(0.2, interval)
        dev_idx = int(effective_gpu_index(args))
        while True:
            try:
                await asyncio.wait_for(infer_done.wait(), timeout=interval)
                return
            except asyncio.TimeoutError:
                pass
            try:
                stats = await loop.run_in_executor(None, sample_gpu_util, dev_idx)
            except Exception:  # pragma: no cover
                logging.exception("gpu sample run_in_executor failed")
                continue
            if stats is None:
                continue
            await publish(
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

    def infer_worker() -> None:
        def _schedule(coro: Any) -> None:
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
            fut.result()

        bundle: dict[str, Any] | None = None
        try:
            print(colored("[infer] 线程启动，开始加载 bundle…", "cyan"), flush=True)

            def on_progress(stage: str, message: str) -> None:
                _schedule(
                    publish(
                        event_to_json(
                            {
                                "type": "server_progress",
                                "run_id": run_id,
                                "stage": stage,
                                "message": message,
                            }
                        ),
                        add_history=False,
                    )
                )

            bundle = load_infer_bundle(args, run_id, on_progress=on_progress)
            meta_msg = bundle["meta_msg"]
            meta_ready["msg"] = meta_msg
            print(colored("[infer] 加载完成，向主循环投递 meta …", "cyan"), flush=True)
            _schedule(publish(meta_msg))

            start_index = bundle["start_index"]
            end = bundle["end"]
            min_step_period = (1.0 / args.max_fps) if args.max_fps and args.max_fps > 0 else 0.0
            last_send_t = 0.0

            for idx in range(start_index, end):
                while infer_paused.is_set():
                    time.sleep(0.05)
                msgs = process_infer_chunk(bundle, idx)
                for msg in msgs:
                    _schedule(publish(msg))
                    if min_step_period > 0:
                        now = time.monotonic()
                        dt = now - last_send_t
                        if dt < min_step_period:
                            time.sleep(min_step_period - dt)
                        last_send_t = time.monotonic()

            done_msg = event_to_json(
                {
                    "type": "done",
                    "phase": "finished",
                    "run_id": run_id,
                    "message": (
                        "推理序列已全部推送完毕；本进程即将关闭 WebSocket 并退出。"
                        "若需再次评估请重新启动本 server。"
                    ),
                    "start_index": int(start_index),
                    "end_index_exclusive": int(end),
                }
            )
            _schedule(publish(done_msg))
            print(colored(f"[infer] 已推送 type=done，序列完毕 run_id={run_id}", "green"), flush=True)
            print(colored(f"[infer] 序列推送完毕 run_id={run_id}", "cyan"), flush=True)
        except Exception as exc:  # pragma: no cover
            logging.exception("推理管线失败: %s", exc)
            err_msg = event_to_json({"type": "error", "run_id": run_id, "message": str(exc)})
            try:
                _schedule(publish(err_msg))
            except Exception:
                pass
        finally:
            stop_pi05_calib_collectors(bundle.get("calib_collectors") if bundle else None)
            print(colored("[infer] 线程退出", "cyan"), flush=True)
            signal_infer_complete()

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
        threading.Thread(target=infer_worker, daemon=True, name="pi05_infer").start()
        await infer_done.wait()
        # 主动关闭所有客户端连接，避免 websockets 在关闭握手超时后打印 ERROR
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
    args = tyro.cli(Args)
    asyncio.run(run_server(args))
