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
from .hints import write_webui_server_hint
from .protocol import LOADING_META_MSG, event_to_json


async def run_server(args: Args) -> None:
    import websockets.asyncio.server as _server

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

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
        finally:
            await broadcaster.unregister(ws)

    loop = asyncio.get_running_loop()
    infer_done = asyncio.Event()

    def signal_infer_complete() -> None:
        loop.call_soon_threadsafe(infer_done.set)

    async def publish(msg: str) -> None:
        broadcaster.add_history(msg)
        await broadcaster.broadcast(msg)

    def infer_worker() -> None:
        def _schedule(coro: Any) -> None:
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
            fut.result()

        bundle: dict[str, Any] | None = None
        try:
            print(colored("[infer] 线程启动，开始加载 bundle…", "cyan"), flush=True)
            bundle = load_infer_bundle(args, run_id)
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
        threading.Thread(target=infer_worker, daemon=True, name="pi05_infer").start()
        await infer_done.wait()
        await asyncio.sleep(0.25)
        print(colored("[main] 推理管线已结束，关闭 WebSocket 并退出进程", "green"), flush=True)
        server.close()
        await server.wait_closed()


def main() -> None:
    args = tyro.cli(Args)
    asyncio.run(run_server(args))
