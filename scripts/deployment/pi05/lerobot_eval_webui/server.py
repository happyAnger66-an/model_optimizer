"""WebSocket 服务：Janus 出站泵、GPU 旁路、推理线程（与 eval_session 编排分离）。"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import websockets.asyncio.server as _server
import websockets.exceptions as _wsex
from termcolor import colored

from .config import Args
from .eval_session import run_infer_worker
from .gpu_stats import run_gpu_stats_loop
from .hints import write_webui_server_hint
from .server_logging import configure_server_logging
from .server_runtime import ServerRuntime, build_server_runtime
from .server_startup import print_server_startup_hints
from .ws_handlers import make_ws_handler


async def _shutdown_after_infer(
    server: Any,
    rt: ServerRuntime,
    gpu_task: asyncio.Task[None] | None,
) -> None:
    clients = rt.broadcaster.snapshot_clients()
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


def _start_infer_thread(rt: ServerRuntime) -> None:
    threading.Thread(
        target=run_infer_worker,
        kwargs={
            "args": rt.args,
            "run_id": rt.run_id,
            "meta_ready": rt.meta_ready,
            "infer_paused": rt.infer_paused,
            "client_connected": rt.client_connected,
            "bridge": rt.bridge,
        },
        daemon=True,
        name="pi05_infer",
    ).start()


async def run_server(args: Args) -> None:
    configure_server_logging()
    rt = build_server_runtime(args)
    handler = make_ws_handler(rt, _wsex)
    pump_task = asyncio.create_task(rt.bridge.drain(), name="outbound_pump")

    print_server_startup_hints(args, write_webui_server_hint(args))

    async with _server.serve(
        handler,
        args.host,
        args.port,
        compression=None,
        max_size=None,
    ) as server:
        print(colored(f"run_id={rt.run_id}（加载完成后会广播完整 meta）", "green"), flush=True)
        gpu_task = None
        if args.gpu_stats_interval_sec and args.gpu_stats_interval_sec > 0:
            gpu_task = asyncio.create_task(
                run_gpu_stats_loop(
                    args=args,
                    run_id=rt.run_id,
                    pump_task=pump_task,
                    publish_direct=rt.publish_direct,
                ),
                name="gpu_stats",
            )
        _start_infer_thread(rt)
        try:
            await pump_task
        finally:
            await _shutdown_after_infer(server, rt, gpu_task)


def main() -> None:
    from .cli_config import parse_args_with_optional_config_file

    args = parse_args_with_optional_config_file(Args)
    asyncio.run(run_server(args))
