"""WebSocket 客户端：路径校验、首包同步、pause/resume 控制。"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from termcolor import colored

from .control_commands import ControlPause, ControlResume, parse_control_message
from .protocol import LOADING_META_MSG, event_to_json
from .server_runtime import ServerRuntime


def handshake_request_path(ws: Any) -> str | None:
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


def paths_equivalent(a: str, b: str) -> bool:
    """``/ws`` 与 ``/ws/`` 视为同一路径。"""
    aa = a.rstrip("/") or "/"
    bb = b.rstrip("/") or "/"
    return aa == bb


async def _send_control_ack(ws: Any, rt: ServerRuntime, action: str, paused: bool) -> None:
    await rt.broadcaster.send_to(
        ws,
        event_to_json({"type": "control_ack", "action": action, "paused": paused}),
    )


async def send_initial_sync(ws: Any, rt: ServerRuntime) -> None:
    if rt.meta_ready["msg"] is None:
        await rt.broadcaster.send_to(ws, LOADING_META_MSG)
    else:
        await rt.broadcaster.send_to(ws, rt.meta_ready["msg"])
        if rt.args.history_size > 0:
            await rt.broadcaster.send_history(ws)
    await _send_control_ack(ws, rt, "sync", rt.infer_paused.is_set())


async def _handle_control_message(ws: Any, msg: dict[str, Any], rt: ServerRuntime) -> None:
    cmd = parse_control_message(msg)
    if isinstance(cmd, ControlPause):
        rt.infer_paused.set()
        print(colored("[infer] 收到 pause：下一 chunk 前将阻塞推理", "yellow"), flush=True)
        await _send_control_ack(ws, rt, "pause", True)
    elif isinstance(cmd, ControlResume):
        rt.infer_paused.clear()
        print(colored("[infer] 收到 resume：继续推理", "green"), flush=True)
        await _send_control_ack(ws, rt, "resume", False)


async def control_message_loop(ws: Any, rt: ServerRuntime, wsex: Any) -> None:
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            if isinstance(msg, dict):
                await _handle_control_message(ws, msg, rt)
    except (wsex.ConnectionClosedOK, wsex.ConnectionClosedError):
        pass


def _notify_first_client(rt: ServerRuntime) -> None:
    if not rt.client_connected.is_set():
        rt.client_connected.set()
        print(
            colored("[main] 已有 WebSocket 客户端连接，开始（或继续）chunk 推理", "green"),
            flush=True,
        )


def make_ws_handler(rt: ServerRuntime, wsex: Any) -> Callable[[Any], Awaitable[None]]:
    async def handler(ws: Any) -> None:
        req_path = handshake_request_path(ws)
        if req_path is None or not paths_equivalent(req_path, rt.args.path):
            await ws.close(code=1008, reason="invalid path")
            return
        await rt.broadcaster.register(ws)
        _notify_first_client(rt)
        try:
            await send_initial_sync(ws, rt)
            await control_message_loop(ws, rt, wsex)
        finally:
            await rt.broadcaster.unregister(ws)

    return handler
