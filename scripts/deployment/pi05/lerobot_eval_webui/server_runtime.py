"""``run_server`` 共享运行时状态与出站直发。"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Any

import janus

from .broadcaster import WebsocketBroadcaster
from .config import Args
from .outbound_bridge import JanusOutboundBridge


@dataclass
class ServerRuntime:
    args: Args
    run_id: str
    broadcaster: WebsocketBroadcaster
    meta_ready: dict[str, Any]
    infer_paused: threading.Event
    client_connected: threading.Event
    outbound_queue: janus.Queue[Any]
    bridge: JanusOutboundBridge

    async def publish_direct(self, msg: str, *, add_history: bool = True) -> None:
        """asyncio 侧消息：不经 Janus，避免从协程向 sync_q 投递。"""
        if add_history:
            self.broadcaster.add_history(msg)
        await self.broadcaster.broadcast(msg)


def build_server_runtime(args: Args) -> ServerRuntime:
    run_id = uuid.uuid4().hex[:12]
    broadcaster = WebsocketBroadcaster(history_size=args.history_size)
    meta_ready: dict[str, Any] = {"msg": None}
    infer_paused = threading.Event()
    client_connected = threading.Event()
    if not bool(getattr(args, "wait_for_client", False)):
        client_connected.set()

    qmax = int(args.outbound_queue_maxsize)
    outbound_queue: janus.Queue[Any] = janus.Queue(qmax if qmax > 0 else 0)
    bridge = JanusOutboundBridge(outbound_queue, broadcaster)
    return ServerRuntime(
        args=args,
        run_id=run_id,
        broadcaster=broadcaster,
        meta_ready=meta_ready,
        infer_paused=infer_paused,
        client_connected=client_connected,
        outbound_queue=outbound_queue,
        bridge=bridge,
    )
