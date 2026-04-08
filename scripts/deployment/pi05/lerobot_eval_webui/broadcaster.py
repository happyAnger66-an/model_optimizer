"""WebSocket 客户端集合与历史回放。"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque


class WebsocketBroadcaster:
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
