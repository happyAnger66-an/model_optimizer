"""WebSocket 客户端集合与历史回放。"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Deque


class WebsocketBroadcaster:
    def __init__(self, *, history_size: int) -> None:
        self._clients: set[Any] = set()
        self._history: Deque[str] = deque(maxlen=max(history_size, 0))
        self._send_locks: dict[int, asyncio.Lock] = {}

    def _lock_for(self, ws: Any) -> asyncio.Lock:
        key = id(ws)
        lock = self._send_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._send_locks[key] = lock
        return lock

    def _drop_lock(self, ws: Any) -> None:
        self._send_locks.pop(id(ws), None)

    def add_history(self, msg: str) -> None:
        if self._history.maxlen and self._history.maxlen > 0:
            self._history.append(msg)

    async def register(self, ws: Any) -> None:
        self._clients.add(ws)

    async def unregister(self, ws: Any) -> None:
        self._clients.discard(ws)
        self._drop_lock(ws)

    async def send_to(self, ws: Any, msg: str) -> bool:
        """向单个客户端发送；与 ``broadcast`` 共用 per-connection 锁。

        ``websockets`` 连接不能由多个协程并发 ``send``；``drain``、``gpu_stats`` 与
        ``send_initial_sync``/``send_history`` 必须串行化，否则大体积 ``step`` 帧可能被破坏。
        """
        async with self._lock_for(ws):
            try:
                await ws.send(msg)
                return True
            except Exception:
                return False

    async def send_history(self, ws: Any) -> None:
        for msg in list(self._history):
            if not await self.send_to(ws, msg):
                await self.unregister(ws)
                return

    async def broadcast(self, msg: str) -> None:
        if not self._clients:
            return
        dead: list[Any] = []
        for ws in list(self._clients):
            if not await self.send_to(ws, msg):
                dead.append(ws)
        for ws in dead:
            await self.unregister(ws)

    def snapshot_clients(self) -> list[Any]:
        """返回当前连接快照，供退出时主动关闭连接。"""
        return list(self._clients)
