"""Janus 队列：推理线程（同步 put）与 asyncio 广播（异步 get）解耦。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import janus

if TYPE_CHECKING:
    from .broadcaster import WebsocketBroadcaster


@dataclass(frozen=True)
class OutboundMessage:
    """待广播的 JSON 文本。"""

    text: str
    add_history: bool = True


class OutboundStop:
    """队列结束标记（单例语义：用 ``is`` 判断）。"""


OUTBOUND_STOP = OutboundStop()


class JanusOutboundBridge:
    """线程侧 ``sync_emit`` / 事件循环侧 ``drain``；实现 ``SyncOutboundPort``（结构性子类型）。"""

    def __init__(self, queue: janus.Queue[Any], broadcaster: WebsocketBroadcaster) -> None:
        self._queue = queue
        self._broadcaster = broadcaster

    def sync_emit(self, text: str, *, add_history: bool = True) -> None:
        self._queue.sync_q.put(OutboundMessage(text=text, add_history=add_history))

    def sync_close(self) -> None:
        """推理线程结束时调用：消费者收到后退出 ``drain``。"""
        self._queue.sync_q.put(OUTBOUND_STOP)

    async def drain(self) -> None:
        """在事件循环任务中运行，直到收到 ``OUTBOUND_STOP``。"""
        try:
            while True:
                item = await self._queue.async_q.get()
                if item is OUTBOUND_STOP:
                    break
                if not isinstance(item, OutboundMessage):
                    continue
                if item.add_history:
                    self._broadcaster.add_history(item.text)
                await self._broadcaster.broadcast(item.text)
        finally:
            self._queue.close()
            await self._queue.wait_closed()
