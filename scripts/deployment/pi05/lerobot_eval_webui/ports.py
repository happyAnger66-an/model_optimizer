"""出站与控制相关端口（Protocol），推理线程依赖抽象而非具体 Janus 实现。"""

from __future__ import annotations

from typing import Protocol


class SyncOutboundPort(Protocol):
    """推理线程侧出站：实现由 ``JanusOutboundBridge`` 等适配器提供。"""

    def sync_emit(self, text: str, *, add_history: bool = True) -> None: ...

    def sync_close(self) -> None: ...
