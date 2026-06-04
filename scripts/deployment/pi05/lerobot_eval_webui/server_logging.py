"""WebSocket 服务日志配置。"""

from __future__ import annotations

import logging


class _SuppressWsHandshakeNoise(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if "opening handshake failed" in record.getMessage():
            return False
        return True


def configure_server_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    filt = _SuppressWsHandshakeNoise()
    for name in ("websockets.server", "websockets.asyncio.server", "websockets"):
        logging.getLogger(name).addFilter(filt)
