"""
LeRobot 离线评估 WebUI — server 侧模块化实现。

- ``protocol``：step/meta 等事件与 JSON
- ``dataset`` / ``media`` / ``action_align``：数据与图像
- ``bundle`` / ``chunk_infer`` / ``tensorrt_backend`` / ``calib``：推理管线
- ``broadcaster`` / ``hints`` / ``server``：WebSocket 与 CLI

协议说明见入口 ``lerobot_eval_webui_server.py`` 顶部文档字符串。
"""

from __future__ import annotations

from typing import Any

from .config import Args
from .protocol import LOADING_META_MSG, StepEvent, event_to_json

__all__ = [
    "Args",
    "LOADING_META_MSG",
    "StepEvent",
    "event_to_json",
    "main",
    "run_server",
]


def main() -> None:
    from .server import main as _main

    _main()


def __getattr__(name: str) -> Any:
    if name == "run_server":
        from .server import run_server as rs

        return rs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
