"""WebSocket 控制面：解析为命令对象（Command）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PauseInference:
    """在下一 chunk 推理前暂停。"""


@dataclass(frozen=True)
class ResumeInference:
    """恢复推理。"""


# server.py 等调用方使用的简短名称（与 PauseInference / ResumeInference 为同一类型）
ControlPause = PauseInference
ControlResume = ResumeInference

ControlCommand = PauseInference | ResumeInference


def parse_control_message(msg: Any) -> ControlCommand | None:
    if not isinstance(msg, dict) or msg.get("type") != "control":
        return None
    act = msg.get("action")
    if act == "pause":
        return PauseInference()
    if act == "resume":
        return ResumeInference()
    return None
