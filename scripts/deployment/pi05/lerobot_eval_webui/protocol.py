"""WebSocket 事件模型与 JSON 序列化（协议 v1）。"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Literal


@dataclasses.dataclass(frozen=True)
class StepEvent:
    type: Literal["step"]
    run_id: str
    episode_id: int
    global_index: int
    k_in_chunk: int
    is_chunk_start: bool
    action_horizon: int
    prompt: str | None
    gt_action: list[float]
    pred_action: list[float]
    metrics: dict[str, Any]
    images: dict[str, str] | None
    server_timing: dict[str, float] | None


def event_to_json(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False, separators=(",", ":"))


LOADING_META_MSG = event_to_json(
    {
        "type": "meta",
        "phase": "loading",
        "message": "正在加载数据集与策略（首次连接可能较慢）；加载完成后会再推送完整 meta 与 step 流。",
    }
)
