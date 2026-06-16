"""WebSocket 事件模型与 JSON 序列化（协议 v1）。"""

from __future__ import annotations

import dataclasses
import json
import math
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
    pred_action_trt: list[float] | None = None
    pred_action_ptq: list[float] | None = None


def _sanitize_json_value(obj: Any) -> Any:
    """将 NaN/Inf 转为 ``null``，保证输出可被浏览器 ``JSON.parse`` 解析。

    Python ``json.dumps`` 默认会把 ``float('nan')`` 写成裸 ``NaN``（非 RFC 8259），
    浏览器端会整帧解析失败；``meta``/``gpu_stats`` 无此问题，``step.metrics`` 里
    的 running 统计（如 ``rel_p99_dim``）在样本不足时常含 NaN。
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_json_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json_value(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_json_value(v) for v in obj]
    return obj


def event_to_json(event: dict[str, Any]) -> str:
    clean = _sanitize_json_value(event)
    return json.dumps(clean, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


LOADING_META_MSG = event_to_json(
    {
        "type": "meta",
        "phase": "loading",
        "message": "服务端正在初始化：将依次推送各加载步骤（server_progress），完成后下发完整 meta 与 step 流。",
    }
)
