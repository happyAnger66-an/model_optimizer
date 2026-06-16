"""WebUI 协议 JSON：须可被浏览器 JSON.parse（禁止裸 NaN/Inf）。"""

from __future__ import annotations

import json
import math

from lerobot_eval_webui.protocol import event_to_json


def test_event_to_json_replaces_nan_and_inf() -> None:
    raw = event_to_json(
        {
            "type": "step",
            "metrics": {
                "rel_p99_dim": [float("nan"), 1.5, float("inf")],
                "mse": float("nan"),
            },
        }
    )
    assert "NaN" not in raw
    assert "Infinity" not in raw
    parsed = json.loads(raw)
    assert parsed["metrics"]["rel_p99_dim"] == [None, 1.5, None]
    assert parsed["metrics"]["mse"] is None


def test_event_to_json_finite_floats_unchanged() -> None:
    raw = event_to_json({"type": "gpu_stats", "gpu_util_pct": 56.0})
    assert json.loads(raw)["gpu_util_pct"] == 56.0
