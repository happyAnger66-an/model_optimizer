"""策略对象上的 PyTorch PI0 模块解析。"""

from __future__ import annotations

from typing import Any


def policy_torch_model(policy: Any) -> Any:
    """解析 PyTorch 策略上的 PI0 模块：openpi ``Policy`` 为 ``_model``，少数封装为 ``_policy._model``。"""
    m = getattr(policy, "_model", None)
    if m is not None:
        return m
    inner = getattr(policy, "_policy", None)
    if inner is not None:
        return getattr(inner, "_model", None)
    return None
