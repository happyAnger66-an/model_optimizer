# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pi05 TensorRT 子图入口耗时统计（装饰器）。
#
# 环境变量（在 ``load_model`` / ``_setup_trt_engine`` 之前设置生效）
# ----------------------------------------------------------------
# - ``MO_TRT_HOOK_STATS`` 或 ``PI05_TRT_HOOK_STATS``：设为 ``1``/``true``/``yes`` 等则
#   对挂接的 TRT 包装函数计时并累计。
# - ``MO_TRT_HOOK_STATS_PRINT``：在进程退出时打印汇总；默认 ``1``（开启）。
#   设为 ``0``/``false``/``off`` 等则只记录、退出时不打印（可调用 ``dump_trt_hook_stats``）。

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

_ENV_ENABLE = ("MO_TRT_HOOK_STATS", "PI05_TRT_HOOK_STATS")
_ENV_PRINT = "MO_TRT_HOOK_STATS_PRINT"

_LOCK = threading.Lock()
_LAT_MS: dict[str, list[float]] = defaultdict(list)
_ATEXIT_REGISTERED = False


def env_trt_hook_stats_enabled() -> bool:
    for k in _ENV_ENABLE:
        v = os.environ.get(k, "").strip().lower()
        if v in ("1", "true", "yes", "y", "on"):
            return True
    return False


def env_trt_hook_stats_print_on_exit() -> bool:
    raw = os.environ.get(_ENV_PRINT, "1").strip().lower()
    return raw not in ("0", "false", "no", "n", "off", "")


def _register_atexit_dump() -> None:
    global _ATEXIT_REGISTERED
    with _LOCK:
        if _ATEXIT_REGISTERED:
            return
        _ATEXIT_REGISTERED = True

    def _dump() -> None:
        if not env_trt_hook_stats_enabled():
            return
        if env_trt_hook_stats_print_on_exit():
            dump_trt_hook_stats()

    atexit.register(_dump)


def dump_trt_hook_stats() -> str:
    """将当前累计的 TRT 钩子耗时打印到 logging 并返回文本（毫秒：mean / p50 / p99 / n）。"""
    lines = ["Pi05 TRT hook timer (ms):"]
    with _LOCK:
        for name in sorted(_LAT_MS.keys()):
            arr = np.asarray(_LAT_MS[name], dtype=np.float64)
            if arr.size == 0:
                lines.append(f"  {name}: n=0")
                continue
            lines.append(
                f"  {name}: n={int(arr.size)} mean={float(np.mean(arr)):.4f}ms "
                f"p50={float(np.percentile(arr, 50)):.4f}ms "
                f"p99={float(np.percentile(arr, 99)):.4f}ms"
            )
    msg = "\n".join(lines)
    logger.info("%s", msg)
    return msg


def clear_trt_hook_stats() -> None:
    with _LOCK:
        _LAT_MS.clear()


F = TypeVar("F", bound=Callable[..., Any])


def trt_hook_timer(name: str) -> Callable[[F], F]:
    """若统计开关打开，包装 ``fn`` 并记录 wall-time（毫秒）；否则原样返回 ``fn``。"""

    def deco(fn: F) -> F:
        if not env_trt_hook_stats_enabled():
            return fn

        _register_atexit_dump()

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                with _LOCK:
                    _LAT_MS[name].append(float(dt_ms))

        return wrapper  # type: ignore[return-value]

    return deco
