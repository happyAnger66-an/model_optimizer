# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# 非侵入式 PI0 PyTorch 推理阶段耗时统计：通过环境变量开启，在 ``Pi05TensorRTExecutor.load_model``
# 末尾对 ``self.pi05_model`` 实例打补丁，不修改 openpi 源码。
#
# 环境变量
# --------
# - ``PI05_PI0_PROFILE`` 或 ``MO_PI0_STAGE_PROFILE``：设为 ``1``/``true``/``yes`` 等则启用。
# - ``PI05_PROFILE_WARMUP``：非负整数，前 N 次完整 ``sample_actions`` 调用**不写入**统计（默认 ``0``）。
#
# 进程退出时（``atexit``）打印各阶段 ``mean / p90 / p99 / min / max``（毫秒）。

from __future__ import annotations

import atexit
import inspect
import logging
import os
import threading
import time
import types
from collections import defaultdict
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_ENV_ENABLE_KEYS = ("PI05_PI0_PROFILE", "MO_PI0_STAGE_PROFILE")
_ENV_WARMUP = "PI05_PROFILE_WARMUP"

_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False
_LATEST_PROFILER: Optional["Pi0StageProfiler"] = None


def env_profile_enabled() -> bool:
    for k in _ENV_ENABLE_KEYS:
        v = os.environ.get(k, "").strip().lower()
        if v in ("1", "true", "yes", "y", "on"):
            return True
    return False


def env_warmup_skips() -> int:
    raw = os.environ.get(_ENV_WARMUP, "0").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("Invalid %s=%r, using 0", _ENV_WARMUP, raw)
        return 0


def _format_line(name: str, values: list[float]) -> str:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return f"  {name}: n=0 (no samples after warmup)"
    return (
        f"  {name}: n={int(a.size)} mean={float(np.mean(a)):.4f}ms "
        f"p90={float(np.percentile(a, 90)):.4f}ms p99={float(np.percentile(a, 99)):.4f}ms "
        f"min={float(np.min(a)):.4f}ms max={float(np.max(a)):.4f}ms"
    )


class Pi0StageProfiler:
    """在 ``PI0Pytorch`` 实例上包装 ``sample_actions`` / ``_preprocess_observation`` /
    ``paligemma_with_expert.forward`` / ``denoise_step``。"""

    KEY_SA = "sample_actions"
    KEY_PRE = "_preprocess_observation"
    KEY_PGE_FWD = "paligemma_with_expert.forward"
    KEY_DENOISE = "denoise_step"

    def __init__(self, model: Any, warmup_skips: int) -> None:
        self.model = model
        self.warmup_skips = int(warmup_skips)
        self._sa_invocation = 0
        self._record_this_call = False
        self.latencies: dict[str, list[float]] = defaultdict(list)

    def _append_ms(self, key: str, dt_ms: float) -> None:
        if not self._record_this_call:
            return
        with _LOCK:
            self.latencies[key].append(float(dt_ms))

    def install(self) -> None:
        m = self.model
        prof = self

        if getattr(m, "_pi05_stage_profiler", None) is not None:
            logger.warning(
                "Pi0 stage profiler: already installed on this model, skip duplicate install"
            )
            return

        # --- _preprocess_observation ---
        # ``m._preprocess_observation`` 已是绑定方法，勿再传入 ``self``。
        _orig_pre = m._preprocess_observation
        _pre_has_train = "train" in inspect.signature(_orig_pre).parameters

        def _pre(self, observation, *, train=True):
            if not prof._record_this_call:
                if _pre_has_train:
                    return _orig_pre(observation, train=train)
                return _orig_pre(observation)
            t0 = time.perf_counter()
            try:
                if _pre_has_train:
                    return _orig_pre(observation, train=train)
                return _orig_pre(observation)
            finally:
                prof._append_ms(prof.KEY_PRE, (time.perf_counter() - t0) * 1000.0)

        m._preprocess_observation = types.MethodType(_pre, m)

        # --- PaliGemmaWithExpertModel.forward (gemma_pytorch) ---
        pgm = m.paligemma_with_expert
        _orig_pge_fwd = pgm.forward

        def _pge_fwd(self, *args, **kwargs):
            if not prof._record_this_call:
                return _orig_pge_fwd(*args, **kwargs)
            t0 = time.perf_counter()
            try:
                return _orig_pge_fwd(*args, **kwargs)
            finally:
                prof._append_ms(prof.KEY_PGE_FWD, (time.perf_counter() - t0) * 1000.0)

        pgm.forward = types.MethodType(_pge_fwd, pgm)

        # --- denoise_step（可能已是 TRT MethodType）---
        _orig_den = m.denoise_step

        def _den(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
            if not prof._record_this_call:
                return _orig_den(
                    state, prefix_pad_masks, past_key_values, x_t, timestep
                )
            t0 = time.perf_counter()
            try:
                return _orig_den(
                    state, prefix_pad_masks, past_key_values, x_t, timestep
                )
            finally:
                prof._append_ms(prof.KEY_DENOISE, (time.perf_counter() - t0) * 1000.0)

        m.denoise_step = types.MethodType(_den, m)

        # --- sample_actions（最外层；须最后包装，且须在 TRT / eager 恢复之后）---
        _orig_sa = m.sample_actions

        def _sa(self, device, observation, noise=None, num_steps=10):
            prof._sa_invocation += 1
            prof._record_this_call = prof._sa_invocation > prof.warmup_skips
            t0 = time.perf_counter()
            try:
                return _orig_sa(
                    device, observation, noise=noise, num_steps=num_steps
                )
            finally:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if prof._record_this_call:
                    prof._append_ms(prof.KEY_SA, dt_ms)
                prof._record_this_call = False

        m.sample_actions = types.MethodType(_sa, m)

        m._pi05_stage_profiler = self
        _register_atexit_dump(self)

    def dump_summary(self) -> None:
        lines = [
            "Pi0 stage profiler summary (ms, post-warmup):",
            f"  warmup_skips(sample_actions)={self.warmup_skips}",
            _format_line(self.KEY_SA, self.latencies[self.KEY_SA]),
            _format_line(self.KEY_PRE, self.latencies[self.KEY_PRE]),
            _format_line(self.KEY_PGE_FWD, self.latencies[self.KEY_PGE_FWD]),
            _format_line(self.KEY_DENOISE, self.latencies[self.KEY_DENOISE]),
        ]
        msg = "\n".join(lines)
        logger.info("%s", msg)


def _register_atexit_dump(prof: Pi0StageProfiler) -> None:
    global _LATEST_PROFILER, _ATEXIT_REGISTERED
    with _LOCK:
        _LATEST_PROFILER = prof
        if _ATEXIT_REGISTERED:
            return
        _ATEXIT_REGISTERED = True

    def _dump() -> None:
        global _LATEST_PROFILER
        p = _LATEST_PROFILER
        if p is not None:
            try:
                p.dump_summary()
            except Exception as exc:
                logger.warning("Pi0 stage profiler atexit dump failed: %s", exc)

    atexit.register(_dump)


def maybe_install_pi0_stage_profiler(model: Any) -> Pi0StageProfiler | None:
    """若环境变量启用，在 ``model``（``PI0Pytorch``）上安装阶段计时；否则返回 ``None``。"""
    if not env_profile_enabled():
        return None
    w = env_warmup_skips()
    prof = Pi0StageProfiler(model, warmup_skips=w)
    prof.install()
    logger.info(
        "Pi0 stage profiler installed (warmup_skip_sample_actions=%s). "
        "Unset %s to disable.",
        w,
        "/".join(_ENV_ENABLE_KEYS),
    )
    return prof
