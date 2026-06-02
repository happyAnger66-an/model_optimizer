"""分阶段 wall-time 统计：可嵌套 key、按 step 展开、统一汇总行格式。

Key 约定（点分路径，便于分组打印）::

    embed_prefix          # 单次 sample 的 embed_prefix wall time
    prefix_llm            # prefix KV 前向
    flashrt.setup         # 首次 backend / setup_prompt
    denoise.total         # 整段 backend.run()（10 步扩散一次跑完）
    denoise.step.0        # pipeline 内第 0 步（flow-matching step）
    denoise.step.9        # 第 9 步

汇总输出与 webui ``[summary:engine]`` 对齐::

    [summary:engine] denoise.total   n=38 mean=...
    [summary:engine] denoise.step    n=380 mean=...   # 所有 step 样本池化
    [summary:engine] denoise.step.0  n=38 mean=...
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# 默认打印顺序：先高层阶段，再 denoise 聚合，再逐步展开。
_DEFAULT_SUMMARY_ORDER: tuple[str, ...] = (
    "embed_prefix",
    "prefix_llm",
    "flashrt.setup",
    "denoise.calibrate",
    "denoise.total",
    "denoise.step",
)


def perf_line_ms(values: Sequence[float]) -> str:
    """毫秒列表 → ``n=.. mean=.. p50=.. p90=.. p99=.. ms``。"""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return "n=0"
    return (
        f"n={int(arr.size)} mean={float(np.mean(arr)):.2f} "
        f"p50={float(np.percentile(arr, 50)):.2f} "
        f"p90={float(np.percentile(arr, 90)):.2f} "
        f"p99={float(np.percentile(arr, 99)):.2f} ms"
    )


def _step_index(key: str) -> int:
    """从 ``denoise.step.3`` 解析 step 下标；非 step key 返回 -1。"""
    m = re.fullmatch(r"denoise\.step\.(\d+)", key)
    return int(m.group(1)) if m else -1


@dataclass
class StagePerfCollector:
    """可复用的阶段耗时采集器（线程内单实例即可；非线程安全）。"""

    enabled: bool = True
    summary_prefix: str = "[summary:engine]"
    _samples: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def record(self, stage: str, dt_ms: float) -> None:
        if not self.enabled:
            return
        self._samples[stage].append(float(dt_ms))

    @contextmanager
    def timed(self, stage: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, (time.perf_counter() - t0) * 1000.0)

    def record_step(self, step: int, dt_ms: float, *, prefix: str = "denoise") -> None:
        """记录扩散单步耗时，key 为 ``{prefix}.step.{step}``。"""
        self.record(f"{prefix}.step.{int(step)}", dt_ms)

    def pooled(self, pattern_prefix: str) -> list[float]:
        """合并所有以 ``pattern_prefix`` 开头的 key 的样本（用于 step 池化统计）。"""
        out: list[float] = []
        for k, vals in self._samples.items():
            if k == pattern_prefix or k.startswith(pattern_prefix + "."):
                out.extend(vals)
        return out

    def keys_for_summary(self, extra_order: Sequence[str] | None = None) -> list[str]:
        """生成汇总打印用的 key 列表（去重、有序）。"""
        order = list(extra_order or _DEFAULT_SUMMARY_ORDER)
        seen: set[str] = set()
        keys: list[str] = []

        def _add(k: str) -> None:
            if k not in seen and self._samples.get(k):
                seen.add(k)
                keys.append(k)

        for k in order:
            if k == "denoise.step":
                # 池化行：有任意 denoise.step.N 时出现
                if any(_step_index(x) >= 0 for x in self._samples):
                    _add("denoise.step")
                continue
            _add(k)

        # 其余未列出的 key 按字母序追加
        for k in sorted(self._samples.keys()):
            if k not in seen and _step_index(k) < 0:
                _add(k)

        # denoise.step.N 按 step 编号排序
        step_keys = sorted(
            (k for k in self._samples if _step_index(k) >= 0),
            key=_step_index,
        )
        for k in step_keys:
            _add(k)

        return keys

    def values_for_key(self, key: str) -> list[float]:
        if key == "denoise.step":
            return self.pooled("denoise.step")
        return list(self._samples.get(key, []))

    def format_summary_lines(
        self,
        *,
        prefix: str | None = None,
        extra_order: Sequence[str] | None = None,
    ) -> list[str]:
        """返回可打印的汇总行（不含换色）。"""
        tag = prefix if prefix is not None else self.summary_prefix
        lines: list[str] = []
        for key in self.keys_for_summary(extra_order):
            vals = self.values_for_key(key)
            if not vals:
                continue
            # 打印名：denoise.total → denoise.total；denoise.step.3 → denoise.step.3
            label = key if key != "denoise.step" else "denoise.step"
            lines.append(f"{tag} {label:<16} {perf_line_ms(vals)}")
        return lines

    def merge_from(self, other: StagePerfCollector | None) -> None:
        if other is None or not other.enabled:
            return
        for k, vals in other._samples.items():
            self._samples[k].extend(vals)

    def clear(self) -> None:
        self._samples.clear()

    def to_dict(self) -> dict[str, list[float]]:
        return {k: list(v) for k, v in self._samples.items()}

    @classmethod
    def from_dict(cls, data: dict[str, list[float]], *, enabled: bool = True) -> StagePerfCollector:
        c = cls(enabled=enabled)
        for k, vals in data.items():
            c._samples[k].extend(float(x) for x in vals)
        return c


def _lines_from_perf_holder(obj: Any) -> list[str]:
    if obj is None:
        return []
    if isinstance(obj, StagePerfCollector):
        return obj.format_summary_lines()
    fn = getattr(obj, "format_perf_summary_lines", None)
    if callable(fn):
        return fn()
    sp = getattr(obj, "stage_perf", None)
    if sp is not None and sp is not obj:
        return _lines_from_perf_holder(sp)
    return []


def format_collector_from_policy(policy: Any) -> list[str]:
    """从 policy / model / native_executor 上挂载的 :class:`StagePerfCollector` 取汇总行。"""
    if policy is None:
        return []
    seen: set[int] = set()
    candidates: list[Any] = [policy]
    model = getattr(policy, "_model", None)
    if model is not None:
        candidates.append(model)
    native_ex = getattr(policy, "_native_executor", None)
    if native_ex is not None:
        candidates.append(native_ex)
    for obj in candidates:
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        for attr in ("_stage_perf", "stage_perf"):
            lines = _lines_from_perf_holder(getattr(obj, attr, None))
            if lines:
                return lines
        lines = _lines_from_perf_holder(obj)
        if lines:
            return lines
    return []


def format_perf_from_bundle(bundle: dict[str, Any] | None) -> list[str]:
    """webui bundle 汇总：优先 ``bundle['native_executor']``，再回退 policy。"""
    if not bundle:
        return []
    native_ex = bundle.get("native_executor")
    lines = _lines_from_perf_holder(native_ex)
    if lines:
        return lines
    return format_collector_from_policy(bundle.get("policy"))
