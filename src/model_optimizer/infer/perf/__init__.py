"""推理阶段耗时统计（可复用于 native / TRT overlay / webui 汇总）。"""

from .stage_perf import (
    StagePerfCollector,
    format_collector_from_policy,
    format_perf_from_bundle,
    perf_line_ms,
)

__all__ = [
    "StagePerfCollector",
    "perf_line_ms",
    "format_collector_from_policy",
    "format_perf_from_bundle",
]
