"""推理阶段耗时统计（可复用于 native / TRT overlay / webui 汇总）。"""

from .stage_perf import (
    KEY_SAMPLE_ACTIONS,
    StagePerfCollector,
    format_collector_from_policy,
    format_perf_from_bundle,
    perf_line_ms,
    wrap_sample_actions_with_stage_perf,
)

__all__ = [
    "KEY_SAMPLE_ACTIONS",
    "StagePerfCollector",
    "perf_line_ms",
    "format_collector_from_policy",
    "format_perf_from_bundle",
    "wrap_sample_actions_with_stage_perf",
]
