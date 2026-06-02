"""推理阶段耗时统计（可复用于 native / TRT overlay / webui 汇总）。"""

from .stage_perf import (
    KEY_POLICY_ALIGN,
    KEY_POLICY_INFER,
    KEY_POLICY_POSTPROCESS,
    KEY_POLICY_PREPROCESS,
    KEY_SAMPLE_ACTIONS,
    StagePerfCollector,
    _lines_from_perf_holder,
    format_collector_from_policy,
    format_perf_from_bundle,
    install_infer_stage_perf,
    perf_line_ms,
    stage_perf_from_policy,
    wrap_policy_infer_with_stage_perf,
    wrap_sample_actions_with_stage_perf,
)

__all__ = [
    "KEY_POLICY_ALIGN",
    "KEY_POLICY_INFER",
    "KEY_POLICY_POSTPROCESS",
    "KEY_POLICY_PREPROCESS",
    "KEY_SAMPLE_ACTIONS",
    "StagePerfCollector",
    "perf_line_ms",
    "format_collector_from_policy",
    "format_perf_from_bundle",
    "install_infer_stage_perf",
    "stage_perf_from_policy",
    "wrap_policy_infer_with_stage_perf",
    "wrap_sample_actions_with_stage_perf",
    "_lines_from_perf_holder",
]
