"""加载数据集、策略与 meta 消息（推理线程内调用）。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .bundle_common import BundleProgress
from .bundle_dataset import load_dataset_bundle
from .bundle_meta import finalize_meta_and_bundle
from .bundle_policies import load_policy_bundle
from .bundle_validate import validate_infer_bundle_args
from .config import Args

# 向后兼容：旧代码若从 bundle 导入辅助函数仍可用。
from .bundle_common import (  # noqa: F401
    install_compare_pt_stage_perf as _install_compare_pt_stage_perf,
    trt_trt_second_engine_filenames as _trt_trt_second_engine_filenames,
    trt_vit_scale_fix_kw as _trt_vit_scale_fix_kw,
)


def load_infer_bundle(
    args: Args,
    run_id: str,
    *,
    on_progress: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """在专用推理线程中执行：数据集 + policy + TRT，避免阻塞 asyncio 事件循环。

    on_progress(stage_id, message)：可选，供 WebUI 推送加载步骤（stage_id 稳定、便于 client 去重/排序）。
    """
    progress = BundleProgress(on_progress)
    validate_infer_bundle_args(args)
    ds = load_dataset_bundle(args, progress)
    policies = load_policy_bundle(args, ds.train_cfg, progress)
    return finalize_meta_and_bundle(args, run_id, ds, policies, progress)
