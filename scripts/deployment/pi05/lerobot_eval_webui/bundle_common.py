"""bundle 加载共用工具（进度回调、策略创建、TRT 辅助）。"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from openpi.policies import policy_config
from openpi.training import config as _config

from .config import Args


class BundleProgress:
    def __init__(self, on_progress: Callable[[str, str], None] | None) -> None:
        self._on_progress = on_progress

    def emit(self, stage: str, msg: str) -> None:
        if self._on_progress is not None:
            self._on_progress(stage, msg)


def trt_vit_scale_fix_kw(args: Args) -> bool | None:
    """YAML ``trt_vit_scale_fix: true`` 时显式开启；默认 ``False`` 时不写入 cfg，仍可读环境变量。"""
    if bool(getattr(args, "trt_vit_scale_fix", False)):
        return True
    return None


def trt_trt_second_engine_filenames(args: Args) -> dict[str, str]:
    """第二路引擎文件名：``trt_trt_second_*`` 非空优先，否则回退到主路 ``*_engine``。"""

    def _coalesce(second: str, primary: str) -> str:
        s = (second or "").strip()
        return s if s else (primary or "")

    return {
        "vit_engine": _coalesce(str(getattr(args, "trt_trt_second_vit_engine", "")), args.vit_engine),
        "llm_engine": _coalesce(str(getattr(args, "trt_trt_second_llm_engine", "")), args.llm_engine),
        "expert_engine": _coalesce(str(getattr(args, "trt_trt_second_expert_engine", "")), args.expert_engine),
        "denoise_engine": _coalesce(str(getattr(args, "trt_trt_second_denoise_engine", "")), args.denoise_engine),
        "embed_prefix_engine": _coalesce(
            str(getattr(args, "trt_trt_second_embed_prefix_engine", "")),
            args.embed_prefix_engine,
        ),
    }


def policy_torch_model_for_perf(pol: Any) -> Any | None:
    m = getattr(pol, "_model", None)
    if m is not None:
        return m
    inner = getattr(pol, "_policy", None)
    if inner is not None:
        return getattr(inner, "_model", None)
    return None


def install_compare_pt_stage_perf(policy: Any, *, warmup_skips: int) -> None:
    """compare 第一路 PyTorch：安装与 TRT 路对齐的阶段耗时（sample_actions / denoise_step 等）。"""
    if not bool(os.environ.get("MO_PI0_STAGE_PROFILE", "").strip()):
        return
    model = policy_torch_model_for_perf(policy)
    if model is None:
        return
    try:
        from model_optimizer.infer.perf import StagePerfCollector, install_infer_stage_perf
        from model_optimizer.infer.tensorrt.pi0_stage_profiler import maybe_install_pi0_stage_profiler

        sp = StagePerfCollector(enabled=True, summary_prefix="[summary:model:pt]")
        install_infer_stage_perf(policy, model, sp, warmup_skips=int(warmup_skips))
        maybe_install_pi0_stage_profiler(model)
        setattr(policy, "_stage_perf", sp)
    except Exception as exc:
        logging.warning("compare_mode：PyTorch 路阶段 profiling 安装失败: %s", exc)


def apply_trt_stage_profile_env(args: Args) -> None:
    if bool(getattr(args, "trt_enable_stage_profile", True)):
        os.environ["MO_PI0_STAGE_PROFILE"] = "1"
        os.environ["PI05_PROFILE_WARMUP"] = str(int(getattr(args, "trt_stage_profile_warmup", 10)))


def apply_trt_hook_profile_env(args: Args, *, print_on_exit: bool) -> None:
    if bool(getattr(args, "trt_enable_hook_profile", True)):
        os.environ["MO_TRT_HOOK_STATS"] = "1"
        if print_on_exit:
            os.environ["MO_TRT_HOOK_STATS_PRINT"] = "1"
        else:
            os.environ.setdefault("MO_TRT_HOOK_STATS_PRINT", "0")


def create_trained_policy(train_cfg: Any, args: Args) -> Any:
    return policy_config.create_trained_policy(
        train_cfg,
        args.checkpoint,
        pytorch_device=args.device,
    )


def log_policy_ready(policy: Any, label: str) -> None:
    from termcolor import colored

    try:
        pd = getattr(policy, "_pytorch_device", None)
        is_pt = getattr(policy, "_is_pytorch_model", None)
        print(
            colored(f"[infer] policy({label}) 就绪 is_pytorch={is_pt} device={pd!r}", "cyan"),
            flush=True,
        )
    except Exception:
        pass


def get_train_config(args: Args) -> Any:
    return _config.get_config(args.config)
