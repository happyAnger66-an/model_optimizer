"""Native / FlashRT 运行时挂载参数。"""

from __future__ import annotations

from typing import Any

from .config import Args
from .native_backend import load_native_runtime


def native_runtime_kwargs(args: Args, *, sample_actions_warmup_skips: int) -> dict[str, Any]:
    return {
        "precision": args.precision,
        "use_cuda_graph": bool(getattr(args, "native_use_cuda_graph", True)),
        "full_loop_graph": bool(getattr(args, "native_full_loop_graph", False)),
        "graph_warmup": int(getattr(args, "native_graph_warmup", 3)),
        "compile_expert": bool(getattr(args, "native_compile_expert", False)),
        "enable_expert": bool(getattr(args, "native_enable_expert", True)),
        "enable_denoise": bool(getattr(args, "native_enable_denoise", True)),
        "quant_spec_path": str(getattr(args, "native_quant_spec_path", "") or ""),
        "recalib_enable": bool(getattr(args, "native_recalib_enable", False)),
        "recalib_max_samples": int(getattr(args, "native_recalib_max_samples", 0)),
        "recalib_percentile": float(getattr(args, "native_recalib_percentile", 99.9)),
        "flashrt_decoder": bool(getattr(args, "native_flashrt_decoder", False)),
        "flashrt_build_dir": str(getattr(args, "native_flashrt_build_dir", "") or ""),
        "flashrt_fmha_so": str(getattr(args, "native_flashrt_fmha_so", "") or ""),
        "flashrt_use_fp8": bool(getattr(args, "native_flashrt_use_fp8", True)),
        "flashrt_act_scales_path": str(getattr(args, "native_flashrt_act_scales_path", "") or ""),
        "flashrt_calibrate": bool(getattr(args, "native_flashrt_calibrate", False)),
        "flashrt_calib_samples": int(getattr(args, "native_flashrt_calib_samples", 8)),
        "sample_actions_warmup_skips": int(sample_actions_warmup_skips),
    }


def load_native_overlay(
    policy: Any,
    args: Args,
    *,
    sample_actions_warmup_skips: int,
) -> Any:
    return load_native_runtime(
        policy,
        **native_runtime_kwargs(args, sample_actions_warmup_skips=sample_actions_warmup_skips),
    )


def attach_native_executor(policy: Any, native_executor: Any | None) -> None:
    if native_executor is None:
        return
    try:
        setattr(policy, "_native_executor", native_executor)
    except Exception:
        pass
