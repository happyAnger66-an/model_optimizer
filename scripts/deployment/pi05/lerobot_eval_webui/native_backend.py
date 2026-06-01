"""Native decoder 运行时挂载（Pi0.5）。"""

from __future__ import annotations

from typing import Any, Literal


def load_native_runtime(
    policy: Any,
    *,
    precision: Literal["fp16", "bf16", "fp32"],
    use_cuda_graph: bool = True,
    full_loop_graph: bool = False,
    graph_warmup: int = 3,
    compile_expert: bool = False,
    enable_expert: bool = True,
    enable_denoise: bool = True,
    quant_spec_path: str = "",
    recalib_enable: bool = False,
    recalib_max_samples: int = 0,
    recalib_percentile: float = 99.9,
    flashrt_decoder: bool = False,
    flashrt_build_dir: str = "",
    flashrt_fmha_so: str = "",
    flashrt_use_fp8: bool = True,
    flashrt_act_scales_path: str = "",
    flashrt_calibrate: bool = False,
    flashrt_calib_samples: int = 8,
) -> None:
    import addict
    import torch

    from model_optimizer.infer.native.pi05_executor import Pi05NativeExecutor

    if precision == "fp16":
        prec = torch.float16
    elif precision == "bf16":
        prec = torch.bfloat16
    else:
        prec = torch.float32

    executor = Pi05NativeExecutor(policy, prec)
    cfg = {
        "enable_expert": bool(enable_expert),
        "enable_denoise": bool(enable_denoise),
        "use_cuda_graph": bool(use_cuda_graph),
        "full_loop_graph": bool(full_loop_graph),
        "graph_warmup": int(graph_warmup),
        "compile_expert": bool(compile_expert),
        "perf": True,
        "quant_spec_path": str(quant_spec_path or ""),
        "recalib_enable": bool(recalib_enable),
        "recalib_max_samples": int(recalib_max_samples),
        "recalib_percentile": float(recalib_percentile),
        "flashrt_decoder": bool(flashrt_decoder),
        "flashrt_build_dir": str(flashrt_build_dir or ""),
        "flashrt_fmha_so": str(flashrt_fmha_so or ""),
        "flashrt_use_fp8": bool(flashrt_use_fp8),
        "flashrt_act_scales_path": str(flashrt_act_scales_path or ""),
        "flashrt_calibrate": bool(flashrt_calibrate),
        "flashrt_calib_samples": int(flashrt_calib_samples),
    }
    executor.load_model(addict.Dict(cfg))

