"""Native decoder 运行时挂载（Pi0.5）。"""

from __future__ import annotations

from typing import Any, Literal


def load_native_runtime(
    policy: Any,
    *,
    precision: Literal["fp16", "bf16", "fp32"],
    use_cuda_graph: bool = True,
    graph_warmup: int = 3,
    compile_expert: bool = False,
    enable_expert: bool = True,
    enable_denoise: bool = True,
    quant_spec_path: str = "",
    recalib_enable: bool = False,
    recalib_max_samples: int = 0,
    recalib_percentile: float = 99.9,
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
        "graph_warmup": int(graph_warmup),
        "compile_expert": bool(compile_expert),
        "perf": True,
        "quant_spec_path": str(quant_spec_path or ""),
        "recalib_enable": bool(recalib_enable),
        "recalib_max_samples": int(recalib_max_samples),
        "recalib_percentile": float(recalib_percentile),
    }
    executor.load_model(addict.Dict(cfg))

