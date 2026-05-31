"""TensorRT 引擎加载（Pi0.5）。"""

from __future__ import annotations

from typing import Any, Literal


def load_tensorrt_engines(
    policy: Any,
    *,
    engine_path: str,
    precision: Literal["fp16", "bf16", "fp32"],
    vit_engine: str,
    llm_engine: str,
    expert_engine: str,
    denoise_engine: str,
    embed_prefix_engine: str,
    vit_batch_views: bool = False,
    trt_perf: bool = True,
    trt_cuda_graph: bool = False,
    trt_cuda_graph_warmup: int = 3,
    denoise_adarms_precompute: bool = False,
) -> None:
    import addict
    import torch

    from model_optimizer.infer.tensorrt.pi05_executor import Pi05TensorRTExecutor

    if precision == "fp16":
        prec = torch.float16
    elif precision == "bf16":
        prec = torch.bfloat16
    else:
        prec = torch.float32

    executor = Pi05TensorRTExecutor(policy, prec)
    cfg: dict = {"engine_path": engine_path}
    if vit_engine:
        cfg["vit_engine"] = vit_engine
    if expert_engine:
        cfg["expert_engine"] = expert_engine
    if llm_engine:
        cfg["llm_engine"] = llm_engine
    if denoise_engine:
        cfg["denoise_engine"] = denoise_engine
    if embed_prefix_engine:
        cfg["embed_prefix_engine"] = embed_prefix_engine
    if vit_batch_views:
        cfg["vit_batch_views"] = True
    cfg["trt_perf"] = bool(trt_perf)
    cfg["trt_cuda_graph"] = bool(trt_cuda_graph)
    cfg["trt_cuda_graph_warmup"] = int(trt_cuda_graph_warmup)
    cfg["denoise_adarms_precompute"] = bool(denoise_adarms_precompute)
    executor.load_model(addict.Dict(cfg))
