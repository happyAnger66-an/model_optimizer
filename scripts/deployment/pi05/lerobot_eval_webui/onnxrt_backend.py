"""ONNX Runtime 引擎加载（Pi0.5）。"""

from __future__ import annotations

from typing import Any


def load_onnxrt_engines(
    policy: Any,
    *,
    engine_path: str,
    vit_engine: str = "",
    llm_engine: str = "",
    expert_engine: str = "",
    denoise_engine: str = "",
    embed_prefix_engine: str = "",
) -> None:
    import addict

    from model_optimizer.infer.onnxrt.pi05_executor import Pi05OnnxRTExecutor

    executor = Pi05OnnxRTExecutor(policy)
    cfg: dict[str, str] = {"engine_path": engine_path}
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
    executor.load_model(addict.Dict(cfg))
