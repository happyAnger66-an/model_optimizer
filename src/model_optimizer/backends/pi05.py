"""Pi0.5 backend installers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from model_optimizer.architectures.pi05 import PI05_ARCHITECTURE_NAME
from model_optimizer.infer.server.config import ServerConfig

from .base import BackendInstaller, ProgressCallback
from .registry import register_backend_installer


def _resolve_precision(precision: str):
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


class Pi05TensorRTBackendInstaller(BackendInstaller):
    architecture = PI05_ARCHITECTURE_NAME
    backend = "tensorrt"

    def install(self, policy: Any, config: ServerConfig, on_progress: ProgressCallback) -> None:
        import addict

        from model_optimizer.infer.tensorrt.pi05_executor import Pi05TensorRTExecutor

        prec = _resolve_precision(config.precision)
        on_progress("tensorrt", "加载 TensorRT 引擎 …")
        executor = Pi05TensorRTExecutor(policy, prec)

        resolved = config.resolve_stages()
        trt_cfg: dict[str, Any] = {"engine_path": config.tensorrt.engine_path}
        attr2stage = {
            "vit_engine": "vit",
            "llm_engine": "llm",
            "expert_engine": "expert",
            "denoise_engine": "denoise",
            "embed_prefix_engine": "embed_prefix",
        }
        for attr, stage in attr2stage.items():
            if resolved.get(stage) != self.backend:
                continue
            val = getattr(config.tensorrt, attr, "")
            if val:
                trt_cfg[attr] = val

        if getattr(config.tensorrt, "denoise_adarms_precompute", False):
            trt_cfg["denoise_adarms_precompute"] = True
        if getattr(config.tensorrt, "vit_batch_views", False):
            trt_cfg["vit_batch_views"] = True
        if getattr(config.tensorrt, "llm_kv_only", False):
            trt_cfg["llm_kv_only"] = True
        llm_expected_seq_len = int(getattr(config.tensorrt, "llm_expected_seq_len", 0) or 0)
        if llm_expected_seq_len > 0:
            trt_cfg["llm_expected_seq_len"] = llm_expected_seq_len

        # Mixed stage matrix: vit=flashrt switches only the visual stack while
        # the remaining selected stages still use TensorRT.
        if resolved.get("vit") == "flashrt":
            ckpt = config.flashrt.checkpoint_dir or str(Path(config.tensorrt.engine_path).parent)
            trt_cfg["use_flashrt_siglip_embed_prefix"] = True
            trt_cfg["flashrt_checkpoint_dir"] = str(Path(ckpt).expanduser().resolve())
            trt_cfg["num_views"] = int(config.flashrt.num_views)
            trt_cfg["flashrt_siglip_use_cuda_graph"] = bool(config.flashrt.use_cuda_graph)
            on_progress("flashrt", f"vit 阶段切换到 FlashRT SigLIP（ckpt={trt_cfg['flashrt_checkpoint_dir']}）")

        executor.load_model(addict.Dict(trt_cfg))
        on_progress("tensorrt", "TensorRT 引擎已就绪")


class Pi05NativeBackendInstaller(BackendInstaller):
    architecture = PI05_ARCHITECTURE_NAME
    backend = "native"

    def install(self, policy: Any, config: ServerConfig, on_progress: ProgressCallback) -> None:
        import addict

        from model_optimizer.infer.native.pi05_executor import Pi05NativeExecutor

        resolved = config.resolve_stages()
        enable_expert = resolved.get("expert") == self.backend
        enable_denoise = resolved.get("denoise") == self.backend
        if not (enable_expert or enable_denoise):
            return

        prec = _resolve_precision(config.precision)
        on_progress("native", "加载 Native decoder 运行时 …")
        executor = Pi05NativeExecutor(policy, prec)
        native_cfg = {
            "enable_expert": bool(enable_expert),
            "enable_denoise": bool(enable_denoise),
            "use_cuda_graph": bool(getattr(config.native, "use_cuda_graph", True)),
            "full_loop_graph": bool(getattr(config.native, "full_loop_graph", False)),
            "graph_warmup": int(getattr(config.native, "graph_warmup", 3)),
            "compile_expert": bool(getattr(config.native, "compile_expert", False)),
            "perf": bool(getattr(config.native, "perf", True)),
            "quant_spec_path": str(getattr(config.native, "quant_spec_path", "") or ""),
            "recalib_enable": bool(getattr(config.native, "recalib_enable", False)),
            "recalib_max_samples": int(getattr(config.native, "recalib_max_samples", 0)),
            "recalib_percentile": float(getattr(config.native, "recalib_percentile", 99.9)),
        }
        executor.load_model(addict.Dict(native_cfg))
        on_progress("native", "Native decoder 已就绪")


class Pi05OnnxRTBackendInstaller(BackendInstaller):
    architecture = PI05_ARCHITECTURE_NAME
    backend = "onnxrt"

    def install(self, policy: Any, config: ServerConfig, on_progress: ProgressCallback) -> None:
        import addict

        from model_optimizer.infer.onnxrt.pi05_executor import Pi05OnnxRTExecutor

        prec = _resolve_precision(config.precision)
        on_progress("onnxrt", "加载 ONNX Runtime 引擎 …")
        executor = Pi05OnnxRTExecutor(policy, prec)

        ort_cfg: dict[str, Any] = {"engine_path": config.onnxrt.engine_path}
        for attr in ("vit_engine", "llm_engine", "expert_engine", "denoise_engine", "embed_prefix_engine"):
            val = getattr(config.onnxrt, attr, "")
            if val:
                ort_cfg[attr] = val

        if getattr(config.onnxrt, "denoise_adarms_precompute", False):
            ort_cfg["denoise_adarms_precompute"] = True

        executor.load_model(addict.Dict(ort_cfg))
        on_progress("onnxrt", "ONNX Runtime 引擎已就绪")


def register() -> None:
    register_backend_installer(Pi05TensorRTBackendInstaller())
    register_backend_installer(Pi05NativeBackendInstaller())
    register_backend_installer(Pi05OnnxRTBackendInstaller())


register()
