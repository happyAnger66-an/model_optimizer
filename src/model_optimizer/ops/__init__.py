# Copyright 2026 the model_optimizer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Runtime extensions (custom ops, patches) used without modifying upstream model files."""

from .siglip_mlp import (
    SiglipMlpCustomOpWrapper,
    patch_vision_siglip_mlp_custom_op,
    siglip_mlp_eager,
)
from .siglip_mlp_plugin import (
    register_siglip_mlp_plugin_onnx_symbolic_functions,
    siglip_mlp_plugin,
)
from .siglip_ffn_fp8 import (
    SiglipFfFp8FlashrtMlpWrapper,
    TinySiglipFfFp8Mlp,
    discover_siglip_ffn_fp8_flashrt_plugin_so,
    discover_siglip_ffn_fp8_plugin_so,
    discover_siglip_mlp_trt_plugin_so,
    gpu_supports_fp8_trt,
    patch_vision_siglip_ffn_fp8_flashrt_custom_op,
    siglip_ffn_fp8_eager,
)
from .siglip_ffn_fp8_flashrt_export import (
    register_siglip_ffn_fp8_flashrt_plugin_onnx_symbolic_functions,
)
from .siglip_ffn_fp8_onnx import (
    TinySiglipMlpTrt,
    build_siglip_mlp_trt_onnx_from_module,
    register_siglip_mlp_trt_onnx_schema,
    register_siglip_ffn_fp8_onnx_schema,
    build_siglip_ffn_fp8_onnx_from_module,
)
from .gemma_fused_gated_mlp import (
    FusedGemmaMLP,
    patch_decoder_fused_gated_mlp,
    gemma_fused_gated_mlp_eager,
)
from .gemma_fused_gated_mlp_plugin import (
    gemma_fused_gated_mlp_plugin,
    register_gemma_fused_gated_mlp_onnx_symbolic_functions,
)

__all__ = [
    "SiglipMlpCustomOpWrapper",
    "patch_vision_siglip_mlp_custom_op",
    "siglip_mlp_eager",
    "register_siglip_mlp_plugin_onnx_symbolic_functions",
    "siglip_mlp_plugin",
    "SiglipFfFp8FlashrtMlpWrapper",
    "patch_vision_siglip_ffn_fp8_flashrt_custom_op",
    "register_siglip_ffn_fp8_flashrt_plugin_onnx_symbolic_functions",
    "siglip_ffn_fp8_eager",
    "TinySiglipFfFp8Mlp",
    "discover_siglip_ffn_fp8_flashrt_plugin_so",
    "discover_siglip_ffn_fp8_plugin_so",
    "discover_siglip_mlp_trt_plugin_so",
    "gpu_supports_fp8_trt",
    "TinySiglipMlpTrt",
    "register_siglip_mlp_trt_onnx_schema",
    "build_siglip_mlp_trt_onnx_from_module",
    "register_siglip_ffn_fp8_onnx_schema",
    "build_siglip_ffn_fp8_onnx_from_module",
    "FusedGemmaMLP",
    "patch_decoder_fused_gated_mlp",
    "gemma_fused_gated_mlp_eager",
    "gemma_fused_gated_mlp_plugin",
    "register_gemma_fused_gated_mlp_onnx_symbolic_functions",
]
