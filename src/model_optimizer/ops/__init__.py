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
    TinySiglipFfFp8Mlp,
    discover_siglip_ffn_fp8_plugin_so,
    gpu_supports_fp8_trt,
    siglip_ffn_fp8_eager,
)
from .siglip_ffn_fp8_onnx import (
    build_siglip_ffn_fp8_onnx_from_module,
    register_siglip_ffn_fp8_onnx_schema,
)

__all__ = [
    "SiglipMlpCustomOpWrapper",
    "patch_vision_siglip_mlp_custom_op",
    "siglip_mlp_eager",
    "register_siglip_mlp_plugin_onnx_symbolic_functions",
    "siglip_mlp_plugin",
    "siglip_ffn_fp8_eager",
    "TinySiglipFfFp8Mlp",
    "discover_siglip_ffn_fp8_plugin_so",
    "gpu_supports_fp8_trt",
    "register_siglip_ffn_fp8_onnx_schema",
    "build_siglip_ffn_fp8_onnx_from_module",
]
