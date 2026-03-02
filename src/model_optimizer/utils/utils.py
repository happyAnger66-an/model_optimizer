import json
import torch.nn as nn
from addict import Dict

from modelopt.torch.quantization.utils import is_quantized_linear

def is_nvfp4_quantized(quant_cfg) -> bool:
    if "*input_quantizer" in quant_cfg["quant_cfg"]:
        input_quantize_cfg = quant_cfg["quant_cfg"]["*input_quantizer"]
        if input_quantize_cfg["num_bits"][0] == 2:
            return True
    return False

def is_nvfp4_linear(module: nn.Module) -> bool:
    """Check if the module is a quantized linear layer with NVFP4 quantization. The test is designed for identification purpose only, not designed to be comprehensive.
    Adapted from TensorRT Model Optimizer: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/modelopt/torch/_deploy/utils/torch_onnx.py
    """
    if is_quantized_linear(module):
        return module.input_quantizer.block_sizes is not None and module.input_quantizer.block_sizes.get(
            "scale_bits", None) == (4, 3)
    return False


def set_dynamic_quant(model: nn.Module, dtype: str) -> None:
    """Set quantization for nvfp4 and mxfp8 quantization."""
    for module in model.modules():
        if is_nvfp4_linear(module):
            module.input_quantizer._trt_high_precision_dtype = "Half" if dtype == "fp16" else "BFloat16"
            module.input_quantizer._onnx_quantizer_type = "dynamic"
            module.weight_quantizer._onnx_quantizer_type = "static"


def is_fp4_quantized(model: nn.Module) -> bool:
    """Check if the model is quantized in NVFP4 mode."""
    for _, module in model.named_modules():
        if (hasattr(module, "input_quantizer")
                and module.input_quantizer.block_sizes
                and module.input_quantizer.block_sizes.get("scale_bits",
                                                           None) == (4, 3)):
            return True
    return False


def _flat_num_bits(num_bits_str):
    return tuple([int(bit) for bit in num_bits_str.split(',')])


def normalize_quant_cfg(quant_cfg):
    num_bits = quant_cfg["quant_cfg"]["*weight_quantizer"]['num_bits']
    quant_cfg["quant_cfg"]["*weight_quantizer"]['num_bits'] = _flat_num_bits(
        num_bits)

    num_bits = quant_cfg["quant_cfg"]["*input_quantizer"]['num_bits']
    quant_cfg["quant_cfg"]["*input_quantizer"]['num_bits'] = _flat_num_bits(
        num_bits)

    quant_mode = quant_cfg["quant_mode"]
    del quant_cfg["quant_mode"]
    return quant_mode, quant_cfg


def load_quant_json(json_file):
    with open(json_file) as f:
        return Dict(json.load(f))


def get_model_input_shape(model_name):
    """Get the input shape from timm model configuration."""
    data_config = {}
    input_size = data_config["input_size"]
    return (1, *tuple(input_size))  # Add batch dimension
