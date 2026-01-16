import json
from addict import Dict

def _flat_num_bits(num_bits_str):
    return tuple([int(bit) for bit in num_bits_str.split(',')])

def normalize_quant_cfg(quant_cfg):
    num_bits = quant_cfg["quant_cfg"]["*weight_quantizer"]['num_bits']
    quant_cfg["quant_cfg"]["*weight_quantizer"]['num_bits'] = _flat_num_bits(num_bits)
    
    num_bits = quant_cfg["quant_cfg"]["*input_quantizer"]['num_bits']
    quant_cfg["quant_cfg"]["*input_quantizer"]['num_bits'] = _flat_num_bits(num_bits)

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