import copy

from modelopt.torch.quantization import NVFP4_DEFAULT_CFG

from model_optimizer.quantization.cfg import add_nvfp4_input_layernorm_explicit

QUANT_CFG = copy.deepcopy(NVFP4_DEFAULT_CFG)
add_nvfp4_input_layernorm_explicit(QUANT_CFG["quant_cfg"])