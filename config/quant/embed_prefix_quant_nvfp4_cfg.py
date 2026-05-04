import copy
from modelopt.torch.quantization import NVFP4_DEFAULT_CFG 
QUANT_CFG = copy.deepcopy(NVFP4_DEFAULT_CFG)

#QUANT_CFG["algorithm"] = "max"
#QUANT_CFG["mode"] = "int8"
QUANT_CFG["quant_cfg"]["input_layernorm"] = {"enable": False}
QUANT_CFG["quant_cfg"]["post_attention_layernorm"] = {"enable": False}
QUANT_CFG["quant_cfg"]["norm"] = {"enable": False}

QUANT_CFG["quant_cfg"]["vision_tower.vision_model.embeddings.patch_embedding.input_quantizer"] = {"enable": False}
QUANT_CFG["quant_cfg"]["vision_tower.vision_model.embeddings.patch_embedding.weight_quantizer"] = {"enable": False}