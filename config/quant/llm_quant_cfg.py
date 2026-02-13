from modelopt.torch.quantization import INT8_DEFAULT_CFG, INT8_SMOOTHQUANT_CFG, FP8_DEFAULT_CFG
QUANT_CFG = INT8_DEFAULT_CFG

#QUANT_CFG["algorithm"] = "max"
#QUANT_CFG["mode"] = "int8"
QUANT_CFG["quant_cfg"]["input_layernorm"] = {"enable": False}
QUANT_CFG["quant_cfg"]["post_attention_layernorm"] = {"enable": False}
QUANT_CFG["quant_cfg"]["norm"] = {"enable": False}