import copy

from modelopt.torch.quantization import NVFP4_DEFAULT_CFG

# 与 ModelOpt 默认 NVFP4 一致，并追加按层关闭规则（quant_cfg 列表越靠后越优先）。
QUANT_CFG = copy.deepcopy(NVFP4_DEFAULT_CFG)
QUANT_CFG["quant_cfg"] = list(QUANT_CFG["quant_cfg"]) + [
    # amax 异常大时仅关掉该 Linear 的输入量化（summary 中的全名含此前缀）
    {"quantizer_name": "*layers.17.mlp.down_proj.input_quantizer", "enable": False},
    # 若仍不理想，可一并关掉权重量化：
    # {"quantizer_name": "*layers.17.mlp.down_proj.weight_quantizer", "enable": False},
]
