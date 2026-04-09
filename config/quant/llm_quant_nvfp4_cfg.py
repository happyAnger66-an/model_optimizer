import copy

from modelopt.torch.quantization import NVFP4_DEFAULT_CFG

# 新版 ModelOpt：quant_cfg 为 dict（Pydantic QuantizeConfig）；旧版可能为 list。
QUANT_CFG = copy.deepcopy(NVFP4_DEFAULT_CFG)
_qc = QUANT_CFG["quant_cfg"]

_DISABLE_DOWN_PROJ_IN = "*layers.17.mlp.down_proj.input_quantizer"

if isinstance(_qc, dict):
    QUANT_CFG["quant_cfg"] = {
        **_qc,
        _DISABLE_DOWN_PROJ_IN: {"enable": False},
        # 若仍不理想，可再加权重量化：
        # "*layers.17.mlp.down_proj.weight_quantizer": {"enable": False},
    }
else:
    QUANT_CFG["quant_cfg"] = list(_qc) + [
        {"quantizer_name": _DISABLE_DOWN_PROJ_IN, "enable": False},
    ]
