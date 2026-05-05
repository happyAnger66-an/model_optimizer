import copy

from modelopt.torch.quantization import NVFP4_AWQ_LITE_CFG

# 新版 ModelOpt：quant_cfg 为 dict（Pydantic QuantizeConfig）；旧版可能为 list。
# 策略：全局仍为 NVFP4_DEFAULT_CFG；对 **第 11–17 层** 的主要 Linear 量化器显式覆盖为 FP8（E4M3，与 FP8_DEFAULT_CFG 一致）。
# 第 **0–10 层** 不添加覆盖项，继续走 NVFP4 默认 `*` 规则。

QUANT_CFG = copy.deepcopy(NVFP4_AWQ_LITE_CFG)
_qc = QUANT_CFG["quant_cfg"]
