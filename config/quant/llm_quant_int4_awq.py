import copy

from modelopt.torch.quantization import INT4_AWQ_CFG

# W4A16：块大小默认沿最后一维 128（与 INT4_AWQ_CFG 一致）。若某 checkpoint 的 Linear in_features 不能被 128 整除，
# 需把 block_sizes[-1] 改为能整除 in_features 的因子（例如 64），否则 INT4 内核可能失败。
# Pi05 LLM 在 ``llm.LLM.quantize`` 中对 AWQ 标定会自动暂转 float32，避免 bf16 + awq_lite 下的 CUDA assert。

QUANT_CFG = copy.deepcopy(INT4_AWQ_CFG)
_qc = QUANT_CFG["quant_cfg"]
