# Pi05DenoiseStep FP8 量化配置：AdaRMS Dense 预计算 + fused MLP（gate/up 合并）。
#
# 开启 fused_mlp 后，每层 ``mlp.gate_proj`` / ``mlp.up_proj`` 已合并为单个
# ``mlp.gate_up_proj``（[2I, H]）。``FP8_DEFAULT_CFG`` 会按 **per-tensor**（axis=None）
# 自动量化它，无需额外规则；本文件与 ``denoise_adrams_st_quant_fp8_cfg.py`` 的差异仅在于
# 显式标注 gate_up_proj 走 per-tensor，便于对照。
#
# ⚠️ 为什么不用 per-channel（axis=0）：
#   标准 ONNX 的 FP8(E4M3) 导出只支持 **per-tensor**（ModelOpt
#   ``tensor_quantizer._check_onnx_readiness`` 要求 amax 为标量）。对 gate_up_proj 设
#   axis=0 会得到 [2I,1] 的 amax，在 torch.onnx.export 阶段断言失败：
#   "E4M3 supports ONNX export only for per-tensor quantization"。
#   per-channel FP8 仅在 TRT 插件 QDQ 导出路径（如 cutedsl LLM）可用；denoise 走标准导出，
#   故 gate_up_proj 用 per-tensor。若需更高精度可改用 NVFP4（沿输入维 block 量化，
#   按输出维 concat 不混 block，无 gate/up 量纲混叠问题）。
#
# 其余项（layernorm / dense / action_in_proj / action_out_proj / time_mlp_in/out）与
# adarms 版一致：AdaRMS 的 dense 与 time_mlp 在预计算模式下已移出导出图，量化它们无意义；
# action_in_proj 保持 bf16 以避免 Add(MatMul, bias) 的 BF16/Float 类型冲突。
from modelopt.torch.quantization import FP8_DEFAULT_CFG

QUANT_CFG = FP8_DEFAULT_CFG

# AdaRMS 归一化（自身与其 dense 子层）不量化。
QUANT_CFG["quant_cfg"]["input_layernorm"] = {"enable": False}
QUANT_CFG["quant_cfg"]["post_attention_layernorm"] = {"enable": False}
QUANT_CFG["quant_cfg"]["norm"] = {"enable": False}
QUANT_CFG["quant_cfg"]["*input_layernorm.dense*"] = {"enable": False}
QUANT_CFG["quant_cfg"]["*post_attention_layernorm.dense*"] = {"enable": False}
QUANT_CFG["quant_cfg"]["*norm.dense*"] = {"enable": False}

# action / time 投影保持 bf16（见上方说明）。
_PI05_DENOISE_QUANTIZER_SKIP_NAMES: tuple[str, ...] = (
    "action_in_proj.input_quantizer",
    "action_in_proj.output_quantizer",
    "action_in_proj.weight_quantizer",
    "action_out_proj.input_quantizer",
    "action_out_proj.output_quantizer",
    "action_out_proj.weight_quantizer",
    "time_mlp_in.input_quantizer",
    "time_mlp_in.output_quantizer",
    "time_mlp_in.weight_quantizer",
    "time_mlp_out.input_quantizer",
    "time_mlp_out.output_quantizer",
    "time_mlp_out.weight_quantizer",
)
for name in _PI05_DENOISE_QUANTIZER_SKIP_NAMES:
    QUANT_CFG["quant_cfg"][name] = {"enable": False}

# fused MLP 的 FC1：标准 ONNX FP8 导出必须 per-tensor（axis=None）。
# 这里显式写出仅为可读性；即便不写，FP8_DEFAULT_CFG 的 ``*`` 默认规则也会按 per-tensor 量化。
_FP8_PER_TENSOR = {"num_bits": (4, 3), "axis": None}


def _apply_layerwise_fp8(qc: dict, num_layers: int = 18) -> None:
    for i in range(0, num_layers):
        qc[f"*layers.{i}.mlp.gate_up_proj.weight_quantizer"] = dict(_FP8_PER_TENSOR)
        qc[f"*layers.{i}.mlp.gate_up_proj.input_quantizer"] = dict(_FP8_PER_TENSOR)


_apply_layerwise_fp8(QUANT_CFG["quant_cfg"])
