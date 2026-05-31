# Pi05DenoiseStep FP8 量化配置：AdaRMS Dense 预计算 + fused MLP（gate/up 合并）。
#
# 与 ``denoise_adrams_st_quant_fp8_cfg.py`` 的区别：开启 fused_mlp 后，每层
# ``mlp.gate_proj`` / ``mlp.up_proj`` 已合并为单个 ``mlp.gate_up_proj``（[2I, H]）。
# 若对其用 per-tensor（axis=None）权重量化，gate 段与 up 段共用一个 amax，量纲差异会
# 拉伸 FP8 scale、损失较小幅度分支精度；故改用 **per-output-channel（axis=0）** 权重量化，
# 2I 个输出行各自独立 scale（TRT 原生支持的标准 FP8 组合）。激活仍 per-tensor。
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

# fused MLP 的 FC1：weight 走 per-channel（axis=0），activation 保持 per-tensor。
_FP8_WEIGHT_PC = {"num_bits": (4, 3), "axis": 0}
_FP8_ACT = {"num_bits": (4, 3), "axis": None}

# 与 fused_mlp 合并后的模块名一致：``*layers.{i}.mlp.gate_up_proj``。
_DENOISE_FUSED_SUFFIXES = (
    "mlp.gate_up_proj",
    # "mlp.down_proj",   # 默认 per-tensor FP8 即可，如需 per-channel 可解开。
)


def _apply_layerwise_fp8(qc: dict, num_layers: int = 18) -> None:
    for i in range(0, num_layers):
        for sub in _DENOISE_FUSED_SUFFIXES:
            qc[f"*layers.{i}.{sub}.weight_quantizer"] = dict(_FP8_WEIGHT_PC)
            qc[f"*layers.{i}.{sub}.input_quantizer"] = dict(_FP8_ACT)


_apply_layerwise_fp8(QUANT_CFG["quant_cfg"])
