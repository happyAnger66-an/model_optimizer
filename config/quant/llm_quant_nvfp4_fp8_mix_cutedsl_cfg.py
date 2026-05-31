import copy

from modelopt.torch.quantization import NVFP4_DEFAULT_CFG,FP8_KV_CFG

from model_optimizer.quantization.cfg import add_nvfp4_input_layernorm_explicit

# 新版 ModelOpt：quant_cfg 为 dict（Pydantic QuantizeConfig）；旧版可能为 list。
# 策略：全局仍为 NVFP4_DEFAULT_CFG；对 **第 11–17 层** 的主要 Linear 量化器显式覆盖为 FP8（E4M3，与 FP8_DEFAULT_CFG 一致）。
# 第 **0–10 层** 不添加覆盖项，继续走 NVFP4 默认 `*` 规则。

QUANT_CFG = copy.deepcopy(NVFP4_DEFAULT_CFG)
_qc = QUANT_CFG["quant_cfg"]

# 与 modelopt FP8_DEFAULT_CFG 中 Linear 一致：num_bits=(4,3), axis=None（per-tensor）。
# 用于 **激活**（input_quantizer）。
_FP8_LINEAR = {"num_bits": (4, 3), "axis": None}

# roadmap #9：fused MLP 把 gate/up 在输出维（dim=0）concat 成 gate_up_proj=[2I, H]。
# ⚠️ FP8 weight 只能 per-tensor（axis=None）：ModelOpt 的 FP8 ONNX 导出（标准 QuantizeLinear 与
# 插件 trt::TRT_FP8QuantizeLinear）scale 均为单个标量——export_fp8 直接 `448.0/float(amax)`，
# tensor_quantizer._check_onnx_readiness 也对 tuple 类型 num_bits 强制 `amax.numel()==1`。
# 因此 axis=0（per-channel FP8）会在 torch.onnx.export 阶段断言失败，**不可用于 FP8**。
# 想要细于 per-tensor 的 weight 量化请用 NVFP4（本配置未被覆盖的层即走 NVFP4 的 per-block 量化，
# 沿输入维 block、按输出维 concat 不混 block，天然兼容 fused MLP）。per-channel 仅对 INT8 导出成立。
_FP8_WEIGHT = {"num_bits": (4, 3), "axis": None}

# 与 print_quant_summary / 配置里常用的通配一致：*layers.{i}... 可匹配 model.layers.{i}...
_LLM_LINEAR_SUFFIXES = (
    #"self_attn.q_proj",
    #self_attn.k_proj",
    #self_attn.v_proj",
    #self_attn.o_proj",
    "mlp.gate_up_proj",
    #"mlp.down_proj",
)


def _apply_layerwise_fp8(qc: dict) -> None:
    for i in range(3, 18):
        if i in [6, 10, 14, 17]:
            continue
        for sub in _LLM_LINEAR_SUFFIXES:
            # weight 与 activation 均 per-tensor（axis=None）：FP8 ONNX 导出仅支持 per-tensor。
            qc[f"*layers.{i}.{sub}.weight_quantizer"] = dict(_FP8_WEIGHT)
            qc[f"*layers.{i}.{sub}.input_quantizer"] = dict(_FP8_LINEAR)


if isinstance(_qc, dict):
    merged = dict(_qc)
    _apply_layerwise_fp8(merged)
    add_nvfp4_input_layernorm_explicit(merged)
    QUANT_CFG["quant_cfg"] = merged
else:
    # 旧版 list：在列表末尾追加更具体的 FP8 项（后项覆盖先项）。
    # weight 与 activation 均 per-tensor（FP8 ONNX 导出仅支持 per-tensor，见 dict 分支说明）。
    extra: list = []
    for i in range(0, 18):
        for sub in _LLM_LINEAR_SUFFIXES:
            extra.append(
                {
                    "quantizer_name": f"*layers.{i}.{sub}.weight_quantizer",
                    "cfg": dict(_FP8_WEIGHT),
                }
            )
            extra.append(
                {
                    "quantizer_name": f"*layers.{i}.{sub}.input_quantizer",
                    "cfg": dict(_FP8_LINEAR),
                }
            )
    QUANT_CFG["quant_cfg"] = list(_qc) + extra


FP8_ATTN = {
    "quant_cfg": {
        "*q_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True
        },
        "*k_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True
        },
        "*v_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "enable": True
        },
    }
}
QUANT_CFG["quant_cfg"].update(FP8_KV_CFG["quant_cfg"])
QUANT_CFG["quant_cfg"].update(FP8_ATTN["quant_cfg"])