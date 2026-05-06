import copy

from modelopt.torch.quantization import NVFP4_DEFAULT_CFG

from model_optimizer.quantization.cfg import add_nvfp4_input_layernorm_explicit

# 新版 ModelOpt：quant_cfg 为 dict（Pydantic QuantizeConfig）；旧版可能为 list。
# 策略：全局仍为 NVFP4_DEFAULT_CFG；对 **第 11–17 层** 的主要 Linear 量化器显式覆盖为 FP8（E4M3，与 FP8_DEFAULT_CFG 一致）。
# 第 **0–10 层** 不添加覆盖项，继续走 NVFP4 默认 `*` 规则。

QUANT_CFG = copy.deepcopy(NVFP4_DEFAULT_CFG)
_qc = QUANT_CFG["quant_cfg"]

# 与 modelopt FP8_DEFAULT_CFG 中 Linear 一致：num_bits=(4,3), axis=None
_FP8_LINEAR = {"num_bits": (4, 3), "axis": None}

# 与 print_quant_summary / 配置里常用的通配一致：*layers.{i}... 可匹配 model.layers.{i}...
_LLM_LINEAR_SUFFIXES = (
#    "self_attn.q_proj",
#    "self_attn.k_proj",
#    "self_attn.v_proj",
#    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
#    "mlp.down_proj",
)


def _apply_layerwise_fp8_11_17(qc: dict) -> None:
    for i in range(0, 18):
        for sub in _LLM_LINEAR_SUFFIXES:
            qc[f"*layers.{i}.{sub}.weight_quantizer"] = dict(_FP8_LINEAR)
            qc[f"*layers.{i}.{sub}.input_quantizer"] = dict(_FP8_LINEAR)


if isinstance(_qc, dict):
    merged = dict(_qc)
    _apply_layerwise_fp8_11_17(merged)
    add_nvfp4_input_layernorm_explicit(merged)
    QUANT_CFG["quant_cfg"] = merged
else:
    # 旧版 list：在列表末尾追加更具体的 FP8 项（后项覆盖先项）
    extra: list = []
    for i in range(11, 18):
        for sub in _LLM_LINEAR_SUFFIXES:
            for kind in ("weight_quantizer", "input_quantizer"):
                extra.append(
                    {
                        "quantizer_name": f"*layers.{i}.{sub}.{kind}",
                        "cfg": dict(_FP8_LINEAR),
                    }
                )
    QUANT_CFG["quant_cfg"] = list(_qc) + extra
