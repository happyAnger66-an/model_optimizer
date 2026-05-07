import copy

from modelopt.torch.quantization import NVFP4_DEFAULT_CFG

from model_optimizer.quantization.cfg import add_nvfp4_input_layernorm_explicit

# 新版 ModelOpt：quant_cfg 为 dict（Pydantic QuantizeConfig）；旧版可能为 list。
# 策略：全局仍为 NVFP4_DEFAULT_CFG；对 **第 11–17 层** 的主要 Linear 量化器显式覆盖为 FP8（E4M3，与 FP8_DEFAULT_CFG 一致）。
# 第 **0–10 层** 不添加覆盖项，继续走 NVFP4 默认 `*` 规则。

QUANT_CFG = copy.deepcopy(NVFP4_DEFAULT_CFG)
_qc = QUANT_CFG["quant_cfg"]

QUANT_CFG["quant_cfg"]["input_layernorm"] = {"enable": False}
QUANT_CFG["quant_cfg"]["post_attention_layernorm"] = {"enable": False}
QUANT_CFG["quant_cfg"]["norm"] = {"enable": False}

_PI05_DENOISE_QUANTIZER_SKIP_NAMES: tuple[str, ...] = (
    # As printed by ``mtq.print_quant_summary`` for PI0Pytorch (Pi0.5).
    "action_in_proj.input_quantizer",
    "action_in_proj.output_quantizer",
    "action_in_proj.weight_quantizer",
    "action_out_proj.input_quantizer",
    "action_out_proj.output_quantizer",
    "action_out_proj.weight_quantizer",
#    "time_mlp_in.input_quantizer",
#    "time_mlp_in.output_quantizer",
#    "time_mlp_in.weight_quantizer",
#    "time_mlp_out.input_quantizer",
#    "time_mlp_out.output_quantizer",
#    "time_mlp_out.weight_quantizer",
)

for name in _PI05_DENOISE_QUANTIZER_SKIP_NAMES:
    QUANT_CFG["quant_cfg"][name] = {"enable": False}

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
    "mlp.down_proj",
)


def _apply_layerwise_fp8(qc: dict) -> None:
    for i in range(0, 18):
        for sub in _LLM_LINEAR_SUFFIXES:
            qc[f"*layers.{i}.{sub}.weight_quantizer"] = dict(_FP8_LINEAR)
            qc[f"*layers.{i}.{sub}.input_quantizer"] = dict(_FP8_LINEAR)


if isinstance(_qc, dict):
    merged = dict(_qc)
    _apply_layerwise_fp8(merged)
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
