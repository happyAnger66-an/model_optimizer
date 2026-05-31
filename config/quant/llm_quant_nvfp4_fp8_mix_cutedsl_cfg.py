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
# 若 weight 用 per-tensor（axis=None），gate 与 up 共用一个 amax，量纲差异会拉伸 FP8 scale、
# 损失较小幅度分支的精度。改用 **per-output-channel**（axis=0）后，2I 个输出行各自独立 scale，
# gate 行与 up 行互不影响；per-channel FP8 weight 是 TRT 原生支持的标准组合。
# 注意：纯 NVFP4 层无此问题（NVFP4 weight 沿输入维做 block 量化，按输出维 concat 不混 block）。
_FP8_WEIGHT_PC = {"num_bits": (4, 3), "axis": 0}

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
            # weight 走 per-channel（axis=0），activation 保持 per-tensor（axis=None）。
            qc[f"*layers.{i}.{sub}.weight_quantizer"] = dict(_FP8_WEIGHT_PC)
            qc[f"*layers.{i}.{sub}.input_quantizer"] = dict(_FP8_LINEAR)


if isinstance(_qc, dict):
    merged = dict(_qc)
    _apply_layerwise_fp8(merged)
    add_nvfp4_input_layernorm_explicit(merged)
    QUANT_CFG["quant_cfg"] = merged
else:
    # 旧版 list：在列表末尾追加更具体的 FP8 项（后项覆盖先项）。
    # weight 走 per-channel（axis=0），activation 保持 per-tensor（见 dict 分支说明）。
    extra: list = []
    for i in range(0, 18):
        for sub in _LLM_LINEAR_SUFFIXES:
            extra.append(
                {
                    "quantizer_name": f"*layers.{i}.{sub}.weight_quantizer",
                    "cfg": dict(_FP8_WEIGHT_PC),
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