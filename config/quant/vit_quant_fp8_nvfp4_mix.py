import copy
from modelopt.torch.quantization import FP8_DEFAULT_CFG, NVFP4_DEFAULT_CFG

QUANT_CFG = copy.deepcopy(NVFP4_DEFAULT_CFG)


# 与 modelopt FP8_DEFAULT_CFG 中 Linear 一致：num_bits=(4,3), axis=None
_FP8_LINEAR = {"num_bits": (4, 3), "axis": None}

#QUANT_CFG["algorithm"] = "max"
#QUANT_CFG["mode"] = "int8"
QUANT_CFG["quant_cfg"]["input_layernorm"] = {"enable": False}
QUANT_CFG["quant_cfg"]["post_attention_layernorm"] = {"enable": False}
QUANT_CFG["quant_cfg"]["norm"] = {"enable": False}


QUANT_CFG["quant_cfg"]["vision_tower.vision_model.embeddings.patch_embedding.input_quantizer"] = {"enable": False}
QUANT_CFG["quant_cfg"]["vision_tower.vision_model.embeddings.patch_embedding.weight_quantizer"] = {"enable": False}
QUANT_CFG["quant_cfg"]["multi_modal_projector.linear.input_quantizer"] = {"enable": False}
QUANT_CFG["quant_cfg"]["multi_modal_projector.linear.weight_quantizer"] = {"enable": False}

# Disable quantization on encoder layers[0]: self_attn (q/k/v/out) + MLP fc1/fc2 (matches ``mtq.print_quant_summary`` FQNs).
_VIT_ENCODER_LAYER0_QUANTIZER_SKIP_SUFFIXES: tuple[str, ...] = (
   # "self_attn.k_proj.input_quantizer",
   # "self_attn.k_proj.output_quantizer",
   # "self_attn.k_proj.weight_quantizer",
   # "self_attn.v_proj.input_quantizer",
   # "self_attn.v_proj.output_quantizer",
   # "self_attn.v_proj.weight_quantizer",
   # "self_attn.q_proj.input_quantizer",
   # "self_attn.q_proj.output_quantizer",
   # "self_attn.q_proj.weight_quantizer",
   # "self_attn.out_proj.input_quantizer",
   # "self_attn.out_proj.output_quantizer",
   # "self_attn.out_proj.weight_quantizer",
    "mlp.fc1.input_quantizer",
    "mlp.fc1.output_quantizer",
    "mlp.fc1.weight_quantizer",
    "mlp.fc2.input_quantizer",
    "mlp.fc2.output_quantizer",
    "mlp.fc2.weight_quantizer",
)
_VIT_LAYER0_PREFIX = "vision_tower.vision_model.encoder.layers.{i}."

def _apply_layerwise_fp8(qc: dict) -> None:
    for i in range(0, 27):
        for sub in _VIT_ENCODER_LAYER0_QUANTIZER_SKIP_SUFFIXES:
            qc[f"vision_tower.vision_model.encoder.layers.{i}.{sub}.weight_quantizer"] = dict(_FP8_LINEAR)
            qc[f"vision_tower.vision_model.encoder.layers.{i}.{sub}.input_quantizer"] = dict(_FP8_LINEAR)

if isinstance(QUANT_CFG["quant_cfg"], dict):
    merged = dict(QUANT_CFG["quant_cfg"])
    _apply_layerwise_fp8(merged)
    QUANT_CFG["quant_cfg"] = merged
else:
    # 旧版 list：在列表末尾追加更具体的 FP8 项（后项覆盖先项）
    extra: list = []
    for i in range(0, 27):
        for sub in _VIT_ENCODER_LAYER0_QUANTIZER_SKIP_SUFFIXES:
            for kind in ("weight_quantizer", "input_quantizer"):
                extra.append(
                    {
                        "quantizer_name": f"*layers.{i}.{sub}.{kind}",
                        "cfg": dict(_FP8_LINEAR),
                    }
                )
    QUANT_CFG["quant_cfg"] = list(QUANT_CFG["quant_cfg"]) + extra