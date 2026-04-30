import copy
from modelopt.torch.quantization import INT8_DEFAULT_CFG, INT8_SMOOTHQUANT_CFG, FP8_DEFAULT_CFG

QUANT_CFG = copy.deepcopy(FP8_DEFAULT_CFG)

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
    "self_attn.k_proj.input_quantizer",
    "self_attn.k_proj.output_quantizer",
    "self_attn.k_proj.weight_quantizer",
    "self_attn.v_proj.input_quantizer",
    "self_attn.v_proj.output_quantizer",
    "self_attn.v_proj.weight_quantizer",
    "self_attn.q_proj.input_quantizer",
    "self_attn.q_proj.output_quantizer",
    "self_attn.q_proj.weight_quantizer",
    "self_attn.out_proj.input_quantizer",
    "self_attn.out_proj.output_quantizer",
    "self_attn.out_proj.weight_quantizer",
    "mlp.fc1.input_quantizer",
    "mlp.fc1.output_quantizer",
    "mlp.fc1.weight_quantizer",
    "mlp.fc2.input_quantizer",
    "mlp.fc2.output_quantizer",
    "mlp.fc2.weight_quantizer",
)
_VIT_LAYER0_PREFIX = "vision_tower.vision_model.encoder.layers.{i}."

for i in range(0, 27):
    for sub in _VIT_ENCODER_LAYER0_QUANTIZER_SKIP_SUFFIXES:
        QUANT_CFG["quant_cfg"][f"vision_tower.vision_model.encoder.layers.{i}.{sub}"] = {"enable": False}
