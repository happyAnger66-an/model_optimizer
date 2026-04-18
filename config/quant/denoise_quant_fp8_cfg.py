from modelopt.torch.quantization import INT8_DEFAULT_CFG, INT8_SMOOTHQUANT_CFG, FP8_DEFAULT_CFG
QUANT_CFG = FP8_DEFAULT_CFG

#QUANT_CFG["algorithm"] = "max"
#QUANT_CFG["mode"] = "int8"
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
    "time_mlp_in.input_quantizer",
    "time_mlp_in.output_quantizer",
    "time_mlp_in.weight_quantizer",
    "time_mlp_out.input_quantizer",
    "time_mlp_out.output_quantizer",
    "time_mlp_out.weight_quantizer",
)

for name in _PI05_DENOISE_QUANTIZER_SKIP_NAMES:
    QUANT_CFG["quant_cfg"][name] = {"enable": False}

_DENOISE_LINEAR_SUFFIXES = (
#    "self_attn.q_proj",
#    "self_attn.k_proj",
#    "self_attn.v_proj",
#    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
#    "mlp.down_proj",
)


def _apply_layerwise_fp8_disable(qc: dict) -> None:
    for i in range(0, 18):
        for sub in _DENOISE_LINEAR_SUFFIXES:
            qc[f"*layers.{i}.{sub}.weight_quantizer"] = {"enable": False}
            qc[f"*layers.{i}.{sub}.input_quantizer"] = {"enable": False}

_apply_layerwise_fp8_disable(QUANT_CFG["quant_cfg"])