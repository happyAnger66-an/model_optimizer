import copy
from typing import Any

import modelopt.torch.quantization as mtq
from modelopt.torch.utils import print_rank_0

QUANT_CFG_CHOICES: dict[str, dict[str, Any]] = {
    "int8": mtq.INT8_DEFAULT_CFG,
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
    "int8_wo": mtq.INT8_WEIGHT_ONLY_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    "nvfp4_awq": mtq.NVFP4_AWQ_LITE_CFG,
    "fp8_pb_wo": mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
    "fp8_pc_pt": mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG,
    "w4a8_nvfp4_fp8": mtq.W4A8_NVFP4_FP8_CFG,
    "w4a8_mxfp4_fp8": mtq.W4A8_MXFP4_FP8_CFG,
    "nvfp4_mlp_only": mtq.NVFP4_MLP_ONLY_CFG,
}

KV_QUANT_CFG_CHOICES = {
    "none": "none",
    "fp8": "FP8_KV_CFG",
    "fp8_affine": "FP8_AFFINE_KV_CFG",
    "nvfp4": "NVFP4_KV_CFG",
    "nvfp4_affine": "NVFP4_AFFINE_KV_CFG",
    "nvfp4_rotate": "NVFP4_KV_ROTATE_CFG",
}

def update_quant_cfg_with_kv_cache_quant(
    quant_cfg: dict[str, Any], kv_cache_quant_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Update the quant_cfg with the kv cache quant_cfg."""
    # If quant_cfg["quant_cfg"] is None, it corresponds to only kv cache quantization case
    quant_cfg["quant_cfg"] = quant_cfg.get("quant_cfg", {"default": {"enable": False}})
    quant_cfg["quant_cfg"].update(kv_cache_quant_cfg)

    # Set default algorithm for kv cache quantization if not provided.
    if not quant_cfg.get("algorithm"):
        quant_cfg["algorithm"] = "max"
    print_rank_0(f"Updated quant_cfg with KV cache quantization: {quant_cfg}")
    return quant_cfg

def build_quant_cfg(
    qformat,
    kv_cache_qformat,
    awq_block_size,
    auto_quantize,
    model_type,
    quant_cfg_choices,
    kv_quant_cfg_choices,
):
    quant_cfg = {}
    if not auto_quantize:
        assert qformat in quant_cfg_choices, (
            f"Unsupported quantization format: {qformat} with {kv_cache_qformat} KV cache"
        )

        quant_cfg = quant_cfg_choices[qformat]

        if "awq" in qformat:
            quant_cfg = copy.deepcopy(quant_cfg_choices[qformat])
            weight_quantizer = quant_cfg["quant_cfg"]["*weight_quantizer"]
            if isinstance(weight_quantizer, list):
                weight_quantizer = weight_quantizer[0]
            # If awq_block_size argument is provided, update weight_quantizer
            if awq_block_size:
                weight_quantizer["block_sizes"][-1] = awq_block_size

            # Coarser optimal scale search seems to resolve the overflow in TRT-LLM for some models
            if qformat == "w4a8_awq" and model_type in ["gemma", "mpt"]:
                quant_cfg["algorithm"] = {"method": "awq_lite", "alpha_step": 1}

        enable_quant_kv_cache = kv_cache_qformat != "none"
        print(f"{'Enable' if enable_quant_kv_cache else 'Disable'} KV cache quantization")

        # Check if any bmm_quantizer is in the quant_cfg. If so, we need to enable the bmm_quantizer.
        if enable_quant_kv_cache:
            quant_cfg = update_quant_cfg_with_kv_cache_quant(
                quant_cfg,
                getattr(mtq, kv_quant_cfg_choices[kv_cache_qformat])["quant_cfg"],
            )

        # Gemma 7B has accuracy regression using alpha 1. We set 0.5 instead.
        if model_type == "gemma" and "int8_sq" in qformat:
            quant_cfg["algorithm"] = {"method": "smoothquant", "alpha": 0.5}

        if model_type == "phi4mm":
            # Only quantize the language model
            quant_cfg["quant_cfg"]["*speech*"] = {"enable": False}
            quant_cfg["quant_cfg"]["*audio*"] = {"enable": False}
            quant_cfg["quant_cfg"]["*image*"] = {"enable": False}
            quant_cfg["quant_cfg"]["*vision*"] = {"enable": False}

    return quant_cfg