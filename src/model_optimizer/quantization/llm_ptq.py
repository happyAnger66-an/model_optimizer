import os
import time
import warnings
from typing import Any

import datasets
import torch

from modelopt.torch.utils.dataset_utils import get_dataset_dataloader, \
    create_forward_loop, get_max_batch_size
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import need_calibration
from modelopt.torch.export import (
    get_model_type,
)

from transformers import AutoTokenizer, AutoConfig, \
    PreTrainedTokenizer, \
    PreTrainedTokenizerFast

from .models.utils import get_language_model_from_vl, get_model
from .cfg import QUANT_CFG_CHOICES, KV_QUANT_CFG_CHOICES, \
    build_quant_cfg


def get_tokenizer(ckpt_path, trust_remote_code=False, **kwargs):
    print(f"Initializing tokenizer from {ckpt_path}")

    if "vila" in ckpt_path.lower():
        ckpt_path += "/llm"

    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path, trust_remote_code=trust_remote_code, **kwargs
    )

    # can't set attribute 'pad_token' for "<unk>"
    # We skip this step for Nemo models
    if tokenizer.pad_token != "<unk>" or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    assert tokenizer.pad_token is not None, f"Pad token for {ckpt_path} cannot be set!"

    return tokenizer


def quantize_model(model, quant_cfg, args, calib_dataloader=None, calibration_only=False):
    # The calibration loop for the model can be setup using the modelopt API.
    #
    # Example usage:
    # from modelopt.torch.utils.dataset_utils import create_forward_loop
    # model = ...  # Initialize the model
    # tokenizer = ...  # Initialize the tokenizer
    # quant_cfg = ...  # Setup quantization configuration
    # forward_loop = create_forward_loop(model=model, dataset_name="cnn_dailymail", tokenizer=tokenizer)
    # mtq.quantize(model, quant_cfg, forward_loop=forward_loop)

    # The calibrate_loop is a custom defined method to run the model with the input data.
    # The basic version looks like:
    #
    # def calibrate_loop(model, dataloader):
    #     for data in dataloader:
    #         model(**data)
    #
    # We also provided a util method to generate the forward_loop with additional error handlings.
    use_calibration = args.auto_quantize_bits or need_calibration(quant_cfg)

    if not use_calibration:
        warnings.warn("Dynamic quantization. Calibration skipped.")
    calibrate_loop = create_forward_loop(
        dataloader=calib_dataloader) if use_calibration else None

    assert not (args.auto_quantize_bits and args.inference_pipeline_parallel > 1), (
        "Auto Quantization is not supported for pipeline parallel size > 1"
    )

    print("Starting quantization...")
    start_time = time.time()
    if args.auto_quantize_bits:
        print(f' Not support auto_quantize_bits now.')
        return
    elif calibration_only:
        model = mtq.calibrate(
            model, quant_cfg["algorithm"], forward_loop=calibrate_loop)
    else:
        model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    end_time = time.time()
    print(f"Quantization done. Total time used: {end_time - start_time}s")

    mtq.print_quant_summary(model)
    return model

def export_llm_model(model, llm_dtype, model_config, output_dir):
    inputs_ids = torch.randn(
        (1, 512),
        dtype=torch.float16,
    ).cuda()
    attention_mask = None,
    position_ids = None,
    past_key_values = None,
    inputs_embeds = None,
    use_cache = False
#    attention_mask = torch.ones((1, 32), dtype=torch.int64).cuda()

    full_layer_quant = False
    # Use different filename for full layer quantization
    llm_dtype_suffix = (
        f"{llm_dtype}_full" if (llm_dtype == "nvfp4" and full_layer_quant) else llm_dtype
    )
    onnx_path = f"{output_dir}/llm_{llm_dtype_suffix}.onnx"
    onnx_dir = os.path.dirname(onnx_path)
    os.makedirs(onnx_dir, exist_ok=True)

    start_time = time.time()
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (inputs_ids),
            onnx_path,
            input_names=["input_ids"],
            output_names=["hidden_states"],
            opset_version=19,
            do_constant_folding=True,
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "hidden_states": {0: "batch_size", 1: "sequence_length"},

            },
        )
    end_time = time.time()
    print(f'export llm model to {onnx_path} cost:{end_time - start_time}s')

def llm_quantize(args):
    tokenizer = get_tokenizer(args.model_path)

    default_padding_side = tokenizer.padding_side
    # Left padding usually provides better calibration result.
    tokenizer.padding_side = "left"

    assert tokenizer is not None and isinstance(
        tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)
    ), "The PreTrainedTokenizer must be set"

    # Prepare config kwargs for loading
    #config_kwargs = {"trust_remote_code": trust_remote_code} if trust_remote_code else {}
    config_kwargs = {}

    # Load config once and handle VL model detection
    try:
        hf_config = AutoConfig.from_pretrained(args.model_path, **config_kwargs)
    except Exception as e:
        print(f"Error: Could not load config from {args.model_path}: {e}")
        raise RuntimeError(f"Failed to load model configuration from {args.model_path}") from e

    model = get_model(args.model_path)
    device = model.device

    include_labels = False
    # include_labels = (
    #    args.auto_quantize_bits is not None and args.auto_quantize_method == "gradient"
    # )

    args.batch_size = 1
    args.calib_size = 10
    calib_dataloader = get_dataset_dataloader(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_samples=args.calib_size,
        device=device,
        include_labels=include_labels,
    )
    from ..datasets.dummy_dataloader import LLMDummyDataLoader
    calib_dataloader = LLMDummyDataLoader(args.calib_size)

    model_type = get_model_type(model)
    args.kv_cache_qformat = "fp8"
    args.awq_block_size = 0
    args.auto_quantize_bits = None
    quant_cfg = build_quant_cfg(
        args.qformat,
        args.kv_cache_qformat,
        args.awq_block_size,
        args.auto_quantize_bits,
        model_type,
        QUANT_CFG_CHOICES,
        KV_QUANT_CFG_CHOICES,
    )
    model = quantize_model(model, quant_cfg, args,
                           calib_dataloader, False)
    export_llm_model(model, args.qformat, hf_config, args.export_dir)