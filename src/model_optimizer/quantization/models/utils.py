import os
import sys
import warnings

import torch
from torch import nn

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from accelerate import init_empty_weights, infer_auto_device_map
from accelerate.utils import get_max_memory 

SPECULATIVE_MODEL_LIST = ["Eagle", "Medusa"]

def is_speculative(hf_config):
    """Check if the model architecture is a speculative model."""
    return hf_config.architectures and any(
        name in hf_config.architectures[0] for name in SPECULATIVE_MODEL_LIST
    )

def is_model_on_gpu(model) -> bool:
    """Returns if the model is fully loaded on GPUs."""
    return all("cuda" in str(param.device) for param in model.parameters())

def get_model(
    ckpt_path,
    device="cuda",
    gpu_mem_percentage=0.8,
    trust_remote_code=False,
    use_seq_device_map=False,
    attn_implementation=None,
):
    print(f"Initializing model from {ckpt_path}")

    device_map = "auto"
    if device == "cpu":
        device_map = "cpu"

    # Prepare config kwargs for loading
    config_kwargs = {"trust_remote_code": trust_remote_code} if trust_remote_code else {}

    # Load config once and handle VL model detection
    try:
        hf_config = AutoConfig.from_pretrained(ckpt_path, **config_kwargs)
    except Exception as e:
        print(f"Error: Could not load config from {ckpt_path}: {e}")
        raise RuntimeError(f"Failed to load model configuration from {ckpt_path}") from e
    
    if attn_implementation is not None:
        config_kwargs["attn_implementation"] = attn_implementation

    # Note: Forcibly converting the model precision between bf16 and fp16 may introduce accuracy drop
    model_kwargs = config_kwargs.copy()
    # Don't set torch_dtype for VILA models as they handle it explicitly in their builder
    if "vila" not in ckpt_path.lower():
        model_kwargs.setdefault("torch_dtype", "auto")

    if use_seq_device_map:
        device_map = "sequential"
        # If we use sequential, set max_memory limit to ensure that the model does not occupy the full GPU
        max_memory = get_max_memory()
        max_memory = {key: value * gpu_mem_percentage for key, value in max_memory.items()}
        model_kwargs["max_memory"] = max_memory

    if hf_config.model_type == "bart":
        # device_map "auto" and "cuda" triggers error regarding meta tensor from safetensors
        device_map = None

    if is_speculative(hf_config):
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path,
            device_map=device_map,
            **model_kwargs,
        )
    else:
        architecture = hf_config.architectures[0]

        if not hasattr(transformers, architecture):
            warnings.warn(
                f"Architecture {architecture} not found in transformers: {transformers.__version__}. "
                "Falling back to AutoModelForCausalLM."
            )
            assert trust_remote_code, (
                "Please set trust_remote_code to True if you want to use this architecture"
            )

            auto_model_module = AutoModelForCausalLM
            from_config = auto_model_module.from_config
        else:
            auto_model_module = getattr(transformers, architecture)
            from_config = auto_model_module._from_config

        with init_empty_weights():
            # When computing the device_map, assuming half precision by default,
            # unless specified by the hf_config.
            torch_dtype = getattr(hf_config, "torch_dtype", torch.float16)
            model_kwargs2 = model_kwargs.copy()
            if auto_model_module != AutoModelForCausalLM:
                model_kwargs2.pop("trust_remote_code", None)
            model_kwargs2["torch_dtype"] = torch_dtype
            model_kwargs2.pop("max_memory", None)
            model = from_config(hf_config, **model_kwargs2)

        max_memory = get_max_memory()
        inferred_device_map = infer_auto_device_map(model, max_memory=max_memory)

        on_cpu = "cpu" in inferred_device_map.values()

        if on_cpu:
            for _device in max_memory:
                if isinstance(_device, int):
                    max_memory[_device] *= gpu_mem_percentage

            print(
                "Model does not fit to the GPU mem. "
                f"We apply the following memory limit for calibration: \n{max_memory}\n"
                "If you hit GPU OOM issue, please adjust `gpu_mem_percentage` or "
                "reduce the calibration `batch_size` manually."
            )
            model_kwargs["max_memory"] = max_memory

        model = auto_model_module.from_pretrained(
            ckpt_path,
            device_map=device_map,
            **model_kwargs,
        )
    model.eval()

    # If device_map was disabled (None), manually move model to target device
    if device_map is None and device != "cpu":
        print(f"Moving model to {device} device...")
        model = model.to(device)

    if device == "cuda" and not is_model_on_gpu(model):
        print("Warning: Some parameters are not on a GPU. Calibration can be slow or hit OOM")

    return model

def get_language_model_from_vl(model) -> list[nn.Module] | None:
    """Extract the language model lineage from a Vision-Language Model (VLM).

    This function handles the common patterns for accessing the language model component
    in various VLM architectures. It checks multiple possible locations where the
    language model might be stored.

    Args:
        model: The VLM model instance to extract the language model from

    Returns:
        list: the lineage path towards the language model

    Examples:
        >>> # For LLaVA-style models
        >>> lineage = get_language_model_from_vl(vlm_model)
        >>> # lineage[0] is vlm_model
        >>> # lineage[1] is vlm_model.language_model
    """
    # always prioritize model.model.langauge_model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return [model, model.model, model.model.language_model]

    if hasattr(model, "language_model"):
        return [model, model.language_model]

    # Pattern 3: No language_model found
    return None