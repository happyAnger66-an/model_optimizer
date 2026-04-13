#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.onnx
import onnx
from onnx.external_data_helper import convert_model_to_external_data

import openpi.models_pytorch.pi0_pytorch
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.models.model import IMAGE_KEYS, IMAGE_RESOLUTION
from openpi.models.gemma import PALIGEMMA_VOCAB_SIZE

import modelopt.torch.quantization as mtq
from openpi_on_thor.calibration_data import load_calibration_data

from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.config import QuantizerAttributeConfig

# ---------------------------------------------------------------------------
# Patch load_pytorch to handle dtype mismatches and tied weights across
# different safetensors / transformers versions (e.g. 25.09 vs 26.01 containers).
# ---------------------------------------------------------------------------
import safetensors.torch as _st
from openpi.models_pytorch import pi0_pytorch as _pi0pt

def _load_pytorch_patched(self, train_config, weight_path: str):
    model = _pi0pt.PI0Pytorch(config=train_config.model)
    state_dict = _st.load_file(weight_path)
    model.load_state_dict(state_dict, strict=False)
    return model

import openpi.models.model as _model_mod
for _cls in vars(_model_mod).values():
    if isinstance(_cls, type) and hasattr(_cls, "load_pytorch"):
        _cls.load_pytorch = _load_pytorch_patched
# ---------------------------------------------------------------------------


class QuantizedMatMul(torch.nn.Module):
    """
    Quantized matrix multiplication with QDQ nodes.

    MTQ cannot automatically insert QDQ nodes for MHA matmul operations,
    so we manually manage quantizers for Q@K^T and attn_weights@V.
    """

    def __init__(self):
        super().__init__()
        self.input1_quantizer = None
        self.input2_quantizer = None
        self._quantizers_created = False

    def _create_quantizers(self):
        if not self._quantizers_created:
            self.input1_quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3)))
            self.input2_quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3)))
            self.input1_quantizer.enable_calib()
            self.input1_quantizer.disable_quant()
            self.input2_quantizer.enable_calib()
            self.input2_quantizer.disable_quant()
            self._quantizers_created = True

    def forward(self, input1, input2):
        if not self._quantizers_created:
            self._create_quantizers()

        if self.input1_quantizer is not None:
            input1 = self.input1_quantizer(input1)
        if self.input2_quantizer is not None:
            input2 = self.input2_quantizer(input2)

        output = torch.matmul(input1, input2)
        return output


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value tensors for multi-query/grouped-query attention."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def quantized_eager_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Attention forward with quantized matmul operations."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    if not hasattr(module, "qk_matmul"):
        module.add_module("qk_matmul", QuantizedMatMul())
    if not hasattr(module, "av_matmul"):
        module.add_module("av_matmul", QuantizedMatMul())

    attn_weights = module.qk_matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = module.av_matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def replace_attention_with_quantized_version():
    """Replace eager_attention_forward with quantized version."""
    from transformers.models.gemma import modeling_gemma

    if not hasattr(modeling_gemma, "_original_eager_attention_forward"):
        modeling_gemma._original_eager_attention_forward = modeling_gemma.eager_attention_forward

    modeling_gemma.eager_attention_forward = quantized_eager_attention_forward


def _create_observation_from_inputs(images, img_masks, state, lang_tokens, lang_masks):
    """Helper function to create Observation from tensor inputs."""
    from openpi.models.model import Observation

    images_dict = {IMAGE_KEYS[i]: images[:, i * 3 : (i + 1) * 3] for i in range(len(IMAGE_KEYS))}
    image_masks_dict = {IMAGE_KEYS[i]: img_masks[:, i] for i in range(len(IMAGE_KEYS))}

    return Observation(
        images=images_dict,
        image_masks=image_masks_dict,
        state=state,
        tokenized_prompt=lang_tokens,
        tokenized_prompt_mask=lang_masks,
    )


def postprocess_onnx_model(onnx_path: str, enable_llm_nvfp4: bool = False) -> None:
    """
    Post-process ONNX model for TensorRT compatibility.

    - Cleans up output directory and removes old files
    - Saves model with external data format
    - Converts FP4 QDQ ops to 2DQ format if enable_llm_nvfp4 is True

    Args:
        onnx_path: Path to the ONNX model file
        enable_llm_nvfp4: Enable NVFP4 LLM conversion to 2DQ format (default: False)

    Returns:
        Modified ONNX model
    """

    onnx_model = onnx.load(onnx_path, load_external_data=True)

    if enable_llm_nvfp4:
        try:
            from modelopt.onnx.quantization.qdq_utils import fp4qdq_to_2dq

            print("  Converting LLM NVFP4 ONNX model to 2DQ format...")
            onnx_model = fp4qdq_to_2dq(onnx_model, verbose=True)
            print("  NVFP4 2DQ conversion completed")
            print(f"  ONNX model saved to: {onnx_path} with NVFP4 2DQ format")
        except ImportError:
            print("  Warning: fp4qdq_to_2dq not available in modelopt "
                  f"{__import__('modelopt').__version__}. "
                  "Skipping 2DQ conversion — TensorRT will handle FP4 QDQ nodes directly.")

    onnx_dir = os.path.dirname(onnx_path)
    os.makedirs(onnx_dir, exist_ok=True)

    for filename in os.listdir(onnx_dir):
        if filename.endswith(".onnx") or filename.endswith(".data"):
            continue
        file_path = os.path.join(onnx_dir, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

    convert_model_to_external_data(
        onnx_model,
        all_tensors_to_one_file=True,
        location=os.path.basename(onnx_path).replace(".onnx", ".data"),
    )

    onnx.save(onnx_model, onnx_path)


class ONNXWrapper(torch.nn.Module):
    """Wrapper for ONNX export that converts inputs to Observation format."""

    def __init__(self, model: torch.nn.Module, num_steps: int):
        """Initialize ONNX wrapper.

        Args:
            model: The model to wrap for ONNX export
            num_steps: Number of denoising steps
        """
        super().__init__()
        self.model = model
        self.num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        """Forward pass that converts tensor inputs to Observation format and calls model.sample_actions.

        Args:
            images: Input images tensor
            img_masks: Image masks tensor
            lang_tokens: Language tokens tensor
            lang_masks: Language masks tensor
            state: State tensor
            noise: Noise tensor

        Returns:
            Model output actions
        """
        observation = _create_observation_from_inputs(images, img_masks, state, lang_tokens, lang_masks)
        return self.model.sample_actions(images.device, observation, noise=noise, num_steps=self.num_steps)


def _create_dummy_inputs(
    model_device: torch.device, model_config, compute_dtype: torch.dtype = torch.float16
) -> Tuple:
    """Create dummy inputs for ONNX export.

    Reads parameters from model_config and imported constants instead of hardcoded values.

    Args:
        model_device: Device to create tensors on
        model_config: Model configuration
        compute_dtype: Compute dtype for input tensors (default: torch.float16)

    Returns:
        Tuple of dummy input tensors
    """
    num_images = len(IMAGE_KEYS)
    image_size = IMAGE_RESOLUTION[0]
    action_horizon = model_config.action_horizon
    action_dim = model_config.action_dim
    max_token_len = model_config.max_token_len

    dummy_inputs = (
        torch.randn(
            1,
            num_images * 3,
            image_size,
            image_size,
            dtype=compute_dtype,
            device=model_device,
        ),
        torch.ones(1, num_images, dtype=torch.bool, device=model_device),
        torch.randint(
            0,
            PALIGEMMA_VOCAB_SIZE,
            (1, max_token_len),
            dtype=torch.long,
            device=model_device,
        ),
        torch.ones(1, max_token_len, dtype=torch.bool, device=model_device),
        torch.randn(1, action_dim, dtype=compute_dtype, device=model_device),
        torch.randn(1, action_horizon, action_dim, dtype=compute_dtype, device=model_device),
    )

    print(
        f"  Dummy inputs created: images={dummy_inputs[0].shape} (dtype={compute_dtype}), noise={dummy_inputs[5].shape} (dtype={compute_dtype})"
    )
    return dummy_inputs


def patch_model_for_export(model, compute_dtype=torch.float16):
    """
    Patch model to add compute_dtype support without modifying original code.

    Args:
        model: PI0Pytorch model instance
        compute_dtype: Compute dtype, default torch.float16

    Returns:
        Patched model
    """
    import types

    model.compute_dtype = compute_dtype

    def make_att_2d_masks_hook(pad_masks, att_masks):
        """TensorRT-compatible version of make_att_2d_masks with explicit int64 casting."""
        if att_masks.ndim != 2:
            raise ValueError(att_masks.ndim)
        if pad_masks.ndim != 2:
            raise ValueError(pad_masks.ndim)

        att_masks_int64 = att_masks.to(dtype=torch.int64)
        cumsum = torch.cumsum(att_masks_int64, dim=1)
        att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
        pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
        return att_2d_masks & pad_2d_masks

    def sample_noise_hook(self, shape, device):
        """Sample noise tensor with compute_dtype.

        Args:
            shape: Shape of the noise tensor
            device: Device to create the tensor on

        Returns:
            Noise tensor
        """
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=self.compute_dtype,
            device=device,
        )

    def sample_time_hook(self, bsize, device):
        """Sample time steps with compute_dtype.

        Args:
            bsize: Batch size
            device: Device to create the tensor on

        Returns:
            Time tensor
        """
        from openpi.models_pytorch.pi0_pytorch import sample_beta

        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=self.compute_dtype, device=device)

    def sample_actions_hook(self, device, observation, noise=None, num_steps=10):
        """Sample actions from the model using TensorRT-compatible operations.

        Args:
            device: Device to run inference on
            observation: Input observation
            noise: Optional noise tensor for sampling
            num_steps: Number of denoising steps

        Returns:
            Sampled actions
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks_hook(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks.to(dtype=torch.int64), dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt_f32 = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise.float()
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt_f32 / 2:
            expanded_time = time.to(self.compute_dtype).expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t.to(self.compute_dtype),
                expanded_time,
            )

            x_t = x_t + dt_f32 * v_t.float()
            time += dt_f32
        return x_t.to(self.compute_dtype)

    def denoise_step_hook(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        """Perform one denoising step using TensorRT-compatible operations.

        Args:
            state: Input state
            prefix_pad_masks: Prefix padding masks
            past_key_values: Cached key-value pairs
            x_t: Current noisy actions
            timestep: Current timestep

        Returns:
            Velocity prediction
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks_hook(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks.to(dtype=torch.int64), dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=self.compute_dtype)
        return self.action_out_proj(suffix_out)

    model.sample_noise = types.MethodType(sample_noise_hook, model)
    model.sample_time = types.MethodType(sample_time_hook, model)
    model.sample_actions = types.MethodType(sample_actions_hook, model)
    model.denoise_step = types.MethodType(denoise_step_hook, model)

    print(f"  Model patched with compute_dtype={compute_dtype}")
    return model


def quantize_model(
    model: torch.nn.Module,
    dummy_inputs: Tuple,
    calibration_data=None,
    num_steps: int = 10,
    enable_llm_nvfp4: bool = False,
    quantize_attention_matmul: bool = True,
) -> torch.nn.Module:
    """Quantize model using NVIDIA modelopt (FP8 with optional NVFP4 for LLM layers).

    Args:
        model: PyTorch model to quantize
        dummy_inputs: Dummy inputs for calibration (fallback)
        calibration_data: DataLoader with calibration data (preferred), or None
        num_steps: Number of denoising steps for the model
        enable_llm_nvfp4: Enable NVFP4 quantization for LLM layers (default: False)
        quantize_attention_matmul: Enable QDQ nodes for attention matmul operations (default: True)

    Returns:
        Quantized model (FP8 base, with optional NVFP4 LLM layers)
    """
    print("  Quantizing model to FP8 using NVIDIA modelopt...")

    if quantize_attention_matmul:
        replace_attention_with_quantized_version()

    quant_cfg = mtq.FP8_DEFAULT_CFG
    quant_cfg["quant_cfg"]["nn.Conv2d"] = {"*": {"enable": False}}

    if enable_llm_nvfp4:
        print("  Enabling NVFP4 quantization for LLM layers...")
        quant_cfg["quant_cfg"]["paligemma_with_expert.paligemma.model.language_model.layers.*"] = {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        }

        quant_cfg["quant_cfg"][
            "paligemma_with_expert.paligemma.model.language_model.layers.*.output_quantizer"
        ] = {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": False,
        }

    if calibration_data is not None:
        num_samples = len(calibration_data.dataset) if hasattr(calibration_data, "dataset") else "unknown"
        print(f"  Using {num_samples} real calibration samples from dataset")

        def forward_loop(mdl):
            mdl.eval()
            for batch_idx, (observation, noise) in enumerate(calibration_data):
                with torch.no_grad():
                    try:
                        device = next(mdl.parameters()).device
                        _ = mdl.sample_actions(device, observation, noise=noise, num_steps=num_steps)
                        if (batch_idx + 1) % 10 == 0:
                            print(f"    Processed {batch_idx + 1}/{num_samples} calibration samples")
                    except Exception as e:
                        print(f"    Warning: Calibration batch {batch_idx} forward failed: {e}")
                        continue
    else:
        print("  Using dummy inputs for calibration")

        def forward_loop(mdl):
            wrapper = ONNXWrapper(mdl, num_steps)
            wrapper(*dummy_inputs)

    print("  Running quantization with calibration...")
    quantized_model = mtq.quantize(model, quant_cfg, forward_loop=forward_loop)

    print("\n  Quantization Summary:")
    mtq.print_quant_summary(quantized_model)

    print("  FP8 quantization completed")

    if enable_llm_nvfp4:
        from modelopt.torch.quantization.utils import is_quantized_linear

        for module in quantized_model.modules():
            assert not isinstance(module, torch.nn.Linear) or is_quantized_linear(module)
            if isinstance(module, torch.nn.Linear):
                module.input_quantizer._trt_high_precision_dtype = "Half"
                module.input_quantizer._onnx_quantizer_type = "dynamic"
                module.output_quantizer._onnx_quantizer_type = "dynamic"
                module.weight_quantizer._onnx_quantizer_type = "static"

    return quantized_model


def _prepare_model_for_export(
    model: torch.nn.Module,
    precision: str = "fp8",
    dummy_inputs: Tuple = None,
    config_obj=None,
    checkpoint_dir: str = None,
    num_calibration_samples: int = 32,
    num_steps: int = 10,
    enable_llm_nvfp4: bool = False,
    quantize_attention_matmul: bool = True,
) -> torch.nn.Module:
    """Prepare model for ONNX export by converting to specified precision and eval mode.

    Args:
        model: PyTorch model to prepare
        precision: Model precision ("fp8")
        dummy_inputs: Dummy inputs for FP8 calibration (required for fp8)
        config_obj: Training config object (for loading calibration data)
        checkpoint_dir: Path to model checkpoint directory (for loading calibration policy)
        num_calibration_samples: Number of calibration samples to use for FP8
        num_steps: Number of denoising steps
        enable_llm_nvfp4: Enable NVFP4 quantization for LLM layers (default: False)
        quantize_attention_matmul: Enable QDQ nodes for attention matmul operations (default: True)

    Returns:
        Prepared model
    """
    model.eval()

    model = patch_model_for_export(model, compute_dtype=torch.float16)
    model = model.to(torch.float16)

    if precision.lower() == "fp8":
        if dummy_inputs is None:
            raise ValueError("dummy_inputs required for FP8 quantization")

        device = next(model.parameters()).device
        calibration_data = None
        if config_obj is not None and checkpoint_dir is not None:
            calibration_data = load_calibration_data(
                config_obj,
                checkpoint_dir,
                num_calibration_samples,
                str(device),
                compute_dtype=torch.float16,
            )

        model = quantize_model(
            model, dummy_inputs, calibration_data, num_steps, enable_llm_nvfp4, quantize_attention_matmul
        )
        dtype_str = "float8 (quantized from float16)"
        if enable_llm_nvfp4:
            dtype_str += " with NVFP4 LLM"
    else:
        raise ValueError(
            "Only FP8 precision is supported. The Pi0.5 model uses BF16 natively and "
            "FP16 has insufficient dynamic range. Use: --precision fp8 --enable_llm_nvfp4 --quantize_attention_matmul"
        )

    device = next(model.parameters()).device
    print(f"  Model device: {device}, dtype: {dtype_str}")

    if hasattr(model.sample_actions, "_torchdynamo_inline"):
        uncompiled = openpi.models_pytorch.pi0_pytorch.PI0Pytorch.sample_actions
        model.sample_actions = lambda *args, **kwargs: uncompiled(model, *args, **kwargs)
    return model


def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    model_config,
    num_steps: int = 10,
    precision: str = "fp8",
    config_obj=None,
    checkpoint_dir: str = None,
    num_calibration_samples: int = 32,
    enable_llm_nvfp4: bool = False,
    quantize_attention_matmul: bool = True,
) -> torch.nn.Module:
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        output_path: Output directory path
        model_config: Model configuration (Pi0Config)
        num_steps: Number of denoising steps (default: 10)
        precision: Model precision ("fp8")
        config_obj: Training config object (for FP8 calibration with real data)
        checkpoint_dir: Path to model checkpoint directory (for loading calibration policy)
        num_calibration_samples: Number of calibration samples for FP8 (default: 32)
        enable_llm_nvfp4: Enable NVFP4 quantization for LLM layers (default: False)
        quantize_attention_matmul: Enable QDQ nodes for attention matmul operations (default: True)

    Returns:
        Exported model
    """
    if enable_llm_nvfp4 and precision.lower() == "fp8":
        print(f"Exporting model to ONNX format with precision: {precision.upper()} + NVFP4 LLM...")
    else:
        print(f"Exporting model to ONNX format with precision: {precision.upper()}...")

    device = next(model.parameters()).device
    dummy_inputs = _create_dummy_inputs(device, model_config, torch.float16)

    model = _prepare_model_for_export(
        model,
        precision,
        dummy_inputs,
        config_obj,
        checkpoint_dir,
        num_calibration_samples,
        num_steps,
        enable_llm_nvfp4,
        quantize_attention_matmul,
    )
    device = next(model.parameters()).device

    wrapped_model = ONNXWrapper(model, num_steps)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    if enable_llm_nvfp4 and precision.lower() == "fp8":
        onnx_filename = f"model_{precision.lower()}_nvfp4.onnx"
    else:
        onnx_filename = f"model_{precision.lower()}.onnx"

    onnx_path = onnx_dir / onnx_filename

    print(f"\nExporting to: {onnx_path}")

    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            dummy_inputs,
            str(onnx_path),
            opset_version=19,
            dynamo=False,
            do_constant_folding=True,
            input_names=[
                "images",
                "img_masks",
                "lang_tokens",
                "lang_masks",
                "state",
                "noise",
            ],
            output_names=["actions"],
            dynamic_axes={
                "images": {0: "batch_size"},
                "img_masks": {0: "batch_size"},
                "lang_tokens": {0: "batch_size", 1: "seq_len"},
                "lang_masks": {0: "batch_size", 1: "seq_len"},
                "state": {0: "batch_size"},
                "noise": {0: "batch_size"},
                "actions": {0: "batch_size"},
            },
        )
        postprocess_onnx_model(onnx_path, enable_llm_nvfp4)

    return model


def export_checkpoint_to_onnx(
    checkpoint_dir: str,
    output_path: Path,
    config_name: str = "pi05_droid",
    num_steps: int = 10,
    precision: str = "fp8",
    num_calibration_samples: int = 32,
    enable_llm_nvfp4: bool = False,
    quantize_attention_matmul: bool = True,
) -> torch.nn.Module:
    """
    Export a trained model checkpoint to ONNX format.

    Args:
        checkpoint_dir: PyTorch checkpoint directory path
        output_path: ONNX output directory path
        config_name: Model configuration name
        num_steps: Number of denoising steps
        precision: Model precision ("fp8")
        num_calibration_samples: Number of samples to use for FP8 calibration (default: 32)
        enable_llm_nvfp4: Enable NVFP4 quantization for LLM layers (default: False)
        quantize_attention_matmul: Enable QDQ nodes for attention matmul operations (default: True)

    Returns:
        Exported model
    """
    print(f"Loading model from: {checkpoint_dir}")
    print(f"Output path: {output_path}\n")

    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    config = _config.get_config(config_name)
    policy = policy_config.create_trained_policy(config, checkpoint_dir)

    model = export_to_onnx(
        model=policy._model,
        output_path=output_path,
        model_config=config.model,
        num_steps=num_steps,
        precision=precision,
        config_obj=config,
        checkpoint_dir=checkpoint_dir,
        num_calibration_samples=num_calibration_samples,
        enable_llm_nvfp4=enable_llm_nvfp4,
        quantize_attention_matmul=quantize_attention_matmul,
    )

    print(f"  ONNX model saved to: {output_path}/onnx/")

    return model


def main():
    """Main entry point for ONNX export."""
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX format")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/root/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch",
        help="Path to PyTorch checkpoint directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/root/.cache/openpi/openpi-assets/checkpoints/pi05_droid_pytorch",
        help="Path to output directory for ONNX model",
    )
    parser.add_argument("--config_name", type=str, default="pi05_droid", help="Model configuration name")
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Number of denoising steps (default: 10)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp8",
        choices=["fp8", "FP8"],
        help="Model precision type (default: fp8). FP16 is not supported — the model uses BF16 natively.",
    )
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=32,
        help="Number of dataset samples to use for FP8 calibration (default: 32)",
    )
    parser.add_argument(
        "--enable_llm_nvfp4",
        action="store_true",
        help="Enable NVFP4 quantization for LLM layers (only applies with --precision fp8)",
    )
    parser.add_argument(
        "--quantize_attention_matmul",
        action="store_true",
        help="Enable QDQ nodes for attention matmul operations (only applies with --precision fp8)",
    )

    args = parser.parse_args()

    try:
        export_checkpoint_to_onnx(
            checkpoint_dir=args.checkpoint_dir,
            output_path=Path(args.output_path),
            config_name=args.config_name,
            num_steps=args.num_steps,
            precision=args.precision,
            num_calibration_samples=args.num_calibration_samples,
            enable_llm_nvfp4=args.enable_llm_nvfp4,
            quantize_attention_matmul=args.quantize_attention_matmul,
        )
    except Exception as e:
        print(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
