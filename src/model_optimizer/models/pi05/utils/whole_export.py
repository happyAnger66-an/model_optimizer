#!/usr/bin/env python3
"""Whole-graph Pi0.5 (PI0Pytorch) FP8/NVFP4 quantization and ONNX export helpers.

This is vendored from `third_party/openpi_on_thor/pytorch_to_onnx.py` to avoid
runtime dependency on `openpi_on_thor` for `model-opt export pi05_libero/wrapper`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import onnx
import torch
import torch.onnx
from onnx.external_data_helper import convert_model_to_external_data

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer

from openpi.models.gemma import PALIGEMMA_VOCAB_SIZE
from openpi.models.model import IMAGE_KEYS, IMAGE_RESOLUTION

from .calibration_data import load_calibration_data


def _resolve_attention_mask_neg_cap() -> float | None:
    """Clamp too-negative additive attention masks to avoid NaN in fused kernels.

    PI0/Pi0.5 sometimes uses extremely negative additive masks (~-2.38e38). While
    PyTorch softmax can often handle this in float32, export/quantized paths may
    trigger fused kernels (or TensorRT later) that overflow to NaN. We pre-clamp
    the mask to a configurable negative cap (default: -1e4).

    Set env ``PI05_WHOLE_ATTN_MASK_NEG_CAP`` to override; set to ``none`` to disable.
    """

    raw = os.environ.get("PI05_WHOLE_ATTN_MASK_NEG_CAP", "-1e4").strip().lower()
    if raw in ("none", "null", "disable", "off"):
        return None
    try:
        return float(raw)
    except ValueError:
        return -1e4


def _sanitize_additive_attention_mask(attention_mask: torch.Tensor | None, neg_cap: float | None) -> torch.Tensor | None:
    if attention_mask is None or neg_cap is None:
        return attention_mask
    if not torch.is_floating_point(attention_mask):
        return attention_mask
    # Only clamp the very negative values; keep semantics.
    return attention_mask.clamp(min=float(neg_cap))


class QuantizedMatMul(torch.nn.Module):
    """Quantized matrix multiplication with QDQ nodes.

    MTQ cannot automatically insert QDQ nodes for some MHA matmul operations,
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

        return torch.matmul(input1, input2)


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
    """Replace Gemma eager_attention_forward with quantized version."""

    from transformers.models.gemma import modeling_gemma

    if not hasattr(modeling_gemma, "_original_eager_attention_forward"):
        modeling_gemma._original_eager_attention_forward = modeling_gemma.eager_attention_forward

#    modeling_gemma.eager_attention_forward = quantized_eager_attention_forward


def _create_observation_from_inputs(images, img_masks, state, lang_tokens, lang_masks):
    """Create OpenPI Observation from tensor inputs."""

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
    """Post-process ONNX model for TensorRT compatibility (external data, optional 2DQ)."""

    onnx_model = onnx.load(onnx_path, load_external_data=True)

    if enable_llm_nvfp4:
        try:
            from modelopt.onnx.quantization.qdq_utils import fp4qdq_to_2dq

            print("  Converting LLM NVFP4 ONNX model to 2DQ format...")
            onnx_model = fp4qdq_to_2dq(onnx_model, verbose=True)
            print("  NVFP4 2DQ conversion completed")
        except ImportError:
            print(
                "  Warning: fp4qdq_to_2dq not available in modelopt "
                f"{__import__('modelopt').__version__}. "
                "Skipping 2DQ conversion — TensorRT will handle FP4 QDQ nodes directly."
            )

    onnx_dir = os.path.dirname(onnx_path)
    os.makedirs(onnx_dir, exist_ok=True)

    # cleanup unrelated files in output directory
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
    """Wrapper for ONNX export that converts inputs to Observation and calls sample_actions."""

    def __init__(self, model: torch.nn.Module, num_steps: int):
        super().__init__()
        self.model = model
        self.num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        observation = _create_observation_from_inputs(images, img_masks, state, lang_tokens, lang_masks)
        return self.model.sample_actions(images.device, observation, noise=noise, num_steps=self.num_steps)


def create_dummy_inputs(
    model_device: torch.device, model_config, compute_dtype: torch.dtype = torch.float16
) -> Tuple:
    """Create dummy inputs for ONNX export/calibration based on config/constants."""

    num_images = len(IMAGE_KEYS)
    image_size = IMAGE_RESOLUTION[0]
    action_horizon = model_config.action_horizon
    action_dim = model_config.action_dim
    max_token_len = model_config.max_token_len

    dummy_inputs = (
        torch.randn(1, num_images * 3, image_size, image_size, dtype=compute_dtype, device=model_device),
        torch.ones(1, num_images, dtype=torch.bool, device=model_device),
        torch.randint(0, PALIGEMMA_VOCAB_SIZE, (1, max_token_len), dtype=torch.long, device=model_device),
        torch.ones(1, max_token_len, dtype=torch.bool, device=model_device),
        torch.randn(1, action_dim, dtype=compute_dtype, device=model_device),
        torch.randn(1, action_horizon, action_dim, dtype=compute_dtype, device=model_device),
    )
    print(
        f"  Dummy inputs created: images={dummy_inputs[0].shape} (dtype={compute_dtype}), "
        f"noise={dummy_inputs[5].shape} (dtype={compute_dtype})"
    )
    return dummy_inputs


def patch_model_for_export(model, compute_dtype=torch.float16):
    """Patch PI0Pytorch to be more ONNX/TensorRT export friendly."""

    import types

    model.compute_dtype = compute_dtype
    neg_cap = _resolve_attention_mask_neg_cap()

    def make_att_2d_masks_hook(pad_masks, att_masks):
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
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=self.compute_dtype, device=device)

    def sample_time_hook(self, bsize, device):
        from openpi.models_pytorch.pi0_pytorch import sample_beta

        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=self.compute_dtype, device=device)

    def sample_actions_hook(self, device, observation, noise=None, num_steps=10):
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
        prefix_att_2d_masks_4d = _sanitize_additive_attention_mask(prefix_att_2d_masks_4d, neg_cap)
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
        full_att_2d_masks_4d = _sanitize_additive_attention_mask(full_att_2d_masks_4d, neg_cap)
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
    """Quantize model using NVIDIA modelopt (FP8 with optional NVFP4 for LLM layers)."""

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
        quant_cfg["quant_cfg"]["paligemma_with_expert.paligemma.model.language_model.layers.*.output_quantizer"] = {
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


def prepare_model_for_export(
    model: torch.nn.Module,
    precision: str = "fp8",
    dummy_inputs: Tuple | None = None,
    config_obj=None,
    checkpoint_dir: str | None = None,
    num_calibration_samples: int = 32,
    num_steps: int = 10,
    enable_llm_nvfp4: bool = False,
    quantize_attention_matmul: bool = True,
) -> torch.nn.Module:
    """Prepare model for ONNX export by quantizing to FP8 (optional NVFP4 LLM)."""

    import openpi.models_pytorch.pi0_pytorch

    model.eval()
    model = patch_model_for_export(model, compute_dtype=torch.float16)
    model = model.to(torch.float16)

    if precision.lower() != "fp8":
        raise ValueError(
            "Only FP8 precision is supported. The Pi0.5 model uses BF16 natively and "
            "FP16 has insufficient dynamic range. Use FP8."
        )
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

    model = quantize_model(model, dummy_inputs, calibration_data, num_steps, enable_llm_nvfp4, quantize_attention_matmul)

    if hasattr(model.sample_actions, "_torchdynamo_inline"):
        uncompiled = openpi.models_pytorch.pi0_pytorch.PI0Pytorch.sample_actions
        model.sample_actions = lambda *args, **kwargs: uncompiled(model, *args, **kwargs)
    return model


def export_whole_model_to_onnx(
    model: torch.nn.Module,
    export_dir: str | Path,
    *,
    num_steps: int = 10,
    precision: str = "fp8",
    config_obj=None,
    checkpoint_dir: str | None = None,
    num_calibration_samples: int = 32,
    enable_llm_nvfp4: bool = False,
    quantize_attention_matmul: bool = True,
    opset_version: int = 19,
) -> Path:
    """Quantize (if needed) and export a single end-to-end ONNX graph."""

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = export_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    if enable_llm_nvfp4 and precision.lower() == "fp8":
        onnx_filename = f"model_{precision.lower()}_nvfp4.onnx"
    else:
        onnx_filename = f"model_{precision.lower()}.onnx"
    onnx_path = onnx_dir / onnx_filename

    device = next(model.parameters()).device
    dummy_inputs = create_dummy_inputs(device, model.config, torch.float16)
    model = prepare_model_for_export(
        model,
        precision=precision,
        dummy_inputs=dummy_inputs,
        config_obj=config_obj,
        checkpoint_dir=checkpoint_dir,
        num_calibration_samples=num_calibration_samples,
        num_steps=num_steps,
        enable_llm_nvfp4=enable_llm_nvfp4,
        quantize_attention_matmul=quantize_attention_matmul,
    )

    wrapped_model = ONNXWrapper(model, num_steps)
    print(f"\nExporting whole Pi05 model to: {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            dummy_inputs,
            str(onnx_path),
            opset_version=opset_version,
            dynamo=False,
            do_constant_folding=True,
            input_names=["images", "img_masks", "lang_tokens", "lang_masks", "state", "noise"],
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
        postprocess_onnx_model(str(onnx_path), enable_llm_nvfp4)

    return onnx_path

