"""
π0.5 PaliGemma 语言塔与 model_optimizer 独立 CuTe DSL FMHA D=256 插件对接。

不依赖 TensorRT-Edge-LLM。ONNX 导出使用 ``trt::FmhaD256AttentionPlugin``，
内核来自 ``kernelSrc/fmha_d256_cutedsl``（study_cute/fmha_d256 迁入）。
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from termcolor import colored
from tqdm import tqdm
from transformers.cache_utils import DynamicCache

from model_optimizer.calibrate.pi05_calib_load import open_pi05_calib_for_quantize
from model_optimizer.evaluate.metrics.pi05 import Pi05Metric
from model_optimizer.ops.fmha_d256_attention_plugin import (
    FmhaD256Attention,
    fmha_d256_attention_custom_translation_table,
    register_fmha_d256_onnx_symbolic,
)
from model_optimizer.quantization.quantization_utils import quantize_model
from model_optimizer.utils.utils import is_fp4_quantized, is_nvfp4_quantized, set_dynamic_quant

from ..model import Model

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _gated_residual(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    if gate is None:
        return x + y
    return x + y * gate


def _bf16_to_fp16_inplace(module: nn.Module) -> list[tuple[torch.nn.Parameter | torch.Tensor, torch.Tensor]]:
    swaps: list[tuple[torch.nn.Parameter | torch.Tensor, torch.Tensor]] = []
    for _, p in module.named_parameters():
        if p.dtype != torch.bfloat16:
            continue
        orig = p.detach().clone()
        with torch.no_grad():
            p.data = p.data.to(torch.float16)
        swaps.append((p, orig))
    for _, b in module.named_buffers():
        if not b.is_floating_point() or b.dtype != torch.bfloat16:
            continue
        orig = b.detach().clone()
        with torch.no_grad():
            b.data = b.data.to(torch.float16)
        swaps.append((b, orig))
    return swaps


def _restore_bf16(swaps: list[tuple[torch.nn.Parameter | torch.Tensor, torch.Tensor]]) -> None:
    for t, orig in swaps:
        with torch.no_grad():
            t.data = orig.clone().to(device=t.device)


class GemmaAttentionCuteDsl(nn.Module):
    """包装 GemmaAttention：训练走 native，导出走 FmhaD256Attention。"""

    def __init__(self, native_attn: nn.Module, *, use_fp16: bool = True) -> None:
        super().__init__()
        self.native = native_attn
        self.plugin = FmhaD256Attention(native_attn, use_fp16=use_fp16)

    def forward(self, *args: Any, **kwargs: Any):
        return self.native.forward(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in ("native", "plugin"):
                raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'") from None
            native = self._modules.get("native")
            if native is None:
                raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'") from None
            return getattr(native, name)


def install_gemma_cutedsl_attention_wrappers(gemma_model: nn.Module) -> nn.Module:
    for layer in gemma_model.layers:
        if not isinstance(layer.self_attn, GemmaAttentionCuteDsl):
            layer.self_attn = GemmaAttentionCuteDsl(layer.self_attn)
    return gemma_model


class GemmaModelCuteDslOnnxExport(nn.Module):
    """ONNX 导出：每层 attention 走 FmhaD256AttentionPlugin。"""

    def __init__(self, gemma_model: nn.Module) -> None:
        super().__init__()
        for layer in gemma_model.layers:
            if not isinstance(layer.self_attn, GemmaAttentionCuteDsl):
                raise TypeError("各层 self_attn 须为 GemmaAttentionCuteDsl")
        self.gemma = gemma_model

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del attention_mask
        gemma = self.gemma
        device = inputs_embeds.device
        hidden_states = inputs_embeds.to(torch.float16)

        cfg = gemma.config
        num_kv_heads = cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        if head_dim != 256:
            raise ValueError(
                f"FmhaD256AttentionPlugin requires head_dim=256, got {head_dim}. "
                "Use pi05 Gemma config or add another kernel variant."
            )

        bsz, seq_len, _ = hidden_states.shape
        past_len = seq_len
        past_list: list[torch.Tensor] = []
        for _ in gemma.layers:
            past_list.append(
                torch.zeros(
                    bsz,
                    2,
                    num_kv_heads,
                    past_len,
                    head_dim,
                    dtype=torch.float16,
                    device=device,
                )
            )

        cu_kv_seqlens = torch.arange(bsz + 1, dtype=torch.int32, device=device) * (past_len + seq_len)

        for layer_idx, layer in enumerate(gemma.layers):
            residual = hidden_states
            hidden_states, gate = layer.input_layernorm(hidden_states, None)

            attn_out, past_list[layer_idx] = layer.self_attn.plugin(
                hidden_states,
                past_list[layer_idx],
                cu_kv_seqlens,
            )
            hidden_states = _gated_residual(residual, attn_out, gate)

            residual = hidden_states
            hidden_states, gate = layer.post_attention_layernorm(hidden_states, None)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = _gated_residual(residual, hidden_states, gate)

        hidden_states, _ = gemma.norm(hidden_states, None)

        past_keys = torch.cat([past_list[i][:, 0] for i in range(len(past_list))], dim=0)
        past_values = torch.cat([past_list[i][:, 1] for i in range(len(past_list))], dim=0)
        return past_keys, past_values, hidden_states


class Pi05CuteDslLanguageModel(nn.Module):
    def __init__(self, hf_gemma: nn.Module, *, wrap_attention: bool = True) -> None:
        super().__init__()
        object.__setattr__(self, "_hf_gemma", hf_gemma)
        self.config = hf_gemma.config
        self.embed_tokens = hf_gemma.embed_tokens
        self.layers = hf_gemma.layers
        if wrap_attention:
            install_gemma_cutedsl_attention_wrappers(hf_gemma)

    def forward(self, *args: Any, **kwargs: Any):
        return self._hf_gemma(*args, **kwargs)


class LLMWithCuteDsl(nn.Module, Model):
    """π0.5 LLM 子模块：CuTe DSL FMHA D=256 TRT 插件路径。"""

    def __init__(self, config, llm: Pi05CuteDslLanguageModel, **kwargs):
        nn.Module.__init__(self)
        Model.__init__(self, "pi05_llm_cutedsl", "")
        self.model = llm
        self.device = next(llm.parameters()).device
        self.config = config
        self.model.config._attn_implementation = "eager"

    def get_calibrate_dataset(self, calib_data):
        return open_pi05_calib_for_quantize(calib_data, component="pi05_llm")

    def forward(self, inputs_embeds, attention_mask, position_ids):
        prefix_output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        past_key_caches = prefix_output.past_key_values
        past_keys = []
        past_values = []
        for i in range(self.config.num_hidden_layers):
            past_keys.append(past_key_caches[i][0])
            past_values.append(past_key_caches[i][1])
        past_keys_tensor = torch.cat(past_keys, dim=0).to(torch.bfloat16)
        past_values_tensor = torch.cat(past_values, dim=0).to(torch.bfloat16)
        return past_keys_tensor, past_values_tensor, prefix_output.last_hidden_state

    @classmethod
    def construct_from_name_path(cls, model_name, model_path, train_config=None):
        from .model_pi05 import Pi05Model

        pi05_model = Pi05Model.construct_from_name_path(model_name, model_path, train_config)
        return cls.construct_model(pi05_model)

    @classmethod
    def construct_model(cls, pi05_model, dtype=torch.bfloat16):
        paligemma = pi05_model.paligemma_with_expert.paligemma
        hf = paligemma.get_decoder()
        bundle = Pi05CuteDslLanguageModel(hf, wrap_attention=True)
        return cls(paligemma.config.text_config, bundle)

    def export(self, export_dir, dynamo=True, mode=None):
        del mode  # CuTe DSL 仅一种 plugin 导出路径；与 convert_formt CLI 兼容
        self.eval().cuda()
        os.makedirs(export_dir, exist_ok=True)
        start = time.time()
        print(colored("Start LLMWithCuteDsl export (trt::FmhaD256AttentionPlugin)...", "green"))

        register_fmha_d256_onnx_symbolic()
        gemma = self.model._hf_gemma
        weight_swaps = _bf16_to_fp16_inplace(gemma)
        try:
            export_net = GemmaModelCuteDslOnnxExport(gemma).eval().cuda()
            inputs_embeds = torch.randn((1, 968, 2048), dtype=torch.float16, device="cuda")
            attention_mask = torch.randn((1, 1, 968, 968), dtype=torch.float32, device="cuda")
            position_ids = torch.randint(1, 1000, (1, 968), dtype=torch.int64, device="cuda")

            export_kwargs: dict[str, Any] = {
                "export_params": True,
                "input_names": ["inputs_embeds", "attention_mask", "position_ids"],
                "output_names": ["past_keys", "past_values", "last_hidden_state"],
                "opset_version": 19,
                "dynamo": dynamo,
                "do_constant_folding": True,
                "dynamic_axes": {
                    "inputs_embeds": {0: "batch_size", 1: "seq_len"},
                    "attention_mask": {0: "batch_size", 2: "seq_len", 3: "seq_len"},
                    "position_ids": {0: "batch_size", 1: "seq_len"},
                    "past_keys": {2: "seq_len"},
                    "past_values": {2: "seq_len"},
                    "last_hidden_state": {0: "batch_size", 1: "seq_len"},
                },
            }
            if dynamo:
                tbl = fmha_d256_attention_custom_translation_table()
                if tbl:
                    export_kwargs["custom_translation_table"] = tbl

            with torch.inference_mode():
                torch.onnx.export(
                    export_net,
                    (inputs_embeds, attention_mask, position_ids),
                    f"{export_dir}/llm.onnx",
                    **export_kwargs,
                )
        finally:
            _restore_bf16(weight_swaps)

        end = time.time()
        print(colored(f"LLMWithCuteDsl export done cost:{end - start}s", "green"))
        return self

    @classmethod
    def export_onnx(cls, pi_model, export_dir):
        del pi_model.paligemma_with_expert.gemma_expert
        llm_model = cls.construct_model(pi_model, dtype=torch.float16)
        llm_model.export(export_dir, dynamo=True)
        return llm_model

    def val(self, val_data, batch_size, output_dir):
        val_datas = self.get_calibrate_dataset(val_data)

        def val_loop(model, output_datas):
            for data in tqdm(val_datas, desc="Val"):
                if isinstance(data, dict):
                    data = {k: v.to(model.device) for k, v in data.items()}
                    outputs = model(**data)
                else:
                    outputs = model(data.to(model.device))
                output_datas.append(
                    {
                        "past_keys": outputs[0].float().cpu().numpy(),
                        "past_values": outputs[1].float().cpu().numpy(),
                        "last_hidden_state": outputs[2].float().cpu().numpy(),
                    }
                )

        if self.is_quantized:
            val_loop(self, self.val_datas_after)
            return Pi05Metric(self.val_datas_after)
        val_loop(self, self.val_datas_before)
        return Pi05Metric(self.val_datas_before)

    def quantize(self, quant_cfg, calib_data, export_dir, *, measure_quant_error=False):
        calib_dataloader = self.get_calibrate_dataset(calib_data)
        quantize_model(self, quant_cfg, calib_dataloader, measure_quant_error=measure_quant_error)
        self.is_quantized = True
        set_dynamic_quant(self, "fp16")
        self.export(export_dir, dynamo=False)
