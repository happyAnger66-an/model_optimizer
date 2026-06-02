# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pi0.5 TensorRT 子图挂载：配置、校验与各 stage 的 hook 工厂（供 ``Pi05TensorRTExecutor`` 调用）。"""

from __future__ import annotations

import math
import numbers
import os
import pathlib
import types
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from termcolor import colored
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling

from .trt_hook_timer import trt_hook_timer
from .trt_torch import Engine

_TRT_ATTN_MASK_NEG_CAP_KEY = "trt_attention_mask_neg_cap"


def cfg_get(config: Any, key: str, default: Any) -> Any:
    """安全读取 config（兼容 ``addict.Dict`` 与普通对象）。"""
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


@dataclass(frozen=True)
class TrtRuntimeOptions:
    """TRT ``Engine`` 共用的 perf / CUDA Graph 开关。"""

    perf: bool = True
    use_cuda_graph: bool = False
    cuda_graph_warmup: int = 3
    perf_warmup: int = 20
    perf_print_interval: int = 50

    @classmethod
    def from_config(cls, config: Any) -> TrtRuntimeOptions:
        return cls(
            perf=bool(cfg_get(config, "trt_perf", True)),
            use_cuda_graph=bool(cfg_get(config, "trt_cuda_graph", False)),
            cuda_graph_warmup=int(cfg_get(config, "trt_cuda_graph_warmup", 3) or 3),
            perf_warmup=int(cfg_get(config, "trt_perf_warmup", 20) or 20),
            perf_print_interval=int(
                cfg_get(config, "trt_perf_print_interval", 50) or 50
            ),
        )

    def engine_kwargs(self) -> dict[str, Any]:
        return {
            "perf": self.perf,
            "use_cuda_graph": self.use_cuda_graph,
            "cuda_graph_warmup": self.cuda_graph_warmup,
            "perf_warmup": self.perf_warmup,
            "perf_print_interval": self.perf_print_interval,
        }


@dataclass(frozen=True)
class EmbedPrefixOptions:
    """``embed_prefix`` 三路互斥策略的配置视图。"""

    vit_batch_views: bool
    use_flashrt_siglip: bool
    embed_prefix_engine_name: str | None
    vit_engine_name: str | None

    @classmethod
    def from_config(cls, config: Any) -> EmbedPrefixOptions:
        return cls(
            vit_batch_views=bool(cfg_get(config, "vit_batch_views", False)),
            use_flashrt_siglip=bool(
                cfg_get(config, "use_flashrt_siglip_embed_prefix", False)
            ),
            embed_prefix_engine_name=cfg_get(config, "embed_prefix_engine", None)
            or None,
            vit_engine_name=cfg_get(config, "vit_engine", None) or None,
        )

    def validate(self) -> None:
        if self.use_flashrt_siglip and self.embed_prefix_engine_name:
            raise ValueError(
                "config.use_flashrt_siglip_embed_prefix=True is incompatible with "
                "config.embed_prefix_engine (TRT whole-graph embed_prefix). "
                "Disable one of them."
            )
        if self.vit_batch_views and (
            self.use_flashrt_siglip or self.embed_prefix_engine_name
        ):
            raise ValueError(
                "config.vit_batch_views=True is incompatible with "
                "use_flashrt_siglip_embed_prefix / embed_prefix_engine "
                "(those already own the vision/embed_prefix stage). Disable one."
            )
        if self.vit_batch_views and not self.vit_engine_name:
            raise ValueError(
                "config.vit_batch_views=True requires config.vit_engine "
                "(the batched SigLIP TRT engine to call once for all views)."
            )


def make_engine(
    engine_path: str,
    engine_name: str,
    opts: TrtRuntimeOptions,
    *,
    return_wrap: Callable[[Any], Any] | None = None,
) -> Engine:
    return Engine(
        os.path.join(engine_path, engine_name),
        return_wrap=return_wrap,
        **opts.engine_kwargs(),
    )


def llm_outputs_to_dynamic_cache(
    outputs: dict[str, torch.Tensor],
    wrap_past_key_values: Callable[[torch.Tensor, torch.Tensor], DynamicCache],
) -> DynamicCache:
    """将 LLM TRT 输出（旧 ``past_keys/past_values`` 或 ``present_key_values.N``）转为 ``DynamicCache``。"""
    if "past_keys" in outputs and "past_values" in outputs:
        return wrap_past_key_values(outputs["past_keys"], outputs["past_values"])

    present_keys: list[torch.Tensor] = []
    present_vals: list[torch.Tensor] = []
    i = 0
    while f"present_key_values.{i}" in outputs:
        kv = outputs[f"present_key_values.{i}"]
        present_keys.append(kv[:, 0])
        present_vals.append(kv[:, 1])
        i += 1
    if i == 0:
        raise KeyError(
            "LLM engine outputs missing both (past_keys,past_values) and "
            "present_key_values.{i}."
        )
    input_keys = torch.cat(present_keys, dim=0)
    input_values = torch.cat(present_vals, dim=0)
    return wrap_past_key_values(input_keys, input_values)


def resolve_trt_attention_mask_neg_cap(config: Any) -> float | None:
    """读取 ``trt_attention_mask_neg_cap``，默认 ``-1e4``；``None`` 表示关闭裁剪。"""
    default = -1e4
    if config is None:
        return default
    if isinstance(config, Mapping):
        raw = config.get(_TRT_ATTN_MASK_NEG_CAP_KEY, default)
    else:
        raw = getattr(config, _TRT_ATTN_MASK_NEG_CAP_KEY, default)
    if raw is None:
        return None
    if isinstance(raw, numbers.Real) and not isinstance(raw, bool):
        return float(raw)
    warnings.warn(
        f"Ignoring invalid {_TRT_ATTN_MASK_NEG_CAP_KEY}={raw!r} ({type(raw).__name__}), "
        f"using {default}",
        UserWarning,
        stacklevel=2,
    )
    return default


def sanitize_additive_attention_mask_for_trt(
    attention_mask: torch.Tensor | None,
    neg_cap: float | None,
) -> torch.Tensor | None:
    if attention_mask is None or neg_cap is None:
        return attention_mask
    if not isinstance(neg_cap, numbers.Real) or isinstance(neg_cap, bool):
        raise TypeError(
            "trt_attention_mask_neg_cap must be a real number or None, "
            f"got {type(neg_cap).__name__!r}"
        )
    return attention_mask.clamp(min=float(neg_cap))


def vit_scale_fix_enabled_from_env() -> bool:
    """兼容旧部署：``PI05_TRT_VIT_SCALE_FIX=1`` / ``true`` 等。"""
    return os.environ.get("PI05_TRT_VIT_SCALE_FIX", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )


def resolve_trt_vit_scale_fix(config: Any | None) -> bool:
    """是否对 TRT ViT 输出乘 ``sqrt(hidden_size)``（与 HF ``get_image_features`` 对齐）。

    优先级：config 显式设置 ``trt_vit_scale_fix`` > 环境变量 ``PI05_TRT_VIT_SCALE_FIX``。
    """
    if config is not None:
        raw = cfg_get(config, "trt_vit_scale_fix", None)
        if raw is not None:
            return bool(raw)
    return vit_scale_fix_enabled_from_env()


def apply_vit_hidden_scale(out: torch.Tensor, hidden_size: int) -> torch.Tensor:
    scale = float(math.sqrt(float(hidden_size)))
    return (out.to(torch.float32) * scale).to(out.dtype)


class Pi05TrtEngineInstaller:
    """按 stage 将 TRT engine 挂到 ``PI0Pytorch`` 实例（Strategy：每 stage 一个 install 方法）。"""

    def __init__(self, executor: Any) -> None:
        self._ex = executor
        self._model = executor.pi05_model
        self._config = executor.config
        self._engines: dict[str, Engine] = executor._trt_engines
        self._opts = TrtRuntimeOptions.from_config(self._config)
        self._engine_root = str(self._config.engine_path)
        self._vit_scale_fix = resolve_trt_vit_scale_fix(self._config)

    @property
    def opts(self) -> TrtRuntimeOptions:
        return self._opts

    def install_all(self) -> None:
        embed_opts = EmbedPrefixOptions.from_config(self._config)
        embed_opts.validate()

        if self._config.vit_engine:
            self.install_vit(str(self._config.vit_engine))

        if embed_opts.vit_batch_views:
            self.install_embed_prefix_batched()
        elif embed_opts.use_flashrt_siglip:
            self.install_embed_prefix_flashrt()
        elif embed_opts.embed_prefix_engine_name:
            self.install_embed_prefix_whole_graph(embed_opts.embed_prefix_engine_name)

        if self._config.llm_engine:
            self.install_llm(str(self._config.llm_engine))

        if self._config.expert_engine:
            self.install_expert(str(self._config.expert_engine))

        denoise_name = cfg_get(self._config, "denoise_engine", None)
        if denoise_name:
            self.install_denoise(
                str(denoise_name),
                stack_past_kv=self._ex._stack_past_key_value_tensors,
            )

    def install_vit(self, engine_name: str) -> None:
        print(colored(f"replace vision_tower with {engine_name}", "green"))
        if self._vit_scale_fix:
            print(
                colored(
                    "vit: TRT output scale fix enabled (×sqrt(hidden_size), trt_vit_scale_fix / "
                    "PI05_TRT_VIT_SCALE_FIX)",
                    "green",
                )
            )

        def vit_return_wrap(output: dict[str, torch.Tensor]) -> torch.Tensor:
            return output["image_features"]

        vit_engine = make_engine(
            self._engine_root, engine_name, self._opts, return_wrap=vit_return_wrap
        )
        self._engines["vit"] = vit_engine
        paligemma = self._model.paligemma_with_expert.paligemma.model

        def get_image_features(pixel_values: torch.Tensor) -> torch.Tensor:
            out = vit_engine(pixel_values)
            if self._vit_scale_fix:
                try:
                    h = int(paligemma.config.text_config.hidden_size)
                except Exception:
                    h = 2048
                out = apply_vit_hidden_scale(out, h)
            return out

        paligemma.get_image_features = trt_hook_timer("trt.vit.get_image_features")(
            get_image_features
        )

    def install_embed_prefix_batched(self) -> None:
        paligemma_model = self._model.paligemma_with_expert.paligemma.model

        def embed_prefix_vit_batched(
            self_m: Any,
            images: list[torch.Tensor],
            img_masks: list[torch.Tensor],
            lang_tokens: torch.Tensor,
            lang_masks: torch.Tensor,
        ):
            images = list(images)
            img_masks = list(img_masks)
            bview = images[0].shape[0]
            stacked = torch.cat(images, dim=0)
            feats = paligemma_model.get_image_features(stacked)

            embs: list[torch.Tensor] = []
            pad_masks: list[torch.Tensor] = []
            att_masks: list[int] = []
            for i, img_mask in enumerate(img_masks):
                img_emb = feats[i * bview : (i + 1) * bview]
                bsize, num_img_embs = img_emb.shape[:2]
                embs.append(img_emb)
                pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
                att_masks += [0] * num_img_embs

            lang_emb = self_m.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
            embs.append(lang_emb)
            pad_masks.append(lang_masks)
            att_masks += [0] * lang_emb.shape[1]

            embs_cat = torch.cat(embs, dim=1)
            pad_cat = torch.cat(pad_masks, dim=1)
            att = torch.tensor(att_masks, dtype=torch.bool, device=pad_cat.device)
            att = att[None, :].expand(pad_cat.shape[0], len(att_masks))
            return embs_cat, pad_cat, att

        hooked = trt_hook_timer("trt.embed_prefix_vit_batched")(embed_prefix_vit_batched)
        self._model.embed_prefix = types.MethodType(hooked, self._model)
        print(
            colored(
                "embed_prefix: batched SigLIP (all views in one vit engine call)",
                "green",
            )
        )

    def install_embed_prefix_flashrt(self) -> None:
        ckpt = cfg_get(self._config, "flashrt_checkpoint_dir", None)
        if not ckpt:
            ckpt = str(pathlib.Path(self._engine_root).parent)
        ckpt = str(pathlib.Path(ckpt).resolve())
        nv = int(
            cfg_get(
                self._config,
                "num_views",
                cfg_get(self._config, "num_image_views", 2),
            )
        )
        from model_optimizer.infer.flash.siglip import FlashRtSiglipVision

        self._ex._flashrt_siglip = FlashRtSiglipVision(
            ckpt,
            num_views=nv,
            use_cuda_graph=bool(
                cfg_get(self._config, "flashrt_siglip_use_cuda_graph", True)
            ),
        )
        self._ex._orig_embed_prefix_for_flashrt = self._model.embed_prefix
        ex = self._ex

        def embed_prefix_flashrt_hybrid(
            self_m: Any,
            images: list[torch.Tensor],
            img_masks: list[torch.Tensor],
            lang_tokens: torch.Tensor,
            lang_masks: torch.Tensor,
        ):
            v = ex._flashrt_siglip.forward_from_torch_images(images)
            v_b = v.unsqueeze(0)
            pad_parts: list[torch.Tensor] = []
            for m in img_masks:
                m = m.to(device=v_b.device, dtype=v_b.dtype)
                if m.dim() != 2:
                    raise ValueError(f"img_mask must be 2D, got {tuple(m.shape)}")
                if m.shape[1] == 1:
                    m = m[:, None].expand(-1, 256)
                elif m.shape[1] != 256:
                    raise ValueError(
                        "Each img_mask must be [B,1] or [B,256] per view; "
                        f"got {tuple(m.shape)}"
                    )
                pad_parts.append(m)
            img_pad = torch.cat(pad_parts, dim=1)

            le = self_m.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb = le * math.sqrt(float(le.shape[-1]))
            embs = torch.cat([v_b, lang_emb], dim=1)
            pad_masks = torch.cat([img_pad, lang_masks], dim=1)
            bsize = pad_masks.shape[0]
            seq_len = embs.shape[1]
            att_t = torch.zeros(
                (1, seq_len), device=embs.device, dtype=torch.bool
            ).expand(bsize, -1)
            return embs, pad_masks, att_t

        self._model.embed_prefix = types.MethodType(
            embed_prefix_flashrt_hybrid, self._model
        )
        print(
            colored(
                f"embed_prefix: FlashRT SigLIP vision + PyTorch lang (ckpt={ckpt}, nv={nv})",
                "green",
            )
        )

    def install_embed_prefix_whole_graph(self, engine_name: str) -> None:
        print(colored(f"replace embed_prefix with {engine_name}", "green"))
        embed_prefix_engine = make_engine(self._engine_root, engine_name, self._opts)
        self._engines["embed_prefix"] = embed_prefix_engine

        def embed_prefix_trt(
            self_m: Any,
            images: list[torch.Tensor],
            img_masks: list[torch.Tensor],
            lang_tokens: torch.Tensor,
            lang_masks: torch.Tensor,
        ):
            del self_m
            kw: dict[str, torch.Tensor] = {}
            for i, (img, m) in enumerate(zip(images, img_masks, strict=True)):
                kw[f"image_{i}"] = img
                kw[f"image_mask_{i}"] = m
            kw["lang_tokens"] = lang_tokens
            kw["lang_masks"] = lang_masks
            out = embed_prefix_engine(**kw)
            return (
                out["prefix_embs"],
                out["prefix_pad_masks"],
                out["prefix_att_masks"],
            )

        hooked = trt_hook_timer("trt.embed_prefix")(embed_prefix_trt)
        self._model.embed_prefix = types.MethodType(hooked, self._model)

    def install_llm(self, engine_name: str) -> None:
        print(colored(f"replace language_model with {engine_name}", "green"))
        llm_engine = make_engine(self._engine_root, engine_name, self._opts)
        self._engines["llm"] = llm_engine
        ex = self._ex

        def llm_forward(
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            cache_position=None,
            logits_to_keep=None,
            adarms_cond=None,
            **kwargs,
        ):
            del (
                input_ids,
                past_key_values,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                cache_position,
                logits_to_keep,
                adarms_cond,
                kwargs,
            )
            neg_cap = resolve_trt_attention_mask_neg_cap(self._config)
            attention_mask = sanitize_additive_attention_mask_for_trt(
                attention_mask, neg_cap
            )
            outputs = llm_engine(inputs_embeds, attention_mask, position_ids)
            k_v_cache = llm_outputs_to_dynamic_cache(outputs, ex._wrap_past_key_values)
            return BaseModelOutputWithPast(
                last_hidden_state=outputs["last_hidden_state"],
                past_key_values=k_v_cache,
            )

        lang_model = self._model.paligemma_with_expert.paligemma.model.language_model
        lang_model.forward = trt_hook_timer("trt.llm.forward")(llm_forward)

    def install_expert(self, engine_name: str) -> None:
        print(colored(f"replace expert with {engine_name}", "green"))

        def expert_return_wrap(output: dict[str, torch.Tensor]) -> BaseModelOutputWithPooling:
            return BaseModelOutputWithPooling(
                last_hidden_state=output["last_hidden_state"],
            )

        expert_engine = make_engine(
            self._engine_root,
            engine_name,
            self._opts,
            return_wrap=expert_return_wrap,
        )
        self._engines["expert"] = expert_engine

        def expert_forward(
            inputs_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            cache_position=None,
            adarms_cond=None,
            **kwargs,
        ):
            del (
                inputs_ids,
                use_cache,
                output_attentions,
                output_hidden_states,
                cache_position,
                kwargs,
            )
            neg_cap = resolve_trt_attention_mask_neg_cap(self._config)
            attention_mask = sanitize_additive_attention_mask_for_trt(
                attention_mask, neg_cap
            )
            input_keys = torch.cat(
                [past_key_values[i][0] for i in range(len(past_key_values))], dim=0
            )
            input_values = torch.cat(
                [past_key_values[i][1] for i in range(len(past_key_values))], dim=0
            )
            return expert_engine(
                attention_mask,
                position_ids,
                inputs_embeds,
                adarms_cond,
                input_keys,
                input_values,
            )

        expert_model = self._model.paligemma_with_expert.gemma_expert.model
        expert_model.forward = trt_hook_timer("trt.expert.forward")(expert_forward)

    def install_denoise(self, engine_name: str, *, stack_past_kv: Callable) -> None:
        print(colored(f"replace denoise_step with {engine_name}", "green"))
        denoise_engine = make_engine(self._engine_root, engine_name, self._opts)
        self._engines["denoise"] = denoise_engine

        from model_optimizer.infer.pi05_adarms import (
            AdaRmsModulator,
            adarms_precompute_enabled,
        )

        adarms_modulator = (
            AdaRmsModulator(self._model)
            if adarms_precompute_enabled(self._config)
            else None
        )
        if adarms_modulator is not None:
            print(colored("[adarms] denoise host 侧预计算已启用（喂 adarms_mod）", "green"))

        def denoise_step_trt(
            self_m: Any,
            state: Any,
            prefix_pad_masks: torch.Tensor,
            past_key_values: Any,
            x_t: torch.Tensor,
            timestep: torch.Tensor,
        ):
            del self_m, state
            input_keys, input_values = stack_past_kv(past_key_values)
            if adarms_modulator is not None:
                outputs = denoise_engine(
                    prefix_pad_masks=prefix_pad_masks,
                    past_keys=input_keys,
                    past_values=input_values,
                    x_t=x_t,
                    adarms_mod=adarms_modulator(timestep),
                )
            else:
                outputs = denoise_engine(
                    prefix_pad_masks=prefix_pad_masks,
                    past_keys=input_keys,
                    past_values=input_values,
                    x_t=x_t,
                    timestep=timestep,
                )
            if isinstance(outputs, dict):
                return outputs["v_t"]
            return outputs

        hooked = trt_hook_timer("trt.denoise_step")(denoise_step_trt)
        self._model.denoise_step = types.MethodType(hooked, self._model)
