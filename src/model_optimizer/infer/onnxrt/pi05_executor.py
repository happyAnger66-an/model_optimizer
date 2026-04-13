"""Pi05OnnxRTExecutor — Pi0.5 子模块 ONNX Runtime 替换（对标 Pi05TensorRTExecutor）。

使用方式与 TensorRT 版完全对称：

    executor = Pi05OnnxRTExecutor(policy, precision)
    executor.load_model(addict.Dict({
        "engine_path": "/path/to/onnx_models/",
        "vit_engine": "vit.onnx",
        "llm_engine": "llm.onnx",
        "expert_engine": "expert.onnx",
        "denoise_engine": "denoise.onnx",
    }))
    # 之后 policy.infer(obs) 透明走 ORT

可选：在 ``addict.Dict`` 中传入 ``ort_providers``（字符串元组/列表）以覆盖 ONNX Runtime EP 顺序；
NVFP4（dtype 23）建议在环境中启用 ``TensorRTExecutionProvider``。
"""

from __future__ import annotations

import os
import types
from typing import Any

import torch
from termcolor import colored
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling

from ..executor import Executor
from ...models.pi05.model_pi05 import Pi05Model
from .ort_engine import OrtEngine


class Pi05OnnxRTExecutor(Executor):
    """将 Pi0.5 各子模块替换为 ONNX Runtime 推理。

    config 结构与 TensorRT 版相同（engine_path + vit_engine + llm_engine + ...），
    只是引擎文件为 ``.onnx`` 而非 ``.engine``。

    Args:
        policy: openpi Policy 对象。
        precision: 推理精度（bf16 输入会自动转 fp32 给 ORT）。
        config: 可选初始配置。
    """

    def __init__(self, policy: Any, precision: Any = torch.bfloat16, config: Any = None) -> None:
        super().__init__(policy)
        pi05_model = Pi05Model(policy)
        self.pi05_model = pi05_model.model
        self.config = config

    def load_model(self, config: Any = None) -> None:
        if config is None:
            return
        self.config = config
        self._setup_ort_engines()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)

    @staticmethod
    def _wrap_past_key_values(input_keys: torch.Tensor, input_values: torch.Tensor) -> DynamicCache:
        k_v_cache = DynamicCache()
        num_layers = input_keys.shape[0]
        for i in range(num_layers):
            k_v_cache.update(input_keys[i : i + 1], input_values[i : i + 1], i)
        return k_v_cache

    @staticmethod
    def _stack_past_key_value_tensors(past_key_values: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """将 LLM 输出的 KV cache 堆成 ``[num_layers, ...]`` 张量。"""
        if past_key_values is None:
            raise ValueError("past_key_values is None")
        n = len(past_key_values)
        keys, vals = [], []
        for i in range(n):
            entry = past_key_values[i]
            if isinstance(entry, (tuple, list)):
                keys.append(entry[0])
                vals.append(entry[1])
            else:
                raise TypeError(f"Unexpected past_key_values[{i}] type: {type(entry)}")
        return torch.cat(keys, dim=0), torch.cat(vals, dim=0)

    def _ort_providers(self) -> list[str] | None:
        """来自配置的 ORT EP 顺序；未配置时由 ``OrtEngine`` 使用内置默认（含 TensorRT EP，若可用）。"""
        if not self.config:
            return None
        raw = getattr(self.config, "ort_providers", None)
        if raw is None:
            return None
        lst = [str(x) for x in raw]
        return lst if lst else None

    def _setup_ort_engines(self) -> None:
        """按配置替换各子模块 forward 为 OrtEngine 调用。"""
        if not self.config.engine_path:
            return

        if getattr(self.config, "vit_engine", None):
            self._replace_vit()

        if getattr(self.config, "embed_prefix_engine", None):
            self._replace_embed_prefix()

        if getattr(self.config, "llm_engine", None):
            self._replace_llm()

        if getattr(self.config, "expert_engine", None):
            self._replace_expert()

        if getattr(self.config, "denoise_engine", None):
            self._replace_denoise()

    def _replace_vit(self) -> None:
        onnx_path = os.path.join(self.config.engine_path, self.config.vit_engine)
        print(colored(f"[ORT] replace vision_tower with {self.config.vit_engine}", "green"))

        def vit_return_wrap(output: dict) -> torch.Tensor:
            return output["image_features"]

        vit_engine = OrtEngine(onnx_path, return_wrap=vit_return_wrap, perf=True, providers=self._ort_providers())

        def get_image_features(pixel_values: torch.Tensor) -> torch.Tensor:
            return vit_engine(pixel_values)

        self.pi05_model.paligemma_with_expert.paligemma.model.get_image_features = get_image_features

    def _replace_embed_prefix(self) -> None:
        onnx_path = os.path.join(self.config.engine_path, self.config.embed_prefix_engine)
        print(colored(f"[ORT] replace embed_prefix with {self.config.embed_prefix_engine}", "green"))

        embed_prefix_engine = OrtEngine(onnx_path, perf=True, providers=self._ort_providers())

        def embed_prefix_ort(
            self_m: Any,
            images: list[torch.Tensor],
            img_masks: list[torch.Tensor],
            lang_tokens: torch.Tensor,
            lang_masks: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        self.pi05_model.embed_prefix = types.MethodType(embed_prefix_ort, self.pi05_model)

    def _replace_llm(self) -> None:
        onnx_path = os.path.join(self.config.engine_path, self.config.llm_engine)
        print(colored(f"[ORT] replace language_model with {self.config.llm_engine}", "green"))

        llm_engine = OrtEngine(onnx_path, perf=True, providers=self._ort_providers())
        executor_self = self

        def llm_forward(
            input_ids: Any = None,
            attention_mask: Any = None,
            position_ids: Any = None,
            past_key_values: Any = None,
            inputs_embeds: Any = None,
            labels: Any = None,
            use_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            cache_position: Any = None,
            logits_to_keep: Any = None,
            adarms_cond: Any = None,
            **kwargs: Any,
        ) -> BaseModelOutputWithPast:
            outputs = llm_engine(inputs_embeds, attention_mask, position_ids)

            # 兼容两种导出格式（与 TRT 版一致）
            if isinstance(outputs, dict) and "past_keys" in outputs and "past_values" in outputs:
                k_v_cache = executor_self._wrap_past_key_values(
                    outputs["past_keys"], outputs["past_values"],
                )
            else:
                present_keys, present_vals = [], []
                i = 0
                while True:
                    name = f"present_key_values.{i}"
                    if not (isinstance(outputs, dict) and name in outputs):
                        break
                    kv = outputs[name]
                    present_keys.append(kv[:, 0])
                    present_vals.append(kv[:, 1])
                    i += 1
                if i == 0:
                    raise KeyError(
                        "LLM engine outputs missing both (past_keys,past_values) "
                        "and present_key_values.{i}."
                    )
                input_keys = torch.cat(present_keys, dim=0)
                input_values = torch.cat(present_vals, dim=0)
                k_v_cache = executor_self._wrap_past_key_values(input_keys, input_values)

            return BaseModelOutputWithPast(
                last_hidden_state=outputs["last_hidden_state"],
                past_key_values=k_v_cache,
            )

        self.pi05_model.paligemma_with_expert.paligemma.model.language_model.forward = llm_forward

    def _replace_expert(self) -> None:
        onnx_path = os.path.join(self.config.engine_path, self.config.expert_engine)
        print(colored(f"[ORT] replace expert with {self.config.expert_engine}", "green"))

        def expert_return_wrap(output: dict) -> BaseModelOutputWithPooling:
            return BaseModelOutputWithPooling(
                last_hidden_state=output["last_hidden_state"],
            )

        expert_engine = OrtEngine(onnx_path, return_wrap=expert_return_wrap, perf=True, providers=self._ort_providers())

        def expert_forward(
            inputs_ids: Any = None,
            attention_mask: Any = None,
            position_ids: Any = None,
            past_key_values: Any = None,
            inputs_embeds: Any = None,
            use_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            cache_position: Any = None,
            adarms_cond: Any = None,
            **kwargs: Any,
        ) -> BaseModelOutputWithPooling:
            input_keys = torch.cat(
                [past_key_values[i][0] for i in range(len(past_key_values))], dim=0,
            )
            input_values = torch.cat(
                [past_key_values[i][1] for i in range(len(past_key_values))], dim=0,
            )
            return expert_engine(
                attention_mask, position_ids, inputs_embeds, adarms_cond,
                input_keys, input_values,
            )

        self.pi05_model.paligemma_with_expert.gemma_expert.model.forward = expert_forward

    def _replace_denoise(self) -> None:
        onnx_path = os.path.join(self.config.engine_path, self.config.denoise_engine)
        print(colored(f"[ORT] replace denoise_step with {self.config.denoise_engine}", "green"))

        denoise_engine = OrtEngine(onnx_path, perf=True, providers=self._ort_providers())
        executor_self = self

        def denoise_step_ort(
            self_m: Any,
            state: Any,
            prefix_pad_masks: torch.Tensor,
            past_key_values: Any,
            x_t: torch.Tensor,
            timestep: torch.Tensor,
        ) -> torch.Tensor:
            del state  # pi05 embed_suffix 不使用 state
            input_keys, input_values = executor_self._stack_past_key_value_tensors(
                past_key_values,
            )
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

        self.pi05_model.denoise_step = types.MethodType(denoise_step_ort, self.pi05_model)
