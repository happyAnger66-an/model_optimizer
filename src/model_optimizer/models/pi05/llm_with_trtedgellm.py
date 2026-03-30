"""
PI0.5 PaliGemma 语言塔与 TensorRT-Edge-LLM 自定义 Attention 插件的对接。

TensorRT-Edge-LLM 的核心 ONNX/TRT 优化在 **EdgeLLMAttention**（``trt::attention_plugin``）。本模块在
``GemmaModel`` 上仅包装各层 ``self_attn`` 为 :class:`GemmaAttentionTrtEdge`：PI0 前向仍走 **原生**
``GemmaAttention``；ONNX 导出则通过 :class:`GemmaModelEdgeOnnxExport` **显式走** ``edge.forward``，
使图中出现自定义 Attention 算子。

为何不直接 ``torch.onnx.export(self)`` 或 ``export(self.model)``？
  - ``LLMWithTrtEdgeLLM.forward`` → ``Pi05TrtEdgeLanguageModel.forward`` → ``GemmaModel.forward``，
    内部每层调用的是 ``GemmaAttentionTrtEdge.forward``，其 **默认委托 ``native.forward``**，
    仍为 HF 注意力，ONNX 里 **不会** 出现 ``trt::attention_plugin``。
  - ``GemmaModel`` 从未在默认路径上调用 ``.edge.forward``；``edge`` 只为导出/引擎预留。
  - 因此必须用单独子模块 ``GemmaModelEdgeOnnxExport``，在解码循环里 **显式调用**
    ``layer.self_attn.edge.forward``，与 PI0 数值路径拆分。

注意：``EdgeLLMAttention.forward`` 在 PyTorch 中 attention 输出为 **dummy**；ONNX 用于 TRT 构建，
数值对齐请在引擎侧或对比 HF 原生子图。

依赖：将 ``third_party/TensorRT-Edge-LLM`` 加入 ``PYTHONPATH``（或安装对应包）。
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import onnx
import torch
import torch.nn as nn
from tqdm import tqdm

from model_optimizer.evaluate.metrics.pi05 import Pi05Metric
from model_optimizer.quantization.quantization_utils import quantize_model
from model_optimizer.utils.utils import is_fp4_quantized, is_nvfp4_quantized, set_dynamic_quant
from modelopt.onnx.quantization.qdq_utils import fp4qdq_to_2dq
from termcolor import colored
from transformers.cache_utils import DynamicCache

from ..model import Model

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _ensure_trt_edgellm_on_path() -> None:
    root = _repo_root()
    trt = root / "third_party" / "TensorRT-Edge-LLM"
    if trt.is_dir():
        p = str(trt)
        if p not in sys.path:
            sys.path.insert(0, p)


class GemmaAttentionTrtEdge(nn.Module):
    """
    包装 HuggingFace/OpenPI 的 Gemma 自注意力：保留 ``native`` 的 HF 前向，并附加同权的 ``EdgeLLMAttention``。

    - ``forward(...)``：与底层 ``GemmaAttention.forward`` 相同签名，**默认走 native**（eager / cache 与 PI0 一致）。
      对包装器或 ``GemmaModel`` 做 ONNX trace 时仍执行 native，图中 **无** Edge 插件；插件导出见
      :class:`GemmaModelEdgeOnnxExport`。
    - ``edge``：TensorRT-Edge 导出用；``edge.forward`` 需 ``past_key_values``、``rope_rotary_cos_sin`` 等
      Edge 侧张量（参见 ``tensorrt_edgellm/onnx_export/llm_export.py`` 的 dummy 约定）。

    未在 ``native`` 上的属性（如 ``q_proj``、``scaling``、``layer_idx``）通过 ``__getattr__`` 转发到
    ``native``，以兼容 OpenPI ``layer.self_attn.q_proj`` 等写法。
    """

    def __init__(self, native_attn: nn.Module) -> None:
        super().__init__()
        if native_attn is None:
            raise ValueError("native_attn is required")
        # 用属性赋值注册子模块，勿用 add_module：add_module 会 hasattr(self, "native")，
        # 在自定义 __getattr__ 下会先于 _modules 写入触发委托，导致递归或链式 AttributeError。
        self.native = native_attn
        _ensure_trt_edgellm_on_path()
        from tensorrt_edgellm.llm_models.layers.layers import EdgeLLMAttention

        self.edge = EdgeLLMAttention(native_attn, eagle3_draft=False)

    def forward(self, *args: Any, **kwargs: Any):
        return self.native.forward(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            # 勿把 native/edge 的缺失转发到内层 GemmaAttention（避免误解析或掩盖错误）。
            if name in ("native", "edge"):
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'"
                ) from None
            native = self._modules.get("native")
            if native is None:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'"
                ) from None
            return getattr(native, name)


def install_gemma_edge_attention_wrappers(
    gemma_model: nn.Module,
    *,
    skip_if_wrapped: bool = True,
) -> nn.Module:
    """
    就地替换 ``GemmaModel.layers[*].self_attn`` 为 :class:`GemmaAttentionTrtEdge`。

    **会修改**传入的 ``gemma_model``（与 PI0 共用同一 Decoder 时需知悉）。

    Parameters
    ----------
    skip_if_wrapped:
        若该层 ``self_attn`` 已是 :class:`GemmaAttentionTrtEdge` 则跳过。
    """
    if getattr(gemma_model.config, "use_adarms", False):
        raise NotImplementedError(
            "Edge attention 包装路径当前仅适用于 config.use_adarms=False 的 Gemma 塔。"
        )

    for layer in gemma_model.layers:
        if skip_if_wrapped and isinstance(layer.self_attn, GemmaAttentionTrtEdge):
            continue
        layer.self_attn = GemmaAttentionTrtEdge(layer.self_attn)
    return gemma_model


def iter_edge_attention_modules(gemma_model: nn.Module) -> Iterator[Any]:
    """按层顺序产出 ``EdgeLLMAttention``（需先 ``install_gemma_edge_attention_wrappers``）。"""
    for layer in gemma_model.layers:
        attn = layer.self_attn
        if isinstance(attn, GemmaAttentionTrtEdge):
            yield attn.edge
        else:
            raise TypeError(
                "layer.self_attn 不是 GemmaAttentionTrtEdge；请先 install_gemma_edge_attention_wrappers"
            )


def _gated_residual(
    x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor | None
) -> torch.Tensor:
    if gate is None:
        return x + y
    return x + y * gate


class GemmaModelEdgeOnnxExport(nn.Module):
    """
    仅用于 ONNX 导出：复现 Gemma 解码器数据流（RMSNorm + MLP + 最终 norm），
    自注意力一律调用 ``GemmaAttentionTrtEdge.edge.forward``（含 ``trt::attention_plugin``）。

    与 :class:`LLMWithTrtEdgeLLM` / ``self.model`` 分离的原因：
      训练与 PI0 推理必须走 ``GemmaModel`` 自带前向 + ``GemmaAttentionTrtEdge.native``；
      若对 ``self`` 或 ``self.model`` trace，执行路径不变，ONNX 中 **不会出现** Edge 插件。
      本类单独实现一层解码循环并 **只调用 ``edge.forward``**，才能把自定义算子写入图。

    输入接口与 :class:`LLM` 导出一致：``inputs_embeds``、``attention_mask``（4D，导出时仅参与形状/设备）、``position_ids``。
    内部按 ``tensorrt_edgellm.onnx_export.llm_export.create_dummy_inputs`` 的约定构造
    KV / RoPE / ``context_lengths`` 等，以便与 Edge 工具链对齐。
    """

    def __init__(self, gemma_model: nn.Module) -> None:
        super().__init__()
        if getattr(gemma_model.config, "use_adarms", False):
            raise NotImplementedError("GemmaModelEdgeOnnxExport 暂不支持 use_adarms=True")
        for layer in gemma_model.layers:
            if not isinstance(layer.self_attn, GemmaAttentionTrtEdge):
                raise TypeError(
                    "各层 self_attn 须为 GemmaAttentionTrtEdge，请先 install_gemma_edge_attention_wrappers"
                )
        self.gemma = gemma_model

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gemma = self.gemma
        device = inputs_embeds.device
        bsz, seq_len, _ = inputs_embeds.shape
        hidden_states = inputs_embeds
        if (
            len(gemma.layers) > 0
            and gemma.layers[0].self_attn.native.q_proj.weight.dtype == torch.bfloat16
        ):
            hidden_states = hidden_states.to(torch.bfloat16)

        position_ids_i64 = position_ids
        if position_ids_i64.dim() != 2:
            raise ValueError(f"position_ids 期望 [batch, seq]，当前 {tuple(position_ids_i64.shape)}")
        cos, sin = gemma.rotary_emb(hidden_states, position_ids_i64)
        rope_rotary_cos_sin = torch.cat(
            [cos.to(torch.float32), sin.to(torch.float32)], dim=-1
        )

        cfg = gemma.config
        num_kv_heads = cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

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

        context_lengths = torch.full(
            (bsz,),
            past_len + seq_len,
            dtype=torch.int32,
            device=device,
        )
        kvcache_start_index = torch.zeros(bsz, dtype=torch.int32, device=device)
        attention_mask_i32 = torch.ones(
            bsz,
            seq_len,
            seq_len + past_len,
            dtype=torch.int32,
            device=device,
        )
        position_ids_i32 = position_ids_i64.to(torch.int32)

        for layer_idx, layer in enumerate(gemma.layers):
            residual = hidden_states
            hidden_states, gate = layer.input_layernorm(hidden_states, None)
            edge = layer.self_attn.edge
            attn_out, past_list[layer_idx] = edge(
                hidden_states,
                past_list[layer_idx],
                rope_rotary_cos_sin,
                context_lengths,
                kvcache_start_index,
                attention_mask_i32,
                position_ids_i32,
            )
            hidden_states = _gated_residual(residual, attn_out, gate)

            residual = hidden_states
            hidden_states, gate = layer.post_attention_layernorm(hidden_states, None)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = _gated_residual(residual, hidden_states, gate)

        hidden_states, _ = gemma.norm(hidden_states, None)

        past_keys_tensor = torch.cat(
            [past_list[i][:, 0] for i in range(len(past_list))], dim=0
        )
        past_values_tensor = torch.cat(
            [past_list[i][:, 1] for i in range(len(past_list))], dim=0
        )
        return past_keys_tensor, past_values_tensor, hidden_states


class Pi05TrtEdgeLanguageModel(nn.Module):
    """
    挂在 ``PaliGemmaForConditionalGeneration.set_decoder`` 上的语言塔代理。

    - 可选：在初始化时对底层 ``GemmaModel`` 调用 ``install_gemma_edge_attention_wrappers``。
    - ``forward`` / ``embed_tokens`` / ``rotary_emb`` 等与 OpenPI 一致：仍走 **原生 Gemma 图**（attention 为包装后的 native 前向）。

    Parameters
    ----------
    wrap_edge_attention:
        为 True 且在 ``take_decoder_ownership=True`` 或你希望就地改同一 Decoder 时，对 ``hf_gemma`` 安装
        :class:`GemmaAttentionTrtEdge`。为 False 时不改 attention，仅作透明代理。
    take_decoder_ownership:
        True 时用 ``add_module`` 接管 Decoder（配合 ``set_decoder(bundle)``）。
        False 时仅保存引用，不把 Decoder 从原 PaliGemma 上拆下。
    """

    def __init__(
        self,
        hf_gemma: nn.Module,
        *,
        take_decoder_ownership: bool = False,
        wrap_edge_attention: bool = True,
    ):
        super().__init__()
        if take_decoder_ownership:
            self.add_module("_hf_gemma", hf_gemma)
        else:
            object.__setattr__(self, "_hf_gemma", hf_gemma)

        self.config = hf_gemma.config
        self.embed_tokens = hf_gemma.embed_tokens
        self.rotary_emb = hf_gemma.rotary_emb
        self.norm = hf_gemma.norm
        self.layers = hf_gemma.layers
        self.padding_idx = getattr(hf_gemma, "padding_idx", None)
        self.vocab_size = getattr(hf_gemma, "vocab_size", None)

        gc = getattr(hf_gemma, "gradient_checkpointing", False)
        object.__setattr__(self, "gradient_checkpointing", gc)

        if wrap_edge_attention:
            install_gemma_edge_attention_wrappers(hf_gemma)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "gradient_checkpointing":
            object.__setattr__(self, name, value)
            if hasattr(self, "_hf_gemma"):
                if getattr(self._hf_gemma, "gradient_checkpointing", None) is not None:
                    self._hf_gemma.gradient_checkpointing = value
            return
        super().__setattr__(name, value)

    @property
    def device(self) -> torch.device:
        return next(self._hf_gemma.parameters()).device

    def get_input_embeddings(self):
        return self._hf_gemma.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self._hf_gemma.set_input_embeddings(value)

    def forward(self, *args: Any, **kwargs: Any):
        return self._hf_gemma(*args, **kwargs)

    def save_pretrained(self, *args: Any, **kwargs: Any):
        if hasattr(self._hf_gemma, "save_pretrained"):
            return self._hf_gemma.save_pretrained(*args, **kwargs)
        raise AttributeError("底层 GemmaModel 没有 save_pretrained")

    def edge_attention_modules(self) -> list[Any]:
        return list(iter_edge_attention_modules(self._hf_gemma))


def swap_paligemma_language_model_to_trt_edgellm(paligemma: nn.Module) -> Pi05TrtEdgeLanguageModel:
    hf = paligemma.get_decoder()
    bundle = Pi05TrtEdgeLanguageModel(
        hf, take_decoder_ownership=True, wrap_edge_attention=True
    )
    paligemma.set_decoder(bundle)
    return bundle


def swap_pi05_language_model_to_trt_edgellm(pi05_model: nn.Module) -> Pi05TrtEdgeLanguageModel:
    paligemma = pi05_model.paligemma_with_expert.paligemma
    return swap_paligemma_language_model_to_trt_edgellm(paligemma)


class LLMWithTrtEdgeLLM(nn.Module, Model):
    """
    π0.5 语言塔子模块：**独立实现**，不继承 :class:`LLM`。

    - **forward / val / quantize**：与 :class:`LLM` 一致，走 ``Pi05TrtEdgeLanguageModel`` → 原生 ``GemmaModel``，便于在 PI0 中共用 Decoder。
    - **export / export_onnx**：**不能** ``torch.onnx.export(self)``：``self.forward`` 仍走 ``GemmaAttentionTrtEdge.native``。
      实际追踪 :class:`GemmaModelEdgeOnnxExport`（见模块顶部说明与该类 docstring）。

    构造时会 **就地** 将 ``paligemma.get_decoder()`` 的各层 ``self_attn`` 换为 :class:`GemmaAttentionTrtEdge`。
    """

    def __init__(self, config, llm: Pi05TrtEdgeLanguageModel, **kwargs):
        nn.Module.__init__(self)
        Model.__init__(self, "pi05_llm_trt_edge", "")
        self.model = llm
        self.device = llm.device
        self.config = config
        self.model.config._attn_implementation = "eager"

    def _wrap_past_key_values(self, input_keys, input_values):
        k_v_cache = DynamicCache()
        num_layers = input_keys.shape[0]
        for i in range(num_layers):
            k_v_cache.update(input_keys[i : i + 1], input_values[i : i + 1], i)
        return k_v_cache

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

    def _nvfp4_post_processing(self, onnx_path, export_dir):
        with torch.inference_mode():
            self.model.save_pretrained(export_dir)
        if is_fp4_quantized(self):
            t1 = time.time()
            onnx.shape_inference.infer_shapes_path(onnx_path)
            onnx_model = onnx.load(onnx_path)
            print(
                colored(
                    "NVFP4 quantization detected in the model, compressing some weights to NVFP4",
                    "green",
                )
            )
            onnx_model = fp4qdq_to_2dq(onnx_model)
            print(
                colored(
                    "Removing all the files in the output directory except for .json files",
                    "green",
                )
            )
            for file in os.listdir(export_dir):
                if file.endswith(".json"):
                    continue
                os.remove(os.path.join(export_dir, file))
            onnx.save_model(
                onnx_model,
                onnx_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="onnx_model.data",
                convert_attribute=True,
            )
            t2 = time.time()
            print(
                colored(
                    f"NVFP4 quantization post processing cost:{t2 - t1}s",
                    "green",
                )
            )

    def val(self, val_data, batch_size, output_dir):
        val_datas = self.get_calibrate_dataset(val_data)

        def val_loop(model: torch.nn.Module, output_datas) -> None:
            print(f"Val model on {len(val_datas)} samples...")
            pbar = tqdm(val_datas, desc="Val", unit="num_samples")
            for data in pbar:
                if isinstance(data, dict):
                    data = {k: v.to(model.device) for k, v in data.items()}
                    outputs = model(**data)
                else:
                    data = data.to(model.device)
                    outputs = model(data)
                output_datas.append(
                    {
                        "past_keys": outputs[0].to(torch.float32).detach().cpu().numpy(),
                        "past_values": outputs[1].to(torch.float32).detach().cpu().numpy(),
                        "last_hidden_state": outputs[2]
                        .to(torch.float32)
                        .detach()
                        .cpu()
                        .numpy(),
                    }
                )

        if self.is_quantized:
            print(colored("Quantized model val", "green"))
            val_loop(self, self.val_datas_after)
            return Pi05Metric(self.val_datas_after)
        print(colored("Original model val", "green"))
        val_loop(self, self.val_datas_before)
        return Pi05Metric(self.val_datas_before)

    def quantize(self, quant_cfg, calib_data, export_dir):
        calib_dataloader = self.get_calibrate_dataset(calib_data)
        quantize_model(self, quant_cfg, calib_dataloader)
        self.is_quantized = True
        set_dynamic_quant(self, "fp16")
        self.export(export_dir, dynamo=False)
        if is_nvfp4_quantized(quant_cfg):
            print(colored("nvfp4 quantization detected, post processing...", "green"))
            self._nvfp4_post_processing(f"{export_dir}/llm.onnx", export_dir)

    @classmethod
    def construct_from_name_path(cls, model_name, model_path):
        from .model_pi05 import Pi05Model

        pi05_model = Pi05Model.construct_from_name_path(model_name, model_path)
        return cls.construct_model(pi05_model)

    @classmethod
    def construct_model(cls, pi05_model, dtype=torch.bfloat16):
        paligemma = pi05_model.paligemma_with_expert.paligemma
        hf = paligemma.get_decoder()
        bundle = Pi05TrtEdgeLanguageModel(
            hf,
            take_decoder_ownership=False,
            wrap_edge_attention=True,
        )
        llm_model = cls(
            pi05_model.paligemma_with_expert.paligemma.config.text_config,
            bundle,
        )
        llm_model.model.config._attn_implementation = "eager"
        return llm_model

    def export(self, export_dir, dynamo=True):
        self.eval().cuda()
        output_dir = export_dir
        os.makedirs(output_dir, exist_ok=True)
        start = time.time()
        logger.info("Start export onnx (EdgeLLMAttention plugin) ...")
        print(colored("Start LLMWithTrtEdgeLLM export onnx (trt::attention_plugin)...", "green"))

        _ensure_trt_edgellm_on_path()
        from tensorrt_edgellm.llm_models.layers.attention_plugin import (
            register_attention_plugin_onnx_symbolic_functions,
        )

        register_attention_plugin_onnx_symbolic_functions()

        # 不用 torch.onnx.export(self)：self.forward 仍走 GemmaModel + native 注意力。
        # 必须用 GemmaModelEdgeOnnxExport 显式走 layer.self_attn.edge.forward，图中才有 trt::attention_plugin。
        gemma = self.model._hf_gemma
        export_net = GemmaModelEdgeOnnxExport(gemma).eval().cuda()

        inputs_embeds = torch.randn(
            (1, 968, 2048), dtype=torch.bfloat16, device="cuda"
        )
        attention_mask = torch.randn((1, 1, 968, 968), dtype=torch.float32, device="cuda")
        position_ids = torch.randint(1, 1000, (1, 968), dtype=torch.int64, device="cuda")

        onnx_path = f"{output_dir}/llm.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                export_net,
                (inputs_embeds, attention_mask, position_ids),
                onnx_path,
                export_params=True,
                input_names=["inputs_embeds", "attention_mask", "position_ids"],
                output_names=["past_keys", "past_values", "last_hidden_state"],
                opset_version=19,
                dynamo=dynamo,
                do_constant_folding=True,
                dynamic_axes={
                    "inputs_embeds": {0: "batch_size", 1: "seq_len"},
                    "attention_mask": {
                        0: "batch_size",
                        2: "seq_len",
                        3: "seq_len",
                    },
                    "position_ids": {0: "batch_size", 1: "seq_len"},
                    "past_keys": {2: "seq_len"},
                    "past_values": {2: "seq_len"},
                    "last_hidden_state": {0: "batch_size", 1: "seq_len"},
                },
            )

        end = time.time()
        logger.info(f"export onnx to {output_dir} done cost:{end - start}s")
        print(
            colored(
                f"LLMWithTrtEdgeLLM export done to {output_dir} cost:{end - start}s",
                "green",
            )
        )
        return self

    @classmethod
    def export_onnx(cls, pi_model, export_dir):
        del pi_model.paligemma_with_expert.gemma_expert
        llm_model = cls.construct_model(pi_model, dtype=torch.float16)
        llm_model.eval().cuda()
        output_dir = export_dir
        os.makedirs(output_dir, exist_ok=True)
        start = time.time()
        logger.info("Start export onnx (Edge attention)...")

        _ensure_trt_edgellm_on_path()
        from tensorrt_edgellm.llm_models.layers.attention_plugin import (
            register_attention_plugin_onnx_symbolic_functions,
        )

        register_attention_plugin_onnx_symbolic_functions()

        # 同 instance export：必须 trace GemmaModelEdgeOnnxExport，见类文档与 export() 内注释。
        gemma = llm_model.model._hf_gemma
        export_net = GemmaModelEdgeOnnxExport(gemma).eval().cuda()

        inputs_embeds = torch.randn((1, 968, 2048), dtype=torch.float16, device="cuda")
        attention_mask = torch.randn((1, 1, 968, 968), dtype=torch.float32, device="cuda")
        position_ids = torch.randint(1, 1000, (1, 968), dtype=torch.int64, device="cuda")

        with torch.inference_mode():
            torch.onnx.export(
                export_net,
                (inputs_embeds, attention_mask, position_ids),
                f"{output_dir}/llm.onnx",
                input_names=["inputs_embeds", "attention_mask", "position_ids"],
                output_names=["past_keys", "past_values", "last_hidden_state"],
                opset_version=19,
                dynamo=False,
                do_constant_folding=True,
                dynamic_axes={
                    "inputs_embeds": {0: "batch_size", 1: "seq_len"},
                    "attention_mask": {
                        0: "batch_size",
                        2: "seq_len",
                        3: "seq_len",
                    },
                    "position_ids": {0: "batch_size", 1: "seq_len"},
                    "last_hidden_state": {0: "batch_size", 1: "seq_len"},
                },
            )
        end = time.time()
        logger.info(f"export onnx to {output_dir} done cost:{end - start}s")
        return llm_model

    def gemma_decoder(self) -> nn.Module:
        """已安装 Edge attention 包装的 ``GemmaModel``（与 PI0 共用引用）。"""
        return self.model._hf_gemma

    def edge_onnx_export_module(self) -> GemmaModelEdgeOnnxExport:
        """用于自定义 ONNX 的子模块（前向走 ``EdgeLLMAttention.forward``）。"""
        return GemmaModelEdgeOnnxExport(self.model._hf_gemma)

    def edge_attention_modules(self) -> list[Any]:
        if not isinstance(self.model, Pi05TrtEdgeLanguageModel):
            raise TypeError("self.model 不是 Pi05TrtEdgeLanguageModel")
        return self.model.edge_attention_modules()

    def get_edgellm_core(self) -> nn.Module:
        """兼容旧名：等价于 :meth:`gemma_decoder`。"""
        return self.gemma_decoder()
