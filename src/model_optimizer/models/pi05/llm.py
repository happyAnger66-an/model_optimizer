import os
import time
import torch
import onnx
from typing import Optional

from ..model import Model
from ..token import get_tokenizer
from tqdm import tqdm
import logging

from transformers.cache_utils import DynamicCache

from termcolor import colored

from model_optimizer.quantization.quantization_utils import quantize_model
from modelopt.onnx.quantization.qdq_utils import fp4qdq_to_2dq
from model_optimizer.utils.utils import is_fp4_quantized, set_dynamic_quant, is_nvfp4_quantized
from model_optimizer.evaluate.metrics.pi05 import Pi05Metric

logger = logging.getLogger(__name__)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把 KV heads 扩展到 Q heads 数量（GQA/MQA）。"""
    if n_rep == 1:
        return hidden_states
    bsz, num_kv_heads, slen, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(bsz, num_kv_heads, n_rep, slen, head_dim)
        .reshape(bsz, num_kv_heads * n_rep, slen, head_dim)
    )


def _gated_residual(x: torch.Tensor, y: torch.Tensor, gate: Optional[torch.Tensor]) -> torch.Tensor:
    if gate is None:
        return x + y
    return x + y * gate


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # cos/sin: [B, S, D] -> broadcast to [B, 1, S, D]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class GemmaModelNativeOnnxExport(torch.nn.Module):
    """仅用于 ONNX 导出：以“纯张量数据流”显式导出每层 KV cache，避免依赖 DynamicCache.update 的副作用。

    目标输出对齐 TensorRT-Edge-LLM 的 per-layer 形式：
    - last_hidden_state: (batch, seq_len, hidden_size)
    - present_key_values.{i}: (batch, 2, num_kv_heads, seq_len, head_dim)
    """

    def __init__(self, gemma_model: torch.nn.Module):
        super().__init__()
        self.gemma = gemma_model

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        gemma = self.gemma
        hidden_states = inputs_embeds

        # 与 OpenPI GemmaModel.forward 的 dtype 行为保持一致：若第一层权重是 bf16，则激活转 bf16
        if len(gemma.layers) > 0 and gemma.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)

        # 共享 RoPE embedding（cos/sin）
        cos, sin = gemma.rotary_emb(hidden_states, position_ids)

        present_key_values: list[torch.Tensor] = []

        for layer in gemma.layers[: gemma.config.num_hidden_layers]:
            residual = hidden_states
            hidden_states, gate = layer.input_layernorm(hidden_states, None)

            attn = layer.self_attn
            input_shape = hidden_states.shape[:-1]
            head_dim = attn.head_dim
            hidden_shape = (*input_shape, -1, head_dim)

            # [B, heads, S, D]
            query_states = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            # rotary
            query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # per-layer 输出：打包成 [B, 2, Hkv, S, D]，与 EdgeLLM 的 present_key_values.{i} 风格一致
            present_key_values.append(torch.stack([key_states, value_states], dim=1))

            # attention (eager)
            key_states_rep = _repeat_kv(key_states, attn.num_key_value_groups)
            value_states_rep = _repeat_kv(value_states, attn.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states_rep.transpose(2, 3)) * attn.scaling
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, : key_states_rep.shape[-2]]
                attn_weights = attn_weights + causal_mask

            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states_rep)  # [B, q_heads, S, D]
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
            attn_output = attn.o_proj(attn_output)

            hidden_states = _gated_residual(residual, attn_output, gate)

            residual = hidden_states
            hidden_states, gate = layer.post_attention_layernorm(hidden_states, None)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = _gated_residual(residual, hidden_states, gate)

        hidden_states, _ = gemma.norm(hidden_states, None)

        # 输出：last_hidden_state + per-layer present_key_values.{i}
        # 为避免导出路径 dtype 混乱，这里保持与原导出一致：KV 输出为 bf16（由上游控制/可再调整）
        present_key_values = [pkv.to(torch.bfloat16) for pkv in present_key_values]
        return (hidden_states,) + tuple(present_key_values)


class LLM(torch.nn.Module, Model):
    def __init__(self, config, llm, **kwargs):
        super().__init__(**kwargs)
        self.model = llm
        self.device = llm.device
        self.config = config
        self.model.config._attn_implementation = "eager"

    def _wrap_past_key_values(self, input_keys, input_values):
        k_v_cache = DynamicCache()
        num_layers = input_keys.shape[0]
        for i in range(num_layers):
            k_v_cache.update(input_keys[i:i+1], input_values[i:i+1], i)

        return k_v_cache

    def forward(self, inputs_embeds, attention_mask, position_ids):
        prefix_output = self.model(inputs_embeds=inputs_embeds,
                                   attention_mask=attention_mask,
                                   position_ids=position_ids,
                                   use_cache=True)
        past_key_caches = prefix_output.past_key_values

        # ONNX 导出时，DynamicCache 的 __getitem__/循环索引容易被 Dynamo 特化成只捕获第 0 层，
        # 导致最终 past_keys/past_values 只包含第一层。这里避免逐层索引，改为直接拼接底层缓存列表。
        if isinstance(past_key_caches, DynamicCache):
            # transformers.cache_utils.DynamicCache: key_cache/value_cache 为按层排列的 list[Tensor]
            past_keys_tensor = torch.cat(list(past_key_caches.key_cache), dim=0).to(torch.bfloat16)
            past_values_tensor = torch.cat(list(past_key_caches.value_cache), dim=0).to(torch.bfloat16)
        else:
            # 兼容 legacy cache：tuple[num_layers] of (k, v)
            past_keys_tensor = torch.cat([kv[0] for kv in past_key_caches], dim=0).to(torch.bfloat16)
            past_values_tensor = torch.cat([kv[1] for kv in past_key_caches], dim=0).to(torch.bfloat16)

        return past_keys_tensor, past_values_tensor, prefix_output.last_hidden_state

    def _nvfp4_post_processing(self, onnx_path, export_dir):
        with torch.inference_mode():
            self.model.save_pretrained(export_dir)

#        onnx_path = f"{export_dir}/llm.onnx"
        if is_fp4_quantized(self):
            t1 = time.time()
            onnx.shape_inference.infer_shapes_path(onnx_path)
            onnx_model = onnx.load(onnx_path)
            graph = None

            print(
                colored(
                    "NVFP4 quantization detected in the model, \
                        compressing some weights to NVFP4", "green")
            )
            onnx_model = fp4qdq_to_2dq(onnx_model)
            print(
                colored(
                    "Removing all the files in the output directory except for .json files",
                    "green"
                )
            )
            for file in os.listdir(export_dir):
                if file.endswith(".json"):
                    continue
                os.remove(os.path.join(export_dir, file))
            onnx.save_model(onnx_model,
                            onnx_path,
                            save_as_external_data=True,
                            all_tensors_to_one_file=True,
                            location="onnx_model.data",
                            convert_attribute=True)
            t2 = time.time()
            print(
                colored(
                    f"NVFP4 quantization post processing cost:{t2 - t1}s", "green"
                )
            )

    def val(self, val_data, batch_size, output_dir):
        val_datas = self.get_calibrate_dataset(val_data)

        def val_loop(model: torch.nn.Module, output_datas) -> None:
            """
            Val loop that val the model.

            Args:
                model: Model to val
            """
            # Create progress bar for val
            print(f"Val model on {len(val_datas)} samples...")
            pbar = tqdm(val_datas, desc="Val", unit="num_samples")

            for data in pbar:
                if isinstance(data, dict):
                    data = {k: v.to(model.device) for k, v in data.items()}
                    outputs = model(**data)
                else:
                    data = data.to(model.device)
                    outputs = model(data)
                output_datas.append({"past_keys": outputs[0].to(torch.float32).detach().cpu().numpy(),
                                     "past_values": outputs[1].to(torch.float32).detach().cpu().numpy(),
                                     "last_hidden_state": outputs[2].to(torch.float32).detach().cpu().numpy()})

        if self.is_quantized:
            print(colored("Quantized model val", "green"))
            val_loop(self, self.val_datas_after)
            return Pi05Metric(self.val_datas_after)
        else:
            print(colored("Original model val", "green"))
            val_loop(self, self.val_datas_before)
            return Pi05Metric(self.val_datas_before)

    def quantize(self, quant_cfg, calib_data, export_dir):
        # tokenizer = get_tokenizer(model_dir)
        calib_dataloader = self.get_calibrate_dataset(calib_data)
        quantize_model(self, quant_cfg, calib_dataloader)
        self.is_quantized = True
        set_dynamic_quant(self, "fp16")

        self.export(export_dir, dynamo=False)
        onnx_path = f"{export_dir}/llm.onnx"
        if is_nvfp4_quantized(quant_cfg):
            print(colored("nvfp4 quantization detected, post processing...", "green"))
            self._nvfp4_post_processing(onnx_path, export_dir)

    @classmethod
    def construct_from_name_path(cls, model_name, model_path):
        from .model_pi05 import Pi05Model
        pi05_model = Pi05Model.construct_from_name_path(model_name, model_path)
        return cls.construct_model(pi05_model)

    @classmethod
    def construct_model(cls, pi05_model, dtype=torch.bfloat16):
        paligemma = pi05_model.paligemma_with_expert.paligemma
        llm_model = cls(pi05_model.paligemma_with_expert.paligemma.config.text_config,
                        paligemma.get_decoder())
        return llm_model

    def export(self, export_dir, dynamo=True):
        self.eval().cuda()

        output_dir = export_dir
        os.makedirs(output_dir, exist_ok=True)
        start = time.time()
        logger.info("Start export onnx ...")
        print(colored(f"Start LLM export onnx...", "green"))
        inputs_embeds = torch.randn((1, 968, 2048),
                                    dtype=torch.bfloat16,
                                    device="cuda",
                                    )
        attention_mask = torch.randn((1, 1, 968, 968),
                                     dtype=torch.float32,
                                     device="cuda",
                                     )
        position_ids = torch.randint(1, 1000, (1, 968),
                                     dtype=torch.int64,
                                     device="cuda",
                                     )
    # am: torch.Size([1, 1, 968, 968]) - torch.float32, pi: torch.Size([1, 968])-torch.int64 prefix: torch.Size([1, 968, 2048])-torch.bfloat16
        with torch.inference_mode():
            export_net = GemmaModelNativeOnnxExport(self.model).eval().cuda()
            num_layers = int(self.config.num_hidden_layers)
            output_names = ["last_hidden_state"] + [f"present_key_values.{i}" for i in range(num_layers)]
            dynamic_axes = {
                "inputs_embeds": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 2: "seq_len", 3: "seq_len"},
                "position_ids": {0: "batch_size", 1: "seq_len"},
                "last_hidden_state": {0: "batch_size", 1: "seq_len"},
            }
            for i in range(num_layers):
                # [B, 2, Hkv, S, D]
                dynamic_axes[f"present_key_values.{i}"] = {0: "batch_size", 3: "seq_len"}
            torch.onnx.export(
                export_net,
                # Include position_ids in ONNX export
                (inputs_embeds, attention_mask, position_ids),
                f"{output_dir}/llm.onnx",
                export_params=True,
                input_names=["inputs_embeds", "attention_mask",
                             "position_ids"],  # Add position_ids to input names
                output_names=output_names,
                opset_version=19,
                dynamo=dynamo,
                do_constant_folding=True,
                dynamic_axes=dynamic_axes,
            )
        end = time.time()
        logger.info(f"export onnx to {output_dir} done cost:{end - start}s")
        print(
            colored(f"LLM export onnx done to {output_dir} cost:{end - start}s", "green"))
        return self

    @classmethod
    def export_onnx(cls, pi_model, export_dir):
        del pi_model.paligemma_with_expert.gemma_expert

        llm_model = cls.construct_model(pi_model, dtype=torch.float16)
        llm_model.eval().cuda()

        output_dir = export_dir
        os.makedirs(output_dir, exist_ok=True)
        start = time.time()
        logger.info("Start export onnx ...")
        inputs_embeds = torch.randn((1, 968, 2048),
                                    dtype=torch.float16,
                                    device="cuda",
                                    )
        attention_mask = torch.randn((1, 1, 968, 968),
                                     dtype=torch.float32,
                                     device="cuda",
                                     )
        position_ids = torch.randint(1, 1000, (1, 968),
                                     dtype=torch.int64,
                                     device="cuda",
                                     )
    # am: torch.Size([1, 1, 968, 968]) - torch.float32, pi: torch.Size([1, 968])-torch.int64 prefix: torch.Size([1, 968, 2048])-torch.bfloat16
        with torch.inference_mode():
            torch.onnx.export(
                llm_model,
                # Include position_ids in ONNX export
                (inputs_embeds, attention_mask, position_ids),
                f"{output_dir}/llm.onnx",
                input_names=["inputs_embeds", "attention_mask",
                             "position_ids"],  # Add position_ids to input names
                output_names=["last_hidden_state"],
                opset_version=19,
                dynamo=False,
                do_constant_folding=True,
                dynamic_axes={
                    "inputs_embeds": {0: "batch_size", 1: "seq_len"},
                    "attention_mask": {0: "batch_size", 2: "seq_len", 3: "seq_len"},
                    "position_ids": {0: "batch_size", 1: "seq_len"},
                },
            )
        end = time.time()
        logger.info(f"export onnx to {output_dir} done cost:{end - start}s")
        return llm_model
