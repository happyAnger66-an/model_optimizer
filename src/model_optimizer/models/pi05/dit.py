"""π₀.₅ 单次 flow / denoise 步：对齐 openpi PI0Pytorch.denoise_step + embed_suffix（pi05 分支）。"""

from __future__ import annotations

import logging
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored
from transformers.cache_utils import DynamicCache

from ..model import Model
from model_optimizer.calibrate.pi05_calib_load import open_pi05_calib_for_quantize
from model_optimizer.quantization.quantization_utils import quantize_model
from model_optimizer.utils.utils import is_nvfp4_quantized, set_dynamic_quant

logger = logging.getLogger(__name__)

# 与 openpi pi0_pytorch 中 _prepare_attention_masks_4d 一致
_ATTN_MASK_FILL_VALUE = -2.3819763e38


def _get_safe_dtype(target_dtype: torch.dtype, device_type: str) -> torch.dtype:
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """标量时间步 → sin-cos 向量（与 openpi pi0_pytorch 一致）。time 形状为 (batch_size,)。"""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dev = device if device is not None else time.device
    dtype = _get_safe_dtype(torch.float64, dev.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=dev)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(dtype)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """与 openpi pi0_pytorch.make_att_2d_masks 一致。"""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class Pi05DenoiseStep(nn.Module, Model):
    """
    单次去噪步：给定 prefix KV 缓存与当前噪声动作 x_t、时间 t，预测速度场 v_t。

    对应 openpi 中 ``denoise_step``（suffix 仅含 action tokens，pi05 不使用 state token）。
    Euler 更新 ``x_t += dt * v_t`` 应在图外由宿主循环调用。
    """

    def __init__(
        self,
        gemma_expert: nn.Module,
        expert_config,
        action_in_proj: nn.Linear,
        time_mlp_in: nn.Linear,
        time_mlp_out: nn.Linear,
        action_out_proj: nn.Linear,
        *,
        action_horizon: int,
        action_dim: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.gemma_expert = gemma_expert
        self.expert_config = expert_config
        self.action_in_proj = action_in_proj
        self.time_mlp_in = time_mlp_in
        self.time_mlp_out = time_mlp_out
        self.action_out_proj = action_out_proj
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.device = gemma_expert.device

        self.gemma_expert.config._attn_implementation = "eager"  # noqa: SLF001

        suffix_ar = [1] + [0] * (action_horizon - 1)
        self.register_buffer(
            "_suffix_ar_mask",
            torch.tensor(suffix_ar, dtype=torch.int32),
            persistent=False,
        )

    @property
    def model(self):
        """与 Expert 一致，供 Model 基类 NVFP4 等路径使用。"""
        return self.gemma_expert

    def get_calibrate_dataset(self, calib_data):
        # 与 LLM/Vit/Expert 一致：支持 manifest+shards（低内存）与旧 merged .pt
        return open_pi05_calib_for_quantize(calib_data, component="pi05_denoise")

    def val(self, val_data, batch_size, output_dir):
        raise NotImplementedError(
            "Pi05DenoiseStep.val 未实现：需提供 denoise 步校准/对比数据与指标。"
        )

    def export(self, export_dir, export_dtype=torch.bfloat16, dynamo=False, mode=None):
        """导出单次 denoise 步 ONNX，输入输出与 ``forward`` 一致。

        ``past_keys`` / ``past_values`` 与 LLM 导出一致：``torch.cat(..., dim=0)`` 后为
        **4D** ``[num_layers, batch, prefix_seq_len, head_dim]``（head_dim 固定，勿标成动态轴）。

        默认 ``dynamo=False`` 使用传统导出器 + ``dynamic_axes``；若 ``dynamo=True``，须使用
        ``dynamic_shapes``（不再与 ``dynamic_axes`` 混用），避免约束冲突。
        """
        self.eval().cuda()

        output_dir = export_dir
        os.makedirs(output_dir, exist_ok=True)
        start = time.time()
        num_layers = int(self.expert_config.num_hidden_layers)
        prefix_len = 968

        logger.info("Start export denoise onnx ...")
        print(colored("Start Pi05 denoise (Pi05DenoiseStep) export onnx...", "green"))

        prefix_pad_masks = torch.ones(
            (1, prefix_len), dtype=torch.bool, device="cuda"
        )
        past_keys = []
        past_values = []
        for _ in range(num_layers):
            past_keys.append(
                torch.randn((1, 1, prefix_len, 256), dtype=export_dtype, device="cuda")
            )
            past_values.append(
                torch.randn((1, 1, prefix_len, 256), dtype=export_dtype, device="cuda")
            )
        past_keys_tensor = torch.cat(past_keys, dim=0)
        past_values_tensor = torch.cat(past_values, dim=0)

        x_t = torch.randn(
            (1, self.action_horizon, self.action_dim),
            dtype=torch.float32,
            device="cuda",
        )
        timestep = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        output_path = f"{output_dir}/denoise.onnx"
        export_kw: dict = {}
        if dynamo:
            from torch.export import Dim

            batch_dim = Dim("batch", min=1, max=4096)
            prefix_seq_dim = Dim("prefix_seq", min=1, max=4096)
            export_kw["dynamic_shapes"] = {
                "prefix_pad_masks": {0: batch_dim, 1: prefix_seq_dim},
                "past_keys": {1: batch_dim, 2: prefix_seq_dim},
                "past_values": {1: batch_dim, 2: prefix_seq_dim},
                "x_t": {0: batch_dim},
                "timestep": {0: batch_dim},
            }
        else:
            export_kw["dynamic_axes"] = {
                "prefix_pad_masks": {0: "batch_size", 1: "prefix_seq_len"},
                "past_keys": {1: "batch_size", 2: "prefix_seq_len"},
                "past_values": {1: "batch_size", 2: "prefix_seq_len"},
                "x_t": {0: "batch_size"},
                "timestep": {0: "batch_size"},
                "v_t": {0: "batch_size"},
            }

        with torch.inference_mode():
            torch.onnx.export(
                self,
                (prefix_pad_masks, past_keys_tensor, past_values_tensor, x_t, timestep),
                output_path,
                export_params=True,
                input_names=[
                    "prefix_pad_masks",
                    "past_keys",
                    "past_values",
                    "x_t",
                    "timestep",
                ],
                output_names=["v_t"],
                opset_version=19,
                dynamo=dynamo,
                do_constant_folding=True,
                **export_kw,
            )
        end = time.time()
        logger.info("export onnx to %s done cost:%ss", output_dir, end - start)
        print(
            colored(
                f"Pi05 denoise export onnx done to {output_path} cost:{end - start}s",
                "green",
            )
        )
        return self

    def quantize(self, quant_cfg, calib_data, export_dir):
        calib_dataloader = self.get_calibrate_dataset(calib_data)
        quantize_model(self, quant_cfg, calib_dataloader)
        self.is_quantized = True
        set_dynamic_quant(self, "bf16")

        self.export(export_dir, dynamo=False)
        onnx_path = f"{export_dir}/denoise.onnx"
        if is_nvfp4_quantized(quant_cfg):
            print(colored("nvfp4 quantization detected, post processing...", "green"))
            self._nvfp4_post_processing(onnx_path, export_dir)

    def _wrap_past_key_values(
        self, past_keys: torch.Tensor, past_values: torch.Tensor
    ) -> DynamicCache:
        k_v_cache = DynamicCache()
        num_layers = past_keys.shape[0]
        for i in range(num_layers):
            k_v_cache.update(past_keys[i : i + 1], past_values[i : i + 1], i)
        return k_v_cache

    @staticmethod
    def _prepare_attention_masks_4d(att_2d_masks: torch.Tensor) -> torch.Tensor:
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, _ATTN_MASK_FILL_VALUE)

    def _embed_suffix_pi05(
        self, noisy_actions: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """pi05：时间 adaRMS 条件 + action_in_proj，与 openpi embed_suffix（pi05 分支）一致。"""
        if timestep.ndim != 1:
            raise ValueError(f"timestep must be 1D (batch,), got shape {tuple(timestep.shape)}")
        bsize = noisy_actions.shape[0]
        device = noisy_actions.device

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=device,
        )
        time_emb = time_emb.to(dtype=timestep.dtype)

        action_emb = self.action_in_proj(noisy_actions)

        x = self.time_mlp_in(time_emb)
        x = F.silu(x)
        x = self.time_mlp_out(x)
        adarms_cond = F.silu(x)

        action_time_emb = action_emb
        action_time_dim = action_time_emb.shape[1]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=device
        )
        pad_masks = action_time_mask

        att_base = self._suffix_ar_mask.to(device=device).expand(bsize, -1)
        return action_time_emb, pad_masks, att_base, adarms_cond

    def forward(
        self,
        prefix_pad_masks: torch.Tensor,
        past_keys: torch.Tensor,
        past_values: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prefix_pad_masks: bool/float [batch, prefix_len]，与 LLM prefix 一致。
            past_keys: [num_layers, batch, num_kv_heads, prefix_len, head_dim]（与 LLM 导出堆叠方式一致）。
            past_values: 与 past_keys 相同布局。
            x_t: float [batch, action_horizon, action_dim]。
            timestep: float [batch]，与 openpi sample 中 expanded_time 一致。

        Returns:
            v_t: float32 [batch, action_horizon, action_dim]。
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self._embed_suffix_pi05(
            x_t, timestep
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat(
            [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
        )

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1, dtype=torch.int64)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1, dtype=torch.int64) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        expert_dtype = self.gemma_expert.layers[0].self_attn.q_proj.weight.dtype
        suffix_embs = suffix_embs.to(dtype=expert_dtype)

        past = self._wrap_past_key_values(past_keys, past_values)
        outputs = self.gemma_expert(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past,
            inputs_embeds=suffix_embs,
            use_cache=False,
            adarms_cond=adarms_cond,
        )
        suffix_out = outputs.last_hidden_state
        suffix_out = suffix_out[:, -self.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    @classmethod
    def construct_model(cls, pi05_model, dtype: torch.dtype | None = None):
        """
        从 Pi05Model 包装器或已加载的 PI0Pytorch（policy._model）构建，共享子模块权重引用。
        """
        if not getattr(pi05_model.config, "pi05", False):
            raise ValueError("Pi05DenoiseStep 仅支持 config.pi05 is True 的 PI0Pytorch 模型")

        gemma_expert = pi05_model.paligemma_with_expert.gemma_expert.model
        expert_config = pi05_model.paligemma_with_expert.gemma_expert.config

        if dtype is not None:
            logger.debug("construct_model dtype=%s ignored (weights keep loaded dtype)", dtype)

        return cls(
            gemma_expert=gemma_expert,
            expert_config=expert_config,
            action_in_proj=pi05_model.action_in_proj,
            time_mlp_in=pi05_model.time_mlp_in,
            time_mlp_out=pi05_model.time_mlp_out,
            action_out_proj=pi05_model.action_out_proj,
            action_horizon=pi05_model.config.action_horizon,
            action_dim=pi05_model.config.action_dim,
        )

    @classmethod
    def construct_from_name_path(cls, model_name: str, model_path: str):
        from .model_pi05 import Pi05Model

        wrapper = Pi05Model.construct_from_name_path(model_name, model_path)
        return cls.construct_model(wrapper, dtype=torch.bfloat16)
