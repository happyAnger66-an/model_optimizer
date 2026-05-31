"""π₀.₅ 单次 flow / denoise 步：对齐 openpi PI0Pytorch.denoise_step + embed_suffix（pi05 分支）。"""

from __future__ import annotations

import logging
import math
import os
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import colored
from transformers.cache_utils import DynamicCache

from ..model import Model
from model_optimizer.calibrate.pi05_calib_load import open_pi05_calib_for_quantize
from model_optimizer.config.feature_config import FeatureConfig
from model_optimizer.utils.utils import is_nvfp4_quantized, set_dynamic_quant

from ..features import FeatureContext, apply_features, register_feature
from .denoise_onnx_post_export import apply_denoise_onnx_post_export_patches

logger = logging.getLogger(__name__)

# 与 registry.py 注册名一致；用于 FeatureContext.model_name 过滤。
MODEL_NAME = "pi05_libero/denoise"


# 与 openpi pi0_pytorch 中 _prepare_attention_masks_4d 一致
_ATTN_MASK_FILL_VALUE = -2.3819763e38


def _adarms_injected_forward(self, x, cond=None):  # noqa: ARG001
    """``GemmaRMSNorm.forward`` 的"注入版"：消费预计算 modulation，不调用 ``self.dense``。

    供 :class:`Pi05DenoiseStep` 在 AdaRMS Dense 预计算模式下，对 expert 内各 AdaRMS 实例做
    运行时替换（不改原始 ``modeling_gemma.py``）。数学与原 ``forward`` 完全一致，区别仅在于
    ``scale/shift/gate`` 取自外部注入的 ``self._injected_mod`` 而非 ``self.dense(cond)``。
    详见 docs/optimizer/ddup/adarms_pre_compute.md。
    """
    dtype = x.dtype
    normed_inputs = self._norm(x)
    mod = self._injected_mod  # [batch, dim*3]，由宿主在 forward 前按 norm 顺序注入
    if x.dim() == 3:  # [batch, seq, features] → 在 seq 维广播
        mod = mod.unsqueeze(1)
    scale, shift, gate = torch.chunk(mod, 3, dim=-1)
    normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(torch.float32)
    return normed_inputs.to(dtype), gate.to(dtype)


# ── 特性注册：AdaRMS Dense 预计算（roadmap #22） ──────────────────────────────
# 作用对象为 Pi05DenoiseStep 本身（调用其 enable_adarms_precompute）。仅适用 denoise 组件。
def _apply_adarms_precompute(target, params: dict, ctx: FeatureContext) -> None:  # noqa: ARG001
    target.enable_adarms_precompute(True)


register_feature(
    "adarms_dense_precompute",
    default_enabled=False,
    apply_fn=_apply_adarms_precompute,
    description="denoise modulation 只依赖固定步调度 → 预算常量注入，把 dense GEMM 移出导出图。",
    supported_models=(MODEL_NAME,),
)


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
        adarms_precompute: bool = False,
        feature_config: FeatureConfig | None = None,
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

        # AdaRMS Dense 预计算（roadmap #22）：modulation 只依赖固定步调度，可离线预算并注入，
        # 把 expert 内的 dense GEMM 移出导出图。默认关闭，开启时不改原始 modeling_gemma.py。
        self.adarms_precompute = False
        self._adarms_patched = False
        self._adarms_norms_cache: list[nn.Module] = []
        # 供 export()/quantize() 读取 export/quantize 级覆盖。
        self.feature_config = feature_config or FeatureConfig.empty()
        if feature_config is not None:
            # 配置驱动：由特性注册表（adarms_dense_precompute / ...）按 JSON > env > 默认 决定启停。
            ctx = FeatureContext(model_name=MODEL_NAME, dtype=getattr(gemma_expert, "dtype", None))
            apply_features(self, feature_config, ctx)
        elif adarms_precompute:
            # 直接构造（非配置）路径的向后兼容入口。
            self.enable_adarms_precompute(True)

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
        # prefix_len 仅用于生成导出假输入（prefix 维是动态轴，不会约束引擎）；
        # 可经 feature_config.export.prefix_len 覆盖，默认 968。
        prefix_len = int(self.feature_config.export.get("prefix_len", 968))

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

        # AdaRMS 预计算模式：第 5 个输入由 ``timestep`` 改为打包 modulation ``adarms_mod``
        # [num_norms, batch, dim*3]，图内不再含 sinusoid/time_mlp/dense。
        if self.adarms_precompute:
            norms = self._adarms_norm_modules()
            dim3 = int(norms[0].dense.out_features)
            fifth_input = torch.randn(
                (len(norms), 1, dim3), dtype=torch.float32, device="cuda"
            )
            fifth_name = "adarms_mod"
            print(
                colored(
                    f"[adarms] precompute export: {len(norms)} norms × dim*3={dim3}, "
                    "dense GEMM 已移出图",
                    "green",
                )
            )
        else:
            fifth_input = torch.tensor([1.0], dtype=torch.float32, device="cuda")
            fifth_name = "timestep"

        output_path = f"{output_dir}/denoise.onnx"
        export_kw: dict = {}
        if dynamo:
            from torch.export import Dim

            batch_dim = Dim("batch", min=1, max=4096)
            prefix_seq_dim = Dim("prefix_seq", min=1, max=4096)
            fifth_shape = {1: batch_dim} if self.adarms_precompute else {0: batch_dim}
            export_kw["dynamic_shapes"] = {
                "prefix_pad_masks": {0: batch_dim, 1: prefix_seq_dim},
                "past_keys": {1: batch_dim, 2: prefix_seq_dim},
                "past_values": {1: batch_dim, 2: prefix_seq_dim},
                "x_t": {0: batch_dim},
                fifth_name: fifth_shape,
            }
        else:
            fifth_axes = {1: "batch_size"} if self.adarms_precompute else {0: "batch_size"}
            export_kw["dynamic_axes"] = {
                "prefix_pad_masks": {0: "batch_size", 1: "prefix_seq_len"},
                "past_keys": {1: "batch_size", 2: "prefix_seq_len"},
                "past_values": {1: "batch_size", 2: "prefix_seq_len"},
                "x_t": {0: "batch_size"},
                fifth_name: fifth_axes,
                "v_t": {0: "batch_size"},
            }

        with torch.inference_mode():
            torch.onnx.export(
                self,
                (prefix_pad_masks, past_keys_tensor, past_values_tensor, x_t, fifth_input),
                output_path,
                export_params=True,
                input_names=[
                    "prefix_pad_masks",
                    "past_keys",
                    "past_values",
                    "x_t",
                    fifth_name,
                ],
                output_names=["v_t"],
                opset_version=19,
                dynamo=dynamo,
                do_constant_folding=True,
                **export_kw,
            )
        apply_denoise_onnx_post_export_patches(output_path)
        end = time.time()
        logger.info("export onnx to %s done cost:%ss", output_dir, end - start)
        print(
            colored(
                f"Pi05 denoise export onnx done to {output_path} cost:{end - start}s",
                "green",
            )
        )
        return self

    def quantize(self, quant_cfg, calib_data, export_dir, *, measure_quant_error: bool = False):
        calib_dataloader = self.get_calibrate_dataset(calib_data)
        # FP8_KV_CFG / *_bmm_quantizer：ModelOpt 要求 ``mtq.quantize`` 的根模块为 HF PreTrainedModel，
        # 与 LLM.quantize(self.model, ...) 一致；标定仍走完整 ``forward`` 以覆盖 action/time 投影与 expert。
        from model_optimizer.quantization.quantization_utils import quantize_model  # noqa: F401
        from model_optimizer.quantization.quantization_utils import quant_config_targets_hf_bmm_kv  # noqa: F401

        # 标定数据提供 ``timestep`` 而非 ``adarms_mod``，因此校准期必须走默认时间链路；
        # AdaRMS 预计算仅在导出时启用（导出图改用 ``adarms_mod`` 输入、移除 dense GEMM）。
        want_precompute = self.adarms_precompute
        if want_precompute:
            self.enable_adarms_precompute(False)

        if quant_config_targets_hf_bmm_kv(quant_cfg):
            quantize_model(
                self.gemma_expert,
                quant_cfg,
                calib_dataloader,
                forward_context=self,
                measure_quant_error=measure_quant_error,
            )
        else:
            quantize_model(
                self,
                quant_cfg,
                calib_dataloader,
                measure_quant_error=measure_quant_error,
            )
        self.is_quantized = True
        set_dynamic_quant(self, "bf16")

        if want_precompute:
            self.enable_adarms_precompute(True)

        dynamo = bool(self.feature_config.export.get("dynamo", False))
        self.export(export_dir, dynamo=dynamo)
        onnx_path = f"{export_dir}/denoise.onnx"
        if is_nvfp4_quantized(quant_cfg):
            print(colored("nvfp4 quantization detected, post processing...", "green"))
            self._nvfp4_post_processing(onnx_path, export_dir)
        apply_denoise_onnx_post_export_patches(onnx_path)

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

    def _compute_adarms_cond(self, timestep: torch.Tensor) -> torch.Tensor:
        """时间步 → adaRMS 条件向量（sinusoid → time_mlp_in → SiLU → time_mlp_out → SiLU）。

        仅依赖 ``timestep``，与 action token / prefix 无关——这是 AdaRMS Dense 预计算的前提。
        """
        if timestep.ndim != 1:
            raise ValueError(f"timestep must be 1D (batch,), got shape {tuple(timestep.shape)}")
        device = timestep.device
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=device,
        )
        # 与 ``time_mlp_*`` 权重 dtype 对齐：标定数据里 ``timestep`` 可能为 bf16，而投影层常为 fp32，
        # 否则 ``F.linear`` 报 ``mat1 and mat2 must have the same dtype``.
        time_emb = time_emb.to(dtype=self.time_mlp_in.weight.dtype)
        x = self.time_mlp_in(time_emb)
        x = F.silu(x)
        x = self.time_mlp_out(x)
        return F.silu(x)

    def _embed_action(
        self, noisy_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """action_in_proj + suffix 掩码（不含时间条件），预计算模式与默认模式共用。"""
        bsize = noisy_actions.shape[0]
        device = noisy_actions.device
        action_emb = self.action_in_proj(
            noisy_actions.to(dtype=self.action_in_proj.weight.dtype)
        )
        action_time_dim = action_emb.shape[1]
        pad_masks = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        att_base = self._suffix_ar_mask.to(device=device).expand(bsize, -1)
        return action_emb, pad_masks, att_base

    def _embed_suffix_pi05(
        self, noisy_actions: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """pi05：时间 adaRMS 条件 + action_in_proj，与 openpi embed_suffix（pi05 分支）一致。"""
        action_time_emb, pad_masks, att_base = self._embed_action(noisy_actions)
        adarms_cond = self._compute_adarms_cond(timestep)
        return action_time_emb, pad_masks, att_base, adarms_cond

    # ------------------------------------------------------------------
    # AdaRMS Dense 预计算（roadmap #22 / docs/optimizer/ddup/adarms_pre_compute.md）
    # ------------------------------------------------------------------
    def _adarms_norm_modules(self) -> list[nn.Module]:
        """按稳定顺序枚举 expert 内含 ``dense`` 的 AdaRMS 归一化模块。

        顺序：``layer[i].input_layernorm``、``layer[i].post_attention_layernorm``（i 递增），
        最后 ``gemma_expert.norm``。该顺序即预计算 modulation 的打包索引。
        """
        mods: list[nn.Module] = []
        for layer in self.gemma_expert.layers:
            mods.append(layer.input_layernorm)
            mods.append(layer.post_attention_layernorm)
        final_norm = getattr(self.gemma_expert, "norm", None)
        if final_norm is not None:
            mods.append(final_norm)
        return [m for m in mods if getattr(m, "dense", None) is not None]

    def enable_adarms_precompute(self, enable: bool = True) -> "Pi05DenoiseStep":
        """开启/关闭 AdaRMS Dense 预计算模式（运行时替换 expert 内 AdaRMS 的 forward）。"""
        if enable and not self._adarms_patched:
            norms = self._adarms_norm_modules()
            if not norms:
                raise RuntimeError(
                    "AdaRMS precompute 需要 expert 使用自适应 RMSNorm（含 dense），"
                    "但未在 gemma_expert 中找到任何 AdaRMS 模块。"
                )
            for n in norms:
                n._orig_forward_for_adarms = n.forward  # noqa: SLF001
                n.forward = types.MethodType(_adarms_injected_forward, n)
            self._adarms_norms_cache = norms
            self._adarms_patched = True
        elif not enable and self._adarms_patched:
            for n in self._adarms_norms_cache:
                if hasattr(n, "_orig_forward_for_adarms"):
                    n.forward = n._orig_forward_for_adarms  # noqa: SLF001
                    del n._orig_forward_for_adarms
            self._adarms_norms_cache = []
            self._adarms_patched = False
        self.adarms_precompute = bool(enable)
        return self

    @torch.no_grad()
    def precompute_adarms_modulation(self, timestep: torch.Tensor) -> torch.Tensor:
        """离线/宿主侧预算：给定 ``timestep`` [batch]，返回打包 modulation。

        Returns:
            形状 ``[num_norms, batch, dim*3]``，``num_norms`` 与
            :meth:`_adarms_norm_modules` 顺序一致；用真实 ``dense`` 权重计算。
            固定步调度下，对 N 个时间步各调一次即可缓存全部常量（见文档 §4）。
        """
        cond = self._compute_adarms_cond(timestep)
        norms = self._adarms_norm_modules()
        if not norms:
            raise RuntimeError("无 AdaRMS 模块，无法预计算 modulation。")
        # dense 此时可能已被替换；从原始 forward 不影响 dense 权重，直接调 dense 即可。
        mods = [n.dense(cond) for n in norms]
        return torch.stack(mods, dim=0)

    def _inject_modulation(self, adarms_mod: torch.Tensor) -> None:
        """把打包 modulation 按 norm 顺序注入各 AdaRMS 模块，供注入版 forward 读取。"""
        if not self._adarms_patched:
            self.enable_adarms_precompute(True)
        norms = self._adarms_norms_cache
        if adarms_mod.shape[0] != len(norms):
            raise ValueError(
                f"adarms_mod[0]={adarms_mod.shape[0]} 与 AdaRMS 模块数 {len(norms)} 不一致"
            )
        for i, n in enumerate(norms):
            n._injected_mod = adarms_mod[i]  # noqa: SLF001

    def _run_expert(
        self,
        prefix_pad_masks: torch.Tensor,
        past_keys: torch.Tensor,
        past_values: torch.Tensor,
        suffix_embs: torch.Tensor,
        suffix_pad_masks: torch.Tensor,
        suffix_att_masks: torch.Tensor,
        adarms_cond: torch.Tensor | None,
    ) -> torch.Tensor:
        """掩码/位置构造 + expert 前向 + action_out_proj（两种模式共用）。

        ``adarms_cond`` 为 ``None`` 时表示预计算模式：modulation 已通过
        :meth:`_inject_modulation` 注入到各 AdaRMS 模块，expert 内不再调用 ``dense``。
        """
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
            timestep: 默认模式为时间标量 float [batch]（同 openpi expanded_time）；
                AdaRMS 预计算模式（``self.adarms_precompute=True``）下，此入参改为承载打包 modulation
                ``adarms_mod`` [num_norms, batch, dim*3]（导出按位置传入；校准始终走默认时间链路，
                因此 ``timestep`` 这个关键字名须保留以匹配校准 batch 的键）。

        Returns:
            v_t: float32 [batch, action_horizon, action_dim]。
        """
        if self.adarms_precompute:
            suffix_embs, suffix_pad_masks, suffix_att_masks = self._embed_action(x_t)
            self._inject_modulation(timestep)
            return self._run_expert(
                prefix_pad_masks,
                past_keys,
                past_values,
                suffix_embs,
                suffix_pad_masks,
                suffix_att_masks,
                adarms_cond=None,
            )

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self._embed_suffix_pi05(
            x_t, timestep
        )
        return self._run_expert(
            prefix_pad_masks,
            past_keys,
            past_values,
            suffix_embs,
            suffix_pad_masks,
            suffix_att_masks,
            adarms_cond=adarms_cond,
        )

    @classmethod
    def construct_model(
        cls,
        pi05_model,
        dtype: torch.dtype | None = None,
        *,
        adarms_precompute: bool | None = None,
        feature_config: FeatureConfig | None = None,
    ):
        """
        从 Pi05Model 包装器或已加载的 PI0Pytorch（policy._model）构建，共享子模块权重引用。

        启用 AdaRMS Dense 预计算（roadmap #22）的三种方式（优先级 高→低）：
          1. ``feature_config``（JSON 配置文件，``features.adarms_dense_precompute``）；
          2. 显式 ``adarms_precompute=True``；
          3. 环境变量 ``PI05_ADARMS_PRECOMPUTE``（``1/true/yes/on``）。
        """
        if not getattr(pi05_model.config, "pi05", False):
            raise ValueError("Pi05DenoiseStep 仅支持 config.pi05 is True 的 PI0Pytorch 模型")

        gemma_expert = pi05_model.paligemma_with_expert.gemma_expert.model
        expert_config = pi05_model.paligemma_with_expert.gemma_expert.config

        if dtype is not None:
            logger.debug("construct_model dtype=%s ignored (weights keep loaded dtype)", dtype)

        # feature_config 优先；否则回退到布尔/环境变量。
        if feature_config is None and adarms_precompute is None:
            adarms_precompute = os.environ.get(
                "PI05_ADARMS_PRECOMPUTE", ""
            ).strip().lower() in ("1", "true", "yes", "y", "on")

        return cls(
            gemma_expert=gemma_expert,
            expert_config=expert_config,
            action_in_proj=pi05_model.action_in_proj,
            time_mlp_in=pi05_model.time_mlp_in,
            time_mlp_out=pi05_model.time_mlp_out,
            action_out_proj=pi05_model.action_out_proj,
            action_horizon=pi05_model.config.action_horizon,
            action_dim=pi05_model.config.action_dim,
            adarms_precompute=bool(adarms_precompute) if adarms_precompute else False,
            feature_config=feature_config,
        )

    @classmethod
    def construct_from_name_path(
        cls, model_name: str, model_path: str, train_config=None, feature_config=None
    ):
        from .model_pi05 import Pi05Model

        wrapper = Pi05Model.construct_from_name_path(
            model_name, model_path, train_config
        )
        return cls.construct_model(
            wrapper, dtype=torch.bfloat16, feature_config=feature_config
        )
