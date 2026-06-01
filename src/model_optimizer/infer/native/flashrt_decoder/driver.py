"""Pi0.5 Thor decoder 驱动（buffer/dims/KV 适配 + 整循环执行）。

移植自 FlashRT ``frontends/torch/pi05_thor.py`` 的 AE decoder 驱动部分，但**剥离 flash_rt
依赖**：buffer 分配/dims 构建/KV 适配为本仓库自有 torch 代码；kernel 由 :mod:`.kernels`
按 .so 路径加载；权重 repack（FP8 布局）与 AdaRMS 预计算由调用方注入。

关键架构点（与 §7.3 的 per-step ``DenoiseBackend`` 不同）：
- FlashRT ``decoder_forward`` 是 **整 10 步扩散循环一次跑完**（内部 ``for s in range(steps)``，
  并把 ``-1/steps`` 积分烘进 ``action_out_proj``）。
- 因此 FlashRT 港版挂在 **``sample_actions`` 整循环层**，而非 per-step ``denoise_step`` hook。
- 接口对齐点改为：给定 prefix KV（来自 TRT llm，stacked）+ 初始 noise，返回最终 action chunk。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from . import pipeline as _pipeline

_FP16 = torch.float16


def build_decoder_dims(
    *,
    Sa: int,
    Da: int,
    Ha: int,
    layers: int,
    enc_seq: int,
    NH: int = 8,
    HD: int = 256,
    steps: int = 10,
) -> dict[str, int]:
    """构建 ``decoder_forward`` 期望的 dims（与 FlashRT ae_dims 一致）。"""
    return {
        "S": int(Sa),
        "D": int(Da),
        "H": int(Ha),
        "NH": int(NH),
        "HD": int(HD),
        "steps": int(steps),
        "layers": int(layers),
        "enc_seq": int(enc_seq),
        "total_keys": int(enc_seq) + int(Sa),
    }


def fill_prefix_kv_from_trt(
    Kc: torch.Tensor,
    Vc: torch.Tensor,
    past_keys: torch.Tensor,
    past_values: torch.Tensor,
    enc_seq: int,
) -> None:
    """方案(B) KV 适配器：把 TRT llm 的 bf16 stacked KV 填入 FlashRT fp16 slab 的 prefix 区。

    布局对齐（见设计文档 §8.2）：
      - TRT past_keys/values: ``[L, 1, enc_seq, HD]``（或 ``[L, enc_seq, HD]``）bf16
      - FlashRT ``Kc``/``Vc``: ``[L, total_keys, HD]`` fp16，prefix 占 ``[:, :enc_seq, :]``，
        suffix 区 ``[:, enc_seq:, :]`` 由 decoder 自己写。
    仅做 dtype cast + 逐层 strided copy；**RoPE 约定须在 Thor 上验证一致**（K 须已施加 RoPE）。
    """
    k = past_keys[:, 0] if past_keys.dim() == 4 else past_keys
    v = past_values[:, 0] if past_values.dim() == 4 else past_values
    if k.shape[1] != enc_seq:
        raise ValueError(
            f"prefix KV 长度 {k.shape[1]} != enc_seq {enc_seq}；检查 prefix_len/total_keys"
        )
    Kc[:, :enc_seq, :].copy_(k.to(Kc.dtype))
    Vc[:, :enc_seq, :].copy_(v.to(Vc.dtype))


@dataclass
class DecoderBuffers:
    """AE decoder 的设备 buffer 持有者（保活 + 暴露 data_ptr 字典）。"""

    Sa: int
    Da: int
    Ha: int
    layers: int
    enc_seq: int
    HD: int = 256
    NH: int = 8
    device: str = "cuda"
    _tensors: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Sa, Da, Ha = self.Sa, self.Da, self.Ha
        total_keys = self.enc_seq + Sa
        t = self._tensors
        # KV cache（fp16 单 KV head；prefix+suffix）
        t["Kc"] = torch.zeros(self.layers, total_keys, self.HD, dtype=_FP16, device=self.device)
        t["Vc"] = torch.zeros(self.layers, total_keys, self.HD, dtype=_FP16, device=self.device)
        # 工作 buffer（镜像 FlashRT ae_* 命名）
        t["noise"] = torch.zeros(Sa, 32, dtype=_FP16, device=self.device)
        t["x"] = torch.empty(Sa, Da, dtype=_FP16, device=self.device)
        t["xn"] = torch.empty(Sa, Da, dtype=_FP16, device=self.device)
        t["gate"] = torch.empty(Sa, Da, dtype=_FP16, device=self.device)
        t["qkv"] = torch.empty(Sa, 2560, dtype=_FP16, device=self.device)
        t["logits"] = torch.empty(Sa * self.NH, total_keys, dtype=_FP16, device=self.device)
        t["attn_out"] = torch.empty(Sa * self.NH, self.HD, dtype=_FP16, device=self.device)
        t["hid"] = torch.empty(Sa, 2 * Ha, dtype=_FP16, device=self.device)
        t["fg"] = torch.empty(Sa, 2 * Ha, dtype=_FP16, device=self.device)
        t["xn_fp8"] = torch.zeros(Sa * Da, dtype=torch.uint8, device=self.device)
        t["hid_fp8"] = torch.zeros(Sa * Ha, dtype=torch.uint8, device=self.device)
        t["ctx_fp8"] = torch.zeros(Sa * self.NH * self.HD, dtype=torch.uint8, device=self.device)
        # 校准 scratch（离线量化用）
        t["calib_buf"] = torch.zeros(self.layers * 4, dtype=torch.float32, device=self.device)
        t["d_scale"] = torch.zeros(1, dtype=torch.float32, device=self.device)
        t["hidden_scratch"] = torch.empty(Sa * Ha, dtype=_FP16, device=self.device)
        t["fp8_scratch"] = torch.zeros(Sa * max(Da, Ha), dtype=torch.uint8, device=self.device)

    @property
    def Kc(self) -> torch.Tensor:
        return self._tensors["Kc"]

    @property
    def Vc(self) -> torch.Tensor:
        return self._tensors["Vc"]

    @property
    def noise(self) -> torch.Tensor:
        return self._tensors["noise"]

    def as_ptr_dict(self) -> dict[str, int]:
        d = {k: v.reshape(-1).data_ptr() for k, v in self._tensors.items()}
        return d


@dataclass
class DecoderWeights:
    """AE decoder 的权重指针集合（由权重 repack 注入）。

    字段对应 ``pipeline.decoder_forward`` 的 weights 字典。``sa/sf/fs`` 是 AdaRMS
    预计算 buffer（``[steps, layers, S, 3D]`` 展平），由 model_optimizer 现有的
    AdaRMS 预计算产出后写入。
    """

    ain_w: int
    ain_b: int
    sa: int
    qw: int
    ow: int
    sf: int
    gw: int
    dw: int
    aow: int
    aob: int
    fs: int
    rope: int
    w_scales: int
    act_scales: int

    def as_dict(self, Kc_ptr: int, Vc_ptr: int) -> dict[str, int]:
        return {
            "ain_w": self.ain_w,
            "ain_b": self.ain_b,
            "sa": self.sa,
            "qw": self.qw,
            "Kc": Kc_ptr,
            "Vc": Vc_ptr,
            "ow": self.ow,
            "sf": self.sf,
            "gw": self.gw,
            "dw": self.dw,
            "aow": self.aow,
            "aob": self.aob,
            "fs": self.fs,
            "rope": self.rope,
            "w_scales": self.w_scales,
            "act_scales": self.act_scales,
        }


class Pi05ThorDecoderLoop:
    """整 10 步 decoder 循环执行器（FlashRT decoder_forward 港版的运行入口）。

    用法（在 sample_actions 层）：
        loop = Pi05ThorDecoderLoop(ctx, fvk, bufs, weights, dims, use_fp8=True)
        loop.set_prefix_kv(past_keys, past_values)   # 来自 TRT llm
        action = loop.run(noise)                     # 返回 [Sa, 32] 动作 chunk

    权重 repack + AdaRMS 预计算（sa/sf/fs）+ rope 表的填充由调用方负责（见 §8.3 待办）。
    """

    def __init__(
        self,
        ctx: Any,
        fvk: Any,
        bufs: DecoderBuffers,
        weights: DecoderWeights,
        dims: dict[str, int],
        *,
        use_fp8: bool = True,
        stream: int = 0,
    ) -> None:
        self._ctx = ctx
        self._fvk = fvk
        self._bufs = bufs
        self._weights = weights
        self._dims = dims
        self._use_fp8 = bool(use_fp8)
        self._stream = int(stream)

    def set_prefix_kv(self, past_keys: torch.Tensor, past_values: torch.Tensor) -> None:
        fill_prefix_kv_from_trt(
            self._bufs.Kc, self._bufs.Vc, past_keys, past_values, self._dims["enc_seq"]
        )

    def run(self, noise: torch.Tensor) -> torch.Tensor:
        """运行整循环。``noise``: ``[Sa, 32]`` fp16；返回最终 action chunk（``noise`` buffer）。"""
        self._bufs.noise.copy_(noise.to(_FP16).reshape(self._bufs.noise.shape))
        ptr_bufs = self._bufs.as_ptr_dict()
        ptr_weights = self._weights.as_dict(
            ptr_bufs["Kc"], ptr_bufs["Vc"]
        )
        _pipeline.decoder_forward(
            self._ctx,
            self._fvk,
            ptr_bufs,
            ptr_weights,
            self._dims,
            self._stream,
            use_fp8=self._use_fp8,
        )
        return self._bufs.noise

    def calibrate(self, noise: torch.Tensor) -> torch.Tensor:
        """离线量化：跑一次校准前向，返回**跨 10 步取 max** 的 act scales（``[layers*4]`` f32）。

        通过 ``per_step_scales_ptr`` 让 pipeline 落盘每步 scale，再在 torch 侧对 step 轴取 max
        （而非 FlashRT 原版只留最后一步），降低 FP8 激活饱和风险。
        """
        self._bufs.noise.copy_(noise.to(_FP16).reshape(self._bufs.noise.shape))
        ptr_bufs = self._bufs.as_ptr_dict()
        ptr_weights = self._weights.as_dict(ptr_bufs["Kc"], ptr_bufs["Vc"])
        steps = int(self._dims["steps"])
        layers = int(self._dims["layers"])
        device = self._bufs.device
        per_step = torch.zeros(steps, layers * 4, dtype=torch.float32, device=device)
        last_step = torch.zeros(layers * 4, dtype=torch.float32, device=device)
        _pipeline.decoder_forward_calibrate(
            self._ctx,
            self._fvk,
            ptr_bufs,
            ptr_weights,
            self._dims,
            last_step.reshape(-1).data_ptr(),
            self._stream,
            per_step_scales_ptr=per_step.reshape(-1).data_ptr(),
        )
        # step 轴取 max（0 值步——理论上每步都写——用 last_step 兜底）
        cross_step = per_step.amax(dim=0)
        return torch.maximum(cross_step, last_step)
