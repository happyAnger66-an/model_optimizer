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
from typing import TYPE_CHECKING, Any

import torch

from . import pipeline as _pipeline

if TYPE_CHECKING:
    from model_optimizer.infer.perf import StagePerfCollector

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
    """方案(B) KV 适配器：把 TRT/PyTorch LLM KV 填入 FlashRT fp16 slab 的 prefix 区。

    布局对齐（见设计文档 §8.2）：
      - TRT past_keys/values: ``[L, 1, enc_seq, HD]``（或 ``[L, enc_seq, HD]``）bf16
      - FlashRT ``Kc``/``Vc``: ``[L, total_keys, HD]`` fp16，prefix 占 ``[:, :enc_seq, :]``，
        suffix 区 ``[:, enc_seq:, :]`` 由 decoder 自己写。
    FlashRT RoPE kernel 读写 pair-interleaved head 维：
    ``[d0, dHD/2, d1, dHD/2+1, ...]``。PyTorch/HF cache 已经施加 RoPE，
    但仍是 half-split head 维，因此 prefix K 拷贝前必须做同样的 pair-interleave。
    """
    k = past_keys[:, 0] if past_keys.dim() == 4 else past_keys
    v = past_values[:, 0] if past_values.dim() == 4 else past_values
    if k.shape[1] != enc_seq:
        raise ValueError(
            f"prefix KV 长度 {k.shape[1]} != enc_seq {enc_seq}；检查 prefix_len/total_keys"
        )
    hd = int(k.shape[-1])
    if hd % 2 != 0:
        raise ValueError(f"prefix K head_dim must be even for RoPE interleave, got {hd}")
    k_flashrt = (
        k.reshape(*k.shape[:-1], 2, hd // 2)
        .permute(0, 1, 3, 2)
        .reshape_as(k)
    )
    Kc[:, :enc_seq, :].copy_(k_flashrt.to(Kc.dtype))
    Vc[:, :enc_seq, :].copy_(v.to(Vc.dtype))


@dataclass
class DecoderBuffers:
    """AE decoder 的设备端 buffer 持有者（生命周期保活 + 暴露 ``data_ptr`` 字典）。

    作用与定位
    ----------
    FlashRT 的 ``decoder_forward``（见 :mod:`.pipeline`）是**纯指针接口**的 CUDA kernel
    序列：它不分配显存，只接收一组 ``int`` 指针（``bufs`` / ``weights``）在原地读写。
    本类负责：

    1. **一次性预分配** 整个 10 步 × ``layers`` 层扩散循环所需的全部中间 buffer；
    2. **持有 torch.Tensor 引用**（防止被 GC 回收，否则裸指针会悬空）；
    3. 通过 :meth:`as_ptr_dict` 把这些 tensor 拍平成 ``{name: data_ptr}`` 交给 pipeline。

    所有 buffer 在构造时分配、整个 session 复用（每个 chunk 不再 malloc），与 FlashRT
    ``ae_*`` 命名一一对应，便于和原版逐字对照。

    维度约定
    --------
    - ``Sa``      : action suffix 序列长（query 长度，diffusion 的并行 token 数）。
    - ``Da`` (D)  : decoder 隐藏维（残差流宽度）。
    - ``Ha`` (H)  : MLP 中间维（gate/up 投影宽度）。
    - ``layers``  : decoder 层数（pi05 为 18）。
    - ``enc_seq`` : prefix（vit+llm 编码）序列长；KV cache 的前缀区长度。
    - ``HD``      : 每个注意力 head 的维度（256）。
    - ``NH``      : query head 数（8）；KV 为**单头**（GQA/MQA，故 Kc/Vc 不带 NH 维）。
    - ``total_keys = enc_seq + Sa`` : KV cache 总长（prefix + 本次 suffix）。

    魔数说明
    --------
    - ``32``   : action 输入/输出维（``action_in_proj`` 入口、``action_out_proj`` 出口）。
    - ``2560`` : 融合 QKV 投影输出宽度 = ``NH*HD + 2*HD`` = ``Q(8*256) + K(256) + V(256)``。
    - 每层 ``4`` 个 FP8 量化点（见 ``act_scales``/``w_scales`` 的 ``l*4 + {0,1,2,3}``）：
      QKV-in / O-in / gate-up-in / down-in。
    """

    Sa: int  # action suffix 长（query 数）
    Da: int  # decoder 隐藏维 D
    Ha: int  # MLP 中间维 H
    layers: int  # decoder 层数（pi05=18）
    enc_seq: int  # prefix 序列长（KV cache 前缀区）
    HD: int = 256  # per-head 维度
    NH: int = 8  # query head 数（KV 为单头）
    device: str = "cuda"
    # 实际持有的 buffer 集合：name -> tensor。私有，统一经 as_ptr_dict() 取指针。
    _tensors: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Sa, Da, Ha = self.Sa, self.Da, self.Ha
        total_keys = self.enc_seq + Sa
        t = self._tensors

        # ── KV cache（fp16，单 KV head；布局 [layers, total_keys, HD]）──
        # prefix 区 [:, :enc_seq, :] 由 fill_prefix_kv_from_trt 从 TRT llm 写入；
        # suffix 区 [:, enc_seq:, :] 由 decoder C2b（qkv_split_rope_kvcache）逐层写入。
        t["Kc"] = torch.zeros(self.layers, total_keys, self.HD, dtype=_FP16, device=self.device)
        t["Vc"] = torch.zeros(self.layers, total_keys, self.HD, dtype=_FP16, device=self.device)

        # ── 主工作 buffer（镜像 FlashRT ae_* 命名）──
        # noise: [Sa, 32] —— 既是输入噪声，也是**最终 action chunk 输出**（in/out 复用）。
        #        入口 gmm: noise(32) → x(D)；出口 gmm: xn(D) → noise(32)，逐步 Euler 累积。
        t["noise"] = torch.zeros(Sa, 32, dtype=_FP16, device=self.device)
        # x:  [Sa, D] 残差流 / 跨层隐藏状态（每层 gated residual 累加于此）。
        t["x"] = torch.empty(Sa, Da, dtype=_FP16, device=self.device)
        # xn: [Sa, D] AdaRMSNorm 归一化后的激活（送入 QKV / gate-up / 最终 out-proj）。
        t["xn"] = torch.empty(Sa, Da, dtype=_FP16, device=self.device)
        # gate: [Sa, D] AdaRMS 产出的 gate（缩放），用于 gate×residual 融合。
        t["gate"] = torch.empty(Sa, Da, dtype=_FP16, device=self.device)
        # qkv: [Sa, 2560] 融合 QKV 投影输出（Q=NH*HD, K=HD, V=HD）；由 C2 ``fp8_gemm_descale_fp16``
        # 直接写入 fp16（见 ``docs/optimizer/flashrt/fp8_gemm_descale_fp16.md``）。
        t["qkv"] = torch.empty(Sa, 2560, dtype=_FP16, device=self.device)
        # logits: [Sa*NH, total_keys] 注意力打分 scratch（每个 query head 对全部 key）。
        t["logits"] = torch.empty(Sa * self.NH, total_keys, dtype=_FP16, device=self.device)
        # attn_out: [Sa*NH, HD] 既作 Q 输入，也接注意力上下文输出（原地复用）。
        t["attn_out"] = torch.empty(Sa * self.NH, self.HD, dtype=_FP16, device=self.device)
        # hid: [Sa, 2H] MLP 中间缓冲（FP16 路 GEGLU 输出落点）。
        t["hid"] = torch.empty(Sa, 2 * Ha, dtype=_FP16, device=self.device)
        # fg: [Sa, 2H] 通用 GEMM 输出 scratch（O-proj / gate-up / down 输出轮流复用，2H 取最大）。
        t["fg"] = torch.empty(Sa, 2 * Ha, dtype=_FP16, device=self.device)

        # ── FP8 量化中间张量（uint8 承载 e4m3）──
        # xn_fp8:  [Sa*D]       归一化激活的 FP8 量化结果（QKV / gate-up GEMM 输入）。
        t["xn_fp8"] = torch.zeros(Sa * Da, dtype=torch.uint8, device=self.device)
        # hid_fp8: [Sa*H]       MLP 中间激活的 FP8（down GEMM 输入）。
        t["hid_fp8"] = torch.zeros(Sa * Ha, dtype=torch.uint8, device=self.device)
        # ctx_fp8: [Sa*NH*HD]   注意力上下文的 FP8（O-proj GEMM 输入）。
        t["ctx_fp8"] = torch.zeros(Sa * self.NH * self.HD, dtype=torch.uint8, device=self.device)

        # ── 校准 scratch（仅离线量化 decoder_forward_calibrate 使用）──
        # calib_buf: [layers*4] 当步每层 4 个量化点测得的 scale（float32），逐步覆盖。
        t["calib_buf"] = torch.zeros(self.layers * 4, dtype=torch.float32, device=self.device)
        # d_scale: [1] measure_scale_gpu 的单点 amax/scale 输出。
        t["d_scale"] = torch.zeros(1, dtype=torch.float32, device=self.device)
        # hidden_scratch: [Sa*H] 校准时 GEGLU 的 FP16 输出（用于测 amax，不污染 hid_fp8）。
        t["hidden_scratch"] = torch.empty(Sa * Ha, dtype=_FP16, device=self.device)
        # fp8_scratch: [Sa*max(D,H)] measure_scale_gpu 的 FP8 临时区（取 D/H 较大者复用）。
        t["fp8_scratch"] = torch.zeros(Sa * max(Da, Ha), dtype=torch.uint8, device=self.device)

    @property
    def Kc(self) -> torch.Tensor:
        """KV cache 的 Key slab（``[layers, total_keys, HD]`` fp16）。供 prefix 填充。"""
        return self._tensors["Kc"]

    @property
    def Vc(self) -> torch.Tensor:
        """KV cache 的 Value slab（``[layers, total_keys, HD]`` fp16）。供 prefix 填充。"""
        return self._tensors["Vc"]

    @property
    def noise(self) -> torch.Tensor:
        """噪声/动作 in-out buffer（``[Sa, 32]`` fp16）。``run()`` 前写入初值、结束后读结果。"""
        return self._tensors["noise"]

    def as_ptr_dict(self) -> dict[str, int]:
        """把全部 buffer 拍平成 ``{name: device_ptr}``，交给 pipeline 的纯指针 kernel。

        每次调用都按 ``reshape(-1).data_ptr()`` 现取；因 tensor 由本对象持有，指针在
        本实例存活期间稳定有效。
        """
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
        stage_perf: StagePerfCollector | None = None,
    ) -> None:
        self._ctx = ctx
        self._fvk = fvk
        self._bufs = bufs
        self._weights = weights
        self._dims = dims
        self._use_fp8 = bool(use_fp8)
        self._stream = int(stream)
        self._stage_perf = stage_perf

    def set_prefix_kv(self, past_keys: torch.Tensor, past_values: torch.Tensor) -> None:
        if self._stage_perf is None:
            fill_prefix_kv_from_trt(
                self._bufs.Kc, self._bufs.Vc, past_keys, past_values, self._dims["enc_seq"]
            )
            return
        with self._stage_perf.timed("kv.fill_prefix"):
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
            perf=self._stage_perf,
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
