"""FlashRT decoder 后端装配（权重 repack + 预计算 + driver + 离线量化）。

把以下部件组装成可直接驱动 pi05 denoise 整循环的后端，**不依赖 flash_rt python 包**：
- :mod:`.kernels`  —— 按 .so 路径加载 ``fvk`` + ``FvkContext``
- :mod:`.weights`  —— FP8 权重 repack（``_dec_*_flat`` + ``_ae_w_scales``）
- :mod:`.precompute` —— RoPE 表 + AdaRMS 风格（sa/sf/fs）
- :mod:`.driver`   —— buffer/dims/KV 适配 + 整循环执行 ``Pi05ThorDecoderLoop``

集成点：在 ``sample_actions`` 整循环层调用 :meth:`run`（给 prefix KV + 初始 noise，返回 action chunk）。
离线量化：:meth:`calibrate` 跑校准导出 act scales；:meth:`save_act_scales` / :meth:`load_act_scales`。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from model_optimizer.infer.perf import StagePerfCollector

from . import precompute as _pre
from . import weights as _wt
from .driver import DecoderBuffers, DecoderWeights, Pi05ThorDecoderLoop, build_decoder_dims
from .kernels import load_kernels

logger = logging.getLogger(__name__)


class FlashRtDecoderBackend:
    """Pi0.5 Thor FlashRT decoder 后端（仓内移植）。

    Args:
        get_weight: ``key -> Tensor``，从 HF/openpi state_dict 取权重。
        Sa: suffix（action）token 数，即单次输出的 action horizon。
        Da: action-expert 隐藏维（``action_expert.width``）。
        Ha: action-expert FFN 中间维（``action_expert.mlp_dim``）。
        num_layers: decoder 层数（pi05 默认 18）。
        num_q_heads: 注意力 Q 头数（pi05=8；K/V 为 GQA）。
        steps: flow-matching 去噪迭代次数（默认 10）。
        HD: 每头维度 ``head_dim``（pi05=256）。
        use_fp8: True 走静态 FP8；False 走 fp16 baseline。
        device: 权重与 buffer 所在 CUDA 设备。
        build_dir / fmha_so: 传给 kernel loader 的 .so 搜索路径。
        stage_perf: 可选，挂载到 ``Pi05ThorDecoderLoop`` 做分阶段耗时统计。
    """

    def __init__(
        self,
        get_weight: Callable[[str], torch.Tensor],
        *,
        Sa: int,
        Da: int,
        Ha: int,
        num_layers: int = 18,
        num_q_heads: int = 8,
        steps: int = 10,
        HD: int = 256,
        use_fp8: bool = True,
        device: str = "cuda",
        build_dir: str | None = None,
        fmha_so: str | None = None,
        stage_perf: StagePerfCollector | None = None,
    ) -> None:
        # ── 与模型结构绑定的静态超参（整次 eval 不变，与 prefix 长度 enc_seq 无关）──
        self.Sa = int(Sa)  # suffix 时间步数 = action chunk 长度（如 pi05 常为 50）
        self.Da = int(Da)  # action-expert（Gemma decoder）隐藏维，与 openpi ``action_expert.width`` 一致
        self.Ha = int(Ha)  # action-expert FFN 中间维（gate/up 投影的 hidden）
        self.num_layers = int(num_layers)  # decoder 层数（pi05 默认 18）
        self.num_q_heads = int(num_q_heads)  # Q 头数（pi05=8；K/V 为 GQA，每头 1 组 KV）
        self.steps = int(steps)  # flow-matching 去噪步数（整循环 ``decoder_forward`` 迭代次数，默认 10）
        self.HD = int(HD)  # 每头维度 head_dim（pi05=256），RoPE 与 QKV 切分均按此维
        self.use_fp8 = bool(use_fp8)  # True：静态 FP8 kernel + act_scales；False：fp16 baseline
        self.device = device  # CUDA 设备字符串（如 ``cuda`` / ``cuda:0``）
        self._stage_perf = stage_perf  # 可选：``denoise.total`` / ``flashrt.setup`` 等分阶段 wall time

        # FlashRT 自定义算子：``flash_rt_kernels*.so`` 导出的 ``fvk`` 模块与执行上下文
        self._fvk = load_kernels(build_dir, fmha_so=fmha_so)
        self._ctx = self._fvk.FvkContext()  # 持有 kernel launch / workspace 绑定的原生句柄

        # 权重 repack（一次性）
        self._repacked = _wt.repack_decoder_weights(
            get_weight,
            num_layers=self.num_layers,
            num_q_heads=self.num_q_heads,
            steps=self.steps,
            use_fp8=self.use_fp8,
            device=device,
        )
        self._get_weight = get_weight

        # 激活 scales（act_scales）：离线校准产出或加载得到。[layers*4] f32
        self._act_scales = torch.zeros(self.num_layers * 4, dtype=torch.float32, device=device)

        # per-prompt 状态：``__init__`` 只做「与 prompt 无关」的一次性准备（fvk、权重 repack），
        # 下列字段在 ``setup_prompt(enc_seq)`` 里按 prefix 长度填充（见该方法的流程说明）。
        self._enc_seq: int | None = None  # 当前 prefix token 数（= TRT/PyTorch llm KV 的 seq 维）
        self._rope: torch.Tensor | None = None  # decoder suffix RoPE 表 [Sa, HD]，供 qkv_split_rope 用
        self._styles: _pre.AdaRmsStyles | None = None  # flow-matching 每步 AdaRMS 预计算 sa/sf/fs
        self._bufs: DecoderBuffers | None = None  # GPU 工作区 + KV slab（尺寸依赖 enc_seq）
        self._loop: Pi05ThorDecoderLoop | None = None  # 绑好指针后可 ``set_prefix_kv`` + ``run(noise)``

    # ── per-prompt 构建 ────────────────────────────────────────────
    def setup_prompt(self, enc_seq: int) -> None:
        """在 prefix 长度 ``enc_seq`` 已知后，组装 decoder 整循环所需的 per-prompt 资源。

        **调用时机**（见 ``pi05_executor.sample_actions_flashrt``）：
          1. TRT/vit+llm 或 PyTorch 跑完 prefix，得到 ``past_keys/past_values``；
          2. ``enc_seq = past_keys.shape[-2]``（图像+语言 token 总数，因任务而异）；
          3. 在 ``backend.run(past_keys, past_values, noise)`` 之前调用本方法。

        **为何按 prompt 构建**：
          - RoPE 位置从 ``enc_seq`` 起算（suffix 在全局序列的 ``[enc_seq, enc_seq+Sa)``）；
          - KV cache 物理长度 ``total_keys = enc_seq + Sa``，buffer 必须按此分配；
          - ``dims["enc_seq"]`` / ``dims["total_keys"]`` 传入 FlashRT kernel 做 KV offset。

        **与 ``__init__`` 的分工**：
          - ``__init__``：加载 ``flash_rt_kernels``、repack 18 层 FP8 权重（与 enc_seq 无关）；
          - ``setup_prompt``：RoPE 表、AdaRMS 风格、GPU buffer、``Pi05ThorDecoderLoop`` 绑定。

        **缓存**：同一 ``enc_seq`` 且 ``_loop`` 已存在则直接返回（固定相机+固定 prompt 的 eval 可复用）。

        **不在这里做的事**：
          - 不把 TRT prefix KV 写入 ``Kc/Vc``（在 ``run`` → ``loop.set_prefix_kv`` 里按样本拷贝）；
          - 不跑 10 步 denoise（在 ``run`` → ``loop.run`` 里完成）。
        """
        enc_seq = int(enc_seq)

        # ── 0) 短路径：prompt 布局未变则复用已分配的 buffer / loop ──
        if self._enc_seq == enc_seq and self._loop is not None:
            return
        self._enc_seq = enc_seq

        # ── 1) RoPE 表：suffix 各位置的 (cos, sin)，交错布局 [Sa, HD=256] ──
        # FlashRT ``qkv_split_rope_kvcache_fp16`` 按 ``rope[s*HD + pair*2]`` 读 cos/sin。
        # 位置索引 = prefix 结束后的全局下标，故从 kp[enc_seq : enc_seq+Sa] 切片。
        self._rope = _pre.build_dec_rope(enc_seq, self.Sa, device=self.device, head_dim=self.HD)

        # ── 2) AdaRMS 调制风格：10 个 flow-matching 时间步 × 18 层 ──
        # Pi0.5 action-expert 用 time-conditioned RMSNorm（非静态 γ）。
        # 在 CPU/GPU 上预计算 ``sa``(attn 前)、``sf``(FFN 前)、``fs``(最后一层)，
        # 避免 denoise 热路径里重复做 time_mlp + dense modulation。
        # 注：与 enc_seq 无直接关系，但放在此处与 FlashRT ``set_prompt`` 段落一致。
        self._styles = _pre.precompute_adarms_styles(
            self._get_weight,
            Sa=self.Sa,
            Da=self.Da,
            num_layers=self.num_layers,
            steps=self.steps,
            device=self.device,
        )

        # ── 3) GPU 工作 buffer + KV slab ──
        # Kc/Vc: [layers, enc_seq+Sa, HD]；prefix 区 [0:enc_seq) 留给 set_prefix_kv 填入，
        # suffix 区 [enc_seq:total_keys) 由 decoder_forward 每层 qkv_split_rope 写入。
        # 其余 x/qkv/logits/hid/... 为单层 kernel 链路的中间结果（尺寸与 Sa/Da/Ha 相关）。
        self._bufs = DecoderBuffers(
            Sa=self.Sa, Da=self.Da, Ha=self.Ha, layers=self.num_layers,
            enc_seq=enc_seq, HD=self.HD, NH=self.num_q_heads, device=self.device,
        )

        # ── 4) 整循环维度元数据（传给 pipeline.decoder_forward）──
        # enc_seq / total_keys 决定 attention 里 K 的索引与 KV cache offset。
        dims = build_decoder_dims(
            Sa=self.Sa, Da=self.Da, Ha=self.Ha, layers=self.num_layers,
            enc_seq=enc_seq, NH=self.num_q_heads, HD=self.HD, steps=self.steps,
        )

        # ── 5) 把 repack 权重 + 预计算表 收成 CUDA 指针包 ──
        # DecoderWeights 不含 Kc/Vc 本体指针（它们在 bufs 里，run 时再 as_dict 注入）。
        weights = self._build_decoder_weights()

        # ── 6) 绑定执行器：后续 run() 只需 set_prefix_kv + run(noise) ──
        self._loop = Pi05ThorDecoderLoop(
            self._ctx, self._fvk, self._bufs, weights, dims,
            use_fp8=self.use_fp8,
            stage_perf=self._stage_perf,
        )

    def _build_decoder_weights(self) -> DecoderWeights:
        """把 ``__init__`` 的 repack 张量与 ``setup_prompt`` 的预计算表转为 kernel 用的 int 指针。

        调用前必须已完成 ``_rope`` / ``_styles`` 构建（由 ``setup_prompt`` 保证）。
        张量通过 ``data_ptr()`` 保活：``DecoderWeights`` 只存指针，底层 storage 由
        ``self._repacked`` / ``self._styles`` / ``self._rope`` / ``self._act_scales`` 持有。
        """
        assert self._rope is not None and self._styles is not None
        r = self._repacked
        return DecoderWeights(
            ain_w=r.ptr("ain_w"),  # action_in_proj，噪声 → 隐藏维
            ain_b=r.ptr("ain_b"),
            sa=self._styles.sa_all.reshape(-1).data_ptr(),  # 每层 attn 前 AdaRMS scale/shift/gate
            qw=r.ptr("dec_qkv_flat"),  # 18 层 fused QKV（已 interleave_qk + FP8）
            ow=r.ptr("dec_o_flat"),
            sf=self._styles.sf_all.reshape(-1).data_ptr(),  # 每层 FFN 前 AdaRMS
            gw=r.ptr("dec_gu_flat"),  # gate+up 融合
            dw=r.ptr("dec_d_flat"),
            aow=r.ptr("aow"),  # action_out_proj（已烘入 -1/steps 的 Euler 系数）
            aob=r.ptr("aob"),
            fs=self._styles.fs_all.reshape(-1).data_ptr(),  # 最后一层 norm 的 AdaRMS
            rope=self._rope.reshape(-1).data_ptr(),
            w_scales=r.ae_w_scales.reshape(-1).data_ptr(),  # 每层权重量化 scale [layers*4]
            act_scales=self._act_scales.reshape(-1).data_ptr(),  # 离线校准的激活 scale（FP8）
        )

    # ── 推理 ───────────────────────────────────────────────────────
    def run(
        self,
        past_keys: torch.Tensor,
        past_values: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """整 10 步 denoise 循环。返回最终 action chunk ``[Sa, 32]``。"""
        if self._loop is None:
            self.setup_prompt(int(past_keys.shape[-2]))
        assert self._loop is not None
        self._loop.set_prefix_kv(past_keys, past_values)
        return self._loop.run(noise)

    # ── 离线量化 ───────────────────────────────────────────────────
    def reset_act_scales(self) -> None:
        """清零累计的 act scales（多样本标定前调用一次）。"""
        self._act_scales.zero_()

    def accumulate_calibration(
        self, past_keys: torch.Tensor, past_values: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """累计一个样本的标定：跨 10 步取 max 的 scale 再与 ``self._act_scales`` 跨样本取 max。

        多样本/多 prompt 调用本方法即可（KV/noise 每次不同），最后用累计的 ``self._act_scales``。
        """
        self.setup_prompt(int(past_keys.shape[-2]))
        assert self._loop is not None
        self._loop.set_prefix_kv(past_keys, past_values)
        sample_scales = self._loop.calibrate(noise)  # [layers*4] 跨步 max
        torch.maximum(
            self._act_scales, sample_scales.to(self._act_scales.device), out=self._act_scales
        )
        return self._act_scales

    def calibrate(self, past_keys: torch.Tensor, past_values: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """单样本标定（reset + accumulate），向后兼容入口。"""
        self.reset_act_scales()
        return self.accumulate_calibration(past_keys, past_values, noise)

    def save_act_scales(self, path: str) -> None:
        scales = self._act_scales.detach().cpu().tolist()
        meta = {
            "version": 1,
            "kind": "flashrt_decoder_act_scales",
            "num_layers": self.num_layers,
            "use_fp8": self.use_fp8,
            "act_scales": scales,
        }
        Path(path).expanduser().write_text(json.dumps(meta, indent=2))
        logger.info("[flashrt-decoder] act scales saved: %s (%d)", path, len(scales))

    def load_act_scales(self, path: str) -> None:
        meta = json.loads(Path(path).expanduser().read_text())
        scales = meta["act_scales"]
        if len(scales) != self.num_layers * 4:
            raise ValueError(
                f"act_scales 长度 {len(scales)} != layers*4 {self.num_layers * 4}"
            )
        self._act_scales.copy_(torch.tensor(scales, dtype=torch.float32, device=self.device))
        logger.info("[flashrt-decoder] act scales loaded: %s", path)
