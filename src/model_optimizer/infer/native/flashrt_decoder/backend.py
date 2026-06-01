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
from typing import Callable

import torch

from . import precompute as _pre
from . import weights as _wt
from .driver import DecoderBuffers, DecoderWeights, Pi05ThorDecoderLoop, build_decoder_dims
from .kernels import load_kernels

logger = logging.getLogger(__name__)


class FlashRtDecoderBackend:
    """Pi0.5 Thor FlashRT decoder 后端（仓内移植）。

    Args:
        get_weight: ``key -> Tensor``，从 HF/openpi state_dict 取权重。
        Sa: suffix（action）token 数。
        Da: action-expert 隐藏维。
        Ha: action-expert MLP 隐藏维。
        num_layers: decoder 层数（pi05=18）。
        num_q_heads: Q head 数（pi05=8；K/V=1）。
        use_fp8: True 走静态 FP8；False 走 fp16 baseline。
        build_dir / fmha_so: 传给 kernel loader 的 .so 搜索路径。
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
    ) -> None:
        self.Sa = int(Sa)
        self.Da = int(Da)
        self.Ha = int(Ha)
        self.num_layers = int(num_layers)
        self.num_q_heads = int(num_q_heads)
        self.steps = int(steps)
        self.HD = int(HD)
        self.use_fp8 = bool(use_fp8)
        self.device = device

        self._fvk = load_kernels(build_dir, fmha_so=fmha_so)
        self._ctx = self._fvk.FvkContext()

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

        # per-prompt 状态（enc_seq 已知后构建）
        self._enc_seq: int | None = None
        self._rope: torch.Tensor | None = None
        self._styles: _pre.AdaRmsStyles | None = None
        self._bufs: DecoderBuffers | None = None
        self._loop: Pi05ThorDecoderLoop | None = None

    # ── per-prompt 构建 ────────────────────────────────────────────
    def setup_prompt(self, enc_seq: int) -> None:
        """prefix 长度已知时构建 RoPE/AdaRMS/buffers/loop（每个 prompt 调一次）。"""
        enc_seq = int(enc_seq)
        if self._enc_seq == enc_seq and self._loop is not None:
            return
        self._enc_seq = enc_seq

        self._rope = _pre.build_dec_rope(enc_seq, self.Sa, device=self.device, head_dim=self.HD)
        self._styles = _pre.precompute_adarms_styles(
            self._get_weight,
            Sa=self.Sa,
            Da=self.Da,
            num_layers=self.num_layers,
            steps=self.steps,
            device=self.device,
        )
        self._bufs = DecoderBuffers(
            Sa=self.Sa, Da=self.Da, Ha=self.Ha, layers=self.num_layers,
            enc_seq=enc_seq, HD=self.HD, NH=self.num_q_heads, device=self.device,
        )
        dims = build_decoder_dims(
            Sa=self.Sa, Da=self.Da, Ha=self.Ha, layers=self.num_layers,
            enc_seq=enc_seq, NH=self.num_q_heads, HD=self.HD, steps=self.steps,
        )
        weights = self._build_decoder_weights()
        self._loop = Pi05ThorDecoderLoop(
            self._ctx, self._fvk, self._bufs, weights, dims,
            use_fp8=self.use_fp8,
        )

    def _build_decoder_weights(self) -> DecoderWeights:
        assert self._rope is not None and self._styles is not None
        r = self._repacked
        return DecoderWeights(
            ain_w=r.ptr("ain_w"),
            ain_b=r.ptr("ain_b"),
            sa=self._styles.sa_all.reshape(-1).data_ptr(),
            qw=r.ptr("dec_qkv_flat"),
            ow=r.ptr("dec_o_flat"),
            sf=self._styles.sf_all.reshape(-1).data_ptr(),
            gw=r.ptr("dec_gu_flat"),
            dw=r.ptr("dec_d_flat"),
            aow=r.ptr("aow"),
            aob=r.ptr("aob"),
            fs=self._styles.fs_all.reshape(-1).data_ptr(),
            rope=self._rope.reshape(-1).data_ptr(),
            w_scales=r.ae_w_scales.reshape(-1).data_ptr(),
            act_scales=self._act_scales.reshape(-1).data_ptr(),
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
    def calibrate(self, past_keys: torch.Tensor, past_values: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """跑校准前向，把 act scales 写入 ``self._act_scales`` 并返回。"""
        if self._loop is None:
            self.setup_prompt(int(past_keys.shape[-2]))
        assert self._loop is not None
        self._loop.set_prefix_kv(past_keys, past_values)
        self._loop.calibrate(noise, self._act_scales)
        return self._act_scales

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
