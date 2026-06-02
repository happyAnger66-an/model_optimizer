# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashRT Pi0.5 denoise 运行器（仅接管 denoise，前缀阶段仍可走 TRT/PyTorch）。"""

from __future__ import annotations

import math
from pathlib import Path

import torch

try:
    import flash_rt.flash_rt_kernels as fvk
    from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor
    from flash_rt.models.pi05.pipeline_thor import decoder_forward
except ImportError as exc:  # pragma: no cover
    fvk = None  # type: ignore[assignment]
    _IMPORT_ERR = exc
else:
    _IMPORT_ERR = None


def _require_flashrt() -> None:
    if _IMPORT_ERR is not None:
        raise ImportError(
            "FlashRT (flash_rt) is required for FlashRtPi05DenoiseRunner."
        ) from _IMPORT_ERR


class FlashRtPi05DenoiseRunner:
    """用 FlashRT ``decoder_forward`` 执行 Pi0.5 的完整 10-step denoise。"""

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        num_views: int = 2,
        use_cuda_graph: bool = True,
    ) -> None:
        _require_flashrt()
        ckpt = str(Path(checkpoint_dir).expanduser().resolve())
        # 复用 FlashRT 官方 frontend 的权重加载/标尺/style 预计算逻辑。
        self._frontend = Pi05TorchFrontendThor(
            ckpt,
            num_views=int(num_views),
            use_cuda_graph=bool(use_cuda_graph),
        )
        # 仅用于触发 style/calib 初始化；denoise 运行不依赖 prompt 语义。
        self._frontend.set_prompt("")

        self._kc_buf: torch.Tensor | None = None
        self._vc_buf: torch.Tensor | None = None
        self._noise_buf: torch.Tensor | None = None

    @staticmethod
    def _as_layer_kv(
        t: torch.Tensor,
        *,
        name: str,
        expected_layers: int,
        expected_hd: int,
    ) -> torch.Tensor:
        """把 KV 归一化成 ``[L, S, HD]``（当前只支持 B=1、Hkv=1）。"""
        if not torch.is_tensor(t):
            raise TypeError(f"{name} must be Tensor, got {type(t).__name__}")
        if t.dim() < 3:
            raise ValueError(f"{name} rank must be >=3, got shape={tuple(t.shape)}")
        if t.shape[0] != expected_layers:
            raise ValueError(
                f"{name} layer dim mismatch: expect {expected_layers}, got {t.shape[0]}"
            )
        if t.shape[-1] != expected_hd:
            raise ValueError(
                f"{name} head_dim mismatch: expect {expected_hd}, got {t.shape[-1]}"
            )
        seq = int(t.shape[-2])
        middle = t.shape[1:-2]
        flat_middle = math.prod(int(x) for x in middle) if middle else 1
        if flat_middle != 1:
            raise ValueError(
                f"{name} only supports B=1,Hkv=1; got shape={tuple(t.shape)}"
            )
        out = t.reshape(expected_layers, seq, expected_hd).contiguous()
        return out.to(dtype=torch.float16, device="cuda", non_blocking=True)

    def _ensure_buffers(self, *, layers: int, enc_seq: int, noise: torch.Tensor) -> None:
        hd = int(self._frontend.HD)
        need_kv = (layers, enc_seq, hd)
        if self._kc_buf is None or tuple(self._kc_buf.shape) != need_kv:
            self._kc_buf = torch.empty(need_kv, dtype=torch.float16, device="cuda")
            self._vc_buf = torch.empty(need_kv, dtype=torch.float16, device="cuda")

        nshape = tuple(noise.shape)
        if self._noise_buf is None or tuple(self._noise_buf.shape) != nshape:
            self._noise_buf = torch.empty(nshape, dtype=torch.float16, device="cuda")

    def run(
        self,
        input_keys: torch.Tensor,
        input_values: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """执行完整 denoise（10 steps），返回与 ``noise`` 同形状 actions。"""
        layers = int(self._frontend.La)
        hd = int(self._frontend.HD)
        k = self._as_layer_kv(
            input_keys,
            name="past_keys",
            expected_layers=layers,
            expected_hd=hd,
        )
        v = self._as_layer_kv(
            input_values,
            name="past_values",
            expected_layers=layers,
            expected_hd=hd,
        )
        enc_seq = int(k.shape[1])

        noise_fp16 = noise.to(dtype=torch.float16, device="cuda", non_blocking=True).contiguous()
        self._ensure_buffers(layers=layers, enc_seq=enc_seq, noise=noise_fp16)
        assert self._kc_buf is not None and self._vc_buf is not None and self._noise_buf is not None
        self._kc_buf.copy_(k, non_blocking=True)
        self._vc_buf.copy_(v, non_blocking=True)
        self._noise_buf.copy_(noise_fp16, non_blocking=True)

        ae_bufs = {
            "noise": self._noise_buf.data_ptr(),
            "x": self._frontend._ae_x.data_ptr(),
            "xn": self._frontend._ae_xn.data_ptr(),
            "gate": self._frontend._ae_gate.data_ptr(),
            "qkv": self._frontend._ae_qkv.data_ptr(),
            "logits": self._frontend._ae_logits.data_ptr(),
            "attn_out": self._frontend._ae_attn.data_ptr(),
            "hid": self._frontend._ae_hid.data_ptr(),
            "fg": self._frontend._ae_fg.data_ptr(),
            "xn_fp8": self._frontend._ae_xn_fp8.data_ptr(),
            "hid_fp8": self._frontend._ae_hid_fp8.data_ptr(),
            "ctx_fp8": self._frontend._ae_ctx_fp8.data_ptr(),
        }
        ae_weights = {
            "ain_w": self._frontend._ain_w.data_ptr(),
            "ain_b": self._frontend._ain_b.data_ptr(),
            "sa": self._frontend._sa_all.data_ptr(),
            "qw": self._frontend._dec_qkv_flat.data_ptr(),
            "Kc": self._kc_buf.reshape(-1).data_ptr(),
            "Vc": self._vc_buf.reshape(-1).data_ptr(),
            "ow": self._frontend._dec_o_flat.data_ptr(),
            "sf": self._frontend._sf_all.data_ptr(),
            "gw": self._frontend._dec_gu_flat.data_ptr(),
            "dw": self._frontend._dec_d_flat.data_ptr(),
            "aow": self._frontend._aow.data_ptr(),
            "aob": self._frontend._aob.data_ptr(),
            "fs": self._frontend._fs_all.data_ptr(),
            "rope": self._frontend._dec_rope.data_ptr(),
            "w_scales": self._frontend._ae_w_dev.data_ptr(),
            "act_scales": self._frontend._ae_calib_scales.data_ptr(),
        }
        ae_dims = {
            "S": int(self._frontend.Sa),
            "D": int(self._frontend.Da),
            "H": int(self._frontend.Ha),
            "NH": 8,
            "HD": hd,
            "steps": 10,
            "layers": layers,
            "enc_seq": enc_seq,
            "total_keys": int(enc_seq + int(self._frontend.Sa)),
        }
        decoder_forward(
            self._frontend._ctx,
            fvk,
            ae_bufs,
            ae_weights,
            ae_dims,
            stream=0,
            attn=None,
        )
        return self._noise_buf.to(dtype=noise.dtype).view_as(noise)

