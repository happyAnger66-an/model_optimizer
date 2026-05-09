# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashRT Thor 风格 SigLIP 视觉栈（多视角 + CUDA Graph），供 TRT 执行器混合 ``embed_prefix`` 使用。

依赖已安装的 ``flash_rt``（与 ``third_party/FlashRT`` 或 pip 同源）。仅实现 **patch + 27 层 SigLIP + PostLN→encoder 维**；
语言嵌入由调用方（如 ``pi05_executor``）用 PyTorch 拼接。

**Batch 约定（v1）**：每路 ``images[i]`` 形状 ``[B, 3, 224, 224]``，当前要求 **B=1**（与 FlashRT Thor B=1 主路径一致）；**B>1** 需后续接 ``encoder_forward_b2`` 类接口。
"""

from __future__ import annotations

import ctypes
import logging
import math
import pathlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass

try:
    import flash_rt.flash_rt_kernels as fvk
    from flash_rt.core.cuda_buffer import CudaBuffer
    from flash_rt.executors.torch_weights import (
        SafetensorsSource,
        WeightLoader,
        _autodetect_strip_prefix,  # noqa: SLF001 — public loader helper
    )
    from flash_rt.executors.weight_loader import ModelWeightSpec
    from flash_rt.frontends.torch._thor_spec_common import paligemma_siglip_block
    from flash_rt.hardware.thor.shared_primitives import postln_project, siglip_forward
except ImportError as e:  # pragma: no cover
    fvk = None  # type: ignore[misc, assignment]
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None

fp16 = torch.float16


def _require_flash_rt() -> None:
    if _IMPORT_ERR is not None:
        raise ImportError(
            "FlashRT (``flash_rt``) is required for infer.flash.siglip. "
            "Install the FlashRT package or add it to PYTHONPATH."
        ) from _IMPORT_ERR


class FlashRtSiglipVision:
    """多视角 SigLIP + PostLN/投影（无语言槽），可选 CUDA Graph。

    与 ``Pi05TorchFrontendThor._patch_embed_ops`` / ``siglip_forward`` /
    ``_postln_project_ops`` 同构；``postln_project`` 使用 ``S_lang=0``，
    不写语言区，仅将 SigLIP 输出投到 ``D_enc``。
    """

    def __init__(
        self,
        checkpoint_dir: str | pathlib.Path,
        *,
        num_views: int,
        use_cuda_graph: bool = True,
    ) -> None:
        _require_flash_rt()
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.num_views = int(num_views)
        self.use_cuda_graph = bool(use_cuda_graph)
        self._ctx = fvk.FvkContext()
        self._gemm = fvk.GemmRunner()
        self._siglip_graph: Optional[torch.cuda.CUDAGraph] = None
        self._siglip_graph_stream: Optional[torch.cuda.Stream] = None

        safetensors_path = self.checkpoint_dir / "model.safetensors"
        if not safetensors_path.is_file():
            raise FileNotFoundError(
                f"Expected model weights at {safetensors_path} "
                f"(FlashRtSiglipVision loads SigLIP from safetensors)."
            )
        self._load_fmha_optional()
        self._load_weights(safetensors_path)
        self._allocate_runtime_buffers()
        if self.use_cuda_graph:
            self._capture_siglip_graph()

    def _load_fmha_optional(self) -> None:
        ck = self.checkpoint_dir
        fmha_paths = [
            str(ck.parent / "libfmha_fp16_strided.so"),
            str(pathlib.Path(fvk.__file__).resolve().parent / "libfmha_fp16_strided.so"),
            str(
                pathlib.Path(fvk.__file__).resolve().parent.parent.parent
                / "build"
                / "libfmha_fp16_strided.so"
            ),
            "/workspace/libfmha_fp16_strided.so",
        ]
        for p in fmha_paths:
            if pathlib.Path(p).is_file():
                if fvk.load_fmha_strided_library(p) == 0:
                    logger.info("FlashRtSiglipVision: FMHA loaded from %s", p)
                    return
        logger.warning(
            "FlashRtSiglipVision: libfmha_fp16_strided.so not found — "
            "SigLIP will use cuBLAS attention fallback inside fvk."
        )

    def _load_weights(self, safetensors_path: pathlib.Path) -> None:
        from safetensors import safe_open

        spec = ModelWeightSpec(framework="torch", blocks=[paligemma_siglip_block()])
        tgt = SimpleNamespace()
        with safe_open(str(safetensors_path), framework="pt", device="cuda") as sf:
            _strip = _autodetect_strip_prefix(set(sf.keys()))
        _src = SafetensorsSource(str(safetensors_path), device="cuda")
        WeightLoader(source=_src, target=tgt, spec=spec).run()

        # --- Patch embed (HWC im2col order) — same as Pi05TorchFrontendThor ---
        vp = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"

        def g(k: str) -> torch.Tensor:
            k2 = (_strip + k) if _strip else k
            return self._open_st_safe(safetensors_path, k2).to(fp16)

        D_sig = 1152
        pe_w_2d = (
            g(f"{vp}.embeddings.patch_embedding.weight")
            .reshape(D_sig, 3, 14, 14)
            .permute(0, 2, 3, 1)
            .reshape(D_sig, -1)
            .T.contiguous()
        )
        self._pe_w = CudaBuffer.from_numpy(pe_w_2d.cpu().numpy().copy())
        self._pe_b = CudaBuffer.from_numpy(
            g(f"{vp}.embeddings.patch_embedding.bias").cpu().numpy().copy()
        )
        self._pos_emb = CudaBuffer.from_numpy(
            g(f"{vp}.embeddings.position_embedding.weight")[:256].cpu().numpy().copy()
        )

        mp = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
        self._postln_w = g(f"{vp}.post_layernorm.weight")
        self._postln_b = g(f"{vp}.post_layernorm.bias")
        self._proj_w = g(f"{mp}.weight").T.contiguous()
        self._proj_b = g(f"{mp}.bias")

        nv = self.num_views
        S_sig = nv * 256
        H_sig = 4304
        self.sig_S = S_sig
        self.sig_D = D_sig
        self.sig_H = H_sig
        self.sig_NH = 16
        self.sig_HD = 72
        self.sig_L = 27
        self.De = 2048

        self._sig_ln_attn_w = tgt._sig_ln_attn_w
        self._sig_ln_attn_b = tgt._sig_ln_attn_b
        self._sig_qkv_w = tgt._sig_qkv_w
        self._sig_qkv_b = tgt._sig_qkv_b
        self._sig_o_w = tgt._sig_o_w
        self._sig_o_b = tgt._sig_o_b
        self._sig_ln_ffn_w = tgt._sig_ln_ffn_w
        self._sig_ln_ffn_b = tgt._sig_ln_ffn_b
        self._sig_up_w = tgt._sig_up_w
        self._sig_up_b = tgt._sig_up_b
        self._sig_down_w = tgt._sig_down_w
        self._sig_down_b = tgt._sig_down_b
        self._sig_alpha = tgt._sig_alpha

        self._unit_scale = torch.ones(1, dtype=torch.float32, device="cuda")
        self._sig_weights = {
            "ln_attn_w": [w.data_ptr() for w in self._sig_ln_attn_w],
            "ln_attn_b": [w.data_ptr() for w in self._sig_ln_attn_b],
            "qkv_w": [w.data_ptr() for w in self._sig_qkv_w],
            "qkv_b": [w.data_ptr() for w in self._sig_qkv_b],
            "o_w": [w.data_ptr() for w in self._sig_o_w],
            "o_b": [w.data_ptr() for w in self._sig_o_b],
            "ln_ffn_w": [w.data_ptr() for w in self._sig_ln_ffn_w],
            "ln_ffn_b": [w.data_ptr() for w in self._sig_ln_ffn_b],
            "up_w": [w.data_ptr() for w in self._sig_up_w],
            "up_b": [w.data_ptr() for w in self._sig_up_b],
            "down_w": [w.data_ptr() for w in self._sig_down_w],
            "down_b": [w.data_ptr() for w in self._sig_down_b],
            "alpha": self._sig_alpha,
            "unit_scale": self._unit_scale.data_ptr(),
        }

    @staticmethod
    def _open_st_safe(path: pathlib.Path, key: str) -> torch.Tensor:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cuda") as sf:
            return sf.get_tensor(key)

    def _allocate_runtime_buffers(self) -> None:
        nv = self.num_views
        S_sig, D_sig, H_sig = self.sig_S, self.sig_D, self.sig_H
        De = self.De

        self._img_buf = CudaBuffer.device_empty(nv * 224 * 224 * 3, np.float16)
        self._patches_buf = CudaBuffer.device_empty(S_sig * 588, np.float16)

        self._sig_x = torch.zeros(S_sig, D_sig, dtype=fp16, device="cuda")
        self._sig_x_fp8 = torch.zeros(S_sig * D_sig, dtype=torch.uint8, device="cuda")
        self._sig_qkv = torch.empty(S_sig, 3 * D_sig, dtype=fp16, device="cuda")
        self._sig_attn = torch.empty(S_sig, D_sig, dtype=fp16, device="cuda")
        self._sig_hidden = torch.empty(S_sig, H_sig, dtype=fp16, device="cuda")
        self._sig_hid_fp8 = torch.zeros(S_sig * H_sig, dtype=torch.uint8, device="cuda")

        self._sig_bufs = {
            "x": self._sig_x.data_ptr(),
            "x_fp8": self._sig_x_fp8.data_ptr(),
            "qkv": self._sig_qkv.data_ptr(),
            "attn_out": self._sig_attn.data_ptr(),
            "hidden": self._sig_hidden.data_ptr(),
            "hid_fp8": self._sig_hid_fp8.data_ptr(),
        }
        self._sig_dims = {
            "S": S_sig,
            "D": D_sig,
            "H": H_sig,
            "NH": self.sig_NH,
            "HD": self.sig_HD,
            "L": self.sig_L,
            "num_views": nv,
            "seq_per_view": 256,
        }

        self._enc_vision = torch.empty(S_sig, De, dtype=fp16, device="cuda")
        self._postln_scratch = torch.empty(S_sig, max(D_sig, H_sig), dtype=fp16, device="cuda")
        self._dummy_lang = torch.zeros(1, De, dtype=fp16, device="cuda")

    def _patch_embed_ops(self, stream_int: int) -> None:
        S_sig, D_sig = self.sig_S, self.sig_D
        fvk.patch_im2col(
            self._img_buf.ptr.value,
            self._patches_buf.ptr.value,
            self.num_views,
            stream_int,
        )
        self._gemm.fp16_nn(
            self._patches_buf.ptr.value,
            self._pe_w.ptr.value,
            self._sig_x.data_ptr(),
            S_sig,
            D_sig,
            588,
            stream_int,
        )
        fvk.patch_embed_bias_pos(
            self._sig_x.data_ptr(),
            self._pe_b.ptr.value,
            self._pos_emb.ptr.value,
            S_sig,
            D_sig,
            256,
            stream_int,
        )

    def _postln_project_ops(self, stream_int: int) -> None:
        S_sig, D_sig, De = self.sig_S, self.sig_D, self.De
        postln_bufs = {
            "x_sig": self._sig_x.data_ptr(),
            "enc_x": self._enc_vision.data_ptr(),
            "scratch": self._postln_scratch.data_ptr(),
        }
        postln_weights = {
            "ln_w": self._postln_w.data_ptr(),
            "ln_b": self._postln_b.data_ptr(),
            "proj_w": self._proj_w.data_ptr(),
            "proj_b": self._proj_b.data_ptr(),
            "lang_emb": self._dummy_lang.data_ptr(),
        }
        postln_dims = {"S_sig": S_sig, "D_sig": D_sig, "D_enc": De, "S_lang": 0}
        postln_project(self._gemm, fvk, postln_bufs, postln_weights, postln_dims, stream=stream_int)

    def _capture_siglip_graph(self) -> None:
        dummy_img = np.zeros((self.num_views, 224, 224, 3), dtype=np.float16)
        self._img_buf.upload(dummy_img)
        for _ in range(3):
            self._patch_embed_ops(0)
            self._sig_x.zero_()
            siglip_forward(self._gemm, fvk, self._sig_bufs, self._sig_weights, self._sig_dims, stream=0, attn=None)
            self._postln_project_ops(0)
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        self._siglip_graph = torch.cuda.CUDAGraph()
        s_int = stream.cuda_stream
        with torch.cuda.stream(stream):
            self._siglip_graph.capture_begin()
            self._patch_embed_ops(s_int)
            siglip_forward(
                self._gemm,
                fvk,
                self._sig_bufs,
                self._sig_weights,
                self._sig_dims,
                stream=s_int,
                attn=None,
            )
            self._postln_project_ops(s_int)
            self._siglip_graph.capture_end()
        torch.cuda.synchronize()
        self._siglip_graph_stream = stream
        logger.info("FlashRtSiglipVision: CUDA graph captured (S_sig=%d)", self.sig_S)

    def upload_images_nhwc_fp16(self, img_nhwc: np.ndarray) -> None:
        """``img_nhwc`` 形状 ``(num_views, 224, 224, 3)``，dtype float16，行主序 NHWC。"""
        if img_nhwc.shape != (self.num_views, 224, 224, 3):
            raise ValueError(
                f"Expected image shape ({self.num_views}, 224, 224, 3) fp16, got {img_nhwc.shape}"
            )
        if img_nhwc.dtype != np.float16:
            raise TypeError("upload_images_nhwc_fp16 expects np.float16")
        self._img_buf.upload(np.ascontiguousarray(img_nhwc))

    def forward_graph(self) -> torch.Tensor:
        """在已 ``upload_images_nhwc_fp16`` 之后重放图（或 eager 路径）。"""
        if self._siglip_graph is not None and self._siglip_graph_stream is not None:
            self._siglip_graph.replay()
            torch.cuda.synchronize()
        else:
            self._patch_embed_ops(0)
            siglip_forward(self._gemm, fvk, self._sig_bufs, self._sig_weights, self._sig_dims, stream=0, attn=None)
            self._postln_project_ops(0)
            torch.cuda.synchronize()
        return self._enc_vision

    def forward_from_torch_images(self, images: list[torch.Tensor]) -> torch.Tensor:
        """``images``: 长度 ``num_views``，每项 ``[1, 3, 224, 224]`` fp16/bf16/fp32（CHW）。

        返回 ``[S_sig, D_enc]`` fp16（``S_sig = num_views * 256``）。
        """
        if len(images) != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {len(images)}")
        nv = self.num_views
        planes = []
        for im in images:
            if im.dim() != 4 or im.shape[0] != 1 or im.shape[1] != 3:
                raise ValueError(f"Each image must be [1,3,224,224], got {tuple(im.shape)}")
            if im.shape[2] != 224 or im.shape[3] != 224:
                raise ValueError(f"Expected 224x224 images, got {tuple(im.shape)}")
            x = im[0].to(dtype=torch.float16, device="cuda", non_blocking=True).contiguous()
            # CHW -> HWC for patch_im2col
            x = x.permute(1, 2, 0).contiguous()
            planes.append(x.cpu().numpy())  # small; nv<=3
        nhwc = np.stack(planes, axis=0).astype(np.float16, copy=False)
        self.upload_images_nhwc_fp16(nhwc)
        return self.forward_graph()


__all__ = ["FlashRtSiglipVision"]
