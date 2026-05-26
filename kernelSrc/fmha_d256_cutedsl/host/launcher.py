# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Host-side ``@cute.jit`` launcher for d=256 FMHA."""

from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Int32, Int64, Float32

from fmha_d256_cutedsl import fmha_helpers as fmha_utils
from fmha_d256_cutedsl import prefill_helpers as prefill_utils


@cute.jit
def launch(
    self,
    q_iter: cute.Pointer,
    k_iter: cute.Pointer,
    v_iter: cute.Pointer,
    o_iter: cute.Pointer,
    scale_k_iter: cute.Pointer,
    scale_v_iter: cute.Pointer,
    problem_shape: Tuple[Int32, Int32, Int32, Int32, Int32, Int32],
    scale_softmax_log2: Float32,
    scale_output: Float32,
    window_size_left: Optional[Int32],
    window_size_right: Optional[Int32],
    stream: cuda.CUstream,
):
        self._setup_attributes()
        b, s_q, s_k, h_q, h_k, d = problem_shape
        h_r = h_q // h_k
        self.d_r = self.cta_tiler[2] // self.scale_granularity
        # (s, d, ((h_r, h_k), b))
        q_layout = cute.make_layout(
            (s_q, d, ((h_r, h_k), b)),
            stride=(d, 1, ((d * s_q, d * s_q * h_r), h_r * h_k * s_q * d)),
        )
        q = cute.make_tensor(q_iter, q_layout)
        # (s, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        k_layout = cute.make_layout(
            (s_k, d, ((h_r, h_k), b)),
            stride=(d, 1, ((0, d * s_k), h_k * s_k * d)),
        )
        k = cute.make_tensor(k_iter, k_layout)
        # (d, s, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        v_layout = cute.make_layout(
            (d, s_k, ((h_r, h_k), b)),
            stride=(1, d, ((0, d * s_k), h_k * s_k * d)),
        )
        v = cute.make_tensor(v_iter, v_layout)
        # (s, d, ((h_r, h_k), b))
        # set divby for better gmem store vectorization
        o_layout = cute.make_layout(
            (s_q, d, ((h_r, h_k), b)),
            stride=(
                cute.assume(d, divby=256),
                1,
                (
                    (
                        cute.assume(d * s_q, divby=256),
                        cute.assume(d * s_q * h_r, divby=256),
                    ),
                    cute.assume(h_r * h_k * s_q * d, divby=256),
                ),
            ),
        )
        o = cute.make_tensor(o_iter, o_layout)
        # (d_r * s, ((h_r, h_k), b))
        scale_k_layout = cute.make_layout(
            (s_k * self.d_r, ((h_r, h_k), b)),
            stride=(1, ((0, self.d_r * s_k), s_k * self.d_r * h_k)),
        )
        scale_k = cute.make_tensor(scale_k_iter, scale_k_layout)
        # (d_r * s, ((h_r, h_k), b))
        scale_v_layout = cute.make_layout(
            (self.d_r * s_k, ((h_r, h_k), b)),
            stride=(1, ((0, self.d_r * s_k), s_k * self.d_r * h_k)),
        )
        scale_v = cute.make_tensor(scale_v_iter, scale_v_layout)

        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type
        self.p_dtype = self.q_dtype  # pv should has the same dtype
        self.tilePlikeFP32 = self.qk_mma_tiler[1] // Float32.width * self.p_dtype.width
        self.scale_k_dtype = scale_k.element_type
        self.scale_v_dtype = scale_v.element_type

        self.tile_sched_params, grid = fmha_utils.compute_grid(
            o.shape,
            self.cta_tiler,
            self.is_persistent,
        )

        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.o_layout = utils.LayoutEnum.from_tensor(o)
        cta_group = tcgen05.CtaGroup.TWO
        p_major_mode = tcgen05.OperandMajorMode.K
        p_source = tcgen05.OperandSource.TMEM
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            self.pv_mma_tiler[:2],
            p_source,
        )
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )
        self.epi_tile = self.pv_block_tiler[:2]

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.kv_stage,
        )
        k_smem_layout_staged = cute.make_composed_layout(
            cute.make_swizzle(0, 4, 3), 0, k_smem_layout_staged.outer
        )
        k_trans_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.kv_trans_stage,
        )
        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.p_dtype,
            self.qk_acc_stage,
        )
        p_tmem_layout = cute.select(p_tmem_layout_staged, mode=[0, 1, 2])
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.q_dtype,
            self.kv_stage,
        )
        v_smem_layout_staged = cute.make_composed_layout(
            cute.make_swizzle(0, 4, 3), 0, v_smem_layout_staged.outer
        )
        v_trans_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.pv_mma_tiler,
            self.q_dtype,
            self.kv_trans_stage,
        )
        scale_k_smem_layout, self.scale_k_tiler, scale_k_s2r_view_layout = (
            prefill_utils.get_scale_smem_layout(
                self.scale_granularity,
                self.d_r,
                self.qk_mma_tiler,
                self.k_major_mode,
            )
        )
        scale_k_smem_layout_staged = cute.append(
            scale_k_smem_layout,
            cute.make_layout(
                (self.scale_k_stage),
                stride=(cute.cosize(scale_k_smem_layout.outer)),
            ),
        )
        scale_k_s2r_view_layout_staged = cute.append(
            scale_k_s2r_view_layout,
            cute.make_layout(
                (self.scale_k_stage),
                stride=(cute.cosize(scale_k_s2r_view_layout)),
            ),
        )
        scale_v_smem_layout, self.scale_v_tiler, scale_v_s2r_view_layout = (
            prefill_utils.get_scale_smem_layout(
                self.scale_granularity,
                self.d_r,
                self.pv_mma_tiler,
                self.v_major_mode,
            )
        )
        scale_v_smem_layout_staged = cute.append(
            scale_v_smem_layout,
            cute.make_layout(
                (self.scale_v_stage),
                stride=(cute.cosize(scale_v_smem_layout.outer)),
            ),
        )
        scale_v_s2r_view_layout_staged = cute.append(
            scale_v_s2r_view_layout,
            cute.make_layout(
                (self.scale_v_stage),
                stride=(cute.cosize(scale_v_s2r_view_layout)),
            ),
        )

        tma_load_q_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
        # For TMA Async, use one cta to sync with corresponding cta only
        tma_load_kv_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(
            tcgen05.CtaGroup.ONE
        )
        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_q_op,
            q,
            q_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for K
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_kv_op,
            k,
            k_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_scale_k, tma_tensor_scale_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_kv_op,
            scale_k,
            scale_k_smem_layout,
            (self.scale_k_tiler[0] // 2,),
        )

        # TMA load for V
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_kv_op,
            v,
            v_smem_layout,
            self.pv_mma_tiler,
            pv_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_scale_v, tma_tensor_scale_v = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_kv_op,
            scale_v,
            scale_v_smem_layout,
            self.scale_v_tiler,
        )

        self.tma_copy_q_bytes = cute.size_in_bytes(
            self.q_dtype, q_smem_layout
        ) * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_kv_bytes = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        self.tma_copy_scale_k_bytes = cute.size_in_bytes(
            self.scale_k_dtype, scale_k_smem_layout
        )
        self.tma_copy_scale_v_bytes = cute.size_in_bytes(
            self.scale_v_dtype, scale_v_smem_layout
        )

        @cute.struct
        class SharedStorage:
            # Pipeline barriers
            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]
            load_kv_mbar_ptr: cute.struct.MemRange[Int64, self.kv_stage * 2]
            load_scale_k_mbar_ptr: cute.struct.MemRange[Int64, self.scale_k_stage * 2]
            load_scale_v_mbar_ptr: cute.struct.MemRange[Int64, self.scale_v_stage * 2]
            dequant_kv_mbar_ptr: cute.struct.MemRange[Int64, self.kv_trans_stage * 2]
            mma_s_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            p_mma_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            s_corr_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            sum_mbar_ptr: cute.struct.MemRange[Int64, 2]
            mma_o_mbar_ptr: cute.struct.MemRange[Int64, self.pv_acc_stage * 2]
            tmem_dealloc_mbar: Int64
            tmem_holding_buf: Int32

        self.shared_storage = SharedStorage

        grid = cute.round_up(grid, self.cluster_shape_mnk)

        # Launch the kernel synchronously
        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_scale_k,
            tma_tensor_scale_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_scale_v,
            tma_tensor_scale_v,
            o,
            scale_softmax_log2,
            scale_output,
            window_size_left,
            window_size_right,
            self.cluster_layout_vmnk,
            q_smem_layout_staged,
            k_smem_layout_staged,
            k_trans_smem_layout_staged,
            scale_k_smem_layout_staged,
            scale_k_s2r_view_layout_staged,
            p_tmem_layout,
            v_smem_layout_staged,
            v_trans_smem_layout_staged,
            scale_v_smem_layout_staged,
            scale_v_s2r_view_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            None,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )


@cute.jit
def launch_homo(
    self,
    q_iter: cute.Pointer,
    k_iter: cute.Pointer,
    v_iter: cute.Pointer,
    o_iter: cute.Pointer,
    problem_shape: Tuple[Int32, Int32, Int32, Int32, Int32, Int32],
    scale_softmax_log2: Float32,
    scale_output: Float32,
    window_size_left: Optional[Int32],
    window_size_right: Optional[Int32],
    stream: cuda.CUstream,
    cum_seqlen_k: Optional[cute.Tensor] = None,
):
    """Launch homogeneous Q/K/V dtype kernel (no scale tensors)."""
    self._setup_attributes()
    b, s_q, s_k, h_q, h_k, d = problem_shape
    h_r = h_q // h_k
    q_layout = cute.make_layout(
        (s_q, d, ((h_r, h_k), b)),
        stride=(d, 1, ((d * s_q, d * s_q * h_r), h_r * h_k * s_q * d)),
    )
    q = cute.make_tensor(q_iter, q_layout)
    k_layout = cute.make_layout(
        (s_k, d, ((h_r, h_k), b)),
        stride=(d, 1, ((0, d * s_k), h_k * s_k * d)),
    )
    k = cute.make_tensor(k_iter, k_layout)
    v_layout = cute.make_layout(
        (d, s_k, ((h_r, h_k), b)),
        stride=(1, d, ((0, d * s_k), h_k * s_k * d)),
    )
    v = cute.make_tensor(v_iter, v_layout)
    o_layout = cute.make_layout(
        (s_q, d, ((h_r, h_k), b)),
        stride=(
            cute.assume(d, divby=256),
            1,
            (
                (
                    cute.assume(d * s_q, divby=256),
                    cute.assume(d * s_q * h_r, divby=256),
                ),
                cute.assume(h_r * h_k * s_q * d, divby=256),
            ),
        ),
    )
    o = cute.make_tensor(o_iter, o_layout)

    self.q_dtype = q.element_type
    self.k_dtype = k.element_type
    self.v_dtype = v.element_type
    self.o_dtype = o.element_type
    self.p_dtype = self.q_dtype
    self.tilePlikeFP32 = self.qk_mma_tiler[1] // Float32.width * self.p_dtype.width

    self.tile_sched_params, grid = fmha_utils.compute_grid(
        o.shape,
        self.cta_tiler,
        self.is_persistent,
    )

    self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
    self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
    self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
    self.o_layout = utils.LayoutEnum.from_tensor(o)
    cta_group = tcgen05.CtaGroup.TWO
    p_major_mode = tcgen05.OperandMajorMode.K
    p_source = tcgen05.OperandSource.TMEM
    qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        self.q_dtype,
        self.q_major_mode,
        self.k_major_mode,
        self.qk_acc_dtype,
        cta_group,
        self.qk_mma_tiler[:2],
    )
    pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        self.q_dtype,
        p_major_mode,
        self.v_major_mode,
        self.pv_acc_dtype,
        cta_group,
        self.pv_mma_tiler[:2],
        p_source,
    )
    self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
    self.cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout(self.cluster_shape_mnk),
        (qk_tiled_mma.thr_id.shape,),
    )
    self.epi_tile = self.pv_block_tiler[:2]

    q_smem_layout_staged = sm100_utils.make_smem_layout_a(
        qk_tiled_mma, self.qk_mma_tiler, self.q_dtype, self.q_stage,
    )
    k_smem_layout_staged = sm100_utils.make_smem_layout_b(
        qk_tiled_mma, self.qk_mma_tiler, self.q_dtype, self.kv_stage,
    )
    k_smem_layout_staged = cute.make_composed_layout(
        cute.make_swizzle(0, 4, 3), 0, k_smem_layout_staged.outer
    )
    k_trans_smem_layout_staged = sm100_utils.make_smem_layout_b(
        qk_tiled_mma, self.qk_mma_tiler, self.q_dtype, self.kv_trans_stage,
    )
    p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
        pv_tiled_mma, self.pv_mma_tiler, self.p_dtype, self.qk_acc_stage,
    )
    p_tmem_layout = cute.select(p_tmem_layout_staged, mode=[0, 1, 2])
    v_smem_layout_staged = sm100_utils.make_smem_layout_b(
        pv_tiled_mma, self.pv_mma_tiler, self.q_dtype, self.kv_stage,
    )
    v_smem_layout_staged = cute.make_composed_layout(
        cute.make_swizzle(0, 4, 3), 0, v_smem_layout_staged.outer
    )
    v_trans_smem_layout_staged = sm100_utils.make_smem_layout_b(
        pv_tiled_mma, self.pv_mma_tiler, self.q_dtype, self.kv_trans_stage,
    )

    tma_load_q_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
    tma_load_kv_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
        tma_load_q_op, q, q_smem_layout, self.qk_mma_tiler, qk_tiled_mma,
        self.cluster_layout_vmnk.shape,
    )
    k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
        tma_load_kv_op, k, k_smem_layout, self.qk_mma_tiler, qk_tiled_mma,
        self.cluster_layout_vmnk.shape,
    )
    v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
        tma_load_kv_op, v, v_smem_layout, self.pv_mma_tiler, pv_tiled_mma,
        self.cluster_layout_vmnk.shape,
    )

    self.tma_copy_q_bytes = cute.size_in_bytes(
        self.q_dtype, q_smem_layout
    ) * cute.size(qk_tiled_mma.thr_id.shape)
    self.tma_copy_kv_bytes = cute.size_in_bytes(self.q_dtype, k_smem_layout)

    @cute.struct
    class SharedStorageHomo:
        load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]
        load_kv_mbar_ptr: cute.struct.MemRange[Int64, self.kv_stage * 2]
        dequant_kv_mbar_ptr: cute.struct.MemRange[Int64, self.kv_trans_stage * 2]
        mma_s_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
        p_mma_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
        s_corr_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
        sum_mbar_ptr: cute.struct.MemRange[Int64, 2]
        mma_o_mbar_ptr: cute.struct.MemRange[Int64, self.pv_acc_stage * 2]
        tmem_dealloc_mbar: Int64
        tmem_holding_buf: Int32

    self.shared_storage = SharedStorageHomo
    grid = cute.round_up(grid, self.cluster_shape_mnk)

    self.kernel_homo(
        qk_tiled_mma,
        pv_tiled_mma,
        tma_atom_q,
        tma_tensor_q,
        tma_atom_k,
        tma_tensor_k,
        tma_atom_v,
        tma_tensor_v,
        o,
        scale_softmax_log2,
        scale_output,
        window_size_left,
        window_size_right,
        self.cluster_layout_vmnk,
        q_smem_layout_staged,
        k_smem_layout_staged,
        k_trans_smem_layout_staged,
        p_tmem_layout,
        v_smem_layout_staged,
        v_trans_smem_layout_staged,
        self.epi_tile,
        self.tile_sched_params,
        cum_seqlen_k,
    ).launch(
        grid=grid,
        block=[self.threads_per_cta, 1, 1],
        cluster=self.cluster_shape_mnk,
        stream=stream,
        min_blocks_per_mp=1,
    )


@cute.jit
def _call_llm(
    self,
    q_tensor: cute.Tensor,
    kv_cache: cute.Tensor,
    o_tensor: cute.Tensor,
    cum_seqlen_k: cute.Tensor,
    window_size_left: Optional[Int32],
    scale_q: Float32,
    scale_k: Float32,
    scale_v: Float32,
    inv_scale_o: Float32,
    stream: cuda.CUstream,
):
    """LLM entry: BSHD Q/O + packed KV cache ``(B, 2, H_k, cap, D)``."""
    self._setup_attributes()
    scale_softmax = scale_q * scale_k * Float32(self.inv_sqrt_head_dim)
    scale_softmax_log2 = scale_softmax * Float32(self.log2_e)
    scale_output = scale_v * inv_scale_o

    b = q_tensor.layout.shape[0]
    s_q = q_tensor.layout.shape[1]
    h_q = q_tensor.layout.shape[2]
    h_k = kv_cache.layout.shape[2]
    cap = kv_cache.layout.shape[3]
    d = Int32(self.head_dim)
    h_r = h_q // h_k

    q_iter = q_tensor.iterator
    kv_base = kv_cache.iterator
    stride_kv_head = cap * d
    stride_kv_select = h_k * stride_kv_head
    k_iter = kv_base
    v_iter = kv_base + stride_kv_select
    o_iter = o_tensor.iterator

    stride_b_qo = h_r * h_k * s_q * d
    stride_b_kv = 2 * stride_kv_select

    q_layout = cute.make_layout(
        (s_q, d, ((h_r, h_k), b)),
        stride=(d * h_r * h_k, 1, ((d, d * h_r), stride_b_qo)),
    )
    q = cute.make_tensor(q_iter, q_layout)
    k_layout = cute.make_layout(
        (cap, d, ((h_r, h_k), b)),
        stride=(d, 1, ((0, stride_kv_head), stride_b_kv)),
    )
    k = cute.make_tensor(k_iter, k_layout)
    v_layout = cute.make_layout(
        (d, cap, ((h_r, h_k), b)),
        stride=(1, d, ((0, stride_kv_head), stride_b_kv)),
    )
    v = cute.make_tensor(v_iter, v_layout)
    o_layout = cute.make_layout(
        (s_q, d, ((h_r, h_k), b)),
        stride=(d * h_r * h_k, 1, ((d, d * h_r), stride_b_qo)),
    )
    o = cute.make_tensor(o_iter, o_layout)

    self.q_dtype = q.element_type
    self.k_dtype = k.element_type
    self.v_dtype = v.element_type
    self.o_dtype = o.element_type
    self.p_dtype = self.q_dtype
    self.tilePlikeFP32 = self.qk_mma_tiler[1] // Float32.width * self.p_dtype.width

    self.tile_sched_params, grid = fmha_utils.compute_grid(
        o.shape, self.cta_tiler, self.is_persistent,
    )

    self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
    self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
    self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
    self.o_layout = utils.LayoutEnum.from_tensor(o)
    cta_group = tcgen05.CtaGroup.TWO
    p_major_mode = tcgen05.OperandMajorMode.K
    p_source = tcgen05.OperandSource.TMEM
    qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        self.q_dtype, self.q_major_mode, self.k_major_mode,
        self.qk_acc_dtype, cta_group, self.qk_mma_tiler[:2],
    )
    pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
        self.q_dtype, p_major_mode, self.v_major_mode,
        self.pv_acc_dtype, cta_group, self.pv_mma_tiler[:2], p_source,
    )
    self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
    self.cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout(self.cluster_shape_mnk), (qk_tiled_mma.thr_id.shape,),
    )
    self.epi_tile = self.pv_block_tiler[:2]

    q_smem_layout_staged = sm100_utils.make_smem_layout_a(
        qk_tiled_mma, self.qk_mma_tiler, self.q_dtype, self.q_stage,
    )
    k_smem_layout_staged = sm100_utils.make_smem_layout_b(
        qk_tiled_mma, self.qk_mma_tiler, self.q_dtype, self.kv_stage,
    )
    k_smem_layout_staged = cute.make_composed_layout(
        cute.make_swizzle(0, 4, 3), 0, k_smem_layout_staged.outer
    )
    k_trans_smem_layout_staged = sm100_utils.make_smem_layout_b(
        qk_tiled_mma, self.qk_mma_tiler, self.q_dtype, self.kv_trans_stage,
    )
    p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
        pv_tiled_mma, self.pv_mma_tiler, self.p_dtype, self.qk_acc_stage,
    )
    p_tmem_layout = cute.select(p_tmem_layout_staged, mode=[0, 1, 2])
    v_smem_layout_staged = sm100_utils.make_smem_layout_b(
        pv_tiled_mma, self.pv_mma_tiler, self.q_dtype, self.kv_stage,
    )
    v_smem_layout_staged = cute.make_composed_layout(
        cute.make_swizzle(0, 4, 3), 0, v_smem_layout_staged.outer
    )
    v_trans_smem_layout_staged = sm100_utils.make_smem_layout_b(
        pv_tiled_mma, self.pv_mma_tiler, self.q_dtype, self.kv_trans_stage,
    )

    tma_load_q_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
    tma_load_kv_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
        tma_load_q_op, q, q_smem_layout, self.qk_mma_tiler, qk_tiled_mma,
        self.cluster_layout_vmnk.shape,
    )
    k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
        tma_load_kv_op, k, k_smem_layout, self.qk_mma_tiler, qk_tiled_mma,
        self.cluster_layout_vmnk.shape,
    )
    v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
    tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
        tma_load_kv_op, v, v_smem_layout, self.pv_mma_tiler, pv_tiled_mma,
        self.cluster_layout_vmnk.shape,
    )

    self.tma_copy_q_bytes = cute.size_in_bytes(
        self.q_dtype, q_smem_layout
    ) * cute.size(qk_tiled_mma.thr_id.shape)
    self.tma_copy_kv_bytes = cute.size_in_bytes(self.q_dtype, k_smem_layout)

    @cute.struct
    class SharedStorageHomo:
        load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]
        load_kv_mbar_ptr: cute.struct.MemRange[Int64, self.kv_stage * 2]
        dequant_kv_mbar_ptr: cute.struct.MemRange[Int64, self.kv_trans_stage * 2]
        mma_s_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
        p_mma_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
        s_corr_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
        sum_mbar_ptr: cute.struct.MemRange[Int64, 2]
        mma_o_mbar_ptr: cute.struct.MemRange[Int64, self.pv_acc_stage * 2]
        tmem_dealloc_mbar: Int64
        tmem_holding_buf: Int32

    self.shared_storage = SharedStorageHomo
    grid = cute.round_up(grid, self.cluster_shape_mnk)

    self.kernel_homo(
        qk_tiled_mma, pv_tiled_mma,
        tma_atom_q, tma_tensor_q, tma_atom_k, tma_tensor_k,
        tma_atom_v, tma_tensor_v, o,
        scale_softmax_log2, scale_output,
        window_size_left, None,
        self.cluster_layout_vmnk,
        q_smem_layout_staged, k_smem_layout_staged, k_trans_smem_layout_staged,
        p_tmem_layout, v_smem_layout_staged, v_trans_smem_layout_staged,
        self.epi_tile, self.tile_sched_params, cum_seqlen_k,
    ).launch(
        grid=grid,
        block=[self.threads_per_cta, 1, 1],
        cluster=self.cluster_shape_mnk,
        stream=stream,
        min_blocks_per_mp=1,
    )
