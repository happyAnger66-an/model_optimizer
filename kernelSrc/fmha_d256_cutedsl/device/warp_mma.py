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
"""MMA warp body and PV GEMM helper."""

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.typing import Float32
from typing import Optional, Tuple

from fmha_d256_cutedsl import fmha_helpers as fmha_utils
from fmha_d256_cutedsl import prefill_helpers as prefill_utils

@cute.jit
def mma_warp_body(self,
    qk_tiled_mma, pv_tiled_mma, pv_thr_mma, tmem,
    tStS, tSrQ, tSrK_trans, tOtO_staged, tOrV_trans, p_tmem_layout,
    load_q_consumer, dequant_kv_consumer,
    mma_s_producer, p_mma_consumer, mma_o_producer,
    seqlen_q, seqlen_k, window_size_left, window_size_right,
    tile_sched_params,
):
    cute.arch.setmaxregister_decrease(self.num_regs_other)
    tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
        tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
    )
    work_tile = tile_sched.initial_work_tile_info()
    tmem.wait_for_alloc()
    while work_tile.is_valid_tile:
        curr_block_coord = work_tile.tile_idx
        mma_block_coord = (
            curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
            curr_block_coord[1],
            curr_block_coord[2],
        )
        seqlen_kv_loop_steps = fmha_utils.FusedMask.get_trip_count(
            self.mask_type,
            mma_block_coord,
            self.qk_mma_tiler,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )
        load_q_releaser = load_q_consumer.clone()
        pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        if seqlen_kv_loop_steps > 1:
            mma_s_producer, load_q_consumer, dequant_kv_consumer = (
                prefill_utils.mma_qk(  # QK0
                    self.iterations_qk,
                    qk_tiled_mma,
                    (tStS, tSrQ, tSrK_trans),
                    (
                        mma_s_producer,
                        load_q_consumer,
                        None,
                        dequant_kv_consumer,
                    ),
                )
            )
            for i in cutlass.range(1, seqlen_kv_loop_steps - 1, 1, unroll=1):
                mma_s_producer, _, dequant_kv_consumer = (
                    prefill_utils.mma_qk(  # QKi
                        self.iterations_qk,
                        qk_tiled_mma,
                        (tStS, tSrQ, tSrK_trans),
                        (mma_s_producer, None, None, dequant_kv_consumer),
                    )
                )
                (
                    pv_tiled_mma,
                    p_mma_consumer,
                    mma_o_producer,
                    dequant_kv_consumer,
                ) = self.mma_pv(  # PVi
                    (pv_tiled_mma, pv_thr_mma),
                    (tOtO_staged, tStS, tOrV_trans, p_tmem_layout),
                    (p_mma_consumer, mma_o_producer, dequant_kv_consumer),
                )
            mma_s_producer, _, dequant_kv_consumer = (
                prefill_utils.mma_qk(  # QKend needs to release Q
                    self.iterations_qk,
                    qk_tiled_mma,
                    (tStS, tSrQ, tSrK_trans),
                    (
                        mma_s_producer,
                        None,
                        load_q_releaser,
                        dequant_kv_consumer,
                    ),
                )
            )
            (
                pv_tiled_mma,
                p_mma_consumer,
                mma_o_producer,
                dequant_kv_consumer,
            ) = self.mma_pv(  # PVend-1
                (pv_tiled_mma, pv_thr_mma),
                (tOtO_staged, tStS, tOrV_trans, p_tmem_layout),
                (p_mma_consumer, mma_o_producer, dequant_kv_consumer),
            )
        else:
            mma_s_producer, load_q_consumer, dequant_kv_consumer = (
                prefill_utils.mma_qk(  # QK0
                    self.iterations_qk,
                    qk_tiled_mma,
                    (tStS, tSrQ, tSrK_trans),
                    (
                        mma_s_producer,
                        load_q_consumer,
                        load_q_releaser,
                        dequant_kv_consumer,
                    ),
                )
            )
        pv_tiled_mma, p_mma_consumer, mma_o_producer, dequant_kv_consumer = (
            self.mma_pv(  # PVend
                (pv_tiled_mma, pv_thr_mma),
                (tOtO_staged, tStS, tOrV_trans, p_tmem_layout),
                (p_mma_consumer, mma_o_producer, dequant_kv_consumer),
            )
        )
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    mma_s_producer.tail()
    mma_o_producer.tail()


@cute.jit
def mma_pv(
        self,
        mma_args: Tuple,
        tensor_args: Tuple,
        pipeline_args: Tuple,
    ):
        pv_tiled_mma, pv_thr_mma = mma_args
        tOtO_staged, tStS, tOrV_trans, p_tmem_layout = tensor_args
        p_mma_consumer, mma_o_producer, dequant_kv_consumer = pipeline_args
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        is_leader_cta = cta_rank_in_cluster % 2 == 0
        if is_leader_cta:
            p_handle = p_mma_consumer.wait_and_advance()
            o_handle = mma_o_producer.acquire_and_advance()
            pv_whether_acc = pv_tiled_mma.get(tcgen05.Field.ACCUMULATE)
            for iter in cutlass.range(self.iterations_pv, unroll=1):
                v_trans_handle = dequant_kv_consumer.wait_and_advance()
                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, pv_whether_acc)
                tOtO_slice = tOtO_staged[None, None, None, iter]
                tStS_slice = tStS[None, None, None, p_handle.index]
                tP = cute.make_tensor(tStS_slice.iterator, p_tmem_layout.outer)
                tOrP = pv_thr_mma.make_fragment_A(tP)
                tOrP_slice = cute.make_tensor(
                    cute.recast_ptr(tStS_slice.iterator, dtype=self.p_dtype),
                    tOrP.layout,
                )
                tOrV_trans_slice = tOrV_trans[None, None, None, v_trans_handle.index]
                num_kphases = cute.size(tOrV_trans_slice, mode=[2])
                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                    kphase_coord = (None, None, kphase_idx)
                    cute.gemm(
                        pv_tiled_mma,
                        tOtO_slice,
                        tOrP_slice[kphase_coord],
                        tOrV_trans_slice[kphase_coord],
                        tOtO_slice,
                    )
                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                v_trans_handle.release()
            o_handle.commit()
            p_handle.release()
        return pv_tiled_mma, p_mma_consumer, mma_o_producer, dequant_kv_consumer

