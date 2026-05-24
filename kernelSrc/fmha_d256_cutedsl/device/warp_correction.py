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
"""Correction warp body and helpers."""

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
def correction_warp_body(self,
    qk_tiled_mma, qk_thr_mma, tmem, tStS, tOtO_staged, sSum,
    gO_qdl, cO_qdl, scale_softmax_log2, scale_output, seqlen_q, seqlen_k,
    window_size_left, window_size_right, epi_tile,
    s_corr_consumer, mma_o_consumer, sum_consumer,
    tile_sched_params,
):
    cute.arch.setmaxregister_increase(self.num_regs_correction)
    tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
        tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
    )
    work_tile = tile_sched.initial_work_tile_info()
    tmem.allocate(self.num_tmem_alloc_cols)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
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
        gO_staged = gO_qdl[
            None, None, curr_block_coord[0], None, curr_block_coord[2]
        ]
        cO_staged = cO_qdl[
            None, None, curr_block_coord[0], None, curr_block_coord[2]
        ]
        cS = cute.make_identity_tensor(
            (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
        )
        tScS = qk_thr_mma.partition_C(cS)
        # Empty step as the first step is no need for correction
        stats_handle = s_corr_consumer.wait_and_advance()
        stats_handle.release()
        for step in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
            # Oi-1 -> Oi
            mma_o_consumer, s_corr_consumer = self.correction_rescale(
                scale_softmax_log2,
                (s_corr_consumer, tStS, tScS),
                (mma_o_consumer, tOtO_staged, cO_staged),
                epi_tile,
            )
        # O_partial -> O_final
        mma_o_consumer, sum_consumer = self.correction_epilog(
            (seqlen_q, scale_output),
            (sum_consumer, sSum),
            (mma_o_consumer, gO_staged, cO_staged, tOtO_staged),
            epi_tile,
        )
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    tmem.relinquish_alloc_permit()
    tmem.free(tmem_ptr)


@cute.jit
def correction_rescale(
        self,
        scale_softmax_log2: Float32,
        stats_args: tuple,
        o_args: tuple,
        epi_tile: cute.Tile,
    ) -> pipeline.PipelineConsumer:
        (s_corr_consumer, tStS, tScS) = stats_args
        (mma_o_consumer, tOtO_staged, cO_staged) = o_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))

        stats_handle = s_corr_consumer.wait_and_advance()
        tStS_slice = tStS[(None, None), 0, 0, stats_handle.index]
        tScS_slice = tScS[(None, None), 0, 0]
        stats_layout = cute.composition(
            tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], 2))
        )
        stats_c_layout = cute.composition(
            tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], 2))
        )
        tOtStats = cute.make_tensor(
            tStS_slice.iterator + self.tilePlikeFP32, stats_layout
        )
        tOcStats = cute.make_tensor(tScS_slice.iterator, stats_c_layout)
        tmem_load_stats_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(2)),
            self.qk_acc_dtype,
        )
        tiled_tmem_load_stats = tcgen05.make_tmem_copy(tmem_load_stats_atom, tOtStats)
        thr_tmem_load_stats = tiled_tmem_load_stats.get_slice(thread_idx)
        tTMEM_LOADtStats = thr_tmem_load_stats.partition_S(tOtStats)
        tTMEM_LOADcStats = thr_tmem_load_stats.partition_D(tOcStats)
        tTMEM_LOADrStats = cute.make_rmem_tensor(
            tTMEM_LOADcStats.shape, self.qk_acc_dtype
        )
        cute.copy(tiled_tmem_load_stats, tTMEM_LOADtStats, tTMEM_LOADrStats)

        scale = scale_softmax_log2 * (tTMEM_LOADrStats[0] - tTMEM_LOADrStats[1])
        scale = cute.math.exp2(scale, fastmath=True)
        stats_handle.release()
        o_handle = mma_o_consumer.wait_and_advance()
        for iter in cutlass.range(self.iterations_pv, unroll_full=True):
            tOtO = tOtO_staged[(None, None), 0, 0, iter]
            cO = cO_staged[None, None, iter]
            tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
            cO_epi = cute.zipped_divide(cO, epi_tile)
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(16)),
                self.pv_acc_dtype,
            )
            tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_epi)
            thr_load = tmem_tiled_load.get_slice(thread_idx)
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.St32x32bOp(tcgen05.Repetition(16)),
                self.pv_acc_dtype,
            )
            tmem_store_atom = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_epi)
            thr_store = tmem_store_atom.get_slice(thread_idx)
            tTMEM_LOADtO = thr_load.partition_S(tOtO_epi)
            tTMEM_LOADcO = thr_load.partition_D(cO_epi)
            tTMEM_STOREtO = thr_store.partition_D(tOtO_epi)
            tTMrO = cute.make_rmem_tensor_like(
                cute.append(
                    cute.make_layout(tTMEM_LOADcO[None, 0, 0].shape),
                    cute.make_layout(
                        2, stride=cute.size(tTMEM_LOADcO[None, 0, 0].shape)
                    ),
                ),
                self.pv_acc_dtype,
            )
            tTMEM_LOADtO_0 = tTMEM_LOADtO[None, 0, 0]
            cute.copy(tmem_tiled_load, tTMEM_LOADtO_0, tTMrO[None, 0])
            iter_num = cute.size(tTMEM_LOADtO, mode=[1])
            for i in cutlass.range(1, iter_num, unroll_full=True):
                tTMEM_LOADtO_i = tTMEM_LOADtO[None, i, 0]
                cute.copy(tmem_tiled_load, tTMEM_LOADtO_i, tTMrO[None, i % 2])
                for j in cutlass.range(
                    cute.size(tTMrO, mode=[0]), unroll_full=True, vectorize=True
                ):
                    tTMrO[j, (i - 1) % 2] = tTMrO[j, (i - 1) % 2] * scale
                tTMEM_STOREtO_prev_i = tTMEM_STOREtO[None, i - 1, 0]
                cute.copy(
                    tmem_store_atom, tTMrO[None, (i - 1) % 2], tTMEM_STOREtO_prev_i
                )

            for j in cutlass.range(
                cute.size(tTMrO, mode=[0]), unroll_full=True, vectorize=True
            ):
                tTMrO[j, (iter_num - 1) % 2] = tTMrO[j, (iter_num - 1) % 2] * scale
            cute.copy(
                tmem_store_atom,
                tTMrO[None, (iter_num - 1) % 2],
                tTMEM_STOREtO[None, iter_num - 1, 0],
            )
        cute.arch.fence_view_async_tmem_store()
        o_handle.release()
        return mma_o_consumer, s_corr_consumer


@cute.jit
def correction_epilog(
        self,
        value_args: Tuple,
        sum_args: Tuple,
        o_args: Tuple,
        epi_tile: cute.Tile,
    ) -> Tuple[pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        (seqlen_q, scale_output) = value_args
        (sum_consumer, sSum) = sum_args
        (mma_o_consumer, gO_staged, cO_staged, tOtO_staged) = o_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        sum_handle = sum_consumer.wait_and_advance()
        row_sum = sSum[thread_idx]
        cute.arch.fence_view_async_shared()
        sum_handle.release()
        scale = scale_output / row_sum
        o_handle = mma_o_consumer.wait_and_advance()
        for iter in cutlass.range(self.iterations_pv):
            gO = gO_staged[None, None, iter]
            cO = cO_staged[None, None, iter]
            tOtO = tOtO_staged[(None, None), 0, 0, iter]
            tOtO_epi = cute.zipped_divide(tOtO, epi_tile)
            cO_epi = cute.zipped_divide(cO, epi_tile)
            gO_epi = cute.zipped_divide(gO, epi_tile)
            tidx, _, _ = cute.arch.thread_idx()
            thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
            tmem_copy_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.pv_acc_dtype
            )
            tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tOtO_epi)
            thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
            tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_epi)
            tTMEM_LOADgO = thr_tmem_load.partition_D(gO_epi)
            tTMEM_LOADcO = thr_tmem_load.partition_D(cO_epi)
            for i in cutlass.range(cute.size(tTMEM_LOADtO, mode=[1]), unroll_full=True):
                tTMEM_LOADtO_i = tTMEM_LOADtO[None, i, 0]
                tTMEM_LOADgO_i = tTMEM_LOADgO[None, i, 0]
                tTMEM_LOADcO_i = tTMEM_LOADcO[None, i, 0]
                tTMrO = cute.make_rmem_tensor(
                    tTMEM_LOADcO[None, 0, i].shape, self.pv_acc_dtype
                )
                cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
                for j in cutlass.range(
                    cute.size(tTMrO), unroll_full=True, vectorize=True
                ):
                    tTMrO[j] = tTMrO[j] * scale
                tSMrO = cute.make_rmem_tensor(tTMrO.shape, self.o_dtype)
                o_vec = tTMrO.load()
                tSMrO.store(o_vec.to(self.o_dtype))
                if cute.elem_less(tTMEM_LOADcO_i[0][0], seqlen_q):
                    cute.autovec_copy(tSMrO, tTMEM_LOADgO_i)
        o_handle.release()
        return mma_o_consumer, sum_consumer

