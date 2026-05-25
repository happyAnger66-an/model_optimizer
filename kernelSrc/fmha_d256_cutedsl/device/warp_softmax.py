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
"""Softmax warp body and helpers."""

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
def softmax_warp_body(self,
    qk_tiled_mma, qk_thr_mma, tmem, tStS, sSum,
    scale_softmax_log2, seqlen_q, seqlen_k,
    window_size_left, window_size_right,
    mma_s_consumer, p_mma_producer, s_corr_producer, sum_producer,
    tile_sched_params,
):
    """Softmax warp role in the warp-specialized FMHA pipeline.

    This function runs only on the softmax warp group inside the top-level
    FMHA kernel.  The full FMHA pipeline is split across warp roles:

      load / transform warps:
        move Q/K/V tiles into SMEM and, for mixed-input variants, dequantize KV
      mma warp:
        computes score tiles S = Q @ K^T into TMEM
      softmax warps:
        consume S tiles, apply mask + numerically stable softmax, and produce
        probability tiles P for the MMA warp to compute P @ V
      correction warps:
        rescale older partial outputs when the running row max changes, then
        normalize by the final row sum and write O

    The softmax work is streaming over the K/V dimension.  For each Q tile
    assigned by the static tile scheduler, the function iterates over all score
    tiles along K/V, maintaining the per-row softmax statistics:

      row_max = running max(S)
      row_sum = running sum(exp2((S - row_max) * scale))

    ``softmax_step`` performs one S-tile -> P-tile conversion and also sends
    enough row-max metadata to the correction warps so that previously computed
    partial O tiles can be rescaled if a later K/V tile increases ``row_max``.
    After all K/V tiles for the current Q tile have been processed, ``store_sum``
    publishes the final denominator to correction warps for the final
    ``scale_output / row_sum`` normalization.
    """
    # The softmax role is register-heavy: it keeps row statistics, fragments of
    # S/P, and temporary reductions live at the same time.  Request the role's
    # tuned register budget before doing any per-tile work.
    cute.arch.setmaxregister_increase(self.num_regs_softmax)

    # Recreate the same static tile scheduler used by the other warp roles in
    # this CTA.  All roles walk the same work-tile sequence, but each role only
    # performs its own pipeline stage for the current tile.
    tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
        tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
    )
    work_tile = tile_sched.initial_work_tile_info()

    # TMEM is allocated by the MMA/correction side.  Softmax consumes score
    # fragments from TMEM and writes probability fragments back to TMEM, so it
    # must not touch TMEM until the allocation permit has been granted.
    tmem.wait_for_alloc()

    # One loop iteration handles one scheduler-assigned output tile:
    #   curr_block_coord[0] -> Q tile coordinate
    #   curr_block_coord[1] -> head / KV tile scheduler dimension
    #   curr_block_coord[2] -> batch dimension
    # The exact semantic mapping is defined by FmhaStaticTileScheduler.
    while work_tile.is_valid_tile:
        curr_block_coord = work_tile.tile_idx

        # ``curr_block_coord[0]`` includes the 2-CTA MMA partitioning dimension.
        # Fold it back to the logical MMA tile coordinate used by mask/trip-count
        # helpers.  The other coordinates are already logical head/batch coords.
        mma_block_coord = (
            curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
            curr_block_coord[1],
            curr_block_coord[2],
        )

        # Number of K/V score tiles that this Q tile must visit.  The masking
        # helper accounts for sequence lengths, causal/window masks, and
        # bottom-right alignment, so the softmax loop skips tiles that can never
        # contribute.
        seqlen_kv_loop_steps = fmha_utils.FusedMask.get_trip_count(
            self.mask_type,
            mma_block_coord,
            self.qk_mma_tiler,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )

        # Prefix of the K/V loop that is known to be fully unmasked.  For those
        # steps ``softmax_step`` can avoid elementwise predicate checks and use
        # a faster path.  Steps >= ``unmask_steps`` may need causal/sliding-window
        # predicates.
        unmask_steps = fmha_utils.FusedMask.get_unmasked_trip_count(
            self.mask_type,
            mma_block_coord,
            self.qk_mma_tiler,
            seqlen_q,
            seqlen_k,
            window_size_left,
            window_size_right,
        )

        # Build coordinate tensors for the score tile S.  ``cS_base`` is local
        # to a logical QK MMA tile, and ``domain_offset`` shifts it to the
        # absolute Q-row coordinate for this work tile.  These coordinates are
        # later used by ``softmax_step`` for masking and by partition_C to map
        # the warp's fragment view onto the TMEM score layout.
        cS_base = cute.make_identity_tensor(
            (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
        )
        cS = cute.domain_offset(
            (mma_block_coord[0] * self.qk_mma_tiler[0], 0), cS_base
        )
        tScS = qk_thr_mma.partition_C(cS)

        # Running softmax statistics for the current Q tile.  The algorithm is
        # online/stable over K/V tiles:
        #   new_max = max(old_max, max(S_i))
        #   new_sum = old_sum * exp2((old_max - new_max) * scale)
        #             + sum(exp2((S_i - new_max) * scale))
        # ``row_max_prev`` is passed into softmax_step so the correction stage
        # knows how to rescale already accumulated O partials.
        row_max = -Float32.inf
        row_max_prev = -Float32.inf
        row_sum = 0.0

        # Stream over all contributing K/V tiles.  The MMA warp produces one
        # score tile S_i per step via ``mma_s_consumer``.  The softmax warp
        # consumes S_i, writes a probability tile P_i for the MMA warp via
        # ``p_mma_producer``, and publishes row-max transition metadata to the
        # correction warp via ``s_corr_producer``.
        for step in cutlass.range(seqlen_kv_loop_steps, unroll=1):
            # Shift the score coordinates to the current K/V tile.  The row
            # coordinate remains fixed for this Q tile, while the column
            # coordinate advances by the QK MMA N tile size.
            cS_iter = cute.domain_offset((0, step * self.qk_mma_tiler[1]), cS)
            tScS_iter = qk_thr_mma.partition_C(cS_iter)

            # Convert score tile S_i to probability tile P_i, updating the
            # online softmax statistics and all producer/consumer pipeline
            # tokens touched by this step.
            (
                row_max,
                row_sum,
                mma_s_consumer,
                p_mma_producer,
                s_corr_producer,
            ) = self.softmax_step(
                (step >= unmask_steps, window_size_left, window_size_right),
                (
                    row_max_prev,
                    row_sum,
                    seqlen_q,
                    seqlen_k,
                    scale_softmax_log2,
                ),
                (tStS, tScS_iter),
                (mma_s_consumer, p_mma_producer, s_corr_producer),
            )

            # Save the current running max for the next step.  If a later tile
            # increases the max, ``softmax_step`` tells correction warps how
            # much already-computed O partials must be rescaled.
            row_max_prev = row_max

        # All K/V steps for this Q tile are done.  Publish the final denominator
        # to the correction epilogue; it will use this value to normalize and
        # write the final O tile.
        sum_producer = self.store_sum(row_sum, sSum, sum_producer)

        # Advance this role to the next tile assigned by the static scheduler.
        # Other warp roles perform the same scheduler advance after finishing
        # their own responsibilities for the same work tile.
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()

    # No more P tiles or correction-stat tiles will be produced by this softmax
    # role.  Tail the producer pipelines so downstream MMA/correction consumers
    # can drain cleanly instead of waiting for more stages.
    p_mma_producer.tail()
    s_corr_producer.tail()


@cute.jit
def softmax_step(
        self,
        mask_args: Tuple,
        value_args: Tuple,
        tensor_args: Tuple,
        pipeline_args: Tuple,
    ) -> Tuple[Float32, Float32, pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        need_apply_mask, window_size_left, window_size_right = mask_args
        row_max, row_sum, seqlen_q, seqlen_k, scale_softmax_log2 = value_args
        tStS, tScS = tensor_args
        mma_s_consumer, p_mma_producer, s_corr_producer = pipeline_args
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))
        s_handle = mma_s_consumer.wait_and_advance()
        tStS_slice = tStS[(None, None), 0, 0, s_handle.index]
        tScS_slice = tScS[(None, None), 0, 0]
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), self.qk_acc_dtype
        )
        tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_slice)
        thr_load = tmem_tiled_load.get_slice(thread_idx)
        tTMEM_LOADtS = thr_load.partition_S(tStS_slice)
        tTMEM_LOADcS = thr_load.partition_D(tScS_slice)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcS.shape, self.qk_acc_dtype)
        cute.copy(tmem_tiled_load, tTMEM_LOADtS, tTMEM_LOADrS)
        cute.arch.fence_view_async_tmem_load()
        s_handle.release()
        if need_apply_mask:
            fmha_utils.FusedMask.apply_mask(
                self.mask_type,
                tTMEM_LOADrS,
                tTMEM_LOADcS,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
            )
        old_row_max = row_max
        row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, row_max, 0)
        row_max_safe = row_max
        if row_max == -cutlass.Float32.inf:
            row_max_safe = 0.0

        stats_handle = s_corr_producer.acquire_and_advance()
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
        tmem_store_stats_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(2)),
            self.qk_acc_dtype,
        )
        tiled_tmem_store_stats = tcgen05.make_tmem_copy(tmem_store_stats_atom, tOtStats)
        thr_tmem_store_stats = tiled_tmem_store_stats.get_slice(thread_idx)
        tTMEM_STOREcStats = thr_tmem_store_stats.partition_S(tOcStats)
        tTMEM_STORErStats = cute.make_rmem_tensor(
            tTMEM_STOREcStats.shape, self.qk_acc_dtype
        )
        tTMEM_STORErStats[0] = old_row_max
        tTMEM_STORErStats[1] = row_max_safe
        tTMEM_STOREtStats = thr_tmem_store_stats.partition_D(tOtStats)
        cute.copy(tiled_tmem_store_stats, tTMEM_STORErStats, tTMEM_STOREtStats)
        cute.arch.fence_view_async_tmem_store()
        stats_handle.commit()

        scale = scale_softmax_log2
        minus_row_max_scale = (0.0 - row_max_safe) * scale
        tTMEM_STORErP = cute.make_rmem_tensor(tTMEM_LOADrS.shape, self.p_dtype)
        for k in cutlass.range(cute.size(tTMEM_LOADrS), vectorize=True):
            tTMEM_LOADrS[k] = tTMEM_LOADrS[k] * scale + minus_row_max_scale
            tTMEM_LOADrS[k] = cute.math.exp2(tTMEM_LOADrS[k], fastmath=True)
        s_vec = tTMEM_LOADrS.load()
        tTMEM_STORErP.store(s_vec.to(self.p_dtype))

        p_handle = p_mma_producer.acquire_and_advance()
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(32)), self.qk_acc_dtype
        )
        tilePlikeFP32 = tStS_slice.shape[1] // Float32.width * self.p_dtype.width
        tStS_P_layout = cute.composition(
            tStS_slice.layout, cute.make_layout((tStS_slice.shape[0], tilePlikeFP32))
        )
        tStS_P = cute.make_tensor(tStS_slice.iterator, tStS_P_layout)
        tScS_P_layout = cute.composition(
            tScS_slice.layout, cute.make_layout((tScS_slice.shape[0], tilePlikeFP32))
        )
        tScS_P = cute.make_tensor(tScS_slice.iterator, tScS_P_layout)
        tmem_tiled_store = tcgen05.make_tmem_copy(tmem_store_atom, tStS_P)
        thr_store = tmem_tiled_store.get_slice(thread_idx)
        tTMEM_STOREtP = thr_store.partition_D(tStS_P)
        tTMEM_STOREcS = thr_store.partition_S(tScS_P)
        tTMEM_STORErP_ = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErP.iterator, dtype=self.qk_acc_dtype),
            tTMEM_STOREcS.shape,
        )
        cute.copy(tmem_tiled_store, tTMEM_STORErP_, tTMEM_STOREtP)
        cute.arch.fence_view_async_tmem_store()

        p_handle.commit()
        acc_scale_ = scale * (old_row_max - row_max_safe)
        acc_scale = cute.math.exp2(acc_scale_, fastmath=True) * 0.5
        # TODO: calc row sum with TensorSSA
        row_sum *= acc_scale
        local_row_sum_0 = (row_sum, row_sum)
        local_row_sum_1 = (0.0, 0.0)
        local_row_sum_2 = (0.0, 0.0)
        local_row_sum_3 = (0.0, 0.0)
        reduction_unroll = 4
        frg_tile = cute.size(tTMEM_LOADrS) // reduction_unroll
        tTMEM_LOADrS_frg = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(frg_tile))
        for j in cutlass.range_constexpr(0, cute.size(tTMEM_LOADrS_frg, mode=[0]), 2):
            local_row_sum_0 = cute.arch.add_packed_f32x2(
                local_row_sum_0, (tTMEM_LOADrS_frg[j, 0], tTMEM_LOADrS_frg[j + 1, 0])
            )
            local_row_sum_1 = cute.arch.add_packed_f32x2(
                local_row_sum_1, (tTMEM_LOADrS_frg[j, 1], tTMEM_LOADrS_frg[j + 1, 1])
            )
            local_row_sum_2 = cute.arch.add_packed_f32x2(
                local_row_sum_2, (tTMEM_LOADrS_frg[j, 2], tTMEM_LOADrS_frg[j + 1, 2])
            )
            local_row_sum_3 = cute.arch.add_packed_f32x2(
                local_row_sum_3, (tTMEM_LOADrS_frg[j, 3], tTMEM_LOADrS_frg[j + 1, 3])
            )
        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_1)
        local_row_sum_2 = cute.arch.add_packed_f32x2(local_row_sum_2, local_row_sum_3)
        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_2)
        row_sum = local_row_sum_0[0] + local_row_sum_0[1]
        return row_max, row_sum, mma_s_consumer, p_mma_producer, s_corr_producer


@cute.jit
def store_sum(self, row_sum, sSum, sum_producer):
        """Publish the final softmax row sum to correction warps.

        ``softmax_step`` computes the numerically stable denominator for one
        logical attention row handled by the current softmax thread.  The
        correction epilogue needs exactly the same denominator to normalize the
        accumulated ``P @ V`` partial output:

            O = accumulated_output * (scale_output / row_sum)

        ``sSum`` is a small shared-memory handoff buffer indexed by the
        softmax-lane id.  ``sum_producer`` is the producer side of a one-stage
        PipelineAsync created between the softmax warp group and the correction
        warp group.  This function writes the denominator into shared memory and
        commits the pipeline stage so the correction warp can safely consume it.
        """
        # ``tidx`` is the CTA-wide thread id.  Only the softmax warp group
        # participates in this producer pipeline, so remap the CTA-wide id into
        # a dense [0, num_softmax_threads) index.  The correction side uses the
        # same mapping before reading ``sSum[thread_idx]``; this keeps a
        # one-to-one correspondence between a softmax producer thread and the
        # correction consumer thread responsible for that row fragment.
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.softmax_warp_ids))

        # Reserve the single shared-memory pipeline slot.  ``acquire`` provides
        # producer/consumer ordering against the correction warp's
        # ``wait_and_advance``; ``advance`` moves this producer state to the
        # stage that will be committed below.
        sum_handle = sum_producer.acquire_and_advance()

        # Store the final denominator for this row/lane.  At this point
        # ``row_sum`` already includes all KV-loop contributions and the
        # rescaling needed when the running row max changed between tiles.
        sSum[thread_idx] = row_sum

        # Make the shared-memory write visible before committing the pipeline
        # transaction.  Without this fence, the correction warp could observe
        # the committed barrier and read stale ``sSum`` data.
        cute.arch.fence_view_async_shared()

        # Signal that ``sSum[thread_idx]`` is ready.  The correction epilogue
        # waits on the matching consumer handle, reads the row sum, then
        # releases the stage.
        sum_handle.commit()

        # Return the advanced producer token so the caller keeps the pipeline
        # state in sync across work tiles.
        return sum_producer

