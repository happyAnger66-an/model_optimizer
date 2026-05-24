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
"""KV transform warp body (dequant for mixed-input, layout-only for homo dtype)."""

import cutlass
import cutlass.cute as cute
from typing import Optional, Tuple

from fmha_d256_cutedsl import fmha_helpers as fmha_utils
from fmha_d256_cutedsl import prefill_helpers as prefill_utils


@cute.jit
def transform_warp_body(
    self,
    qk_tiled_mma,
    pv_tiled_mma,
    sK,
    sV,
    sK_trans,
    sV_trans,
    sScaleK_s2r_view,
    sScaleV_s2r_view,
    seqlen_q,
    seqlen_k,
    window_size_left,
    window_size_right,
    load_kv_consumer,
    load_scale_k_consumer,
    load_scale_v_consumer,
    dequant_kv_producer,
    tile_sched_params,
    cum_seqlen_k: Optional[cute.Tensor] = None,
):
    cute.arch.setmaxregister_decrease(self.num_regs_transform)
    tile_sched = fmha_utils.create_fmha_static_tile_scheduler(
        tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
    )
    work_tile = tile_sched.initial_work_tile_info()
    qk_thr_mma_leader_cta = qk_tiled_mma.get_slice(0)
    pv_thr_mma_leader_cta = pv_tiled_mma.get_slice(0)
    if cutlass.const_expr(self.is_mixed_input):
        sScaleK_ = qk_thr_mma_leader_cta.partition_B(sScaleK_s2r_view)
        sScaleV_ = pv_thr_mma_leader_cta.partition_B(sScaleV_s2r_view)
    while work_tile.is_valid_tile:
        curr_block_coord = work_tile.tile_idx
        mma_block_coord = (
            curr_block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
            curr_block_coord[1],
            curr_block_coord[2],
        )
        batch_coord = mma_block_coord[2][1]
        effective_seqlen_k = seqlen_k
        if cutlass.const_expr(cum_seqlen_k is not None):
            cuseqlen_k = cum_seqlen_k[batch_coord]
            effective_seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
        seqlen_kv_loop_steps = fmha_utils.FusedMask.get_trip_count(
            self.mask_type,
            mma_block_coord,
            self.qk_mma_tiler,
            seqlen_q,
            effective_seqlen_k,
            window_size_left,
            window_size_right,
        )
        if cutlass.const_expr(self.is_mixed_input):
            load_kv_consumer, load_scale_k_consumer, dequant_kv_producer = (
                prefill_utils.dequant_k(
                    self.iterations_qk,
                    self.transform_warp_ids,
                    (self.k_dtype, self.q_dtype),
                    (sK, sScaleK_, sK_trans),
                    (load_kv_consumer, load_scale_k_consumer, dequant_kv_producer),
                )
            )
        else:
            load_kv_consumer, dequant_kv_producer = prefill_utils.transform_k(
                self.iterations_qk,
                self.transform_warp_ids,
                (self.k_dtype, self.q_dtype),
                (sK, sK_trans),
                (load_kv_consumer, dequant_kv_producer),
            )
        for step in cutlass.range(1, seqlen_kv_loop_steps, 1, unroll=1):
            if cutlass.const_expr(self.is_mixed_input):
                load_kv_consumer, load_scale_k_consumer, dequant_kv_producer = (
                    prefill_utils.dequant_k(
                        self.iterations_qk,
                        self.transform_warp_ids,
                        (self.k_dtype, self.q_dtype),
                        (sK, sScaleK_, sK_trans),
                        (
                            load_kv_consumer,
                            load_scale_k_consumer,
                            dequant_kv_producer,
                        ),
                    )
                )
                load_kv_consumer, load_scale_v_consumer, dequant_kv_producer = (
                    prefill_utils.dequant_v(
                        self.iterations_pv,
                        self.transform_warp_ids,
                        (self.v_dtype, self.q_dtype),
                        (sV, sScaleV_, sV_trans),
                        (
                            load_kv_consumer,
                            load_scale_v_consumer,
                            dequant_kv_producer,
                        ),
                    )
                )
            else:
                load_kv_consumer, dequant_kv_producer = prefill_utils.transform_k(
                    self.iterations_qk,
                    self.transform_warp_ids,
                    (self.k_dtype, self.q_dtype),
                    (sK, sK_trans),
                    (load_kv_consumer, dequant_kv_producer),
                )
                load_kv_consumer, dequant_kv_producer = prefill_utils.transform_v(
                    self.iterations_pv,
                    self.transform_warp_ids,
                    (self.v_dtype, self.q_dtype),
                    (sV, sV_trans),
                    (load_kv_consumer, dequant_kv_producer),
                )
        if cutlass.const_expr(self.is_mixed_input):
            load_kv_consumer, load_scale_v_consumer, dequant_kv_producer = (
                prefill_utils.dequant_v(
                    self.iterations_pv,
                    self.transform_warp_ids,
                    (self.v_dtype, self.q_dtype),
                    (sV, sScaleV_, sV_trans),
                    (load_kv_consumer, load_scale_v_consumer, dequant_kv_producer),
                )
            )
        else:
            load_kv_consumer, dequant_kv_producer = prefill_utils.transform_v(
                self.iterations_pv,
                self.transform_warp_ids,
                (self.v_dtype, self.q_dtype),
                (sV, sV_trans),
                (load_kv_consumer, dequant_kv_producer),
            )
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()
    dequant_kv_producer.tail()
