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
"""Top-level ``@cute.kernel`` shell and warp-role dispatcher."""

from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.typing import Float32, Int32

from fmha_d256_cutedsl import fmha_helpers as fmha_utils

@cute.kernel
def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_scale_k: cute.CopyAtom,
        mScaleK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        tma_atom_scale_v: cute.CopyAtom,
        mScaleV_dkl: cute.Tensor,
        mO_qdl: cute.Tensor,
        scale_softmax_log2: Float32,
        scale_output: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        cluster_layout_vmnk: cute.Layout,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        k_trans_smem_layout_staged: cute.ComposedLayout,
        scale_k_smem_layout_staged: cute.ComposedLayout,
        scale_k_s2r_view_layout_staged: cute.Layout,
        p_tmem_layout: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        v_trans_smem_layout_staged: cute.ComposedLayout,
        scale_v_smem_layout_staged: cute.ComposedLayout,
        scale_v_s2r_view_layout_staged: cute.Layout,
        epi_tile: cute.Tile,
        tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams,
        cum_seqlen_k: Optional[cute.Tensor] = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Prefetch tma desc
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_scale_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_scale_v)
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(qk_tiled_mma.thr_id.shape)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # Alloc
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.load_warp_id])
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.mma_warp_id])
            ),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_kv_producer, load_kv_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.kv_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.load_warp_id])
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.transform_warp_ids) * self.threads_per_warp,
            ),
            tx_count=self.tma_copy_kv_bytes,
            barrier_storage=storage.load_kv_mbar_ptr.data_ptr(),
            tidx=0,
            defer_sync=True,
        ).make_participants()
        load_scale_k_producer, load_scale_k_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.scale_k_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.load_warp_id])
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.transform_warp_ids) * self.threads_per_warp,
            ),
            tx_count=self.tma_copy_scale_k_bytes,
            barrier_storage=storage.load_scale_k_mbar_ptr.data_ptr(),
            tidx=0,
            defer_sync=True,
        ).make_participants()
        load_scale_v_producer, load_scale_v_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.scale_v_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.load_warp_id])
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.transform_warp_ids) * self.threads_per_warp,
            ),
            tx_count=self.tma_copy_scale_v_bytes,
            barrier_storage=storage.load_scale_v_mbar_ptr.data_ptr(),
            tidx=0,
            defer_sync=True,
        ).make_participants()
        dequant_kv_producer, dequant_kv_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.kv_trans_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.transform_warp_ids)
                * self.threads_per_warp
                * self.cluster_shape_mnk[0],
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.mma_warp_id])
            ),
            barrier_storage=storage.dequant_kv_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_s_producer, mma_s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.mma_warp_id])
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.softmax_warp_ids)
                * self.threads_per_warp
                * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_s_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        p_mma_producer, p_mma_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.qk_acc_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.softmax_warp_ids)
                * self.threads_per_warp
                * self.cluster_shape_mnk[0],
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.mma_warp_id])
            ),
            barrier_storage=storage.p_mma_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        s_corr_producer, s_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.softmax_warp_ids),
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.correction_warp_ids),
            ),
            barrier_storage=storage.s_corr_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        sum_producer, sum_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.softmax_warp_ids),
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.correction_warp_ids),
            ),
            barrier_storage=storage.sum_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        mma_o_producer, mma_o_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.pv_acc_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.mma_warp_id])
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.correction_warp_ids)
                * self.threads_per_warp
                * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_o_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=self.threads_per_warp
            * len(
                (self.mma_warp_id, *self.softmax_warp_ids, *self.correction_warp_ids)
            ),
        )
        # Tensor memory dealloc barrier init
        # NOTE: Thor's DSL version (<= 4.4.2) returns a `_Pointer` directly when
        # accessing a scalar struct field (e.g. `storage.tmem_holding_buf`), so
        # the `.ptr` attribute introduced in newer DSL releases does not exist.
        # Dropping `.ptr` works for both old and new variants because the value
        # is already a Pointer the moment we read the struct field.
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.correction_warp_ids[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar,
        )
        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sK_trans = smem.allocate_tensor(
            element_type=self.q_dtype,
            layout=k_trans_smem_layout_staged.outer,
            swizzle=k_trans_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sV_trans_ptr = cute.recast_ptr(
            sK_trans.iterator, v_trans_smem_layout_staged.inner
        )
        sV_trans = cute.make_tensor(sV_trans_ptr, v_trans_smem_layout_staged.outer)
        sQ = smem.allocate_tensor(
            element_type=self.q_dtype,
            layout=q_smem_layout_staged.outer,
            swizzle=q_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sScaleK = smem.allocate_tensor(
            element_type=self.scale_k_dtype,
            layout=scale_k_smem_layout_staged.outer,
            swizzle=scale_k_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sScaleK_s2r_view = cute.make_tensor(
            sScaleK.iterator, scale_k_s2r_view_layout_staged
        )
        sScaleV = smem.allocate_tensor(
            element_type=self.scale_v_dtype,
            layout=scale_v_smem_layout_staged.outer,
            swizzle=scale_v_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sScaleV_s2r_view = cute.make_tensor(
            sScaleV.iterator, scale_v_s2r_view_layout_staged
        )
        sK = smem.allocate_tensor(
            element_type=self.k_dtype,
            layout=k_smem_layout_staged.outer,
            swizzle=k_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sV_ptr = cute.recast_ptr(sK.iterator, v_smem_layout_staged.inner)
        sV = cute.make_tensor(sV_ptr, v_smem_layout_staged.outer)

        sSum = smem.allocate_tensor(
            element_type=self.qk_acc_dtype,
            layout=cute.make_layout(len(self.softmax_warp_ids) * self.threads_per_warp),
            byte_alignment=128,
        )

        qk_thr_mma = qk_tiled_mma.get_slice(mma_tile_coord_v)
        pv_thr_mma = pv_tiled_mma.get_slice(mma_tile_coord_v)
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK_trans = qk_thr_mma.make_fragment_B(sK_trans)
        tOrV_trans = pv_thr_mma.make_fragment_B(sV_trans)
        qk_acc_shape = pv_thr_mma.partition_shape_C(
            (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
        )
        # (atomV, restM, restN, accStage)
        tStS = qk_tiled_mma.make_fragment_C(
            cute.append(qk_acc_shape, self.qk_acc_stage)
        )
        pv_acc_shape = pv_thr_mma.partition_shape_C(
            cute.select(self.pv_mma_tiler, mode=[0, 1])
        )
        # (atomV, restM, restN)
        tOtO = pv_thr_mma.make_fragment_C(pv_acc_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                self.iterations_pv,
                stride=self.pv_mma_tiler[1] // self.tmem_warp_shape_mn[1],
            ),
        )
        tStS = cute.make_tensor(tStS.iterator + self.tmem_s_offset, tStS.layout)
        tOtO_staged = cute.make_tensor(tOtO.iterator + self.tmem_o_offset, tOtO_layout)
        # Local_tile partition global tensors
        q_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # (bM, bK, restM, restK, loopM, loopK, loopL)
        gQ_qdl = cute.flat_divide(mQ_qdl, cute.select(self.qk_mma_tiler, mode=[0, 2]))
        tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
        tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_q,
            block_in_cluster_coord_vmnk[2],
            q_cta_layout,
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tSgQ_qdl, 0, 3),
        )
        kv_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # (bN, bK, loopN, loopK, loopL)
        gK_kdl = cute.flat_divide(mK_kdl, cute.select(self.qk_mma_tiler, mode=[1, 2]))
        tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
        tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_k,
            block_in_cluster_coord_vmnk[1],
            kv_cta_layout,
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tSgK_kdl, 0, 3),
        )
        # (blk, loopBlk, loopL)
        gScaleK_kdl = cute.flat_divide(mScaleK_kdl, self.scale_k_tiler)
        # Deal with 2cta
        gScaleK_kdl_ = cute.logical_divide(gScaleK_kdl, (self.scale_k_tiler[0] // 2,))[
            (None, mma_tile_coord_v), None, None
        ]
        tKsScaleK, tKgScaleK_kdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_scale_k,
            block_in_cluster_coord_vmnk[1],
            kv_cta_layout,
            sScaleK,
            gScaleK_kdl_,
        )
        # (bN, bK, loopN, loopK, loopL)
        gV_dkl = cute.flat_divide(mV_dkl, cute.select(self.pv_mma_tiler, mode=[1, 2]))
        tOgV_dkl = pv_thr_mma.partition_B(gV_dkl)
        tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_v,
            block_in_cluster_coord_vmnk[1],
            kv_cta_layout,
            cute.group_modes(sV, 0, 3),
            cute.group_modes(tOgV_dkl, 0, 3),
        )
        # (bBlk, loopBlk, loopL)
        gScaleV_dkl = cute.flat_divide(mScaleV_dkl, self.scale_v_tiler)
        tVsScaleV, tVgScaleV_dkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_scale_v,
            block_in_cluster_coord_vmnk[1],
            kv_cta_layout,
            sScaleV,
            gScaleV_dkl,
        )
        # (bM, bN, loopM, loopN, loopL)
        gO_qdl = cute.flat_divide(mO_qdl, cute.select(self.pv_block_tiler, mode=[0, 1]))
        cO_qdl = cute.flat_divide(
            cute.make_identity_tensor(mO_qdl.shape),
            cute.select(self.pv_block_tiler, mode=[0, 1]),
        )
        seqlen_q = mQ_qdl.shape[0]
        seqlen_k = mK_kdl.shape[0]
        # Cluster wait
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # Warp dispatch (bodies in fmha_d256.device.warp_*)
        if warp_idx == self.load_warp_id:
            self.load_warp_body(
                qk_tiled_mma, pv_tiled_mma,
                tQgQ_qdl, tKgK_kdl, tKgScaleK_kdl, tVgV_dkl, tVgScaleV_dkl,
                tQsQ, tKsK, tKsScaleK, tVsV, tVsScaleV,
                tma_atom_q, tma_atom_k, tma_atom_v, tma_atom_scale_k, tma_atom_scale_v,
                load_q_producer, load_kv_producer, load_scale_k_producer, load_scale_v_producer,
                seqlen_q, seqlen_k, window_size_left, window_size_right,
                tile_sched_params,
                cum_seqlen_k,
            )

        if warp_idx == self.mma_warp_id:
            self.mma_warp_body(
                qk_tiled_mma, pv_tiled_mma, pv_thr_mma, tmem,
                tStS, tSrQ, tSrK_trans, tOtO_staged, tOrV_trans, p_tmem_layout,
                load_q_consumer, dequant_kv_consumer,
                mma_s_producer, p_mma_consumer, mma_o_producer,
                seqlen_q, seqlen_k, window_size_left, window_size_right,
                tile_sched_params,
            )

        if (
            warp_idx < self.correction_warp_ids[0]
            and warp_idx >= self.softmax_warp_ids[0]
        ):
            self.softmax_warp_body(
                qk_tiled_mma, qk_thr_mma, tmem, tStS, sSum,
                scale_softmax_log2, seqlen_q, seqlen_k,
                window_size_left, window_size_right,
                mma_s_consumer, p_mma_producer, s_corr_producer, sum_producer,
                tile_sched_params,
            )

        if warp_idx < self.mma_warp_id and warp_idx >= self.correction_warp_ids[0]:
            self.correction_warp_body(
                qk_tiled_mma, qk_thr_mma, tmem, tStS, tOtO_staged, sSum,
                gO_qdl, cO_qdl, scale_softmax_log2, scale_output, seqlen_q, seqlen_k,
                window_size_left, window_size_right, epi_tile,
                s_corr_consumer, mma_o_consumer, sum_consumer,
                tile_sched_params,
            )

        if warp_idx < self.softmax_warp_ids[0]:
            self.transform_warp_body(
                qk_tiled_mma, pv_tiled_mma,
                sK, sV, sK_trans, sV_trans, sScaleK_s2r_view, sScaleV_s2r_view,
                seqlen_q, seqlen_k, window_size_left, window_size_right,
                load_kv_consumer, load_scale_k_consumer, load_scale_v_consumer,
                dequant_kv_producer, tile_sched_params,
                cum_seqlen_k,
            )

        if warp_idx > self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

        return


@cute.kernel
def kernel_homo(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        mO_qdl: cute.Tensor,
        scale_softmax_log2: Float32,
        scale_output: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        cluster_layout_vmnk: cute.Layout,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        k_trans_smem_layout_staged: cute.ComposedLayout,
        p_tmem_layout: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        v_trans_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: fmha_utils.FmhaStaticTileSchedulerParams,
        cum_seqlen_k: Optional[cute.Tensor] = None,
    ):
        """Homogeneous Q/K/V dtype kernel (no INT8 dequant / scale pipelines)."""
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(qk_tiled_mma.thr_id.shape)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.load_warp_id])
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.mma_warp_id])
            ),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_kv_producer, load_kv_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.kv_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.load_warp_id])
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.transform_warp_ids) * self.threads_per_warp,
            ),
            tx_count=self.tma_copy_kv_bytes,
            barrier_storage=storage.load_kv_mbar_ptr.data_ptr(),
            tidx=0,
            defer_sync=True,
        ).make_participants()
        dequant_kv_producer, dequant_kv_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.kv_trans_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.transform_warp_ids)
                * self.threads_per_warp
                * self.cluster_shape_mnk[0],
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.mma_warp_id])
            ),
            barrier_storage=storage.dequant_kv_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_s_producer, mma_s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.mma_warp_id])
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.softmax_warp_ids)
                * self.threads_per_warp
                * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_s_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        p_mma_producer, p_mma_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.qk_acc_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.softmax_warp_ids)
                * self.threads_per_warp
                * self.cluster_shape_mnk[0],
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.mma_warp_id])
            ),
            barrier_storage=storage.p_mma_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        s_corr_producer, s_corr_consumer = pipeline.PipelineAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.softmax_warp_ids),
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.correction_warp_ids),
            ),
            barrier_storage=storage.s_corr_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        sum_producer, sum_consumer = pipeline.PipelineAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.softmax_warp_ids),
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_warp * len(self.correction_warp_ids),
            ),
            barrier_storage=storage.sum_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        mma_o_producer, mma_o_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.pv_acc_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len([self.mma_warp_id])
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                len(self.correction_warp_ids)
                * self.threads_per_warp
                * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_o_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=self.threads_per_warp
            * len(
                (self.mma_warp_id, *self.softmax_warp_ids, *self.correction_warp_ids)
            ),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.correction_warp_ids[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar,
        )
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sK_trans = smem.allocate_tensor(
            element_type=self.q_dtype,
            layout=k_trans_smem_layout_staged.outer,
            swizzle=k_trans_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sV_trans_ptr = cute.recast_ptr(
            sK_trans.iterator, v_trans_smem_layout_staged.inner
        )
        sV_trans = cute.make_tensor(sV_trans_ptr, v_trans_smem_layout_staged.outer)
        sQ = smem.allocate_tensor(
            element_type=self.q_dtype,
            layout=q_smem_layout_staged.outer,
            swizzle=q_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sK = smem.allocate_tensor(
            element_type=self.q_dtype,
            layout=k_smem_layout_staged.outer,
            swizzle=k_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sV = smem.allocate_tensor(
            element_type=self.q_dtype,
            layout=v_smem_layout_staged.outer,
            swizzle=v_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sSum = smem.allocate_tensor(
            element_type=self.qk_acc_dtype,
            layout=cute.make_layout(len(self.softmax_warp_ids) * self.threads_per_warp),
            byte_alignment=128,
        )

        qk_thr_mma = qk_tiled_mma.get_slice(mma_tile_coord_v)
        pv_thr_mma = pv_tiled_mma.get_slice(mma_tile_coord_v)
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK_trans = qk_thr_mma.make_fragment_B(sK_trans)
        tOrV_trans = pv_thr_mma.make_fragment_B(sV_trans)
        qk_acc_shape = pv_thr_mma.partition_shape_C(
            (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
        )
        tStS = qk_tiled_mma.make_fragment_C(
            cute.append(qk_acc_shape, self.qk_acc_stage)
        )
        pv_acc_shape = pv_thr_mma.partition_shape_C(
            cute.select(self.pv_mma_tiler, mode=[0, 1])
        )
        tOtO = pv_thr_mma.make_fragment_C(pv_acc_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                self.iterations_pv,
                stride=self.pv_mma_tiler[1] // self.tmem_warp_shape_mn[1],
            ),
        )
        tStS = cute.make_tensor(tStS.iterator + self.tmem_s_offset, tStS.layout)
        tOtO_staged = cute.make_tensor(tOtO.iterator + self.tmem_o_offset, tOtO_layout)
        q_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        gQ_qdl = cute.flat_divide(mQ_qdl, cute.select(self.qk_mma_tiler, mode=[0, 2]))
        tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
        tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_q,
            block_in_cluster_coord_vmnk[2],
            q_cta_layout,
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tSgQ_qdl, 0, 3),
        )
        kv_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        gK_kdl = cute.flat_divide(mK_kdl, cute.select(self.qk_mma_tiler, mode=[1, 2]))
        tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
        tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_k,
            block_in_cluster_coord_vmnk[1],
            kv_cta_layout,
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tSgK_kdl, 0, 3),
        )
        gV_dkl = cute.flat_divide(mV_dkl, cute.select(self.pv_mma_tiler, mode=[1, 2]))
        tOgV_dkl = pv_thr_mma.partition_B(gV_dkl)
        tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_v,
            block_in_cluster_coord_vmnk[1],
            kv_cta_layout,
            cute.group_modes(sV, 0, 3),
            cute.group_modes(tOgV_dkl, 0, 3),
        )
        gO_qdl = cute.flat_divide(mO_qdl, cute.select(self.pv_block_tiler, mode=[0, 1]))
        cO_qdl = cute.flat_divide(
            cute.make_identity_tensor(mO_qdl.shape),
            cute.select(self.pv_block_tiler, mode=[0, 1]),
        )
        seqlen_q = mQ_qdl.shape[0]
        seqlen_k = mK_kdl.shape[0]
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        dummy_scale_k = sK
        dummy_scale_v = sV
        if warp_idx == self.load_warp_id:
            self.load_warp_body(
                qk_tiled_mma, pv_tiled_mma,
                tQgQ_qdl, tKgK_kdl, None, tVgV_dkl, None,
                tQsQ, tKsK, None, tVsV, None,
                tma_atom_q, tma_atom_k, tma_atom_v, None, None,
                load_q_producer, load_kv_producer, None, None,
                seqlen_q, seqlen_k, window_size_left, window_size_right,
                tile_sched_params,
                cum_seqlen_k,
            )

        if warp_idx == self.mma_warp_id:
            self.mma_warp_body(
                qk_tiled_mma, pv_tiled_mma, pv_thr_mma, tmem,
                tStS, tSrQ, tSrK_trans, tOtO_staged, tOrV_trans, p_tmem_layout,
                load_q_consumer, dequant_kv_consumer,
                mma_s_producer, p_mma_consumer, mma_o_producer,
                seqlen_q, seqlen_k, window_size_left, window_size_right,
                tile_sched_params,
            )

        if (
            warp_idx < self.correction_warp_ids[0]
            and warp_idx >= self.softmax_warp_ids[0]
        ):
            self.softmax_warp_body(
                qk_tiled_mma, qk_thr_mma, tmem, tStS, sSum,
                scale_softmax_log2, seqlen_q, seqlen_k,
                window_size_left, window_size_right,
                mma_s_consumer, p_mma_producer, s_corr_producer, sum_producer,
                tile_sched_params,
            )

        if warp_idx < self.mma_warp_id and warp_idx >= self.correction_warp_ids[0]:
            self.correction_warp_body(
                qk_tiled_mma, qk_thr_mma, tmem, tStS, tOtO_staged, sSum,
                gO_qdl, cO_qdl, scale_softmax_log2, scale_output, seqlen_q, seqlen_k,
                window_size_left, window_size_right, epi_tile,
                s_corr_consumer, mma_o_consumer, sum_consumer,
                tile_sched_params,
            )

        if warp_idx < self.softmax_warp_ids[0]:
            self.transform_warp_body(
                qk_tiled_mma, pv_tiled_mma,
                sK, sV, sK_trans, sV_trans, dummy_scale_k, dummy_scale_v,
                seqlen_q, seqlen_k, window_size_left, window_size_right,
                load_kv_consumer, None, None,
                dequant_kv_producer, tile_sched_params,
                cum_seqlen_k,
            )

        if warp_idx > self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

        return
