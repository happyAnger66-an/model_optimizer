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
        """Mixed-input FMHA D=256 prefill kernel for Blackwell SM100/SM110.

        ===========================================================================
        Role in the FMHA pipeline
        ===========================================================================

        This is the **top-level ``@cute.kernel``** entry point — i.e. the function
        that becomes the actual CUDA kernel when ``cute.compile`` lowers the DSL
        IR to PTX/SASS.  It does not compute attention itself; it sets up *all*
        per-CTA state required by the warp-specialized pipeline and then
        dispatches each warp to a dedicated role.

        FMHA computes, for each (batch, head, query row) triple:

            O = softmax( Q @ K^T * scale ) @ V

        On Blackwell this is fused into one kernel and split across five warp
        roles, each occupying a fixed subset of the CTA's warps:

          ┌─────────────────────────────────────────────────────────────────┐
          │  load warp  ──TMA──▶  sQ / sK / sV / sScaleK / sScaleV (SMEM)   │
          │       │                       │                                 │
          │       ▼                       ▼                                 │
          │  transform warps  ◀──── dequant INT8 KV + scale → BF16 ────▶   │
          │       │                       │                                 │
          │       ▼              (sK_trans / sV_trans in SMEM)              │
          │   mma warp  ──tcgen05──▶  S = Q @ K^T   in TMEM (tStS)          │
          │       │                                                         │
          │       ▼                                                         │
          │   softmax warps  ──▶  P = softmax(S)    in TMEM (tStS overlay) │
          │       │                                                         │
          │       ▼                                                         │
          │   mma warp  ──tcgen05──▶  O_partial = P @ V  in TMEM (tOtO)     │
          │       │                                                         │
          │       ▼                                                         │
          │   correction warps  ──▶  rescale + normalize + write gO        │
          └─────────────────────────────────────────────────────────────────┘

        All stages run **concurrently** on different warps; the producer/consumer
        pipelines built below provide async ordering between them.  The kernel
        also uses a 2-CTA cluster (``cluster_shape_mn=(2,1)``) so two CTAs
        cooperate on each tcgen05 MMA tile.

        ===========================================================================
        What this function actually does
        ===========================================================================

        It is **set-up code + dispatcher**, not the per-tile compute:

          1. Prefetch all TMA descriptors so cp.async issued by the load warp does
             not stall.
          2. Compute per-CTA coordinates inside the cluster.
          3. Carve up the shared-memory arena ``self.shared_storage`` into mbar
             slabs (one per pipeline) and the SMEM tensors (sQ/sK/sV/sK_trans/
             sV_trans/sScaleK/sScaleV/sSum).
          4. Construct **9 mbar-backed pipelines** (load_q, load_kv, load_scale_k,
             load_scale_v, dequant_kv, mma_s, p_mma, s_corr, sum, mma_o) wiring
             together the warp roles.  Each pipeline encodes a producer→consumer
             handshake with ``num_stages`` SMEM/TMEM buffers; correctness of the
             pipeline depends on producer/consumer thread counts and tx_count
             (bytes) being exactly right for the data being moved.
          5. Allocate TMEM via ``utils.TmemAllocator`` (correction warp owns the
             allocator, freed at correction tail).
          6. Build the **TMA partitions** for Q/K/V/scaleK/scaleV/O so each warp
             sees its private ``tQsQ``/``tQgQ``-style fragment views.
          7. Wait for cluster-wide pipeline init (``pipeline_init_wait``) so
             every CTA finishes mbar construction before any role starts firing
             arrives/waits.
          8. Dispatch each warp by ``warp_idx`` into one of:
                 load_warp_body / mma_warp_body / softmax_warp_body /
                 correction_warp_body / transform_warp_body
             Each ``*_warp_body`` is a long ``@cute.jit`` function in
             ``device/warp_*.py`` that walks the tile scheduler and performs
             that role's per-tile work.
          9. After dispatch, every non-load warp shrinks its register budget
             so the SM can fit the next CTA wave.

        The five ``*_warp_body`` functions only exchange data via the pipelines
        and SMEM/TMEM tensors prepared here; the kernel function is the single
        place where all of that state actually lives.
        """
        # -------------------------------------------------------------------
        # 1. Warp / CTA / cluster coordinates
        # -------------------------------------------------------------------
        # ``warp_idx`` is the per-CTA warp index (0..warps_per_cta-1).  Wrap it
        # in ``make_warp_uniform`` so the compiler treats it as a uniform
        # predicate — without that, the warp-dispatch ``if`` branches below
        # would generate divergent control flow.
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch the TMA descriptors for every global tensor the load warp
        # will pull.  Only the load warp does this so we don't issue 5×
        # redundant prefetches per CTA.  The prefetch lowers the first TMA
        # bulk-async latency once the load warp starts streaming Q/K/V.
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_scale_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_scale_v)

        # ``bidx`` is the linear block index in grid X.  With a 2-CTA cluster,
        # two consecutive CTAs cooperate on the same MMA tile but partition the
        # tcgen05 V dimension; ``mma_tile_coord_v`` selects which half a given
        # CTA owns (0 or 1).
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(qk_tiled_mma.thr_id.shape)

        # The CTA's rank within its cluster, plus the (V, M, N, K) coordinate
        # in the cluster layout.  TMA partition routines below use these to
        # decide which sub-slab of the global tile this CTA should fetch.
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        # -------------------------------------------------------------------
        # 2. Carve up the shared-memory arena
        # -------------------------------------------------------------------
        # ``self.shared_storage`` is a ``@cute.struct`` declared by
        # host/launcher.py; it lays out every mbar array + tmem holding buf in
        # one contiguous SMEM region.  ``SmemAllocator`` hands out bump-pointer
        # allocations from that region; tensors allocated after this point
        # (sQ, sK, sV, …) come from the same arena.
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # -------------------------------------------------------------------
        # 3. Build the producer/consumer pipelines
        #
        # Each ``PipelineXxx.create(...)`` returns one pair of producer/consumer
        # tokens backed by the mbar storage in ``self.shared_storage``.
        # Naming convention is ``<data>_<source>_<dest>`` (or just ``<data>``):
        #
        #   load_q       : load warp     -> mma warp        (TMA -> tcgen05 A)
        #   load_kv      : load warp     -> transform warps (TMA bulk -> SMEM)
        #   load_scale_k : load warp     -> transform warps (TMA bulk -> SMEM)
        #   load_scale_v : load warp     -> transform warps (TMA bulk -> SMEM)
        #   dequant_kv   : transform     -> mma warp        (SMEM ready -> MMA)
        #   mma_s        : mma warp      -> softmax warps   (S tile in TMEM)
        #   p_mma        : softmax warps -> mma warp        (P tile in TMEM)
        #   s_corr       : softmax warps -> correction      (rescale stats)
        #   sum          : softmax warps -> correction      (final row sum)
        #   mma_o        : mma warp      -> correction      (O_partial in TMEM)
        #
        # ``num_stages`` controls outstanding tiles in flight; ``tx_count`` is
        # the TMA byte count used to seed mbar arrival counts.
        # -------------------------------------------------------------------

        # Q is loaded once per (Q-tile, batch, head) so its stage count
        # (self.q_stage) is small.  PipelineTmaUmma is the variant whose TMA
        # completion signal directly arrives on a tcgen05 (UMMA) consumer.
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

        # K/V is bulk-loaded as INT8 into ``sK``/``sV`` by the load warp.  The
        # whole transform warp group (8 warps x 32 threads) is the consumer:
        # all of those threads cooperate to dequantize INT8 -> BF16 into
        # ``sK_trans``/``sV_trans``.  The arrival count must match this size.
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

        # Per-tile dequant scales for K and V (one BF16 scale per group of
        # ``scale_granularity`` channels).  Same producer/consumer layout as
        # load_kv but different staging because scales are much smaller.
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

        # After transform warps dequantize KV in SMEM, they tell the MMA warp
        # that the BF16 buffers ``sK_trans``/``sV_trans`` are ready.  Cluster
        # layout is propagated so both CTAs in the 2-CTA cluster handshake.
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

        # S = Q @ K^T (in TMEM) handoff: the MMA warp issues tcgen05 MMAs that
        # write S into TMEM; once a stage of S is committed, the softmax warp
        # group can read it.  Consumer count covers all softmax threads in the
        # cluster (cluster_shape_mnk[0] CTAs cooperate).
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

        # P (softmax output) feed back into the MMA warp for the second
        # GEMM P @ V.  Softmax writes P into the same TMEM region; once a stage
        # is committed, the MMA warp can issue P @ V.
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

        # Per-K/V-tile rescale stats handoff from softmax to correction warps.
        # When softmax discovers a new row-max for a row, it must tell the
        # correction warps so that previously accumulated O_partial gets
        # rescaled by exp2(old_max - new_max).
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

        # Final row-sum handoff (single stage).  After softmax finishes all
        # K/V tiles for a Q tile, ``store_sum`` writes the final denominator
        # into ``sSum`` and commits this 1-stage pipeline so correction can
        # divide by it.
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

        # O_partial = P @ V handoff from MMA to correction.  Correction reads
        # the partial output from TMEM, applies rescale + softmax normalization,
        # and writes the final O tile to global memory.
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

        # -------------------------------------------------------------------
        # 4. TMEM allocation barrier
        #
        # tcgen05 TMEM is a shared on-die memory region whose lifetime spans
        # the whole CTA.  The correction warp owns the allocator (because it
        # writes the final O tile last); the MMA and softmax warps must wait
        # on this named barrier before touching TMEM.
        # -------------------------------------------------------------------
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
        # Cluster arrive after barrier init: every CTA in the cluster signals
        # that its mbars are fully constructed.  Matched by pipeline_init_wait
        # below; until that wait returns, no warp may arrive on any mbar.
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        # -------------------------------------------------------------------
        # 5. Allocate SMEM tensors (data, not mbars)
        #
        # Order matters: every ``allocate_tensor`` bumps the SMEM pointer.
        # Some tensors are **aliased** — for example ``sV_trans`` reuses the
        # ``sK_trans`` allocation because K and V dequant buffers do not need
        # to coexist in time; same for ``sV`` overlapping ``sK``.
        # -------------------------------------------------------------------

        # BF16 dequantized K tile (after INT8 -> BF16 transform).  Allocated
        # first so its iterator pointer can be reused by sV_trans below.
        sK_trans = smem.allocate_tensor(
            element_type=self.q_dtype,
            layout=k_trans_smem_layout_staged.outer,
            swizzle=k_trans_smem_layout_staged.inner,
            byte_alignment=128,
        )

        # ``sV_trans`` aliases ``sK_trans`` storage: at any given time only one
        # is live (K used during QK MMA, V during PV MMA).  This halves the
        # SMEM footprint of dequant buffers.
        sV_trans_ptr = cute.recast_ptr(
            sK_trans.iterator, v_trans_smem_layout_staged.inner
        )
        sV_trans = cute.make_tensor(sV_trans_ptr, v_trans_smem_layout_staged.outer)

        # Q tile in SMEM, BF16.
        sQ = smem.allocate_tensor(
            element_type=self.q_dtype,
            layout=q_smem_layout_staged.outer,
            swizzle=q_smem_layout_staged.inner,
            byte_alignment=128,
        )

        # Per-channel dequant scales for K (BF16).  ``s2r_view`` is a logical
        # reshape used by the transform warps when they issue ldsm-like reads
        # into registers; both views share the same SMEM storage.
        sScaleK = smem.allocate_tensor(
            element_type=self.scale_k_dtype,
            layout=scale_k_smem_layout_staged.outer,
            swizzle=scale_k_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sScaleK_s2r_view = cute.make_tensor(
            sScaleK.iterator, scale_k_s2r_view_layout_staged
        )

        # Per-channel dequant scales for V (BF16), mirror of sScaleK.
        sScaleV = smem.allocate_tensor(
            element_type=self.scale_v_dtype,
            layout=scale_v_smem_layout_staged.outer,
            swizzle=scale_v_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sScaleV_s2r_view = cute.make_tensor(
            sScaleV.iterator, scale_v_s2r_view_layout_staged
        )

        # Raw INT8 K tile (before dequant).  ``sV`` aliases the same storage
        # since INT8 K and INT8 V loads do not overlap.
        sK = smem.allocate_tensor(
            element_type=self.k_dtype,
            layout=k_smem_layout_staged.outer,
            swizzle=k_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sV_ptr = cute.recast_ptr(sK.iterator, v_smem_layout_staged.inner)
        sV = cute.make_tensor(sV_ptr, v_smem_layout_staged.outer)

        # Per-row softmax sums published by the softmax warps and read by the
        # correction warps.  One Float32 entry per softmax-lane thread; that
        # exactly maps 1:1 to the row fragments each lane owns.
        sSum = smem.allocate_tensor(
            element_type=self.qk_acc_dtype,
            layout=cute.make_layout(len(self.softmax_warp_ids) * self.threads_per_warp),
            byte_alignment=128,
        )

        # -------------------------------------------------------------------
        # 6. tcgen05 MMA fragments + TMEM accumulators
        #
        # ``qk_thr_mma`` / ``pv_thr_mma`` are this CTA's slice of the tiled
        # MMA (the cluster split happens via ``mma_tile_coord_v``).
        # ``make_fragment_A/B`` create the operand fragment views over SMEM;
        # ``make_fragment_C`` creates the accumulator views over TMEM.
        # -------------------------------------------------------------------
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
        # Apply fixed TMEM offsets so S and O occupy disjoint TMEM regions.
        # The same TMEM tile is later overlaid (in time) by P that the softmax
        # warp writes back; softmax uses a different ``p_tmem_layout`` view.
        tStS = cute.make_tensor(tStS.iterator + self.tmem_s_offset, tStS.layout)
        tOtO_staged = cute.make_tensor(tOtO.iterator + self.tmem_o_offset, tOtO_layout)

        # -------------------------------------------------------------------
        # 7. TMA partitions for Q / K / V / scaleK / scaleV
        #
        # ``flat_divide`` chops each global tensor by the relevant MMA tile.
        # ``tma_partition`` then computes:
        #   - ``tXsX``  : SMEM view this CTA writes to
        #   - ``tXgX_*`` : per-tile global address generator
        # using the CTA's coordinate within the cluster as the partition key.
        # -------------------------------------------------------------------
        q_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # (bM, bK, restM, restK, loopM, loopK, loopL): tile Q by MMA(M)xMMA(K).
        gQ_qdl = cute.flat_divide(mQ_qdl, cute.select(self.qk_mma_tiler, mode=[0, 2]))
        tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
        tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_q,
            block_in_cluster_coord_vmnk[2],
            q_cta_layout,
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tSgQ_qdl, 0, 3),
        )

        # KV uses the cluster's N dimension as the partition axis so two CTAs
        # in a 2-CTA cluster each fetch half of the K/V tile.
        kv_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # (bN, bK, loopN, loopK, loopL): tile K by MMA(N)xMMA(K).
        gK_kdl = cute.flat_divide(mK_kdl, cute.select(self.qk_mma_tiler, mode=[1, 2]))
        tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
        tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_k,
            block_in_cluster_coord_vmnk[1],
            kv_cta_layout,
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tSgK_kdl, 0, 3),
        )
        # ScaleK is much smaller than K: shape (blk, loopBlk, loopL).  The
        # 2-CTA split below picks the half that belongs to this CTA's V slot.
        gScaleK_kdl = cute.flat_divide(mScaleK_kdl, self.scale_k_tiler)
        # 2-CTA fixup: logically split the leading scale dimension into two
        # halves and select ``mma_tile_coord_v`` (0 or 1) for the current CTA.
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
        # Output partition (correction warp will write through this view).
        # ``cO_qdl`` is an identity-coordinate companion used for predication.
        gO_qdl = cute.flat_divide(mO_qdl, cute.select(self.pv_block_tiler, mode=[0, 1]))
        cO_qdl = cute.flat_divide(
            cute.make_identity_tensor(mO_qdl.shape),
            cute.select(self.pv_block_tiler, mode=[0, 1]),
        )

        # Dynamic sequence lengths (Q can be padded BSHD; K is the KV-cache
        # capacity in LLM prefill mode and the real per-batch length is in
        # ``cum_seqlen_k``).
        seqlen_q = mQ_qdl.shape[0]
        seqlen_k = mK_kdl.shape[0]

        # Wait for every CTA in the cluster to finish mbar init.  After this
        # returns, all pipelines above are safe to use across the cluster.
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # -------------------------------------------------------------------
        # 8. Warp role dispatch
        #
        # ``warp_idx`` selects exactly one role per warp.  Each ``*_warp_body``
        # is a long ``@cute.jit`` function in ``device/warp_*.py`` that walks
        # the static tile scheduler and performs that role's per-tile work.
        # All five bodies coexist in the same kernel; ``warp_idx`` decides
        # which executes for any given warp.
        # -------------------------------------------------------------------

        # Load warp: drives all TMA bulk-async copies (Q + INT8 KV + scales).
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

        # MMA warp: issues tcgen05 MMAs S = Q @ K^T and O_partial = P @ V into
        # TMEM, gated by load_q / dequant_kv producers and feeding the softmax
        # and correction warps via mma_s / mma_o pipelines.
        if warp_idx == self.mma_warp_id:
            self.mma_warp_body(
                qk_tiled_mma, pv_tiled_mma, pv_thr_mma, tmem,
                tStS, tSrQ, tSrK_trans, tOtO_staged, tOrV_trans, p_tmem_layout,
                load_q_consumer, dequant_kv_consumer,
                mma_s_producer, p_mma_consumer, mma_o_producer,
                seqlen_q, seqlen_k, window_size_left, window_size_right,
                tile_sched_params,
            )

        # Softmax warps: warp_idx in [softmax_warp_ids[0], correction_warp_ids[0]).
        # Consume S tiles from TMEM, apply masking + numerically stable
        # softmax, produce P back into TMEM, publish rescale stats + final
        # row sum to the correction warps via s_corr / sum.
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

        # Correction warps: warp_idx in [correction_warp_ids[0], mma_warp_id).
        # Apply per-tile rescaling, normalize by row sum, and write the final
        # O tile to global memory through the ``epi_tile`` epilogue tiler.
        # Also owns TMEM alloc/dealloc.
        if warp_idx < self.mma_warp_id and warp_idx >= self.correction_warp_ids[0]:
            self.correction_warp_body(
                qk_tiled_mma, qk_thr_mma, tmem, tStS, tOtO_staged, sSum,
                gO_qdl, cO_qdl, scale_softmax_log2, scale_output, seqlen_q, seqlen_k,
                window_size_left, window_size_right, epi_tile,
                s_corr_consumer, mma_o_consumer, sum_consumer,
                tile_sched_params,
            )

        # Transform warps: warp_idx in [0, softmax_warp_ids[0]) excluding the
        # load warp.  Cooperatively dequantize INT8 KV from sK/sV into BF16
        # sK_trans/sV_trans, gated by load_kv / load_scale_* consumers and
        # publishing readiness via dequant_kv_producer.
        if warp_idx < self.softmax_warp_ids[0]:
            self.transform_warp_body(
                qk_tiled_mma, pv_tiled_mma,
                sK, sV, sK_trans, sV_trans, sScaleK_s2r_view, sScaleV_s2r_view,
                seqlen_q, seqlen_k, window_size_left, window_size_right,
                load_kv_consumer, load_scale_k_consumer, load_scale_v_consumer,
                dequant_kv_producer, tile_sched_params,
                cum_seqlen_k,
            )

        # The load warp keeps its full register budget (it needs few but each
        # ``*_warp_body`` above tunes its own budget via setmaxregister_increase).
        # All other warps reduce their register usage so the SM can hold more
        # CTAs simultaneously (improves occupancy for the next wave).
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
