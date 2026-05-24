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
"""Host-side configuration for CUTLASS d=256 mixed-input FMHA prefill."""

import math
from typing import Tuple, Type

import cutlass
import cutlass.cute as cute

from fmha_d256_cutedsl import fmha_helpers as fmha_utils


class MixedInputFusedMultiHeadAttentionPrefillD256:
    def __init__(
        self,
        scale_granularity: int,
        qk_acc_dtype: Type[cutlass.Numeric],
        pv_acc_dtype: Type[cutlass.Numeric],
        is_persistent: bool,
        mask_type: fmha_utils.MaskEnum,
        is_mixed_input: bool = True,
    ):
        self.is_mixed_input = is_mixed_input
        self.head_dim = 256
        self.inv_sqrt_head_dim = 1.0 / math.sqrt(self.head_dim)
        self.log2_e = math.log2(math.exp(1.0))
        self.qk_acc_dtype = qk_acc_dtype
        self.pv_acc_dtype = pv_acc_dtype
        self.cta_tiler = (128, 128, 256)
        self.qk_mma_tiler = (
            self.cta_tiler[0] * 2,  # default 2cta
            self.cta_tiler[1],
            min(self.cta_tiler[2], 128),  # avoid too large GemmK
        )
        self.pv_mma_tiler = self.qk_mma_tiler  # keep BMM1 & BMM2 at the same pace
        self.pv_block_tiler = (
            self.pv_mma_tiler[0] // 2,  # default 2cta
            self.pv_mma_tiler[1],
            self.pv_mma_tiler[2],
        )
        self.scale_granularity = scale_granularity
        self.iterations_qk = self.cta_tiler[2] // self.qk_mma_tiler[2]
        self.iterations_pv = self.cta_tiler[2] // self.pv_mma_tiler[1]
        self.cluster_shape_mn = (2, 1)  # use 2x1 cluster by default
        self.tmem_warp_shape_mn = (4, 1)
        self.is_persistent = is_persistent
        self.mask_type = mask_type
        self.transform_warp_ids = (0, 1, 2, 3, 4, 5, 6, 7)  # i8 -> bf16 for kv
        self.softmax_warp_ids = (8, 9, 10, 11)  # softmax
        self.correction_warp_ids = (12, 13, 14, 15)  # correction
        self.mma_warp_id = 16  # mma
        self.load_warp_id = 17  # load
        self.empty_warp_ids = (18, 19)  # empty
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        self.tmem_alloc_sync_bar_id = 1
        self.tmem_s_offset = 256
        self.tmem_p_offset = self.tmem_s_offset
        self.tmem_o_offset = 0
        self.num_regs_softmax = 256
        self.num_regs_correction = 112
        self.num_regs_other = 32
        self.num_regs_transform = 40
        self.buffer_align_bytes = 1024
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                *self.transform_warp_ids,
                *self.softmax_warp_ids,
                *self.correction_warp_ids,
                self.load_warp_id,
                self.mma_warp_id,
                *self.empty_warp_ids,
            )
        )

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        self.q_stage = self.iterations_qk
        self.kv_stage = 4
        self.scale_k_stage = self.kv_stage
        self.scale_v_stage = self.kv_stage
        self.qk_acc_stage = 2
        self.pv_acc_stage = 1
        self.kv_trans_stage = 2
