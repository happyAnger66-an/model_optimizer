// Copyright 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace mopt_trt {

/// act_id matches Python ``_encode_hidden_act`` (gelu_pytorch_tanh = 1).
/// io_type: 0 = FP16 (kHALF), 1 = BF16 (kBF16).
int gemma_fused_gated_mlp_cuda(cudaStream_t stream, int32_t act_id, int32_t io_type, int32_t m, int32_t hidden,
    int32_t inter, void const* x, void const* wGateUp, void const* wDown, void* y, void* workspace,
    size_t workspaceBytes);

size_t gemma_fused_gated_mlp_workspace_bytes(int32_t m, int32_t hidden, int32_t inter, int32_t io_type);

} // namespace mopt_trt
