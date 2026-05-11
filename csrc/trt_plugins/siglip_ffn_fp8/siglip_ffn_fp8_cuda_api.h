// Copyright 2026 the model_optimizer team.
//
// SPDX-License-Identifier: Apache-2.0
//
// C++ host API for SigLIP FFN FP8 CUDA path (cuBLASLt + static FP8 quant kernel).

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace mopt_cuda {
struct SiglipFfRtContext;
}
using MoptSiglipFfRt = mopt_cuda::SiglipFfRtContext;

MoptSiglipFfRt* mopt_siglip_ffn_rt_create();
void mopt_siglip_ffn_rt_destroy(MoptSiglipFfRt* rt);

// Upper bound on TRT workspace: hidden_fp16 + hid_fp8 + cuBLASLt scratch (heuristic at S_max).
size_t mopt_siglip_ffn_query_workspace_bytes(int32_t s_max, int32_t d, int32_t h);

// Fused: gelu(up) -> static fp8 quant -> down + bias + residual into y_out.
// Tensor layouts match FlashRT GemmRunner fp8_nn_* (row-major x [S,D], up_w [D,H], down_w [H,D]).
int32_t mopt_siglip_ffn_enqueue(
    MoptSiglipFfRt* rt,
    int32_t s,
    int32_t d,
    int32_t h,
    float alpha_up,
    float alpha_down,
    void const* x_fp8,
    void const* residual_fp16,
    void const* up_w_fp8,
    void const* down_w_fp8,
    void const* up_b_fp16,
    void const* down_b_fp16,
    void const* unit_scale_fp32,
    void* y_fp16,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream);
