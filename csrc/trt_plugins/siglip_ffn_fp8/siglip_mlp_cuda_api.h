// Copyright 2026 the model_optimizer team.
//
// SPDX-License-Identifier: Apache-2.0
//
// CUDA host API for SigLIP MLP TensorRT plugin (Linear → act → Linear).

#pragma once

#include <cstddef>
#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

// dtype: 0=float32, 1=float16, 2=bfloat16 (must match plugin mapping).
// x: [S, D] row-major; fc1_w: [H, D]; fc1_b: [H]; fc2_w: [D, H]; fc2_b: [D]; y: [S, D] row-major.
// act_id matches model_optimizer.ops.siglip_mlp._act_fn_from_id.
int32_t mopt_siglip_mlp_cuda_enqueue(cudaStream_t stream, cublasHandle_t handle, int32_t dtype, int32_t act_id,
    int32_t s, int32_t d, int32_t h, void const* x, void const* fc1_w, void const* fc1_b, void const* fc2_w,
    void const* fc2_b, void* y, void* workspace, size_t workspace_bytes);

// Upper bound: fp32 scratch for activation (S * max(H, D)) + small alignment.
size_t mopt_siglip_mlp_query_workspace_bytes(int32_t s_max, int32_t d, int32_t h);
