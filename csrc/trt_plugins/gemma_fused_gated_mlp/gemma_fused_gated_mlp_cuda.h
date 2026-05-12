// Copyright 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// 性能扩展（未在此头实现，供后续迭代）：
//
// 1) CUTLASS 3.x / CuTe：在 Blackwell/Thor（SM100）上对两次 GEMM 使用 device 侧已调好的 tile（类似
//    “wide / t1 / plain” 分桶），并可探索 epilogue 融合 GeGLU（需单独数值验证与 TRT 插件 ABI 对齐）。
// 2) FP8 权重 + FP16/BF16 累加：吞吐显著高于纯 BF16 GEMM，但需改 ONNX/引擎权重布局与标定流程。
// 3) 双流：仅当存在可重叠的独立算子时有效；本 FFN 链为严格顺序，收益有限。
//
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
