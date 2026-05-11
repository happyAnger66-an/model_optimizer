// Copyright 2026 the model_optimizer team.
//
// SPDX-License-Identifier: Apache-2.0
//
// SigLIP fused MLP for TensorRT: two row-major GEMMs (cuBLAS) + bias + activation (fp32),
// aligned with model_optimizer.ops.siglip_mlp.siglip_mlp_eager.

#include "siglip_mlp_cuda_api.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdint>

#define MOPT_CUBLAS_CHECK(expr)                                                                 \
    do {                                                                                        \
        cublasStatus_t st = (expr);                                                             \
        if (st != CUBLAS_STATUS_SUCCESS) {                                                     \
            return 4;                                                                          \
        }                                                                                       \
    } while (0)

#define MOPT_CUDA_CHECK(expr)                                                                   \
    do {                                                                                        \
        cudaError_t e = (expr);                                                                \
        if (e != cudaSuccess) {                                                                \
            return 6;                                                                           \
        }                                                                                       \
    } while (0)

namespace mopt_cuda {

inline size_t align256(size_t x) { return (x + 255ULL) & ~255ULL; }

__device__ float gelu_tanh_approx(float x) {
    float const c = 0.044715f;
    float const k = 0.7978845608028654f; // sqrt(2/pi)
    float x3 = x * x * x;
    float t = tanhf(k * (x + c * x3));
    return 0.5f * x * (1.0f + t);
}

__device__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

__device__ float act_forward(float v, int act_id) {
    if (act_id == 0 || act_id == 1) {
        return gelu_tanh_approx(v);
    }
    if (act_id == 2) {
        return fmaxf(v, 0.0f);
    }
    if (act_id == 3) {
        return v / (1.0f + expf(-v));
    }
    if (act_id == 4) {
        return v * (1.0f / (1.0f + expf(-1.702f * v)));
    }
    if (act_id == 5) {
        return gelu_exact(v);
    }
    return gelu_tanh_approx(v);
}

__global__ void apply_act_fp32_kernel(float* data, int n, int act_id) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    data[i] = act_forward(data[i], act_id);
}

__global__ void add_bias_lastdim_f32(float* a, float const* b, int s, int last) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = s * last;
    if (idx >= n) {
        return;
    }
    int j = idx % last;
    a[idx] += b[j];
}

__global__ void add_bias_lastdim_f16(__half* a, __half const* b, int s, int last) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = s * last;
    if (idx >= n) {
        return;
    }
    int j = idx % last;
    a[idx] = __hadd(a[idx], b[j]);
}

__global__ void add_bias_lastdim_bf16(__nv_bfloat16* a, __nv_bfloat16 const* b, int s, int last) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = s * last;
    if (idx >= n) {
        return;
    }
    int j = idx % last;
    float av = __bfloat162float(a[idx]);
    float bv = __bfloat162float(b[j]);
    a[idx] = __float2bfloat16_rn(av + bv);
}

__global__ void f16_to_f32_kernel(__half const* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    dst[i] = __half2float(src[i]);
}

__global__ void bf16_to_f32_kernel(__nv_bfloat16 const* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    dst[i] = __bfloat162float(src[i]);
}

__global__ void f32_to_f16_kernel(float const* src, __half* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    dst[i] = __float2half_rn(src[i]);
}

__global__ void f32_to_bf16_kernel(float const* src, __nv_bfloat16* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    dst[i] = __float2bfloat16_rn(src[i]);
}

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

} // namespace mopt_cuda

size_t mopt_siglip_mlp_query_workspace_bytes(int32_t s_max, int32_t d, int32_t h) {
    if (s_max <= 0 || d <= 0 || h <= 0) {
        return 0;
    }
    (void)d;
    int const n_hid = s_max * h;
    size_t const act_f32 = mopt_cuda::align256(static_cast<size_t>(n_hid) * sizeof(float));
    size_t const hid_aux = mopt_cuda::align256(static_cast<size_t>(n_hid) * sizeof(__half));
    return act_f32 + hid_aux;
}

int32_t mopt_siglip_mlp_cuda_enqueue(cudaStream_t stream, cublasHandle_t handle, int32_t dtype, int32_t act_id,
    int32_t s, int32_t d, int32_t h, void const* x, void const* fc1_w, void const* fc1_b, void const* fc2_w,
    void const* fc2_b, void* y, void* workspace, size_t workspace_bytes) {
    if (s <= 0 || d <= 0 || h <= 0 || handle == nullptr) {
        return 1;
    }
    MOPT_CUBLAS_CHECK(cublasSetStream(handle, stream));
    int const n_hid = s * h;
    size_t const need_f32 = mopt_cuda::align256(static_cast<size_t>(n_hid) * sizeof(float));
    size_t const need_half = mopt_cuda::align256(static_cast<size_t>(n_hid) * sizeof(__half));
    size_t const need_total = need_f32 + need_half;
    if (workspace_bytes < need_total) {
        return 2;
    }
    float* act_buf = reinterpret_cast<float*>(workspace);
    int threads = 256;

    float alpha = 1.0f;
    float beta0 = 0.0f;

    if (dtype == 0) {
        float const* x_f = static_cast<float const*>(x);
        float const* w1 = static_cast<float const*>(fc1_w);
        float const* b1 = static_cast<float const*>(fc1_b);
        float const* w2 = static_cast<float const*>(fc2_w);
        float const* b2 = static_cast<float const*>(fc2_b);
        float* y_f = static_cast<float*>(y);
        float* hid = act_buf; // reuse: first n_hid floats for hidden after gemm1, then overwritten by act
        // Gemm1: hid[S,H] = X[S,D] * W1^T ; W1 [H,D] row-major
        MOPT_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, h, s, d, &alpha, w1, d, x_f, d, &beta0, hid, h));
        int n_bias1 = n_hid;
        mopt_cuda::add_bias_lastdim_f32<<<mopt_cuda::ceil_div(n_bias1, threads), threads, 0, stream>>>(
            hid, b1, s, h);
        MOPT_CUDA_CHECK(cudaGetLastError());
        mopt_cuda::apply_act_fp32_kernel<<<mopt_cuda::ceil_div(n_hid, threads), threads, 0, stream>>>(
            hid, n_hid, act_id);
        MOPT_CUDA_CHECK(cudaGetLastError());
        // Gemm2: y[S,D] = hid[S,H] * W2^T ; W2 [D,H]
        MOPT_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, d, s, h, &alpha, w2, h, hid, h, &beta0, y_f, d));
        int n_bias2 = s * d;
        mopt_cuda::add_bias_lastdim_f32<<<mopt_cuda::ceil_div(n_bias2, threads), threads, 0, stream>>>(
            y_f, b2, s, d);
        MOPT_CUDA_CHECK(cudaGetLastError());
        return 0;
    }

    if (dtype == 1) {
        __half const* x_h = static_cast<__half const*>(x);
        __half const* w1 = static_cast<__half const*>(fc1_w);
        __half const* b1 = static_cast<__half const*>(fc1_b);
        __half const* w2 = static_cast<__half const*>(fc2_w);
        __half const* b2 = static_cast<__half const*>(fc2_b);
        __half* y_h = static_cast<__half*>(y);
        __half* hid = reinterpret_cast<__half*>(reinterpret_cast<char*>(workspace) + need_f32);
        MOPT_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, h, s, d, &alpha, w1, CUDA_R_16F, d, x_h,
            CUDA_R_16F, d, &beta0, hid, CUDA_R_16F, h, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        mopt_cuda::add_bias_lastdim_f16<<<mopt_cuda::ceil_div(n_hid, threads), threads, 0, stream>>>(
            hid, b1, s, h);
        MOPT_CUDA_CHECK(cudaGetLastError());
        mopt_cuda::f16_to_f32_kernel<<<mopt_cuda::ceil_div(n_hid, threads), threads, 0, stream>>>(hid, act_buf, n_hid);
        MOPT_CUDA_CHECK(cudaGetLastError());
        mopt_cuda::apply_act_fp32_kernel<<<mopt_cuda::ceil_div(n_hid, threads), threads, 0, stream>>>(
            act_buf, n_hid, act_id);
        MOPT_CUDA_CHECK(cudaGetLastError());
        mopt_cuda::f32_to_f16_kernel<<<mopt_cuda::ceil_div(n_hid, threads), threads, 0, stream>>>(act_buf, hid, n_hid);
        MOPT_CUDA_CHECK(cudaGetLastError());
        MOPT_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, d, s, h, &alpha, w2, CUDA_R_16F, h, hid,
            CUDA_R_16F, h, &beta0, y_h, CUDA_R_16F, d, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        int n_bias2 = s * d;
        mopt_cuda::add_bias_lastdim_f16<<<mopt_cuda::ceil_div(n_bias2, threads), threads, 0, stream>>>(
            y_h, b2, s, d);
        MOPT_CUDA_CHECK(cudaGetLastError());
        return 0;
    }

    if (dtype == 2) {
        __nv_bfloat16 const* x_b = static_cast<__nv_bfloat16 const*>(x);
        __nv_bfloat16 const* w1 = static_cast<__nv_bfloat16 const*>(fc1_w);
        __nv_bfloat16 const* b1 = static_cast<__nv_bfloat16 const*>(fc1_b);
        __nv_bfloat16 const* w2 = static_cast<__nv_bfloat16 const*>(fc2_w);
        __nv_bfloat16 const* b2 = static_cast<__nv_bfloat16 const*>(fc2_b);
        __nv_bfloat16* y_b = static_cast<__nv_bfloat16*>(y);
        __nv_bfloat16* hid = reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<char*>(workspace) + need_f32);
        MOPT_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, h, s, d, &alpha, w1, CUDA_R_16BF, d, x_b,
            CUDA_R_16BF, d, &beta0, hid, CUDA_R_16BF, h, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        mopt_cuda::add_bias_lastdim_bf16<<<mopt_cuda::ceil_div(n_hid, threads), threads, 0, stream>>>(
            hid, b1, s, h);
        MOPT_CUDA_CHECK(cudaGetLastError());
        mopt_cuda::bf16_to_f32_kernel<<<mopt_cuda::ceil_div(n_hid, threads), threads, 0, stream>>>(hid, act_buf, n_hid);
        MOPT_CUDA_CHECK(cudaGetLastError());
        mopt_cuda::apply_act_fp32_kernel<<<mopt_cuda::ceil_div(n_hid, threads), threads, 0, stream>>>(
            act_buf, n_hid, act_id);
        MOPT_CUDA_CHECK(cudaGetLastError());
        mopt_cuda::f32_to_bf16_kernel<<<mopt_cuda::ceil_div(n_hid, threads), threads, 0, stream>>>(act_buf, hid, n_hid);
        MOPT_CUDA_CHECK(cudaGetLastError());
        MOPT_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, d, s, h, &alpha, w2, CUDA_R_16BF, h, hid,
            CUDA_R_16BF, h, &beta0, y_b, CUDA_R_16BF, d, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        int n_bias2 = s * d;
        mopt_cuda::add_bias_lastdim_bf16<<<mopt_cuda::ceil_div(n_bias2, threads), threads, 0, stream>>>(
            y_b, b2, s, d);
        MOPT_CUDA_CHECK(cudaGetLastError());
        return 0;
    }
    return 3;
}
