// Copyright 2026 the model_optimizer team.
//
// SPDX-License-Identifier: Apache-2.0
//
// cuBLASLt FP8 GEMM epilogues + static FP8 quantize, aligned with FlashRT GemmRunner::fp8_nn_gelu_bias /
// fp8_nn_bias_res and quantize_fp8_static_fp16.

#include "siglip_ffn_fp8_cuda_api.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#define MOPT_CUBLAS_CHECK(expr)                                                                 \
    do {                                                                                        \
        cublasStatus_t st = (expr);                                                             \
        if (st != CUBLAS_STATUS_SUCCESS) {                                                     \
            throw std::runtime_error(std::string("cuBLASLt error ") + __FILE__ + ":" +          \
                std::to_string(__LINE__) + " code=" + std::to_string(static_cast<int>(st)));    \
        }                                                                                       \
    } while (0)

#define MOPT_CUDA_CHECK(expr)                                                                   \
    do {                                                                                        \
        cudaError_t e = (expr);                                                               \
        if (e != cudaSuccess) {                                                               \
            throw std::runtime_error(std::string("CUDA error ") + __FILE__ + ":" +              \
                std::to_string(__LINE__) + ": " + cudaGetErrorString(e));                       \
        }                                                                                       \
    } while (0)

namespace mopt_cuda {

// Heuristic workspace cap for cublasLtMatmulAlgoGetHeuristic (must match query vs enqueue cache).
constexpr size_t kLtPrefWorkspaceBytes = 32ULL * 1024ULL * 1024ULL;

inline size_t align256(size_t x) { return (x + 255ULL) & ~255ULL; }

// ---------------------------------------------------------------------------
// FP8 static quant kernel (derived from FlashRT third_party/FlashRT/csrc/kernels/quantize.cu,
// Apache-2.0). Kept structurally identical for bitwise alignment with FlashRT encoder path.
// ---------------------------------------------------------------------------
__global__ void mopt_quantize_fp8_kernel_fp16_row(
    __half const* __restrict__ in, __nv_fp8_e4m3* __restrict__ out, float const* descale_ptr, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= n) {
        return;
    }
    float inv_scale = 1.0f / fmaxf(*descale_ptr, 1e-12f);
    __half2 const* in2 = reinterpret_cast<__half2 const*>(in);
    __half2 vA = in2[i / 2];
    __half2 vB = in2[i / 2 + 1];
    float fv[4] = {__half2float(vA.x), __half2float(vA.y), __half2float(vB.x), __half2float(vB.y)};
    __nv_fp8_e4m3 fp8_pack[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        fp8_pack[j] = __nv_fp8_e4m3(fminf(fmaxf(fv[j] * inv_scale, -448.f), 448.f));
    }
    *reinterpret_cast<uint32_t*>(out + i) = *reinterpret_cast<uint32_t*>(fp8_pack);
}

void launch_quantize_fp8_static_fp16(
    __half const* input, __nv_fp8_e4m3* output, float const* d_scale, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n / 4 + threads - 1) / threads;
    if (blocks < 1) {
        blocks = 1;
    }
    mopt_quantize_fp8_kernel_fp16_row<<<blocks, threads, 0, stream>>>(input, output, d_scale, n);
}

struct GemmKey {
    int type;
    int M;
    int N;
    int K;
    bool operator==(GemmKey const& o) const { return type == o.type && M == o.M && N == o.N && K == o.K; }
};

struct GemmKeyHash {
    size_t operator()(GemmKey const& k) const {
        size_t h = static_cast<size_t>(k.type);
        h ^= static_cast<size_t>(k.M) + 0x9e3779b9ULL + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(k.N) + 0x9e3779b9ULL + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(k.K) + 0x9e3779b9ULL + (h << 6) + (h >> 2);
        return h;
    }
};

struct CachedLtGemm {
    cublasLtMatmulDesc_t matmul_desc{};
    cublasLtMatrixLayout_t A_desc{};
    cublasLtMatrixLayout_t B_desc{};
    cublasLtMatrixLayout_t D_desc{};
    cublasLtMatmulAlgo_t algo{};
    size_t workspace_size{0};
    bool valid{false};

    void destroy() {
        if (!valid) {
            return;
        }
        cublasLtMatmulDescDestroy(matmul_desc);
        cublasLtMatrixLayoutDestroy(A_desc);
        cublasLtMatrixLayoutDestroy(B_desc);
        cublasLtMatrixLayoutDestroy(D_desc);
        valid = false;
    }
};

void create_fp8_nn_gelu_bias_entry(
    cublasLtHandle_t h, int M, int N, int K, CachedLtGemm* out, size_t* heuristic_ws) {
    cublasOperation_t opN = CUBLAS_OP_N;
    MOPT_CUBLAS_CHECK(cublasLtMatmulDescCreate(&out->matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    MOPT_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        out->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
    MOPT_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        out->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_GELU_BIAS;
    MOPT_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        out->matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
    cudaDataType_t btype = CUDA_R_16F;
    MOPT_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        out->matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &btype, sizeof(btype)));
    MOPT_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&out->A_desc, CUDA_R_8F_E4M3, N, K, N));
    MOPT_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&out->B_desc, CUDA_R_8F_E4M3, K, M, K));
    MOPT_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&out->D_desc, CUDA_R_16F, N, M, N));

    cublasLtMatmulPreference_t pref{};
    MOPT_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    MOPT_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &kLtPrefWorkspaceBytes, sizeof(kLtPrefWorkspaceBytes)));
    cublasLtMatmulHeuristicResult_t result{};
    int ret = 0;
    MOPT_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(h, out->matmul_desc, out->A_desc, out->B_desc, out->D_desc,
        out->D_desc, pref, 1, &result, &ret));
    MOPT_CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    if (result.state != CUBLAS_STATUS_SUCCESS || ret < 1) {
        out->destroy();
        throw std::runtime_error("cublasLtMatmulAlgoGetHeuristic failed for fp8_nn_gelu_bias");
    }
    out->algo = result.algo;
    out->workspace_size = result.workspaceSize;
    out->valid = true;
    *heuristic_ws = result.workspaceSize;
}

void create_fp8_nn_bias_res_entry(
    cublasLtHandle_t h, int M, int N, int K, CachedLtGemm* out, size_t* heuristic_ws) {
    cublasOperation_t opN = CUBLAS_OP_N;
    MOPT_CUBLAS_CHECK(cublasLtMatmulDescCreate(&out->matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    MOPT_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        out->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
    MOPT_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        out->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    MOPT_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        out->matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
    MOPT_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&out->A_desc, CUDA_R_8F_E4M3, N, K, N));
    MOPT_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&out->B_desc, CUDA_R_8F_E4M3, K, M, K));
    MOPT_CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&out->D_desc, CUDA_R_16F, N, M, N));

    cublasLtMatmulPreference_t pref{};
    MOPT_CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    MOPT_CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &kLtPrefWorkspaceBytes, sizeof(kLtPrefWorkspaceBytes)));
    cublasLtMatmulHeuristicResult_t result{};
    int ret = 0;
    MOPT_CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(h, out->matmul_desc, out->A_desc, out->B_desc, out->D_desc,
        out->D_desc, pref, 1, &result, &ret));
    MOPT_CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    if (result.state != CUBLAS_STATUS_SUCCESS || ret < 1) {
        out->destroy();
        throw std::runtime_error("cublasLtMatmulAlgoGetHeuristic failed for fp8_nn_bias_res");
    }
    out->algo = result.algo;
    out->workspace_size = result.workspaceSize;
    out->valid = true;
    *heuristic_ws = result.workspaceSize;
}

size_t query_lt_workspace_pair(int32_t s_max, int32_t d, int32_t h) {
    cublasLtHandle_t hnd{};
    if (cublasLtCreate(&hnd) != CUBLAS_STATUS_SUCCESS) {
        return 4ULL * 1024ULL * 1024ULL;
    }
    size_t w_gelu = 0;
    size_t w_down = 0;
    try {
        CachedLtGemm g1;
        create_fp8_nn_gelu_bias_entry(hnd, s_max, h, d, &g1, &w_gelu);
        g1.destroy();
        CachedLtGemm g2;
        create_fp8_nn_bias_res_entry(hnd, s_max, d, h, &g2, &w_down);
        g2.destroy();
    } catch (...) {
        cublasLtDestroy(hnd);
        return 4ULL * 1024ULL * 1024ULL;
    }
    cublasLtDestroy(hnd);
    size_t mx = w_gelu > w_down ? w_gelu : w_down;
    return mx + 65536ULL;
}

struct SiglipFfRtContext {
    cublasLtHandle_t handle{};
    std::mutex mu;
    std::unordered_map<GemmKey, std::unique_ptr<CachedLtGemm>, GemmKeyHash> cache{};

    void clear_cache() {
        for (auto& p : cache) {
            if (p.second) {
                p.second->destroy();
            }
        }
        cache.clear();
    }

    CachedLtGemm& get_gelu(int M, int N, int K) {
        GemmKey key{102, M, N + 3000000, K};
        std::lock_guard<std::mutex> lk(mu);
        auto it = cache.find(key);
        if (it != cache.end()) {
            return *it->second;
        }
        auto ent = std::make_unique<CachedLtGemm>();
        size_t hw = 0;
        create_fp8_nn_gelu_bias_entry(handle, M, N, K, ent.get(), &hw);
        auto ins = cache.emplace(key, std::move(ent));
        return *ins.first->second;
    }

    CachedLtGemm& get_bias_res(int M, int N, int K) {
        GemmKey key{101, M, N + 4000000, K};
        std::lock_guard<std::mutex> lk(mu);
        auto it = cache.find(key);
        if (it != cache.end()) {
            return *it->second;
        }
        auto ent = std::make_unique<CachedLtGemm>();
        size_t hw = 0;
        create_fp8_nn_bias_res_entry(handle, M, N, K, ent.get(), &hw);
        auto ins = cache.emplace(key, std::move(ent));
        return *ins.first->second;
    }
};

} // namespace mopt_cuda

MoptSiglipFfRt* mopt_siglip_ffn_rt_create() {
    auto* rt = new mopt_cuda::SiglipFfRtContext();
    if (cublasLtCreate(&rt->handle) != CUBLAS_STATUS_SUCCESS) {
        delete rt;
        return nullptr;
    }
    return rt;
}

void mopt_siglip_ffn_rt_destroy(MoptSiglipFfRt* rt) {
    if (rt == nullptr) {
        return;
    }
    rt->clear_cache();
    cublasLtDestroy(rt->handle);
    delete rt;
}

size_t mopt_siglip_ffn_query_workspace_bytes(int32_t s_max, int32_t d, int32_t h) {
    if (s_max <= 0 || d <= 0 || h <= 0) {
        return 0;
    }
    size_t internal = mopt_cuda::align256(static_cast<size_t>(s_max) * static_cast<size_t>(h) * sizeof(__half));
    internal += mopt_cuda::align256(static_cast<size_t>(s_max) * static_cast<size_t>(h) * sizeof(__nv_fp8_e4m3));
    size_t lt = mopt_cuda::query_lt_workspace_pair(s_max, d, h);
    return internal + mopt_cuda::align256(lt);
}

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
    cudaStream_t stream) {
    if (rt == nullptr || s <= 0 || d <= 0 || h <= 0) {
        return 1;
    }
    try {
        size_t off0 = 0;
        size_t hidden_bytes = static_cast<size_t>(s) * static_cast<size_t>(h) * sizeof(__half);
        size_t off1 = mopt_cuda::align256(off0 + hidden_bytes);
        size_t hid_fp8_bytes = static_cast<size_t>(s) * static_cast<size_t>(h) * sizeof(__nv_fp8_e4m3);
        size_t off2 = mopt_cuda::align256(off1 + hid_fp8_bytes);
        if (workspace_size < off2 + 1) {
            return 2;
        }
        void* lt_ptr = reinterpret_cast<char*>(workspace) + off2;
        size_t lt_cap = workspace_size - off2;

        __half* hidden = reinterpret_cast<__half*>(reinterpret_cast<char*>(workspace) + off0);
        __nv_fp8_e4m3* hid_fp8 = reinterpret_cast<__nv_fp8_e4m3*>(reinterpret_cast<char*>(workspace) + off1);

        if (static_cast<void const*>(y_fp16) != residual_fp16) {
            MOPT_CUDA_CHECK(cudaMemcpyAsync(y_fp16, residual_fp16,
                static_cast<size_t>(s) * static_cast<size_t>(d) * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
        }

        auto& g_gelu = rt->get_gelu(s, h, d);
        auto& g_down = rt->get_bias_res(s, d, h);
        size_t lt_need = g_gelu.workspace_size > g_down.workspace_size ? g_gelu.workspace_size : g_down.workspace_size;
        if (lt_need > lt_cap) {
            return 3;
        }
        MOPT_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            g_gelu.matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &up_b_fp16, sizeof(up_b_fp16)));
        float beta0 = 0.0f;
        MOPT_CUBLAS_CHECK(cublasLtMatmul(rt->handle, g_gelu.matmul_desc, &alpha_up, up_w_fp8, g_gelu.A_desc, x_fp8,
            g_gelu.B_desc, &beta0, hidden, g_gelu.D_desc, hidden, g_gelu.D_desc, &g_gelu.algo, lt_ptr,
            g_gelu.workspace_size, stream));

        int n_elts = s * h;
        mopt_cuda::launch_quantize_fp8_static_fp16(hidden, hid_fp8,
            reinterpret_cast<float const*>(unit_scale_fp32), n_elts, stream);
        MOPT_CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            g_down.matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &down_b_fp16, sizeof(down_b_fp16)));
        float beta1 = 1.0f;
        MOPT_CUBLAS_CHECK(cublasLtMatmul(rt->handle, g_down.matmul_desc, &alpha_down, down_w_fp8, g_down.A_desc,
            hid_fp8, g_down.B_desc, &beta1, y_fp16, g_down.D_desc, y_fp16, g_down.D_desc, &g_down.algo, lt_ptr,
            g_down.workspace_size, stream));
    } catch (...) {
        return 5;
    }
    return 0;
}
