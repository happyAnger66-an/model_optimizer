// Copyright 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
//
// Gemma GeGLU FFN（fp16/bf16）：两路 cuBLASLt row-major GEMM（D = A @ B^T）+ 中间 GeGLU（tanh）逐点。
//
// ---------------------------------------------------------------------------
// 优化方案（借鉴 FlashRT / CUTLASS 思路，而非照搬其 FP8 管线）
// ---------------------------------------------------------------------------
// 1) GEMM 主体：继续用 cuBLASLt（成熟、与 TRT 插件 ABI 简单），通过「更大 workspace 上限 + 多启发式候选」
//    让库在 Thor/大 N（gate_up 的 N=2I）上更容易选到 Tensor Core 友好算法。
// 2) 内存带宽：GeGLU 位于两次 GEMM 之间，对 z[M,2I] 读一遍、写 h[M,I]；用向量化（一次处理 4 列）降低指令与访存开销。
// 3) CUTLASS 3.x / CuTe（下一阶段）：对固定形状（如 M=968,H=2048,I=16384）可引入 CUTLASS device_gemm + 可选 epilogue
//    融合激活；需单独 CMake 目标与数值对齐测试。参见 ``gemma_fused_gated_mlp_cuda.h`` 注释。
// 4) FP8 权重路径（可选）：吞吐潜力最大，但需改权重/标定与 ONNX，超出本文件范围。
// ---------------------------------------------------------------------------

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "gemma_fused_gated_mlp_cuda.h"

namespace mopt_trt {

namespace {

#define MOPT_CUBLAS_LT_CHECK(expr)                                                                                    \
    do {                                                                                                              \
        cublasStatus_t st = (expr);                                                                                   \
        if (st != CUBLAS_STATUS_SUCCESS) {                                                                            \
            return 10 + static_cast<int>(st);                                                                         \
        }                                                                                                             \
    } while (0)

constexpr size_t kLtPrefWorkspaceGateUp = 64ULL * 1024ULL * 1024ULL; // gate_up：大 N=2*I，允许更大 scratch
constexpr size_t kLtPrefWorkspaceDown = 32ULL * 1024ULL * 1024ULL; // down：相对较小
constexpr size_t kLtPrefWorkspaceBytes = (kLtPrefWorkspaceGateUp > kLtPrefWorkspaceDown)
    ? kLtPrefWorkspaceGateUp
    : kLtPrefWorkspaceDown;
constexpr int kLtHeuristicMaxResults = 16;
constexpr int kCacheCap = 384;

inline size_t align256(size_t x) { return (x + 255ULL) & ~255ULL; }

__device__ __forceinline__ float gelu_tanh_f(float g) {
    return g / (1.0f + expf(-1.5957691216057308f * g * (1.0f + 0.044715f * g * g)));
}

// ---------------------------------------------------------------------------
// Fallback naive GEMM (correctness if cuBLASLt heuristic yields no algo)
// ---------------------------------------------------------------------------
__global__ void gemm_x_wgt_fp16(
    __half const* __restrict__ x, __half const* __restrict__ w, __half* __restrict__ z, int M, int H, int I2) {
    int mj = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * I2;
    if (mj >= total) {
        return;
    }
    int const row = mj / I2;
    int const col = mj % I2;
    float acc = 0.f;
    for (int k = 0; k < H; ++k) {
        acc += __half2float(x[row * H + k]) * __half2float(w[col * H + k]);
    }
    z[mj] = __float2half(acc);
}

__global__ void gemm_x_wgt_bf16(
    __nv_bfloat16 const* __restrict__ x, __nv_bfloat16 const* __restrict__ w, __nv_bfloat16* __restrict__ z, int M,
    int H, int I2) {
    int mj = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * I2;
    if (mj >= total) {
        return;
    }
    int const row = mj / I2;
    int const col = mj % I2;
    float acc = 0.f;
    for (int k = 0; k < H; ++k) {
        acc += __bfloat162float(x[row * H + k]) * __bfloat162float(w[col * H + k]);
    }
    z[mj] = __float2bfloat16(acc);
}

__global__ void gelu_mul_split_fp16(__half const* __restrict__ z, __half* __restrict__ h, int M, int inter) {
    int mi = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * inter;
    if (mi >= total) {
        return;
    }
    int const row = mi / inter;
    int const col = mi % inter;
    int const base = row * 2 * inter;
    float const g = __half2float(z[base + col]);
    float const u = __half2float(z[base + inter + col]);
    h[mi] = __float2half(gelu_tanh_f(g) * u);
}

__global__ void gelu_mul_split_bf16(
    __nv_bfloat16 const* __restrict__ z, __nv_bfloat16* __restrict__ h, int M, int inter) {
    int mi = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * inter;
    if (mi >= total) {
        return;
    }
    int const row = mi / inter;
    int const col = mi % inter;
    int const base = row * 2 * inter;
    float const g = __bfloat162float(z[base + col]);
    float const u = __bfloat162float(z[base + inter + col]);
    h[mi] = __float2bfloat16(gelu_tanh_f(g) * u);
}

//! 每个线程处理 4 列：降低 GeGLU 带宽与指令开销（要求 ``inter % 4 == 0``，Pi05 I=16384 满足）。
__global__ void gelu_mul_split_fp16_x4(__half const* __restrict__ z, __half* __restrict__ h, int M, int inter) {
    int const vec_cols = inter / 4;
    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    int const nvec = M * vec_cols;
    if (tid >= nvec) {
        return;
    }
    int const row = tid / vec_cols;
    int const v = tid % vec_cols;
    int const col = v * 4;
    int const base = row * 2 * inter;
    __half const* zg = z + base + col;
    __half const* zu = z + base + inter + col;
    __half* out = h + row * inter + col;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float const g = __half2float(zg[i]);
        float const u = __half2float(zu[i]);
        out[i] = __float2half(gelu_tanh_f(g) * u);
    }
}

__global__ void gelu_mul_split_bf16_x4(
    __nv_bfloat16 const* __restrict__ z, __nv_bfloat16* __restrict__ h, int M, int inter) {
    int const vec_cols = inter / 4;
    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    int const nvec = M * vec_cols;
    if (tid >= nvec) {
        return;
    }
    int const row = tid / vec_cols;
    int const v = tid % vec_cols;
    int const col = v * 4;
    int const base = row * 2 * inter;
    __nv_bfloat16 const* zg = z + base + col;
    __nv_bfloat16 const* zu = z + base + inter + col;
    __nv_bfloat16* out = h + row * inter + col;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float const g = __bfloat162float(zg[i]);
        float const u = __bfloat162float(zu[i]);
        out[i] = __float2bfloat16(gelu_tanh_f(g) * u);
    }
}

__global__ void gemm_h_wgt_fp16(
    __half const* __restrict__ h, __half const* __restrict__ w, __half* __restrict__ y, int M, int inter, int Hout) {
    int md = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * Hout;
    if (md >= total) {
        return;
    }
    int const row = md / Hout;
    int const col = md % Hout;
    float acc = 0.f;
    for (int k = 0; k < inter; ++k) {
        acc += __half2float(h[row * inter + k]) * __half2float(w[col * inter + k]);
    }
    y[md] = __float2half(acc);
}

__global__ void gemm_h_wgt_bf16(__nv_bfloat16 const* __restrict__ h, __nv_bfloat16 const* __restrict__ w,
    __nv_bfloat16* __restrict__ y, int M, int inter, int Hout) {
    int md = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * Hout;
    if (md >= total) {
        return;
    }
    int const row = md / Hout;
    int const col = md % Hout;
    float acc = 0.f;
    for (int k = 0; k < inter; ++k) {
        acc += __bfloat162float(h[row * inter + k]) * __bfloat162float(w[col * inter + k]);
    }
    y[md] = __float2bfloat16(acc);
}

// ---------------------------------------------------------------------------
// cuBLASLt: D_row(M,N) = A_row(M,K) @ B_row(N,K)^T  (same as FlashRT fp8_gemm comment block)
// ---------------------------------------------------------------------------
struct LtRowNtGemm {
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

struct LtNtKey {
    int dev{};
    int8_t io{};
    int8_t which{}; // 0 = gate_up, 1 = down
    int32_t m{};
    int32_t n{};
    int32_t k{};

    bool operator==(LtNtKey const& o) const {
        return dev == o.dev && io == o.io && which == o.which && m == o.m && n == o.n && k == o.k;
    }
};

struct LtNtKeyHash {
    size_t operator()(LtNtKey const& x) const noexcept {
        size_t h = static_cast<size_t>(static_cast<uint32_t>(x.dev));
        h ^= static_cast<size_t>(x.io) << 8;
        h ^= static_cast<size_t>(x.which) << 16;
        h ^= static_cast<size_t>(x.m) * 0x9e3779b97f4a7c15ULL;
        h ^= static_cast<size_t>(x.n) * 0x85ebca6b;
        h ^= static_cast<size_t>(x.k) * 0xc2b2ae35;
        return h;
    }
};

std::mutex g_lt_handle_mu;
std::mutex g_nt_cache_mu;
std::unordered_map<int, cublasLtHandle_t> g_lt_by_dev;
std::unordered_map<LtNtKey, std::unique_ptr<LtRowNtGemm>, LtNtKeyHash> g_nt_cache;

cublasLtHandle_t get_lt_for_device(int dev) {
    std::lock_guard<std::mutex> lk(g_lt_handle_mu);
    auto it = g_lt_by_dev.find(dev);
    if (it != g_lt_by_dev.end()) {
        return it->second;
    }
    cublasLtHandle_t h{};
    if (cublasLtCreate(&h) != CUBLAS_STATUS_SUCCESS) {
        return nullptr;
    }
    g_lt_by_dev[dev] = h;
    return h;
}

void clear_nt_cache_unlocked() {
    for (auto& e : g_nt_cache) {
        if (e.second) {
            e.second->destroy();
        }
    }
    g_nt_cache.clear();
}

cudaDataType_t io_cuda_type(int io_type) {
    return io_type == 1 ? CUDA_R_16BF : CUDA_R_16F;
}

/// Build D(M,N) = A(M,K) @ B(N,K)^T with row-major A,B,D (PyTorch Linear weights [N,K]).
/// ``which``：0 = gate_up（更大 workspace 上限），1 = down。
bool build_lt_row_nt(cublasLtHandle_t lt, int io_type, int32_t which, int M, int N, int K, LtRowNtGemm* out) {
    cudaDataType_t dt = io_cuda_type(io_type);
    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    cublasOperation_t op_N = CUBLAS_OP_N;
    cublasOperation_t op_T = CUBLAS_OP_T;

    if (cublasLtMatmulDescCreate(&out->matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    if (cublasLtMatmulDescSetAttribute(out->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N))
        != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulDescDestroy(out->matmul_desc);
        return false;
    }
    if (cublasLtMatmulDescSetAttribute(out->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_T, sizeof(op_T))
        != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulDescDestroy(out->matmul_desc);
        return false;
    }

    if (cublasLtMatrixLayoutCreate(&out->A_desc, dt, static_cast<uint64_t>(M), static_cast<uint64_t>(K), K)
        != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulDescDestroy(out->matmul_desc);
        return false;
    }
    if (cublasLtMatrixLayoutSetAttribute(out->A_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order))
        != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatrixLayoutDestroy(out->A_desc);
        cublasLtMatmulDescDestroy(out->matmul_desc);
        return false;
    }

    if (cublasLtMatrixLayoutCreate(&out->B_desc, dt, static_cast<uint64_t>(N), static_cast<uint64_t>(K), K)
        != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatrixLayoutDestroy(out->A_desc);
        cublasLtMatmulDescDestroy(out->matmul_desc);
        return false;
    }
    if (cublasLtMatrixLayoutSetAttribute(out->B_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order))
        != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatrixLayoutDestroy(out->B_desc);
        cublasLtMatrixLayoutDestroy(out->A_desc);
        cublasLtMatmulDescDestroy(out->matmul_desc);
        return false;
    }

    if (cublasLtMatrixLayoutCreate(&out->D_desc, dt, static_cast<uint64_t>(M), static_cast<uint64_t>(N), N)
        != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatrixLayoutDestroy(out->B_desc);
        cublasLtMatrixLayoutDestroy(out->A_desc);
        cublasLtMatmulDescDestroy(out->matmul_desc);
        return false;
    }
    if (cublasLtMatrixLayoutSetAttribute(out->D_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order))
        != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatrixLayoutDestroy(out->D_desc);
        cublasLtMatrixLayoutDestroy(out->B_desc);
        cublasLtMatrixLayoutDestroy(out->A_desc);
        cublasLtMatmulDescDestroy(out->matmul_desc);
        return false;
    }

    cublasLtMatmulPreference_t pref{};
    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatrixLayoutDestroy(out->D_desc);
        cublasLtMatrixLayoutDestroy(out->B_desc);
        cublasLtMatrixLayoutDestroy(out->A_desc);
        cublasLtMatmulDescDestroy(out->matmul_desc);
        return false;
    }
    size_t const pref_ws = (which == 0) ? kLtPrefWorkspaceGateUp : kLtPrefWorkspaceDown;
    cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &pref_ws, sizeof(pref_ws));

    cublasLtMatmulHeuristicResult_t heurs[kLtHeuristicMaxResults]{};
    int nret = 0;
    cublasStatus_t gh = cublasLtMatmulAlgoGetHeuristic(
        lt, out->matmul_desc, out->A_desc, out->B_desc, out->D_desc, out->D_desc, pref, kLtHeuristicMaxResults, heurs, &nret);
    cublasLtMatmulPreferenceDestroy(pref);
    if (gh != CUBLAS_STATUS_SUCCESS || nret < 1) {
        cublasLtMatrixLayoutDestroy(out->D_desc);
        cublasLtMatrixLayoutDestroy(out->B_desc);
        cublasLtMatrixLayoutDestroy(out->A_desc);
        cublasLtMatmulDescDestroy(out->matmul_desc);
        return false;
    }

    int pick = -1;
    size_t best_ws = 0;
    for (int i = 0; i < nret; ++i) {
        if (heurs[i].state != CUBLAS_STATUS_SUCCESS) {
            continue;
        }
        if (heurs[i].workspaceSize > pref_ws) {
            continue;
        }
        if (pick < 0 || heurs[i].workspaceSize > best_ws) {
            pick = i;
            best_ws = heurs[i].workspaceSize;
        }
    }
    if (pick < 0) {
        for (int i = 0; i < nret; ++i) {
            if (heurs[i].state == CUBLAS_STATUS_SUCCESS) {
                pick = i;
                break;
            }
        }
    }
    if (pick < 0) {
        cublasLtMatrixLayoutDestroy(out->D_desc);
        cublasLtMatrixLayoutDestroy(out->B_desc);
        cublasLtMatrixLayoutDestroy(out->A_desc);
        cublasLtMatmulDescDestroy(out->matmul_desc);
        return false;
    }

    out->algo = heurs[pick].algo;
    out->workspace_size = heurs[pick].workspaceSize;
    out->valid = true;
    return true;
}

LtRowNtGemm* get_or_create_nt_entry(int dev, int io_type, int which, int M, int N, int K, cublasLtHandle_t lt) {
    LtNtKey key{dev, static_cast<int8_t>(io_type), static_cast<int8_t>(which), M, N, K};
    std::lock_guard<std::mutex> lk(g_nt_cache_mu);
    auto it = g_nt_cache.find(key);
    if (it != g_nt_cache.end()) {
        return it->second.get();
    }
    if (static_cast<int>(g_nt_cache.size()) >= kCacheCap) {
        clear_nt_cache_unlocked();
    }
    auto up = std::make_unique<LtRowNtGemm>();
    if (!build_lt_row_nt(lt, io_type, key.which, M, N, K, up.get())) {
        return nullptr;
    }
    LtRowNtGemm* raw = up.get();
    g_nt_cache.emplace(key, std::move(up));
    return raw;
}

int launch_lt_row_nt(cublasLtHandle_t lt, LtRowNtGemm const& g, void const* A, void const* B, void* D, float alpha,
    float beta, void* lt_ws, size_t lt_ws_bytes, cudaStream_t stream) {
    if (g.workspace_size > lt_ws_bytes) {
        return 20;
    }
    MOPT_CUBLAS_LT_CHECK(cublasLtMatmul(lt, g.matmul_desc, &alpha, A, g.A_desc, B, g.B_desc, &beta, D, g.D_desc, D,
        g.D_desc, &g.algo, lt_ws, g.workspace_size, stream));
    return 0;
}

void launch_gelu_dispatch(cudaStream_t stream, int io_type, void const* zptr, void* hptr, int m, int inter) {
    int const threads = 256;
    int const total = m * inter;
    if ((inter % 4) == 0) {
        int const nvec = m * (inter / 4);
        dim3 const grid((nvec + threads - 1) / threads);
        if (io_type == 0) {
            gelu_mul_split_fp16_x4<<<grid, threads, 0, stream>>>(
                reinterpret_cast<__half const*>(zptr), reinterpret_cast<__half*>(hptr), m, inter);
        } else {
            gelu_mul_split_bf16_x4<<<grid, threads, 0, stream>>>(reinterpret_cast<__nv_bfloat16 const*>(zptr),
                reinterpret_cast<__nv_bfloat16*>(hptr), m, inter);
        }
    } else {
        dim3 const grid((total + threads - 1) / threads);
        if (io_type == 0) {
            gelu_mul_split_fp16<<<grid, threads, 0, stream>>>(
                reinterpret_cast<__half const*>(zptr), reinterpret_cast<__half*>(hptr), m, inter);
        } else {
            gelu_mul_split_bf16<<<grid, threads, 0, stream>>>(reinterpret_cast<__nv_bfloat16 const*>(zptr),
                reinterpret_cast<__nv_bfloat16*>(hptr), m, inter);
        }
    }
}

void launch_naive_fp16(cudaStream_t stream, int m, int hidden, int inter, void const* x, void const* wgu, void* z,
    void* h, void const* wd, void* y) {
    int const i2 = inter * 2;
    int const threads = 256;
    dim3 const g1((m * i2 + threads - 1) / threads);
    dim3 const g3((m * hidden + threads - 1) / threads);
    gemm_x_wgt_fp16<<<g1, threads, 0, stream>>>(
        reinterpret_cast<__half const*>(x), reinterpret_cast<__half const*>(wgu), reinterpret_cast<__half*>(z), m,
        hidden, i2);
    launch_gelu_dispatch(stream, 0, z, h, m, inter);
    gemm_h_wgt_fp16<<<g3, threads, 0, stream>>>(reinterpret_cast<__half const*>(h),
        reinterpret_cast<__half const*>(wd), reinterpret_cast<__half*>(y), m, inter, hidden);
}

void launch_naive_bf16(cudaStream_t stream, int m, int hidden, int inter, void const* x, void const* wgu, void* z,
    void* h, void const* wd, void* y) {
    int const i2 = inter * 2;
    int const threads = 256;
    dim3 const g1((m * i2 + threads - 1) / threads);
    dim3 const g3((m * hidden + threads - 1) / threads);
    gemm_x_wgt_bf16<<<g1, threads, 0, stream>>>(reinterpret_cast<__nv_bfloat16 const*>(x),
        reinterpret_cast<__nv_bfloat16 const*>(wgu), reinterpret_cast<__nv_bfloat16*>(z), m, hidden, i2);
    launch_gelu_dispatch(stream, 1, z, h, m, inter);
    gemm_h_wgt_bf16<<<g3, threads, 0, stream>>>(reinterpret_cast<__nv_bfloat16 const*>(h),
        reinterpret_cast<__nv_bfloat16 const*>(wd), reinterpret_cast<__nv_bfloat16*>(y), m, inter, hidden);
}

} // namespace

size_t gemma_fused_gated_mlp_workspace_bytes(int32_t m, int32_t hidden, int32_t inter, int32_t io_type) {
    (void)io_type;
    int64_t const i2 = static_cast<int64_t>(inter) * 2;
    size_t const el = 2;
    size_t const zbytes = static_cast<size_t>(m * i2) * el;
    size_t const hbytes = static_cast<size_t>(m * inter) * el;
    return align256(zbytes) + align256(hbytes) + kLtPrefWorkspaceBytes;
}

int gemma_fused_gated_mlp_cuda(cudaStream_t stream, int32_t act_id, int32_t io_type, int32_t m, int32_t hidden,
    int32_t inter, void const* x, void const* wGateUp, void const* wDown, void* y, void* workspace,
    size_t workspaceBytes) {
    (void)act_id;
    if (m <= 0 || hidden <= 0 || inter <= 0) {
        return 2;
    }
    int const i2 = inter * 2;
    size_t const need = gemma_fused_gated_mlp_workspace_bytes(m, hidden, inter, io_type);
    if (workspaceBytes < need || workspace == nullptr) {
        return 3;
    }

    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        return 4;
    }
    cublasLtHandle_t lt = get_lt_for_device(dev);
    if (lt == nullptr) {
        if (io_type == 0) {
            launch_naive_fp16(stream, m, hidden, inter, x, wGateUp, workspace, static_cast<char*>(workspace) + align256(
                                                                                  static_cast<size_t>(m * i2) * 2),
                wDown, y);
        } else if (io_type == 1) {
            launch_naive_bf16(stream, m, hidden, inter, x, wGateUp, workspace, static_cast<char*>(workspace) + align256(
                                                                                   static_cast<size_t>(m * i2) * 2),
                wDown, y);
        } else {
            return 4;
        }
        return cudaGetLastError() == cudaSuccess ? 0 : 1;
    }

    char* ws = reinterpret_cast<char*>(workspace);
    size_t const zbytes = static_cast<size_t>(m * i2) * 2;
    size_t const hbytes = static_cast<size_t>(m * inter) * 2;
    size_t const off_z = 0;
    size_t const off_h = align256(zbytes);
    size_t const off_lt = align256(off_h + hbytes);
    void* zptr = ws + off_z;
    void* hptr = ws + off_h;
    void* lt_ptr = ws + off_lt;
    size_t lt_cap = workspaceBytes > off_lt ? workspaceBytes - off_lt : 0;

    LtRowNtGemm* g1 = get_or_create_nt_entry(dev, io_type, 0, m, i2, hidden, lt);
    LtRowNtGemm* g2 = get_or_create_nt_entry(dev, io_type, 1, m, hidden, inter, lt);

    if (g1 == nullptr || g2 == nullptr || g1->workspace_size > lt_cap || g2->workspace_size > lt_cap) {
        if (io_type == 0) {
            launch_naive_fp16(stream, m, hidden, inter, x, wGateUp, zptr, hptr, wDown, y);
        } else if (io_type == 1) {
            launch_naive_bf16(stream, m, hidden, inter, x, wGateUp, zptr, hptr, wDown, y);
        } else {
            return 4;
        }
        return cudaGetLastError() == cudaSuccess ? 0 : 1;
    }

    float alpha = 1.f;
    float beta = 0.f;
    int st = launch_lt_row_nt(lt, *g1, x, wGateUp, zptr, alpha, beta, lt_ptr, lt_cap, stream);
    if (st != 0) {
        return st;
    }

    launch_gelu_dispatch(stream, io_type, zptr, hptr, m, inter);

    st = launch_lt_row_nt(lt, *g2, hptr, wDown, y, alpha, beta, lt_ptr, lt_cap, stream);
    if (st != 0) {
        return st;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

} // namespace mopt_trt
