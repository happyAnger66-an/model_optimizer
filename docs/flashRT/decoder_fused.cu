// ================================================================
// FlashRT — Decoder fused kernels (FP16)
// Direct port of pi05 engine ae_forward_static kernels.
// ================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cmath>

// ── C1: Fused AdaRMSNorm → FP8 with static scale ──
//
// OpenPI / Pi0.5 equivalent (per decoder layer, before self-attention):
//   hidden_states, gate = layer.input_layernorm(x, adarms_cond)
//   # GemmaRMSNorm: modulation = dense(adarms_cond); scale,shift,gate = chunk(...)
//   # x_norm = rmsnorm(x) * (1 + scale) + shift
//   # then q_proj/k_proj/v_proj(x_norm)  →  downstream C2 fp8_gemm_descale (QKV GEMM)
//
// FlashRT folds dense(adarms_cond) into precomputed ``style`` buffer (``sa`` in pipeline):
//   style[r, 0:D)   = scale
//   style[r, D:2D)  = shift
//   style[r, 2D:3D)  = gate   (written to gate_out; used later at C4→5 gated residual)
//
// This kernel fuses in ONE launch:
//   1) RMSNorm on x
//   2) Ada affine: * (1+scale) + shift
//   3) Static FP8 quant using descale_ptr (= act_scale_qkv, per-tensor s_act)
//   4) Store gate for later (not consumed here)
//
// Launch: <<<S, 256>>> — one CUDA block per token row (r = 0 .. S-1).
//
// Args:
//   x            [S, D] fp16  residual stream (bufs['x']); read-only here
//   style        [S, 3D] fp16 precomputed AdaRMS modulation (sa_ptr for step/layer)
//   out          [S, D] fp8  quantized norm output (xn_fp8) → C2 QKV GEMM input
//   gate_out     [S, D] fp16 gate from style; saved for C4→5: x += o_proj_out * gate
//   S            sequence length (= action_horizon tokens)
//   D            hidden dim (= action expert width)
//   descale_ptr  device float* → s_act; quant: out ≈ normed / s_act (C2 descales in GEMM epilogue)
__global__ void fused_adarms_fp8_static_fp16_kernel(
    const __half* __restrict__ x, const __half* __restrict__ style,
    __nv_fp8_e4m3* __restrict__ out, __half* __restrict__ gate_out,
    int S, int D, const float* __restrict__ descale_ptr) {
    // One block ↔ one action token row r ∈ [0, S).
    int r = blockIdx.x;
    if (r >= S) return;

    // Pointers into row r of x and the 3×D style vector for this token.
    const __half* row = x + r * D;
    const __half* sc = style + r * 3 * D;  // scale[D]
    const __half* sh = sc + D;             // shift[D]
    const __half* gt = sh + D;             // gate[D]

    // ── Phase 1: RMSNorm — compute sum of squares Σ_i x_i² for this row ──
    // Each thread accumulates a partial sum over columns strided by blockDim.x.
    float sum_sq = 0;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(row[i]);
        sum_sq += v * v;
    }

    // Warp- and block-level reduction to get total sum_sq for row r.
    // (256 threads → 8 warps; two-stage shuffle + shared mem, same pattern as pi05 engine.)
    __shared__ float shv[8];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    // Intra-warp reduce:
    for (int o = 16; o > 0; o >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, o);
    if (!lane) shv[wid] = sum_sq;
    __syncthreads();
    // Inter-warp reduce (warp 0 only):
    if (!wid) {
        sum_sq = (lane < (blockDim.x + 31) / 32) ? shv[lane] : 0;
        for (int o = 16; o > 0; o >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, o);
    }
    __syncthreads();
    if (!threadIdx.x) shv[0] = sum_sq;
    __syncthreads();

    // rstd = 1 / sqrt(mean(x²) + eps)  — inverse RMS scale (eps = 1e-6).
    float rstd = rsqrtf(shv[0] / D + 1e-6f);

    // FP8 quant factor: inv_scale = 1 / s_act (clamped descale from calibration).
    // Matches: x_fp8 = clamp(normed / s_act, -448, 448) in e4m3fn range.
    float inv_scale = 1.0f / fmaxf(*descale_ptr, 1e-12f);

    // ── Phase 2: AdaRMS affine + FP8 quant + gate store ──
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        // Step A: RMSNorm — v = x_i / RMS(x)
        float v = __half2float(row[i]) * rstd;

        // Step B: Ada modulation (DiT-style adaLN on RMSNorm output):
        //   normed_i = v * (1 + scale_i) + shift_i
        float normed = v * (1.0f + __half2float(sc[i])) + __half2float(sh[i]);

        // Step C: Static FP8 quant → xn_fp8 (input to C2 fp8_gemm_descale_fp16).
        // C2 multiplies by the same s_act in GEMM epilogue to recover fp16 magnitude.
        out[r * D + i] = __nv_fp8_e4m3(fminf(fmaxf(normed * inv_scale, -448.0f), 448.0f));

        // Step D: Pass through gate (from precomputed style, not used until after attention).
        gate_out[r * D + i] = __float2half(__half2float(gt[i]));
    }
}

// ── C4→C5: Fused gate×residual + AdaRMSNorm → FP8 ──
// Combines: residual += gemm_out * gate, RMSNorm(residual) * (1+scale) + shift → FP8
__global__ void gate_res_adarms_fp8_static_fp16_kernel(
    const __half* __restrict__ gemm_out, const __half* __restrict__ prev_gate,
    __half* __restrict__ residual, const __half* __restrict__ style,
    __nv_fp8_e4m3* __restrict__ fp8_out, __half* __restrict__ gate_out,
    int S, int D, const float* __restrict__ descale_ptr) {
    int r = blockIdx.x; if (r >= S) return;
    const __half* sc = style + r * 3 * D;
    const __half* sh = sc + D;
    const __half* gt = sh + D;
    extern __shared__ float shv[];
    float sum_sq = 0;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float res = __half2float(residual[r*D+i]) + __half2float(gemm_out[r*D+i]) * __half2float(prev_gate[r*D+i]);
        residual[r*D+i] = __float2half(res);
        sum_sq += res * res;
    }
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    for (int o = 16; o > 0; o >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, o);
    if (!lane) shv[wid] = sum_sq; __syncthreads();
    if (!wid) { sum_sq = (lane < (blockDim.x+31)/32) ? shv[lane] : 0;
        for (int o = 16; o > 0; o >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, o); }
    __syncthreads(); if (!threadIdx.x) shv[0] = sum_sq; __syncthreads();
    float rstd = rsqrtf(shv[0] / D + 1e-6f);
    float inv_scale = 1.0f / fmaxf(*descale_ptr, 1e-12f);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(residual[r*D+i]) * rstd;
        float normed = v * (1.0f + __half2float(sc[i])) + __half2float(sh[i]);
        fp8_out[r*D+i] = __nv_fp8_e4m3(fminf(fmaxf(normed * inv_scale, -448.0f), 448.0f));
        gate_out[r*D+i] = __float2half(__half2float(gt[i]));
    }
}

// ── C6: Merged GeGLU → FP8 with static scale ──
// Reads from merged [S, 2H] buffer: first H = gate, second H = up
// Applies GELU (tanh approx) to gate, multiply by up, quantize to FP8
// NOTE: pi05 names this "silu" but it's actually GELU tanh approximation
__global__ void geglu_fp8_static_fp16_kernel(
    const __half* __restrict__ merged, __nv_fp8_e4m3* __restrict__ out,
    int S, int H, const float* __restrict__ descale_ptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S * H) return;
    int s = i / H, h = i % H;
    float gv = __half2float(merged[s * 2 * H + h]);
    float uv = __half2float(merged[s * 2 * H + H + h]);
    // GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // pi05 uses: x / (1 + exp(-1.5957691216057308 * x * (1 + 0.044715 * x^2)))
    float gelu = gv / (1.0f + expf(-1.5957691216057308f * gv * (1.0f + 0.044715f * gv * gv)));
    float val = gelu * uv;
    float inv_scale = 1.0f / fmaxf(*descale_ptr, 1e-12f);
    out[i] = __nv_fp8_e4m3(fminf(fmaxf(val * inv_scale, -448.0f), 448.0f));
}

// ── Simple gate × residual (last layer, no norm) ──
__global__ void gate_res_fp16_kernel(const __half* __restrict__ gemm_out,
                                      const __half* __restrict__ gate,
                                      __half* __restrict__ residual, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    residual[i] = __float2half(__half2float(residual[i]) + __half2float(gemm_out[i]) * __half2float(gate[i]));
}

// ── AdaRMSNorm (BF16/FP16 output, for final step) ──
__global__ void adarms_fp16_kernel(const __half* __restrict__ x, const __half* __restrict__ style,
                                    __half* __restrict__ out, __half* __restrict__ gate_out,
                                    int S, int D) {
    int r = blockIdx.x; if (r >= S) return;
    const __half* row = x + r * D;
    const __half* sc = style + r * 3 * D;
    const __half* sh = sc + D;
    const __half* gt = sh + D;
    float sum_sq = 0;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(row[i]); sum_sq += v * v;
    }
    __shared__ float shv[8];
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32;
    for (int o = 16; o > 0; o >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, o);
    if (!lane) shv[wid] = sum_sq; __syncthreads();
    if (!wid) { sum_sq = (lane < (blockDim.x+31)/32) ? shv[lane] : 0;
        for (int o = 16; o > 0; o >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, o); }
    __syncthreads(); if (!threadIdx.x) shv[0] = sum_sq; __syncthreads();
    float rstd = rsqrtf(shv[0] / D + 1e-6f);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __half2float(row[i]) * rstd;
        out[r*D+i] = __float2half(v * (1.0f + __half2float(sc[i])) + __half2float(sh[i]));
        gate_out[r*D+i] = __float2half(__half2float(gt[i]));
    }
}

// ── Host wrappers ──

void fused_adarms_fp8_static_fp16(const __half* x, const __half* style,
                                    __nv_fp8_e4m3* out, __half* gate_out,
                                    int S, int D, const float* descale_ptr,
                                    cudaStream_t stream) {
    fused_adarms_fp8_static_fp16_kernel<<<S, 256, 0, stream>>>(x, style, out, gate_out, S, D, descale_ptr);
}

void gate_res_adarms_fp8_static_fp16(const __half* gemm_out, const __half* prev_gate,
                                       __half* residual, const __half* style,
                                       __nv_fp8_e4m3* fp8_out, __half* gate_out,
                                       int S, int D, const float* descale_ptr,
                                       cudaStream_t stream) {
    gate_res_adarms_fp8_static_fp16_kernel<<<S, 256, 8*sizeof(float), stream>>>(
        gemm_out, prev_gate, residual, style, fp8_out, gate_out, S, D, descale_ptr);
}

void geglu_fp8_static_fp16(const __half* merged, __nv_fp8_e4m3* out,
                             int S, int H, const float* descale_ptr,
                             cudaStream_t stream) {
    geglu_fp8_static_fp16_kernel<<<(S*H + 255)/256, 256, 0, stream>>>(merged, out, S, H, descale_ptr);
}

void gate_res_fp16(const __half* gemm_out, const __half* gate,
                    __half* residual, int n, cudaStream_t stream) {
    gate_res_fp16_kernel<<<(n + 255)/256, 256, 0, stream>>>(gemm_out, gate, residual, n);
}

void adarms_fp16(const __half* x, const __half* style,
                  __half* out, __half* gate_out, int S, int D,
                  cudaStream_t stream) {
    adarms_fp16_kernel<<<S, 256, 0, stream>>>(x, style, out, gate_out, S, D);
}

// ── Simple bias add: x[i] += b[i % D] (pi05 bias_k) ──
__global__ void add_bias_fp16_kernel(__half* x, const __half* b, int S, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < S * D) x[i] = __float2half(__half2float(x[i]) + __half2float(b[i % D]));
}

void add_bias_fp16(__half* x, const __half* b, int S, int D, cudaStream_t stream) {
    add_bias_fp16_kernel<<<(S*D + 255)/256, 256, 0, stream>>>(x, b, S, D);
}

// ── gmm: cuBLAS NN GEMM with beta parameter (pi05 gmm) ──
// C = alpha * A @ B + beta * C  (FP16)
// A: (M, K), B: (K, N) row-major, C: (M, N)
// Stateless: receives cuBLAS handle from caller (FvkContext).
void gmm_fp16(cublasHandle_t handle, const __half* A, const __half* B, __half* C,
               int M, int N, int K, float beta, cudaStream_t stream) {
    cublasSetStream(handle, stream);
    float alpha = 1.0f;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K, &alpha, B, CUDA_R_16F, N, A, CUDA_R_16F, K,
        &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// ── FP8 GEMM with device descale pointers → FP16 output ──
// Exact port of pi05 gmm_fp8_kn_descale: cached descriptor + algo per (M,N,K).
// Scale pointers updated per-call. Layout: col-major matching pi05.
#include <cublasLt.h>
#include <unordered_map>

static cublasLtHandle_t g_fp8_lt = nullptr;
static void* g_fp8_ws = nullptr;
static size_t g_fp8_ws_sz = 32 * 1024 * 1024;  // Match production (32MB)

struct LtGemmKey {
    int M, N, K;
    bool operator==(const LtGemmKey& o) const { return M==o.M && N==o.N && K==o.K; }
};
struct LtGemmKeyHash {
    size_t operator()(const LtGemmKey& k) const {
        size_t h = std::hash<int>()(k.M);
        h ^= std::hash<int>()(k.N) + 0x9e3779b9 + (h<<6) + (h>>2);
        h ^= std::hash<int>()(k.K) + 0x9e3779b9 + (h<<6) + (h>>2);
        return h;
    }
};
struct CachedLtGemm {
    cublasLtMatmulDesc_t desc;
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatmulAlgo_t algo;
};
static std::unordered_map<LtGemmKey, CachedLtGemm, LtGemmKeyHash> g_lt_cache;

void fp8_gemm_descale_fp16(const void* A_fp8, const void* B_fp8, void* C_fp16,
                             int M, int N, int K,
                             const float* act_descale, const float* w_descale,
                             cudaStream_t stream) {
    if (!g_fp8_lt) { cublasLtCreate(&g_fp8_lt); cudaMalloc(&g_fp8_ws, g_fp8_ws_sz); }

    LtGemmKey key{M, N, K};
    auto it = g_lt_cache.find(key);
    if (it == g_lt_cache.end()) {
        CachedLtGemm cg;
        cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, N, K, N);
        cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, K, M, K);
        cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_16F, N, M, N);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &g_fp8_ws_sz, sizeof(g_fp8_ws_sz));
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        cublasLtMatmulAlgoGetHeuristic(g_fp8_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc,
                                        pref, 1, &result, &ret);
        cg.algo = result.algo;
        cublasLtMatmulPreferenceDestroy(pref);
        g_lt_cache[key] = cg;
        it = g_lt_cache.find(key);
    }
    auto& cg = it->second;
    cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_descale, sizeof(w_descale));
    cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &act_descale, sizeof(act_descale));
    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(g_fp8_lt, cg.desc, &alpha, B_fp8, cg.Adesc, A_fp8, cg.Bdesc,
                    &beta, C_fp16, cg.Cdesc, C_fp16, cg.Cdesc,
                    &cg.algo, g_fp8_ws, g_fp8_ws_sz, stream);
}

// FP32 output variant
void fp8_gemm_descale_f32out(const void* A_fp8, const void* B_fp8, void* C_fp32,
                              int M, int N, int K,
                              const float* act_descale, const float* w_descale,
                              cudaStream_t stream) {
    if (!g_fp8_lt) { cublasLtCreate(&g_fp8_lt); cudaMalloc(&g_fp8_ws, g_fp8_ws_sz); }

    LtGemmKey key{M, N + 9200000, K};  // unique key for f32out
    auto it = g_lt_cache.find(key);
    if (it == g_lt_cache.end()) {
        CachedLtGemm cg;
        cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, N, K, N);
        cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, K, M, K);
        cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_32F, N, M, N);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &g_fp8_ws_sz, sizeof(g_fp8_ws_sz));
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        cublasLtMatmulAlgoGetHeuristic(g_fp8_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc,
                                        pref, 1, &result, &ret);
        if (ret == 0) printf("[fp8_gemm_descale_f32out] Heuristic FAILED for [%d,%d,%d]\n", M, N, K);
        cg.algo = result.algo;
        cublasLtMatmulPreferenceDestroy(pref);
        g_lt_cache[key] = cg;
        it = g_lt_cache.find(key);
    }
    auto& cg = it->second;
    cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_descale, sizeof(w_descale));
    cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &act_descale, sizeof(act_descale));
    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(g_fp8_lt, cg.desc, &alpha, B_fp8, cg.Adesc, A_fp8, cg.Bdesc,
                    &beta, C_fp32, cg.Cdesc, C_fp32, cg.Cdesc,
                    &cg.algo, g_fp8_ws, g_fp8_ws_sz, stream);
}

// BF16 output variant — for models trained in BF16 with activations exceeding
// FP16 range (Pi0-FAST decode_step). Same FP8 inputs and per-tensor descales as
// the FP16 variant; only the C matrix dtype is BF16.
void fp8_gemm_descale_bf16out(const void* A_fp8, const void* B_fp8, void* C_bf16,
                               int M, int N, int K,
                               const float* act_descale, const float* w_descale,
                               cudaStream_t stream) {
    if (!g_fp8_lt) { cublasLtCreate(&g_fp8_lt); cudaMalloc(&g_fp8_ws, g_fp8_ws_sz); }

    LtGemmKey key{M, N + 9100000, K};  // unique key for bf16out (avoid clash with fp16/f32out)
    auto it = g_lt_cache.find(key);
    if (it == g_lt_cache.end()) {
        CachedLtGemm cg;
        cublasLtMatmulDescCreate(&cg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        cublasLtMatrixLayoutCreate(&cg.Adesc, CUDA_R_8F_E4M3, N, K, N);
        cublasLtMatrixLayoutCreate(&cg.Bdesc, CUDA_R_8F_E4M3, K, M, K);
        cublasLtMatrixLayoutCreate(&cg.Cdesc, CUDA_R_16BF, N, M, N);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &g_fp8_ws_sz, sizeof(g_fp8_ws_sz));
        cublasLtMatmulHeuristicResult_t result; int ret = 0;
        cublasLtMatmulAlgoGetHeuristic(g_fp8_lt, cg.desc, cg.Adesc, cg.Bdesc, cg.Cdesc, cg.Cdesc,
                                        pref, 1, &result, &ret);
        if (ret == 0) printf("[fp8_gemm_descale_bf16out] Heuristic FAILED for [%d,%d,%d]\n", M, N, K);
        cg.algo = result.algo;
        cublasLtMatmulPreferenceDestroy(pref);
        g_lt_cache[key] = cg;
        it = g_lt_cache.find(key);
    }
    auto& cg = it->second;
    cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_descale, sizeof(w_descale));
    cublasLtMatmulDescSetAttribute(cg.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &act_descale, sizeof(act_descale));
    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(g_fp8_lt, cg.desc, &alpha, B_fp8, cg.Adesc, A_fp8, cg.Bdesc,
                    &beta, C_bf16, cg.Cdesc, C_bf16, cg.Cdesc,
                    &cg.algo, g_fp8_ws, g_fp8_ws_sz, stream);
}
