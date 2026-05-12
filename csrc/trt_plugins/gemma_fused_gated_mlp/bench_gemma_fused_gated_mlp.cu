// Copyright 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
//
// CUDA micro-benchmark for ``gemma_fused_gated_mlp_cuda`` (Pi05-ish shapes, Jetson Thor / SM100+).
//
// Default problem (token-major M = B*T):
//   M=968, hidden=2048, intermediate=16384  (x [1,968,2048], gate_up [32768,2048], down [2048,16384] in PyTorch layout)
//
// Build (with TRT plugin tree):
//   cmake -S csrc -B csrc/build -DMODEL_OPTIMIZER_BUILD_TRT_PLUGINS=ON -DTENSORRT_ROOT=...
//   cmake --build csrc/build --target bench_gemma_fused_gated_mlp -j
//
// Run:
//   ./bench_gemma_fused_gated_mlp [M] [H] [inter] [warmup] [iters]   # all ints; bf16 io_type=1
//
// Env (optional overrides):
//   GEMMA_BENCH_BF16=0  -> fp16 (io_type=0)

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gemma_fused_gated_mlp_cuda.h"

namespace {

#define BENCH_CUDA_CHECK(expr)                                                                                        \
    do {                                                                                                              \
        cudaError_t st = (expr);                                                                                      \
        if (st != cudaSuccess) {                                                                                     \
            fprintf(stderr, "[bench] CUDA error %s at %s:%d\n", cudaGetErrorString(st), __FILE__, __LINE__);         \
            std::exit(1);                                                                                             \
        }                                                                                                             \
    } while (0)

static int getenv_int(char const* k, int def_v) {
    char const* s = std::getenv(k);
    if (s == nullptr || s[0] == '\0') {
        return def_v;
    }
    return std::atoi(s);
}

static double median_ms(std::vector<double>& v) {
    if (v.empty()) {
        return 0.0;
    }
    std::sort(v.begin(), v.end());
    size_t const n = v.size();
    if ((n & 1U) != 0) {
        return v[n / 2];
    }
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

} // namespace

int main(int argc, char** argv) {
    int M = 968;
    int H = 2048;
    int inter = 16384;
    int warmup = 20;
    int iters = 200;
    if (argc > 1) {
        M = std::atoi(argv[1]);
    }
    if (argc > 2) {
        H = std::atoi(argv[2]);
    }
    if (argc > 3) {
        inter = std::atoi(argv[3]);
    }
    if (argc > 4) {
        warmup = std::atoi(argv[4]);
    }
    if (argc > 5) {
        iters = std::atoi(argv[5]);
    }

    int const io_type = getenv_int("GEMMA_BENCH_BF16", 1) != 0 ? 1 : 0;
    int const act_id = 1;
    size_t const elem = 2; // bf16/fp16 both 2 bytes
    int const i2 = inter * 2;

    size_t const bytes_x = static_cast<size_t>(M) * static_cast<size_t>(H) * elem;
    size_t const bytes_wgu = static_cast<size_t>(i2) * static_cast<size_t>(H) * elem;
    size_t const bytes_wd = static_cast<size_t>(H) * static_cast<size_t>(inter) * elem;
    size_t const bytes_y = static_cast<size_t>(M) * static_cast<size_t>(H) * elem;

    size_t const ws = mopt_trt::gemma_fused_gated_mlp_workspace_bytes(M, H, inter, io_type);

    void *d_x{}, *d_wgu{}, *d_wd{}, *d_y{}, *d_ws{};
    BENCH_CUDA_CHECK(cudaMalloc(&d_x, bytes_x));
    BENCH_CUDA_CHECK(cudaMalloc(&d_wgu, bytes_wgu));
    BENCH_CUDA_CHECK(cudaMalloc(&d_wd, bytes_wd));
    BENCH_CUDA_CHECK(cudaMalloc(&d_y, bytes_y));
    BENCH_CUDA_CHECK(cudaMalloc(&d_ws, ws));

    BENCH_CUDA_CHECK(cudaMemset(d_x, 0x3b, bytes_x));
    BENCH_CUDA_CHECK(cudaMemset(d_wgu, 0x2b, bytes_wgu));
    BENCH_CUDA_CHECK(cudaMemset(d_wd, 0x1b, bytes_wd));
    BENCH_CUDA_CHECK(cudaMemset(d_y, 0, bytes_y));
    BENCH_CUDA_CHECK(cudaMemset(d_ws, 0, ws));

    cudaStream_t stream{};
    BENCH_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // One dry run to populate cuBLASLt heuristic cache (same as steady-state after warm cache).
    int st0 = mopt_trt::gemma_fused_gated_mlp_cuda(stream, act_id, io_type, M, H, inter, d_x, d_wgu, d_wd, d_y, d_ws, ws);
    BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (st0 != 0) {
        fprintf(stderr, "[bench] gemma_fused_gated_mlp_cuda returned %d (see plugin cuda fallback paths)\n", st0);
    }

    for (int i = 0; i < warmup; ++i) {
        int st = mopt_trt::gemma_fused_gated_mlp_cuda(
            stream, act_id, io_type, M, H, inter, d_x, d_wgu, d_wd, d_y, d_ws, ws);
        if (st != 0) {
            fprintf(stderr, "[bench] warmup iter %d failed: %d\n", i, st);
            std::exit(2);
        }
    }
    BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iters));

    cudaEvent_t ev0{}, ev1{};
    BENCH_CUDA_CHECK(cudaEventCreate(&ev0));
    BENCH_CUDA_CHECK(cudaEventCreate(&ev1));

    for (int i = 0; i < iters; ++i) {
        BENCH_CUDA_CHECK(cudaEventRecord(ev0, stream));
        int st = mopt_trt::gemma_fused_gated_mlp_cuda(
            stream, act_id, io_type, M, H, inter, d_x, d_wgu, d_wd, d_y, d_ws, ws);
        BENCH_CUDA_CHECK(cudaEventRecord(ev1, stream));
        BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (st != 0) {
            fprintf(stderr, "[bench] timed iter %d failed: %d\n", i, st);
            std::exit(3);
        }
        float ms = 0.f;
        BENCH_CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        samples.push_back(static_cast<double>(ms));
    }

    double sum = 0.0;
    for (double ms : samples) {
        sum += ms;
    }
    double const mean = sum / static_cast<double>(samples.size());
    double const med = median_ms(samples);

    // Approximate FLOPs for two GEMMs (GeGLU elementwise treated as negligible vs GEMMs at this size).
    int64_t const flops_g1 = static_cast<int64_t>(M) * static_cast<int64_t>(i2) * static_cast<int64_t>(H) * 2;
    int64_t const flops_g2 = static_cast<int64_t>(M) * static_cast<int64_t>(H) * static_cast<int64_t>(inter) * 2;
    int64_t const flops = flops_g1 + flops_g2;

    double const tflops_mean = (static_cast<double>(flops) / (mean * 1e-3)) / 1e12;
    double const tflops_med = (static_cast<double>(flops) / (med * 1e-3)) / 1e12;

    char const* const dt = (io_type == 1) ? "bf16" : "fp16";
    std::printf(
        "gemma_fused_gated_mlp_cuda bench | dtype=%s M=%d H=%d inter=%d | mean_ms=%.4f median_ms=%.4f | "
        "approx_gemm_tflops_mean=%.3f approx_gemm_tflops_median=%.3f | iters=%d warmup=%d ws_bytes=%zu\n",
        dt,
        M,
        H,
        inter,
        mean,
        med,
        tflops_mean,
        tflops_med,
        iters,
        warmup,
        ws);

    BENCH_CUDA_CHECK(cudaEventDestroy(ev0));
    BENCH_CUDA_CHECK(cudaEventDestroy(ev1));
    BENCH_CUDA_CHECK(cudaStreamDestroy(stream));
    BENCH_CUDA_CHECK(cudaFree(d_x));
    BENCH_CUDA_CHECK(cudaFree(d_wgu));
    BENCH_CUDA_CHECK(cudaFree(d_wd));
    BENCH_CUDA_CHECK(cudaFree(d_y));
    BENCH_CUDA_CHECK(cudaFree(d_ws));
    return 0;
}
