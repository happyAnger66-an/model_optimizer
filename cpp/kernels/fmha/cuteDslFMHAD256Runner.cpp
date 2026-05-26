/*
 * Copyright 2025 the model_optimizer team.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "fmha/cuteDslFMHAD256Runner.h"

#include <mutex>

#ifdef CUTE_DSL_FMHA_ENABLED
#include "cutedsl_all.h"
#endif

namespace model_opt
{

bool CuteDslFMHAD256Runner::sModulesLoaded = false;

bool CuteDslFMHAD256Runner::canImplement(int32_t headDim, int32_t smVersion)
{
    if (headDim != 256)
    {
        return false;
    }
    return smVersion >= 100;
}

#ifdef CUTE_DSL_FMHA_ENABLED

static fmha_d256_homo_bf16_Kernel_Module_t sModuleBf16{};
static fmha_d256_homo_fp16_Kernel_Module_t sModuleFp16{};
static std::mutex sLoadMutex;

bool CuteDslFMHAD256Runner::loadKernelModules()
{
    std::lock_guard<std::mutex> lock(sLoadMutex);
    if (sModulesLoaded)
    {
        return true;
    }
    fmha_d256_homo_bf16_Kernel_Module_Load(&sModuleBf16);
    fmha_d256_homo_fp16_Kernel_Module_Load(&sModuleFp16);
    sModulesLoaded = true;
    return true;
}

void CuteDslFMHAD256Runner::unloadKernelModules()
{
    std::lock_guard<std::mutex> lock(sLoadMutex);
    if (!sModulesLoaded)
    {
        return;
    }
    fmha_d256_homo_bf16_Kernel_Module_Unload(&sModuleBf16);
    fmha_d256_homo_fp16_Kernel_Module_Unload(&sModuleFp16);
    sModulesLoaded = false;
}

static int32_t runVariant(
    FmhaD256Params const& p, cudaStream_t stream, bool useFp16)
{
    int32_t const windowSizeLeft{0};
    float const scaleQ{1.0F};
    float const scaleK{1.0F};
    float const scaleV{1.0F};
    float const invScaleO{1.0F};

    if (useFp16)
    {
        fmha_d256_homo_fp16_Tensor_q_tensor_t q{};
        fmha_d256_homo_fp16_Tensor_kv_cache_t kv{};
        fmha_d256_homo_fp16_Tensor_o_tensor_t o{};
        fmha_d256_homo_fp16_Tensor_cum_seqlen_k_t cu{};

        q.data = const_cast<void*>(p.qPtr);
        kv.data = const_cast<void*>(p.kvPtr);
        o.data = p.oPtr;
        cu.data = const_cast<void*>(static_cast<void const*>(p.cuKvSeqLens));

        q.dynamic_shapes[0] = p.batchSize;
        q.dynamic_shapes[1] = p.seqLenQ;
        q.dynamic_shapes[2] = p.numHeadsQ;
        q.dynamic_shapes[3] = p.headDim;

        kv.dynamic_shapes[0] = p.batchSize;
        kv.dynamic_shapes[1] = 2;
        kv.dynamic_shapes[2] = p.numHeadsKv;
        kv.dynamic_shapes[3] = p.kvCacheCapacity;
        kv.dynamic_shapes[4] = p.headDim;

        o.dynamic_shapes[0] = p.batchSize;
        o.dynamic_shapes[1] = p.seqLenQ;
        o.dynamic_shapes[2] = p.numHeadsQ;
        o.dynamic_shapes[3] = p.headDim;

        cu.dynamic_shapes[0] = p.batchSize + 1;

        return cute_dsl_fmha_d256_homo_fp16_wrapper(
            &sModuleFp16, &q, &kv, &o, &cu, windowSizeLeft, scaleQ, scaleK,
            scaleV, invScaleO, stream);
    }

    fmha_d256_homo_bf16_Tensor_q_tensor_t q{};
    fmha_d256_homo_bf16_Tensor_kv_cache_t kv{};
    fmha_d256_homo_bf16_Tensor_o_tensor_t o{};
    fmha_d256_homo_bf16_Tensor_cum_seqlen_k_t cu{};

    q.data = const_cast<void*>(p.qPtr);
    kv.data = const_cast<void*>(p.kvPtr);
    o.data = p.oPtr;
    cu.data = const_cast<void*>(static_cast<void const*>(p.cuKvSeqLens));

    q.dynamic_shapes[0] = p.batchSize;
    q.dynamic_shapes[1] = p.seqLenQ;
    q.dynamic_shapes[2] = p.numHeadsQ;
    q.dynamic_shapes[3] = p.headDim;

    kv.dynamic_shapes[0] = p.batchSize;
    kv.dynamic_shapes[1] = 2;
    kv.dynamic_shapes[2] = p.numHeadsKv;
    kv.dynamic_shapes[3] = p.kvCacheCapacity;
    kv.dynamic_shapes[4] = p.headDim;

    o.dynamic_shapes[0] = p.batchSize;
    o.dynamic_shapes[1] = p.seqLenQ;
    o.dynamic_shapes[2] = p.numHeadsQ;
    o.dynamic_shapes[3] = p.headDim;

    cu.dynamic_shapes[0] = p.batchSize + 1;

    return cute_dsl_fmha_d256_homo_bf16_wrapper(
        &sModuleBf16, &q, &kv, &o, &cu, windowSizeLeft, scaleQ, scaleK,
        scaleV, invScaleO, stream);
}

int32_t CuteDslFMHAD256Runner::run(FmhaD256Params const& params, cudaStream_t stream)
{
    if (!sModulesLoaded && !loadKernelModules())
    {
        return -1;
    }
    return runVariant(params, stream, params.useFp16);
}

#else

bool CuteDslFMHAD256Runner::loadKernelModules()
{
    return false;
}

void CuteDslFMHAD256Runner::unloadKernelModules() {}

int32_t CuteDslFMHAD256Runner::run(FmhaD256Params const&, cudaStream_t)
{
    return -1;
}

#endif

} // namespace model_opt
