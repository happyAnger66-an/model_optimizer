/*
 * Copyright 2025 the model_optimizer team.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace model_opt
{

struct FmhaD256Params
{
    void const* qPtr{nullptr};
    void const* kvPtr{nullptr};
    void* oPtr{nullptr};
    int32_t const* cuKvSeqLens{nullptr};
    int32_t batchSize{0};
    int32_t seqLenQ{0};
    int32_t numHeadsQ{0};
    int32_t numHeadsKv{0};
    int32_t kvCacheCapacity{0};
    int32_t headDim{256};
    bool useFp16{false};
};

class CuteDslFMHAD256Runner
{
public:
    static bool canImplement(int32_t headDim, int32_t smVersion);
    static bool loadKernelModules();
    static void unloadKernelModules();

    static int32_t run(FmhaD256Params const& params, cudaStream_t stream);

private:
    static bool sModulesLoaded;
};

} // namespace model_opt
