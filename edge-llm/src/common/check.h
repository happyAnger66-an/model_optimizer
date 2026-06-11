// SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                                 \
    do                                                                                   \
    {                                                                                    \
        cudaError_t const status_ = (call);                                              \
        if (status_ != cudaSuccess)                                                      \
        {                                                                                \
            std::ostringstream oss_;                                                     \
            oss_ << "CUDA error " << cudaGetErrorName(status_) << " ("                   \
                 << cudaGetErrorString(status_) << ") at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(oss_.str());                                        \
        }                                                                                \
    } while (0)

#define EDGE_CHECK(cond, msg)                                                            \
    do                                                                                   \
    {                                                                                    \
        if (!(cond))                                                                     \
        {                                                                                \
            std::ostringstream oss_;                                                     \
            oss_ << "Check failed: " << #cond << " — " << msg << " at " << __FILE__      \
                 << ":" << __LINE__;                                                     \
            throw std::runtime_error(oss_.str());                                        \
        }                                                                                \
    } while (0)
