// SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <NvInfer.h>

#include <cstdio>

namespace pi05
{

//! 简化版 TRT logger（参考 TensorRT-Edge-LLM cpp/common/logger）。
class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, char const* msg) noexcept override;

    static TrtLogger& instance();
};

#define LOG_INFO(...)                                                                    \
    do                                                                                   \
    {                                                                                    \
        std::fprintf(stdout, "[pi05][I] ");                                              \
        std::fprintf(stdout, __VA_ARGS__);                                               \
        std::fprintf(stdout, "\n");                                                      \
    } while (0)

#define LOG_WARN(...)                                                                    \
    do                                                                                   \
    {                                                                                    \
        std::fprintf(stderr, "[pi05][W] ");                                              \
        std::fprintf(stderr, __VA_ARGS__);                                               \
        std::fprintf(stderr, "\n");                                                      \
    } while (0)

#define LOG_ERROR(...)                                                                   \
    do                                                                                   \
    {                                                                                    \
        std::fprintf(stderr, "[pi05][E] ");                                              \
        std::fprintf(stderr, __VA_ARGS__);                                               \
        std::fprintf(stderr, "\n");                                                      \
    } while (0)

} // namespace pi05
