// SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
#include "common/logger.h"

namespace pi05
{

void TrtLogger::log(Severity severity, char const* msg) noexcept
{
    switch (severity)
    {
    case Severity::kINTERNAL_ERROR:
    case Severity::kERROR: std::fprintf(stderr, "[trt][E] %s\n", msg); break;
    case Severity::kWARNING: std::fprintf(stderr, "[trt][W] %s\n", msg); break;
    case Severity::kINFO:
    case Severity::kVERBOSE:
    default:
        // 默认静默 INFO/VERBOSE，避免反序列化日志刷屏。
        break;
    }
}

TrtLogger& TrtLogger::instance()
{
    static TrtLogger logger;
    return logger;
}

} // namespace pi05
