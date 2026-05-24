/*
 * Copyright 2025 the model_optimizer team.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <NvInferPlugin.h>

// Plugin registration happens via REGISTER_TENSORRT_PLUGIN in FmhaD256AttentionPlugin.cpp

extern "C" bool initLibModelOptPlugin()
{
    return true;
}
