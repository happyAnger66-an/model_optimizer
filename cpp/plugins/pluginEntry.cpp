/*
 * Copyright 2025 the model_optimizer team.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <NvInferPlugin.h>

#include "fmha_d256/FmhaD256AttentionPlugin.h"

extern "C" bool initLibModelOptPlugin()
{
    return true;
}

extern "C" TENSORRTAPI void setLoggerFinder(nvinfer1::ILoggerFinder*)
{
}

extern "C" TENSORRTAPI nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& nbCreators)
{
    // TensorRT 10's IPluginRegistry::loadLibrary() looks for this C ABI symbol
    // when loading a library via trtexec --dynamicPlugins. Static registration
    // via REGISTER_TENSORRT_PLUGIN alone is not enough for that path.
    static model_opt::FmhaD256AttentionPluginCreator creator{};
    static nvinfer1::IPluginCreatorInterface* creators[] = {
        static_cast<nvinfer1::IPluginCreatorInterface*>(&creator)};
    nbCreators = 1;
    return creators;
}
