/*
 * Copyright 2025 the model_optimizer team.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "FmhaD256AttentionPlugin.h"

#include "fmha/cuteDslFMHAD256Runner.h"

#include <cstring>

using namespace nvinfer1;

namespace model_opt
{

static char const* kPluginName = "FmhaD256AttentionPlugin";
static char const* kPluginVersion = "1";

FmhaD256AttentionPlugin::FmhaD256AttentionPlugin(
    int32_t numQHeads, int32_t numKvHeads, int32_t headDim, bool useFp16)
    : mNumQHeads(numQHeads)
    , mNumKvHeads(numKvHeads)
    , mHeadDim(headDim)
    , mUseFp16(useFp16)
{
}

FmhaD256AttentionPlugin::FmhaD256AttentionPlugin(void const* data, size_t length)
{
    auto const* d = reinterpret_cast<uint8_t const*>(data);
    std::memcpy(&mNumQHeads, d, sizeof(mNumQHeads));
    d += sizeof(mNumQHeads);
    std::memcpy(&mNumKvHeads, d, sizeof(mNumKvHeads));
    d += sizeof(mNumKvHeads);
    std::memcpy(&mHeadDim, d, sizeof(mHeadDim));
    d += sizeof(mHeadDim);
    std::memcpy(&mUseFp16, d, sizeof(mUseFp16));
    (void)length;
}

char const* FmhaD256AttentionPlugin::getPluginType() const noexcept
{
    return kPluginName;
}

char const* FmhaD256AttentionPlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

int32_t FmhaD256AttentionPlugin::getNbOutputs() const noexcept
{
    return 2; // attn_out, updated_kv
}

int32_t FmhaD256AttentionPlugin::initialize() noexcept
{
    if (!CuteDslFMHAD256Runner::canImplement(mHeadDim, 110))
    {
        return -1;
    }
    return CuteDslFMHAD256Runner::loadKernelModules() ? 0 : -1;
}

void FmhaD256AttentionPlugin::terminate() noexcept
{
    CuteDslFMHAD256Runner::unloadKernelModules();
}

size_t FmhaD256AttentionPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNumQHeads) + sizeof(mNumKvHeads) + sizeof(mHeadDim) + sizeof(mUseFp16);
}

void FmhaD256AttentionPlugin::serialize(void* buffer) const noexcept
{
    auto* d = reinterpret_cast<uint8_t*>(buffer);
    std::memcpy(d, &mNumQHeads, sizeof(mNumQHeads));
    d += sizeof(mNumQHeads);
    std::memcpy(d, &mNumKvHeads, sizeof(mNumKvHeads));
    d += sizeof(mNumKvHeads);
    std::memcpy(d, &mHeadDim, sizeof(mHeadDim));
    d += sizeof(mHeadDim);
    std::memcpy(d, &mUseFp16, sizeof(mUseFp16));
}

void FmhaD256AttentionPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2DynamicExt* FmhaD256AttentionPlugin::clone() const noexcept
{
    return new FmhaD256AttentionPlugin(mNumQHeads, mNumKvHeads, mHeadDim, mUseFp16);
}

void FmhaD256AttentionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* FmhaD256AttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

DataType FmhaD256AttentionPlugin::getOutputDataType(
    int32_t, DataType const* inputTypes, int32_t) const noexcept
{
    return inputTypes[0];
}

bool FmhaD256AttentionPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    (void)nbInputs;
    (void)nbOutputs;
    if (pos == 2)
    {
        return inOut[pos].type == DataType::kINT32 && inOut[pos].format == TensorFormat::kLINEAR;
    }
    DataType dt = mUseFp16 ? DataType::kHALF : DataType::kBF16;
    return inOut[pos].type == dt && inOut[pos].format == TensorFormat::kLINEAR;
}

void FmhaD256AttentionPlugin::configurePlugin(
    DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept
{
}

size_t FmhaD256AttentionPlugin::getWorkspaceSize(
    PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) const noexcept
{
    return 0;
}

int32_t FmhaD256AttentionPlugin::enqueue(
    PluginTensorDesc const* inputDesc, PluginTensorDesc const*,
    void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept
{
    auto const& qDesc = inputDesc[0].dims;
    auto const& kvDesc = inputDesc[1].dims;

    FmhaD256Params params{};
    params.qPtr = inputs[0];
    params.kvPtr = inputs[1];
    params.cuKvSeqLens = static_cast<int32_t const*>(inputs[2]);
    params.oPtr = outputs[0];
    params.batchSize = qDesc.d[0];
    params.seqLenQ = qDesc.d[1];
    params.numHeadsQ = qDesc.d[2];
    params.headDim = qDesc.d[3];
    params.numHeadsKv = kvDesc.d[2];
    params.kvCacheCapacity = kvDesc.d[3];
    params.useFp16 = mUseFp16;

    int32_t status = CuteDslFMHAD256Runner::run(params, stream);
    if (status != 0)
    {
        return status;
    }

    // Output 1: pass-through updated KV cache (in-place semantics for export graph).
    cudaMemcpyAsync(outputs[1], inputs[1], inputDesc[1].dims.d[0] * inputDesc[1].dims.d[1]
            * inputDesc[1].dims.d[2] * inputDesc[1].dims.d[3] * inputDesc[1].dims.d[4]
            * (mUseFp16 ? 2 : 2),
        cudaMemcpyDeviceToDevice, stream);
    return 0;
}

DimsExprs FmhaD256AttentionPlugin::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t, IExprBuilder&) noexcept
{
    if (outputIndex == 0)
    {
        return inputs[0];
    }
    return inputs[1];
}

// ---- Creator ----

FmhaD256AttentionPluginCreator::FmhaD256AttentionPluginCreator()
{
    mFields.emplace_back(PluginField{"num_q_heads", nullptr, PluginFieldType::kINT32, 1});
    mFields.emplace_back(PluginField{"num_kv_heads", nullptr, PluginFieldType::kINT32, 1});
    mFields.emplace_back(PluginField{"head_dim", nullptr, PluginFieldType::kINT32, 1});
    mFields.emplace_back(PluginField{"use_fp16", nullptr, PluginFieldType::kINT32, 1});
    mFC.nbFields = static_cast<int32_t>(mFields.size());
    mFC.fields = mFields.data();
}

char const* FmhaD256AttentionPluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

char const* FmhaD256AttentionPluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

PluginFieldCollection const* FmhaD256AttentionPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* FmhaD256AttentionPluginCreator::createPlugin(
    char const*, PluginFieldCollection const* fc) noexcept
{
    int32_t numQHeads = 8;
    int32_t numKvHeads = 1;
    int32_t headDim = 256;
    int32_t useFp16 = 1;
    for (int32_t i = 0; i < fc->nbFields; ++i)
    {
        auto const& f = fc->fields[i];
        if (std::strcmp(f.name, "num_q_heads") == 0)
        {
            numQHeads = *static_cast<int32_t const*>(f.data);
        }
        if (std::strcmp(f.name, "num_kv_heads") == 0)
        {
            numKvHeads = *static_cast<int32_t const*>(f.data);
        }
        if (std::strcmp(f.name, "head_dim") == 0)
        {
            headDim = *static_cast<int32_t const*>(f.data);
        }
        if (std::strcmp(f.name, "use_fp16") == 0)
        {
            useFp16 = *static_cast<int32_t const*>(f.data);
        }
    }
    return new FmhaD256AttentionPlugin(numQHeads, numKvHeads, headDim, useFp16 != 0);
}

IPluginV2* FmhaD256AttentionPluginCreator::deserializePlugin(
    char const*, void const* serialData, size_t serialLength) noexcept
{
    return new FmhaD256AttentionPlugin(serialData, serialLength);
}

void FmhaD256AttentionPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace ? pluginNamespace : "";
}

char const* FmhaD256AttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(FmhaD256AttentionPluginCreator);

} // namespace model_opt
