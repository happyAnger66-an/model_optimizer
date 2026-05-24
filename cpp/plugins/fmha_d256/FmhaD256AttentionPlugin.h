/*
 * Copyright 2025 the model_optimizer team.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace model_opt
{

class FmhaD256AttentionPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    FmhaD256AttentionPlugin(int32_t numQHeads, int32_t numKvHeads, int32_t headDim, bool useFp16);

    FmhaD256AttentionPlugin(void const* data, size_t length);

    ~FmhaD256AttentionPlugin() override = default;

    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    void configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(
        nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

private:
    int32_t mNumQHeads{};
    int32_t mNumKvHeads{};
    int32_t mHeadDim{};
    bool mUseFp16{};
    std::string mNamespace;
};

class FmhaD256AttentionPluginCreator : public nvinfer1::IPluginCreator
{
public:
    FmhaD256AttentionPluginCreator();

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(char const* pluginNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

private:
    nvinfer1::PluginFieldCollection mFC{};
    std::vector<nvinfer1::PluginField> mFields;
    std::string mNamespace;
};

} // namespace model_opt
