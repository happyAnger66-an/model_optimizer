// Copyright 2026 the model_optimizer team.
//
// SPDX-License-Identifier: Apache-2.0
//
// TensorRT IPluginV2DynamicExt: fused SigLIP MLP (Linear → act → Linear), ONNX ``trt::SiglipMlpPlugin`` /
// PyTorch ``trt::siglip_mlp_plugin``.

#include <NvInfer.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "siglip_mlp_cuda_api.h"

namespace mopt_trt {

int32_t constexpr kNbInputs = 5;

inline int32_t volume_except_last(nvinfer1::Dims const& dims) noexcept {
    if (dims.nbDims <= 1) {
        return 1;
    }
    int32_t v = 1;
    for (int32_t i = 0; i < dims.nbDims - 1; ++i) {
        v *= dims.d[i];
    }
    return v;
}

inline int32_t last_dim(nvinfer1::Dims const& dims) noexcept {
    if (dims.nbDims < 1) {
        return 0;
    }
    return dims.d[dims.nbDims - 1];
}

inline int32_t trt_dtype_to_cuda_enum(nvinfer1::DataType t) noexcept {
    if (t == nvinfer1::DataType::kFLOAT) {
        return 0;
    }
    if (t == nvinfer1::DataType::kHALF) {
        return 1;
    }
    if (t == nvinfer1::DataType::kBF16) {
        return 2;
    }
    return -1;
}

class SiglipMlpPlugin final : public nvinfer1::IPluginV2DynamicExt {
public:
    explicit SiglipMlpPlugin(int32_t act_id)
        : m_act_id(act_id) {}

    nvinfer1::AsciiChar const* getPluginType() const noexcept override { return "SiglipMlpPlugin"; }

    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return "1"; }

    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t getNbInputs() const noexcept override { return kNbInputs; }

    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        (void)exprBuilder;
        (void)outputIndex;
        (void)nbInputs;
        nvinfer1::DimsExprs out{};
        out.nbDims = inputs[0].nbDims;
        for (int32_t i = 0; i < out.nbDims; ++i) {
            out.d[i] = inputs[0].d[i];
        }
        return out;
    }

    bool supportsFormatCombination(int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override {
        (void)nbInputs;
        (void)nbOutputs;
        nvinfer1::PluginTensorDesc const& d = inOut[pos];
        if (d.format != nvinfer1::TensorFormat::kLINEAR) {
            return false;
        }
        if (d.type != nvinfer1::DataType::kFLOAT && d.type != nvinfer1::DataType::kHALF
            && d.type != nvinfer1::DataType::kBF16) {
            return false;
        }
        if (pos == 0) {
            return true;
        }
        return d.type == inOut[0].type;
    }

    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override {
        (void)out;
        (void)nbOutputs;
        if (nbInputs < kNbInputs) {
            return;
        }
        m_d = last_dim(in[0].desc.dims);
        if (in[1].desc.dims.nbDims >= 2) {
            m_h = in[1].desc.dims.d[0];
        }
        int32_t s_max = 1;
        if (in[0].desc.dims.nbDims > 1) {
            for (int32_t i = 0; i < in[0].desc.dims.nbDims - 1; ++i) {
                int32_t mx = in[0].max.d[i];
                int32_t use = mx > 0 ? mx : (in[0].desc.dims.d[i] > 0 ? in[0].desc.dims.d[i] : 1);
                s_max *= use;
            }
        }
        m_s_max = s_max > 0 ? s_max : 1;
        if (m_d > 0 && m_h > 0 && m_s_max > 0) {
            m_workspace_bytes = mopt_siglip_mlp_query_workspace_bytes(m_s_max, m_d, m_h);
        }
    }

    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override {
        (void)inputs;
        (void)nbInputs;
        (void)outputs;
        (void)nbOutputs;
        return m_workspace_bytes;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
        (void)outputDesc;
        if (m_cublas == nullptr) {
            return 1;
        }
        int32_t const s = volume_except_last(inputDesc[0].dims);
        int32_t d = m_d;
        int32_t h = m_h;
        if (inputDesc[0].dims.nbDims >= 1) {
            d = last_dim(inputDesc[0].dims);
        }
        if (inputDesc[1].dims.nbDims >= 2) {
            h = inputDesc[1].dims.d[0];
        }
        int32_t const dtype = trt_dtype_to_cuda_enum(inputDesc[0].type);
        if (dtype < 0) {
            return 7;
        }
        return mopt_siglip_mlp_cuda_enqueue(stream, m_cublas, dtype, m_act_id, s, d, h, inputs[0], inputs[1],
            inputs[2], inputs[3], inputs[4], outputs[0], workspace, m_workspace_bytes);
    }

    size_t getSerializationSize() const noexcept override { return sizeof(int32_t); }

    void serialize(void* buffer) const noexcept override { std::memcpy(buffer, &m_act_id, sizeof(int32_t)); }

    void destroy() noexcept override {
        terminate();
        delete this;
    }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override { return new SiglipMlpPlugin(m_act_id); }

    void setPluginNamespace(nvinfer1::AsciiChar const* pluginNamespace) noexcept override {
        m_ns = pluginNamespace != nullptr ? pluginNamespace : "";
    }

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return m_ns.c_str(); }

    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override {
        (void)index;
        (void)nbInputs;
        return inputTypes[0];
    }

    int32_t initialize() noexcept override {
        if (cublasCreate(&m_cublas) != CUBLAS_STATUS_SUCCESS) {
            m_cublas = nullptr;
            return -1;
        }
        return 0;
    }

    void terminate() noexcept override {
        if (m_cublas != nullptr) {
            cublasDestroy(m_cublas);
            m_cublas = nullptr;
        }
    }

private:
    int32_t m_act_id{0};
    int32_t m_d{0};
    int32_t m_h{0};
    int32_t m_s_max{1};
    size_t m_workspace_bytes{0};
    cublasHandle_t m_cublas{nullptr};
    std::string m_ns{};
};

class SiglipMlpPluginCreator final : public nvinfer1::IPluginCreator {
public:
    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return "SiglipMlpPlugin"; }

    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return "1"; }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override {
        static nvinfer1::PluginFieldCollection fc{0, nullptr};
        return &fc;
    }

    nvinfer1::IPluginV2* createPlugin(
        nvinfer1::AsciiChar const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override {
        (void)name;
        int32_t act_id = 0;
        if (fc != nullptr && fc->nbFields > 0 && fc->fields != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                nvinfer1::PluginField const& f = fc->fields[i];
                if (f.name == nullptr || f.data == nullptr) {
                    continue;
                }
                if ((std::strcmp(f.name, "act_id") == 0 || std::strcmp(f.name, "act_id_i") == 0)
                    && f.type == nvinfer1::PluginFieldType::kINT32 && f.length >= 1) {
                    act_id = *static_cast<int32_t const*>(f.data);
                }
            }
        }
        return new SiglipMlpPlugin(act_id);
    }

    nvinfer1::IPluginV2* deserializePlugin(
        nvinfer1::AsciiChar const* name, void const* serialData, size_t serialLength) noexcept override {
        (void)name;
        if (serialLength < sizeof(int32_t) || serialData == nullptr) {
            return nullptr;
        }
        int32_t act_id{};
        std::memcpy(&act_id, serialData, sizeof(int32_t));
        return new SiglipMlpPlugin(act_id);
    }

    void setPluginNamespace(nvinfer1::AsciiChar const* pluginNamespace) noexcept override {
        m_creator_ns = pluginNamespace != nullptr ? pluginNamespace : "";
    }

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return m_creator_ns.c_str(); }

private:
    std::string m_creator_ns{};
};

} // namespace mopt_trt

using mopt_trt::SiglipMlpPluginCreator;
REGISTER_TENSORRT_PLUGIN(SiglipMlpPluginCreator);
