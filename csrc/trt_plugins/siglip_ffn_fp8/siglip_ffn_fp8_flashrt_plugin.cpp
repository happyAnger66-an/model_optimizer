// Copyright 2026 the model_optimizer team.
//
// SPDX-License-Identifier: Apache-2.0
//
// TensorRT IPluginV2DynamicExt: SigLIP encoder FFN FP8 via FlashRT static bridge
// (``frt_siglip_ffn_fp8_*`` / ``GemmRunner`` + ``quantize_fp8_static_fp16``).

#include <NvInfer.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "flash_rt_trt_api.h"

namespace mopt_trt {

int32_t constexpr kNbInputs = 7;

class SiglipFfFp8FlashrtPlugin final : public nvinfer1::IPluginV2DynamicExt {
public:
    SiglipFfFp8FlashrtPlugin(float alpha_up, float alpha_down)
        : m_alpha_up(alpha_up)
        , m_alpha_down(alpha_down) {}

    nvinfer1::AsciiChar const* getPluginType() const noexcept override { return "SiglipFfFp8FlashrtPlugin"; }

    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return "1"; }

    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t getNbInputs() const noexcept override { return kNbInputs; }

    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        (void)exprBuilder;
        (void)outputIndex;
        (void)nbInputs;
        nvinfer1::DimsExprs out{};
        out.nbDims = inputs[1].nbDims;
        for (int32_t i = 0; i < out.nbDims; ++i) {
            out.d[i] = inputs[1].d[i];
        }
        return out;
    }

    bool supportsFormatCombination(int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override {
        (void)nbInputs;
        (void)nbOutputs;
        if (pos == 0 || pos == 2 || pos == 3) {
            return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR && inOut[pos].type == nvinfer1::DataType::kFP8;
        }
        if (pos == 1 || pos == 4 || pos == 5 || pos == 7) {
            return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR
                && inOut[pos].type == nvinfer1::DataType::kHALF;
        }
        if (pos == 6) {
            return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR
                && inOut[pos].type == nvinfer1::DataType::kFLOAT;
        }
        return false;
    }

    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override {
        (void)out;
        (void)nbOutputs;
        if (nbInputs < kNbInputs) {
            return;
        }
        if (in[2].desc.dims.nbDims >= 2) {
            m_d = in[2].desc.dims.d[0];
            m_h = in[2].desc.dims.d[1];
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
            m_workspace_bytes = frt_siglip_ffn_fp8_workspace_bytes(m_s_max, m_d, m_h);
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
        if (m_rt == nullptr) {
            return 1;
        }
        int32_t s = 1;
        if (inputDesc[0].dims.nbDims > 1) {
            s = 1;
            for (int32_t i = 0; i < inputDesc[0].dims.nbDims - 1; ++i) {
                s *= inputDesc[0].dims.d[i];
            }
        } else if (inputDesc[0].dims.nbDims == 1) {
            s = 1;
        }
        int32_t d = m_d;
        int32_t h = m_h;
        if (inputDesc[0].dims.nbDims >= 2) {
            d = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];
        }
        if (inputDesc[2].dims.nbDims >= 2) {
            h = inputDesc[2].dims.d[1];
        }
        return frt_siglip_ffn_fp8_enqueue(m_rt, s, d, h, m_alpha_up, m_alpha_down, inputs[0], inputs[1], inputs[2],
            inputs[3], inputs[4], inputs[5], inputs[6], outputs[0], workspace, m_workspace_bytes, stream);
    }

    size_t getSerializationSize() const noexcept override { return sizeof(float) * 2; }

    void serialize(void* buffer) const noexcept override {
        std::memcpy(buffer, &m_alpha_up, sizeof(float));
        std::memcpy(static_cast<char*>(buffer) + sizeof(float), &m_alpha_down, sizeof(float));
    }

    void destroy() noexcept override {
        terminate();
        delete this;
    }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
        return new SiglipFfFp8FlashrtPlugin(m_alpha_up, m_alpha_down);
    }

    void setPluginNamespace(nvinfer1::AsciiChar const* pluginNamespace) noexcept override {
        m_ns = pluginNamespace != nullptr ? pluginNamespace : "";
    }

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return m_ns.c_str(); }

    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override {
        (void)index;
        (void)inputTypes;
        (void)nbInputs;
        return nvinfer1::DataType::kHALF;
    }

    int32_t initialize() noexcept override {
        m_rt = frt_gemm_runner_create();
        return m_rt != nullptr ? 0 : -1;
    }

    void terminate() noexcept override {
        if (m_rt != nullptr) {
            frt_gemm_runner_destroy(m_rt);
            m_rt = nullptr;
        }
    }

private:
    float m_alpha_up{1.0F};
    float m_alpha_down{1.0F};
    int32_t m_d{0};
    int32_t m_h{0};
    int32_t m_s_max{1};
    size_t m_workspace_bytes{0};
    void* m_rt{nullptr};
    std::string m_ns{};
};

class SiglipFfFp8FlashrtPluginCreator final : public nvinfer1::IPluginCreator {
public:
    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return "SiglipFfFp8FlashrtPlugin"; }

    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return "1"; }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override {
        static nvinfer1::PluginFieldCollection fc{0, nullptr};
        return &fc;
    }

    nvinfer1::IPluginV2* createPlugin(
        nvinfer1::AsciiChar const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override {
        (void)name;
        float a_up = 1.0F;
        float a_dn = 1.0F;
        if (fc != nullptr && fc->nbFields > 0 && fc->fields != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                nvinfer1::PluginField const& f = fc->fields[i];
                if (f.name == nullptr || f.data == nullptr) {
                    continue;
                }
                if (std::strcmp(f.name, "alpha_up") == 0 && f.type == nvinfer1::PluginFieldType::kFLOAT32
                    && f.length >= 1) {
                    a_up = *static_cast<float const*>(f.data);
                }
                if (std::strcmp(f.name, "alpha_down") == 0 && f.type == nvinfer1::PluginFieldType::kFLOAT32
                    && f.length >= 1) {
                    a_dn = *static_cast<float const*>(f.data);
                }
            }
        }
        return new SiglipFfFp8FlashrtPlugin(a_up, a_dn);
    }

    nvinfer1::IPluginV2* deserializePlugin(
        nvinfer1::AsciiChar const* name, void const* serialData, size_t serialLength) noexcept override {
        (void)name;
        if (serialLength < sizeof(float) * 2 || serialData == nullptr) {
            return nullptr;
        }
        float a_up{};
        float a_dn{};
        std::memcpy(&a_up, serialData, sizeof(float));
        std::memcpy(&a_dn, static_cast<char const*>(serialData) + sizeof(float), sizeof(float));
        return new SiglipFfFp8FlashrtPlugin(a_up, a_dn);
    }

    void setPluginNamespace(nvinfer1::AsciiChar const* pluginNamespace) noexcept override {
        m_creator_ns = pluginNamespace != nullptr ? pluginNamespace : "";
    }

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return m_creator_ns.c_str(); }

private:
    std::string m_creator_ns{};
};

} // namespace mopt_trt

using mopt_trt::SiglipFfFp8FlashrtPluginCreator;
REGISTER_TENSORRT_PLUGIN(SiglipFfFp8FlashrtPluginCreator);
