// Copyright 2026 the model_optimizer team.
//
// SPDX-License-Identifier: Apache-2.0
//
// TensorRT IPluginV2DynamicExt: fused PaliGemma text FFN (GeGLU / gelu_pytorch_tanh), fp16/bf16.
// Verbose I/O: set MODEL_OPTIMIZER_GEMMA_TRT_PLUGIN_VERBOSE=1 for stderr logs in configurePlugin.

#include <NvInfer.h>
#include <NvInferPlugin.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "gemma_fused_gated_mlp_cuda.h"

namespace mopt_trt {

namespace {

bool gemma_trt_plugin_verbose() noexcept {
    char const* const e = ::getenv("MODEL_OPTIMIZER_GEMMA_TRT_PLUGIN_VERBOSE");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
}

char const* data_type_str(nvinfer1::DataType t) noexcept {
    switch (t) {
    case nvinfer1::DataType::kFLOAT:
        return "fp32";
    case nvinfer1::DataType::kHALF:
        return "fp16";
    case nvinfer1::DataType::kBF16:
        return "bf16";
    default:
        return "other";
    }
}

void print_dims(char const* label, nvinfer1::Dims const& d) noexcept {
    std::fprintf(stderr, "%s nbDims=%d [", label, static_cast<int>(d.nbDims));
    for (int32_t i = 0; i < d.nbDims; ++i) {
        std::fprintf(stderr, "%s%d", i > 0 ? ", " : "", static_cast<int>(d.d[i]));
    }
    std::fprintf(stderr, "]\n");
}

} // namespace

class GemmaFusedGatedMlpPlugin final : public nvinfer1::IPluginV2DynamicExt {
public:
    explicit GemmaFusedGatedMlpPlugin(int32_t act_id)
        : m_act_id(act_id) {}

    GemmaFusedGatedMlpPlugin(void const* serialData, size_t serialLength) {
        if (serialLength >= sizeof(int32_t) && serialData != nullptr) {
            std::memcpy(&m_act_id, serialData, sizeof(int32_t));
        }
    }

    nvinfer1::AsciiChar const* getPluginType() const noexcept override { return "GemmaFusedGatedMlp"; }

    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return "1"; }

    int32_t getNbOutputs() const noexcept override { return 1; }

    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        (void)outputIndex;
        (void)exprBuilder;
        if (nbInputs < 3 || inputs == nullptr) {
            nvinfer1::DimsExprs bad{};
            bad.nbDims = 0;
            return bad;
        }
        nvinfer1::DimsExprs out{};
        out.nbDims = inputs[0].nbDims;
        for (int32_t i = 0; i < out.nbDims - 1; ++i) {
            out.d[i] = inputs[0].d[i];
        }
        out.d[out.nbDims - 1] = inputs[2].d[0];
        return out;
    }

    bool supportsFormatCombination(int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override {
        int32_t const n = nbInputs + nbOutputs;
        if (pos < 0 || pos >= n) {
            return false;
        }
        nvinfer1::DataType const t0 = inOut[0].type;
        nvinfer1::DataType const t = inOut[pos].type;
        if (t != nvinfer1::DataType::kHALF && t != nvinfer1::DataType::kBF16) {
            return false;
        }
        if (t != t0) {
            return false;
        }
        return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    }

    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override {
        (void)out;
        (void)nbOutputs;
        if (nbInputs < 3 || in == nullptr) {
            return;
        }
        nvinfer1::Dims const xd = in[0].desc.dims;
        nvinfer1::Dims const wd = in[1].desc.dims;
        nvinfer1::Dims const dd = in[2].desc.dims;
        if (xd.nbDims < 1 || wd.nbDims != 2 || dd.nbDims != 2) {
            if (gemma_trt_plugin_verbose()) {
                std::fprintf(stderr, "[GemmaFusedGatedMlp] configurePlugin: bad rank (xd.nbDims=%d wd.nbDims=%d "
                               "dd.nbDims=%d)\n",
                    static_cast<int>(xd.nbDims), static_cast<int>(wd.nbDims), static_cast<int>(dd.nbDims));
            }
            return;
        }
        int32_t const hidden = xd.d[xd.nbDims - 1];
        int32_t const inter = wd.d[0] / 2;
        if (gemma_trt_plugin_verbose()) {
            std::fprintf(stderr, "[GemmaFusedGatedMlp] configurePlugin I/O contract: kLINEAR row-major; "
                           "x[...,H], gate_up[2I,H], down[H,I]; CUDA uses leading M = prod(batch dims).\n");
            for (int32_t i = 0; i < nbInputs; ++i) {
                nvinfer1::PluginTensorDesc const& d = in[i].desc;
                std::fprintf(stderr, "  in[%d] type=%s format=%d (kLINEAR=%d)\n", static_cast<int>(i),
                    data_type_str(d.type), static_cast<int>(d.format), static_cast<int>(nvinfer1::TensorFormat::kLINEAR));
                print_dims("    desc.dims", d.dims);
                print_dims("    min", in[i].min);
                print_dims("    opt", in[i].opt);
                print_dims("    max", in[i].max);
            }
        }
        if (wd.d[1] != hidden || dd.d[0] != hidden || dd.d[1] != inter || wd.d[0] != 2 * inter) {
            if (gemma_trt_plugin_verbose()) {
                std::fprintf(stderr, "[GemmaFusedGatedMlp] configurePlugin: dimension mismatch vs contract "
                               "(hidden=%d inter=%d): gate_up[%d,%d] down[%d,%d] (expect gate_up[2I,H] down[H,I])\n",
                    static_cast<int>(hidden), static_cast<int>(inter), static_cast<int>(wd.d[0]),
                    static_cast<int>(wd.d[1]), static_cast<int>(dd.d[0]), static_cast<int>(dd.d[1]));
            }
        }
        m_hidden = hidden;
        m_inter = inter;
        m_io_type = in[0].desc.type == nvinfer1::DataType::kBF16 ? 1 : 0;
        int64_t prod = 1;
        nvinfer1::Dims const& dmax = in[0].max;
        nvinfer1::Dims const& dopt = in[0].opt;
        for (int32_t i = 0; i < xd.nbDims - 1; ++i) {
            int32_t hi = dmax.d[i];
            if (hi <= 0) {
                hi = dopt.d[i];
            }
            if (hi <= 0) {
                hi = 1;
            }
            prod *= static_cast<int64_t>(hi);
        }
        m_m_leading = static_cast<int32_t>(std::min<int64_t>(prod, static_cast<int64_t>(INT_MAX)));
    }

    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override {
        (void)outputs;
        (void)nbOutputs;
        if (nbInputs < 3 || inputs == nullptr) {
            return 0;
        }
        int64_t m = 1;
        nvinfer1::Dims const xd = inputs[0].dims;
        bool use_cached_leading = false;
        for (int32_t i = 0; i < xd.nbDims - 1; ++i) {
            int64_t const v = xd.d[i];
            if (v <= 0) {
                use_cached_leading = true;
                break;
            }
            m *= v;
        }
        if (use_cached_leading) {
            m = m_m_leading;
        }
        int32_t const mi = static_cast<int32_t>(std::min<int64_t>(m, static_cast<int64_t>(INT_MAX)));
        int32_t const hidden = xd.d[xd.nbDims - 1];
        int32_t const inter = inputs[1].dims.d[0] / 2;
        int32_t const io = inputs[0].type == nvinfer1::DataType::kBF16 ? 1 : 0;
        return gemma_fused_gated_mlp_workspace_bytes(mi, hidden, inter, io);
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
        (void)outputDesc;
        int64_t m64 = 1;
        nvinfer1::Dims const xd = inputDesc[0].dims;
        for (int32_t i = 0; i < xd.nbDims - 1; ++i) {
            m64 *= static_cast<int64_t>(xd.d[i]);
        }
        int32_t const m = static_cast<int32_t>(std::min<int64_t>(m64, static_cast<int64_t>(INT_MAX)));
        size_t const need = gemma_fused_gated_mlp_workspace_bytes(m, m_hidden, m_inter, m_io_type);
        int const st = gemma_fused_gated_mlp_cuda(stream, m_act_id, m_io_type, m, m_hidden, m_inter, inputs[0],
            inputs[1], inputs[2], outputs[0], workspace, need);
        return st;
    }

    size_t getSerializationSize() const noexcept override { return sizeof(int32_t); }

    void serialize(void* buffer) const noexcept override { std::memcpy(buffer, &m_act_id, sizeof(int32_t)); }

    void destroy() noexcept override { delete this; }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override { return new GemmaFusedGatedMlpPlugin(m_act_id); }

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

    int32_t initialize() noexcept override { return 0; }

    void terminate() noexcept override {}

private:
    int32_t m_act_id{1};
    int32_t m_hidden{0};
    int32_t m_inter{0};
    int32_t m_io_type{0};
    //! Product of non-hidden leading dims (upper bound from configurePlugin) for dynamic-shape workspace.
    int32_t m_m_leading{1};
    std::string m_ns{};
};

class GemmaFusedGatedMlpPluginCreator final : public nvinfer1::IPluginCreator {
public:
    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return "GemmaFusedGatedMlp"; }

    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return "1"; }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override {
        static nvinfer1::PluginField fields[] = {
            {"act_id", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
        };
        static nvinfer1::PluginFieldCollection fc{1, fields};
        return &fc;
    }

    nvinfer1::IPluginV2* createPlugin(
        nvinfer1::AsciiChar const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override {
        (void)name;
        int32_t act_id = 1;
        if (fc != nullptr && fc->nbFields > 0 && fc->fields != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                nvinfer1::PluginField const& f = fc->fields[i];
                if (f.name != nullptr && std::strcmp(f.name, "act_id") == 0 && f.type == nvinfer1::PluginFieldType::kINT32
                    && f.data != nullptr && f.length >= 1) {
                    act_id = *static_cast<int32_t const*>(f.data);
                }
            }
        }
        return new GemmaFusedGatedMlpPlugin(act_id);
    }

    nvinfer1::IPluginV2* deserializePlugin(
        nvinfer1::AsciiChar const* name, void const* serialData, size_t serialLength) noexcept override {
        (void)name;
        return new GemmaFusedGatedMlpPlugin(serialData, serialLength);
    }

    void setPluginNamespace(nvinfer1::AsciiChar const* pluginNamespace) noexcept override {
        m_creator_ns = pluginNamespace != nullptr ? pluginNamespace : "";
    }

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return m_creator_ns.c_str(); }

private:
    std::string m_creator_ns{};
};

//! ONNX 自定义算子域为 ``trt::GemmaFusedGatedMlp`` 时，引擎反序列化常在命名空间 ``"trt"`` 下
//! ``getCreator``；仅注册 ``""`` 会导致 ``Cannot find plugin ... namespace``。使用两个 Creator
//! 实例分别注册 ``""`` 与 ``"trt"``（与 ``loadLibrary`` / ``getCreators`` 一致）。
static GemmaFusedGatedMlpPluginCreator g_gemma_fused_gated_mlp_plugin_creator_default{};
static GemmaFusedGatedMlpPluginCreator g_gemma_fused_gated_mlp_plugin_creator_trt{};

struct GemmaFusedGatedMlpPluginRegisterOnce {
    GemmaFusedGatedMlpPluginRegisterOnce() noexcept {
        // ``getPluginRegistry`` 在 ``NvInferRuntime.h`` 中为全局 ``extern "C"``，不在 ``nvinfer1`` 内。
        nvinfer1::IPluginRegistry* reg = ::getPluginRegistry();
        if (reg == nullptr) {
            return;
        }
        auto try_register = [&](GemmaFusedGatedMlpPluginCreator& creator, char const* ns) noexcept {
            if (reg->getCreator("GemmaFusedGatedMlp", "1", ns) != nullptr) {
                return;
            }
            creator.setPluginNamespace(ns);
            (void) reg->registerCreator(static_cast<nvinfer1::IPluginCreatorInterface&>(creator), ns);
        };
        try_register(g_gemma_fused_gated_mlp_plugin_creator_default, "");
        try_register(g_gemma_fused_gated_mlp_plugin_creator_trt, "trt");
    }
};

static GemmaFusedGatedMlpPluginRegisterOnce g_gemma_fused_gated_mlp_plugin_register_once{};

} // namespace mopt_trt

//!
//! TensorRT 10+ ``IPluginRegistry::loadLibrary`` 会 ``dlsym(getCreators)``；若 .so 未导出该符号则报
//! API Usage Error。此处与官方动态插件约定一致，导出 ``getCreators`` / ``setLoggerFinder``；并在库
//! 加载时用 ``registerCreator`` 注册（``ctypes.CDLL`` 路径不一定会调用 ``loadLibrary``）。
//!
extern "C" TENSORRTAPI void setLoggerFinder(nvinfer1::ILoggerFinder* finder) noexcept {
    (void) finder;
}

extern "C" TENSORRTAPI nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& nbCreators) noexcept {
    nbCreators = 2;
    static nvinfer1::IPluginCreatorInterface* const kCreators[] = {
        &mopt_trt::g_gemma_fused_gated_mlp_plugin_creator_default,
        &mopt_trt::g_gemma_fused_gated_mlp_plugin_creator_trt,
    };
    return kCreators;
}
