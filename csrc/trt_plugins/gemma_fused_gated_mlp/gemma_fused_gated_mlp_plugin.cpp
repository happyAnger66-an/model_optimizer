// Copyright 2026 the model_optimizer team.
//
// SPDX-License-Identifier: Apache-2.0
//
// TensorRT IPluginV2DynamicExt: fused PaliGemma text FFN (GeGLU / gelu_pytorch_tanh), fp16/bf16.
//
// - Version "1": inputs ``x``, ``gate_up_weight``, ``down_weight`` (图权重边，易触发 Myelin ``__myl_Move_*``)。
// - Version "2": input ``x`` only; weights from ONNX tensor attributes → PluginField → 序列化进引擎；
//   ``initialize()`` 一次 H2D，``enqueue`` 不再经图输入搬运权重。
//
// Verbose I/O: ``MODEL_OPTIMIZER_GEMMA_TRT_PLUGIN_VERBOSE=1`` → ``configurePlugin`` 打 stderr。

#include <NvInfer.h>
#include <NvInferPlugin.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

//! Little-endian magic ``'G'<<0 | 'F'<<8 | 'M'<<16 | '2'<<24``.
static constexpr uint32_t kSerialMagicV2 = 0x324D4647u;

static int32_t io_type_from_desc(nvinfer1::DataType t) noexcept {
    return t == nvinfer1::DataType::kBF16 ? 1 : 0;
}

} // namespace

class GemmaFusedGatedMlpPlugin final : public nvinfer1::IPluginV2DynamicExt {
public:
    //! External weights path (v1).
    explicit GemmaFusedGatedMlpPlugin(int32_t act_id)
        : m_act_id(act_id)
        , m_baked_weights(false)
        , m_plugin_version("1") {}

    //! Deserialize: v1 = 4-byte ``act_id`` only; v2 = ``kSerialMagicV2`` + host weight payload.
    GemmaFusedGatedMlpPlugin(void const* serialData, size_t serialLength, bool /*expect_v2*/) {
        if (serialLength >= 8 && serialData != nullptr) {
            uint32_t magic = 0;
            std::memcpy(&magic, serialData, sizeof(uint32_t));
            if (magic == kSerialMagicV2) {
                parse_serial_v2(serialData, serialLength);
                return;
            }
        }
        if (serialLength >= sizeof(int32_t) && serialData != nullptr) {
            std::memcpy(&m_act_id, serialData, sizeof(int32_t));
        }
        m_baked_weights = false;
        m_plugin_version = "1";
    }

    //! Baked weights from ONNX ``PluginField`` (v2).
    GemmaFusedGatedMlpPlugin(int32_t act_id, int32_t hidden, int32_t inter, int32_t io_type,
        std::vector<uint8_t>&& gate_up_host, std::vector<uint8_t>&& down_host)
        : m_act_id(act_id)
        , m_hidden(hidden)
        , m_inter(inter)
        , m_io_type(io_type)
        , m_baked_weights(true)
        , m_plugin_version("2")
        , m_host_gate_up(std::move(gate_up_host))
        , m_host_down(std::move(down_host)) {}

    nvinfer1::AsciiChar const* getPluginType() const noexcept override { return "GemmaFusedGatedMlp"; }

    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return m_plugin_version.c_str(); }

    int32_t getNbOutputs() const noexcept override { return 1; }

    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        (void)outputIndex;
        (void)exprBuilder;
        if (inputs == nullptr) {
            nvinfer1::DimsExprs bad{};
            bad.nbDims = 0;
            return bad;
        }
        if (nbInputs == 1) {
            nvinfer1::DimsExprs out{};
            out.nbDims = inputs[0].nbDims;
            for (int32_t i = 0; i < out.nbDims; ++i) {
                out.d[i] = inputs[0].d[i];
            }
            return out;
        }
        if (nbInputs < 3) {
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
        if (in == nullptr) {
            return;
        }
        if (m_baked_weights) {
            if (nbInputs < 1) {
                return;
            }
            nvinfer1::Dims const xd = in[0].desc.dims;
            if (xd.nbDims < 1) {
                return;
            }
            int32_t const xh = xd.d[xd.nbDims - 1];
            if (gemma_trt_plugin_verbose()) {
                std::fprintf(stderr, "[GemmaFusedGatedMlp] v2 configurePlugin (baked weights): hidden=%d inter=%d\n",
                    static_cast<int>(m_hidden), static_cast<int>(m_inter));
                print_dims("  in[0].desc.dims", in[0].desc.dims);
            }
            if (xh != m_hidden && m_hidden > 0) {
                if (gemma_trt_plugin_verbose()) {
                    std::fprintf(stderr, "[GemmaFusedGatedMlp] v2: x last dim %d != baked hidden %d\n",
                        static_cast<int>(xh), static_cast<int>(m_hidden));
                }
            }
            m_io_type = io_type_from_desc(in[0].desc.type);
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
            return;
        }

        if (nbInputs < 3) {
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
            std::fprintf(stderr, "[GemmaFusedGatedMlp] v1 configurePlugin: kLINEAR row-major; "
                           "x[...,H], gate_up[2I,H], down[H,I].\n");
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
                               "(hidden=%d inter=%d): gate_up[%d,%d] down[%d,%d]\n",
                    static_cast<int>(hidden), static_cast<int>(inter), static_cast<int>(wd.d[0]),
                    static_cast<int>(wd.d[1]), static_cast<int>(dd.d[0]), static_cast<int>(dd.d[1]));
            }
        }
        m_hidden = hidden;
        m_inter = inter;
        m_io_type = io_type_from_desc(in[0].desc.type);
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
        if (inputs == nullptr || nbInputs < 1) {
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
        int32_t inter = m_inter;
        if (!m_baked_weights && nbInputs >= 2) {
            inter = inputs[1].dims.d[0] / 2;
        }
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
        int32_t const io = inputDesc[0].type == nvinfer1::DataType::kBF16 ? 1 : 0;
        int32_t const hidden = xd.d[xd.nbDims - 1];
        int32_t inter = m_inter;
        if (!m_baked_weights) {
            inter = inputDesc[1].dims.d[0] / 2;
        }
        size_t const need = gemma_fused_gated_mlp_workspace_bytes(m, hidden, inter, io);
        void const* wgu = m_baked_weights ? static_cast<void const*>(m_d_gate_up) : inputs[1];
        void const* wdn = m_baked_weights ? static_cast<void const*>(m_d_down) : inputs[2];
        int const st = gemma_fused_gated_mlp_cuda(
            stream, m_act_id, io, m, hidden, inter, inputs[0], wgu, wdn, outputs[0], workspace, need);
        return st;
    }

    size_t getSerializationSize() const noexcept override {
        if (!m_baked_weights) {
            return sizeof(int32_t);
        }
        return sizeof(uint32_t) + sizeof(int32_t) * 4 + sizeof(uint64_t) * 2 + m_host_gate_up.size() + m_host_down.size();
    }

    void serialize(void* buffer) const noexcept override {
        if (!m_baked_weights) {
            std::memcpy(buffer, &m_act_id, sizeof(int32_t));
            return;
        }
        uint8_t* p = static_cast<uint8_t*>(buffer);
        uint32_t const magic = kSerialMagicV2;
        std::memcpy(p, &magic, sizeof(uint32_t));
        p += sizeof(uint32_t);
        std::memcpy(p, &m_act_id, sizeof(int32_t));
        p += sizeof(int32_t);
        std::memcpy(p, &m_hidden, sizeof(int32_t));
        p += sizeof(int32_t);
        std::memcpy(p, &m_inter, sizeof(int32_t));
        p += sizeof(int32_t);
        std::memcpy(p, &m_io_type, sizeof(int32_t));
        p += sizeof(int32_t);
        uint64_t gu = static_cast<uint64_t>(m_host_gate_up.size());
        uint64_t dn = static_cast<uint64_t>(m_host_down.size());
        std::memcpy(p, &gu, sizeof(uint64_t));
        p += sizeof(uint64_t);
        std::memcpy(p, &dn, sizeof(uint64_t));
        p += sizeof(uint64_t);
        if (gu > 0) {
            std::memcpy(p, m_host_gate_up.data(), static_cast<size_t>(gu));
            p += static_cast<size_t>(gu);
        }
        if (dn > 0) {
            std::memcpy(p, m_host_down.data(), static_cast<size_t>(dn));
        }
    }

    void destroy() noexcept override { delete this; }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
        if (!m_baked_weights) {
            return new GemmaFusedGatedMlpPlugin(m_act_id);
        }
        auto* p = new GemmaFusedGatedMlpPlugin(m_act_id, m_hidden, m_inter, m_io_type,
            std::vector<uint8_t>(m_host_gate_up), std::vector<uint8_t>(m_host_down));
        p->m_m_leading = m_m_leading;
        return p;
    }

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
        if (!m_baked_weights || m_d_gate_up != nullptr) {
            return 0;
        }
        if (m_host_gate_up.empty() || m_host_down.empty()) {
            return -1;
        }
        cudaError_t e1 = cudaMalloc(&m_d_gate_up, m_host_gate_up.size());
        cudaError_t e2 = cudaMalloc(&m_d_down, m_host_down.size());
        if (e1 != cudaSuccess || e2 != cudaSuccess) {
            return -2;
        }
        e1 = cudaMemcpy(m_d_gate_up, m_host_gate_up.data(), m_host_gate_up.size(), cudaMemcpyHostToDevice);
        e2 = cudaMemcpy(m_d_down, m_host_down.data(), m_host_down.size(), cudaMemcpyHostToDevice);
        if (e1 != cudaSuccess || e2 != cudaSuccess) {
            return -3;
        }
        return 0;
    }

    void terminate() noexcept override {
        if (m_d_gate_up != nullptr) {
            (void)cudaFree(m_d_gate_up);
            m_d_gate_up = nullptr;
        }
        if (m_d_down != nullptr) {
            (void)cudaFree(m_d_down);
            m_d_down = nullptr;
        }
    }

private:
    void parse_serial_v2(void const* serialData, size_t serialLength) {
        uint8_t const* p = static_cast<uint8_t const*>(serialData);
        size_t off = sizeof(uint32_t);
        if (serialLength < off + sizeof(int32_t) * 4 + sizeof(uint64_t) * 2) {
            return;
        }
        std::memcpy(&m_act_id, p + off, sizeof(int32_t));
        off += sizeof(int32_t);
        std::memcpy(&m_hidden, p + off, sizeof(int32_t));
        off += sizeof(int32_t);
        std::memcpy(&m_inter, p + off, sizeof(int32_t));
        off += sizeof(int32_t);
        std::memcpy(&m_io_type, p + off, sizeof(int32_t));
        off += sizeof(int32_t);
        uint64_t gu = 0;
        uint64_t dn = 0;
        std::memcpy(&gu, p + off, sizeof(uint64_t));
        off += sizeof(uint64_t);
        std::memcpy(&dn, p + off, sizeof(uint64_t));
        off += sizeof(uint64_t);
        if (off + gu + dn > serialLength) {
            return;
        }
        m_host_gate_up.resize(static_cast<size_t>(gu));
        m_host_down.resize(static_cast<size_t>(dn));
        if (gu > 0) {
            std::memcpy(m_host_gate_up.data(), p + off, static_cast<size_t>(gu));
            off += static_cast<size_t>(gu);
        }
        if (dn > 0) {
            std::memcpy(m_host_down.data(), p + off, static_cast<size_t>(dn));
        }
        m_baked_weights = true;
        m_plugin_version = "2";
    }

    int32_t m_act_id{1};
    int32_t m_hidden{0};
    int32_t m_inter{0};
    int32_t m_io_type{0};
    int32_t m_m_leading{1};
    bool m_baked_weights{false};
    std::string m_plugin_version{"1"};
    std::vector<uint8_t> m_host_gate_up{};
    std::vector<uint8_t> m_host_down{};
    void* m_d_gate_up{nullptr};
    void* m_d_down{nullptr};
    std::string m_ns{};
};

class GemmaFusedGatedMlpPluginCreatorV1 final : public nvinfer1::IPluginCreator {
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
        if (serialLength != sizeof(int32_t)) {
            return nullptr;
        }
        return new GemmaFusedGatedMlpPlugin(serialData, serialLength, false);
    }

    void setPluginNamespace(nvinfer1::AsciiChar const* pluginNamespace) noexcept override {
        m_creator_ns = pluginNamespace != nullptr ? pluginNamespace : "";
    }

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return m_creator_ns.c_str(); }

private:
    std::string m_creator_ns{};
};

//! ONNX-TensorRT ``FallbackPluginImporter`` 优先读取节点 **字符串** 属性 ``plugin_version`` / ``plugin_namespace``
//!（缺省 ``"1"`` / ``""``）再 ``getCreator``（见 ``onnx-tensorrt/onnxOpCheckers.cpp``）。``"19"`` 与
//! ``model_optimizer.ops.gemma_fused_gated_mlp_plugin.ONNX_OPSET_VERSION`` 对齐，作兜底注册。
static constexpr char const* kGemmaFusedGatedMlpOnnxOpsetVersion = "19";

static bool onnx_plugin_field_collection_has_baked_weights(nvinfer1::PluginFieldCollection const* fc) noexcept {
    if (fc == nullptr || fc->nbFields <= 0 || fc->fields == nullptr) {
        return false;
    }
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        nvinfer1::PluginField const& f = fc->fields[i];
        if (f.name == nullptr) {
            continue;
        }
        if (std::strcmp(f.name, "hidden_dim") == 0 || std::strcmp(f.name, "inter_dim") == 0
            || std::strcmp(f.name, "gate_up_weight") == 0 || std::strcmp(f.name, "down_weight") == 0) {
            return true;
        }
    }
    return false;
}

class GemmaFusedGatedMlpPluginCreatorV2 final : public nvinfer1::IPluginCreator {
public:
    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return "GemmaFusedGatedMlp"; }

    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return "2"; }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override {
        static nvinfer1::PluginField fields[] = {
            {"act_id", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
            {"hidden_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
            {"inter_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
            {"gate_up_weight", nullptr, nvinfer1::PluginFieldType::kFLOAT16, 0},
            {"down_weight", nullptr, nvinfer1::PluginFieldType::kFLOAT16, 0},
        };
        static nvinfer1::PluginFieldCollection fc{5, fields};
        return &fc;
    }

    static bool copy_weight_field(nvinfer1::PluginField const& f, std::vector<uint8_t>& out_host) noexcept {
        if (f.data == nullptr || f.length <= 0) {
            return false;
        }
        size_t const bytes = static_cast<size_t>(f.length)
            * ((f.type == nvinfer1::PluginFieldType::kBF16 || f.type == nvinfer1::PluginFieldType::kFLOAT16) ? 2u
                : 0u);
        if (bytes == 0) {
            return false;
        }
        out_host.resize(bytes);
        std::memcpy(out_host.data(), f.data, bytes);
        return true;
    }

    nvinfer1::IPluginV2* createPlugin(
        nvinfer1::AsciiChar const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override {
        (void)name;
        int32_t act_id = 1;
        int32_t hidden = 0;
        int32_t inter = 0;
        std::vector<uint8_t> gu;
        std::vector<uint8_t> dn;
        bool have_gu = false;
        bool have_dn = false;
        if (fc != nullptr && fc->nbFields > 0 && fc->fields != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                nvinfer1::PluginField const& f = fc->fields[i];
                if (f.name == nullptr || f.data == nullptr) {
                    continue;
                }
                if (std::strcmp(f.name, "act_id") == 0 && f.type == nvinfer1::PluginFieldType::kINT32 && f.length >= 1) {
                    act_id = *static_cast<int32_t const*>(f.data);
                } else if (std::strcmp(f.name, "hidden_dim") == 0 && f.type == nvinfer1::PluginFieldType::kINT32
                    && f.length >= 1) {
                    hidden = *static_cast<int32_t const*>(f.data);
                } else if (std::strcmp(f.name, "inter_dim") == 0 && f.type == nvinfer1::PluginFieldType::kINT32
                    && f.length >= 1) {
                    inter = *static_cast<int32_t const*>(f.data);
                } else if (std::strcmp(f.name, "gate_up_weight") == 0
                    && (f.type == nvinfer1::PluginFieldType::kBF16 || f.type == nvinfer1::PluginFieldType::kFLOAT16)) {
                    have_gu = copy_weight_field(f, gu);
                } else if (std::strcmp(f.name, "down_weight") == 0
                    && (f.type == nvinfer1::PluginFieldType::kBF16 || f.type == nvinfer1::PluginFieldType::kFLOAT16)) {
                    have_dn = copy_weight_field(f, dn);
                }
            }
        }
        if (hidden <= 0 || inter <= 0 || !have_gu || !have_dn) {
            return nullptr;
        }
        uint64_t const expect_gu = static_cast<uint64_t>(2) * static_cast<uint64_t>(inter)
            * static_cast<uint64_t>(hidden) * 2u;
        uint64_t const expect_dn = static_cast<uint64_t>(hidden) * static_cast<uint64_t>(inter) * 2u;
        if (static_cast<uint64_t>(gu.size()) != expect_gu || static_cast<uint64_t>(dn.size()) != expect_dn) {
            return nullptr;
        }
        int32_t io_type = 0;
        if (fc != nullptr && fc->nbFields > 0 && fc->fields != nullptr) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                nvinfer1::PluginField const& f = fc->fields[i];
                if (f.name != nullptr && std::strcmp(f.name, "gate_up_weight") == 0) {
                    io_type = (f.type == nvinfer1::PluginFieldType::kBF16) ? 1 : 0;
                    break;
                }
            }
        }
        return new GemmaFusedGatedMlpPlugin(act_id, hidden, inter, io_type, std::move(gu), std::move(dn));
    }

    nvinfer1::IPluginV2* deserializePlugin(
        nvinfer1::AsciiChar const* name, void const* serialData, size_t serialLength) noexcept override {
        (void)name;
        if (serialLength < 8 || serialData == nullptr) {
            return nullptr;
        }
        uint32_t magic = 0;
        std::memcpy(&magic, serialData, sizeof(uint32_t));
        if (magic != kSerialMagicV2) {
            return nullptr;
        }
        size_t off = sizeof(uint32_t);
        if (serialLength < off + sizeof(int32_t) * 4 + sizeof(uint64_t) * 2) {
            return nullptr;
        }
        int32_t act_id = 0;
        int32_t hidden = 0;
        int32_t inter = 0;
        int32_t io_type = 0;
        uint64_t gu = 0;
        uint64_t dn = 0;
        uint8_t const* p = static_cast<uint8_t const*>(serialData);
        std::memcpy(&act_id, p + off, sizeof(int32_t));
        off += sizeof(int32_t);
        std::memcpy(&hidden, p + off, sizeof(int32_t));
        off += sizeof(int32_t);
        std::memcpy(&inter, p + off, sizeof(int32_t));
        off += sizeof(int32_t);
        std::memcpy(&io_type, p + off, sizeof(int32_t));
        off += sizeof(int32_t);
        std::memcpy(&gu, p + off, sizeof(uint64_t));
        off += sizeof(uint64_t);
        std::memcpy(&dn, p + off, sizeof(uint64_t));
        off += sizeof(uint64_t);
        if (hidden <= 0 || inter <= 0) {
            return nullptr;
        }
        uint64_t const expect_gu = static_cast<uint64_t>(2) * static_cast<uint64_t>(inter)
            * static_cast<uint64_t>(hidden) * 2u;
        uint64_t const expect_dn = static_cast<uint64_t>(hidden) * static_cast<uint64_t>(inter) * 2u;
        if (gu != expect_gu || dn != expect_dn) {
            return nullptr;
        }
        if (off + gu + dn != serialLength) {
            return nullptr;
        }
        (void)act_id;
        (void)io_type;
        return new GemmaFusedGatedMlpPlugin(serialData, serialLength, true);
    }

    void setPluginNamespace(nvinfer1::AsciiChar const* pluginNamespace) noexcept override {
        m_creator_ns = pluginNamespace != nullptr ? pluginNamespace : "";
    }

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return m_creator_ns.c_str(); }

private:
    std::string m_creator_ns{};
};

static GemmaFusedGatedMlpPluginCreatorV1 g_gemma_fused_gated_mlp_plugin_creator_v1_default{};
static GemmaFusedGatedMlpPluginCreatorV1 g_gemma_fused_gated_mlp_plugin_creator_v1_trt{};
static GemmaFusedGatedMlpPluginCreatorV2 g_gemma_fused_gated_mlp_plugin_creator_v2_default{};
static GemmaFusedGatedMlpPluginCreatorV2 g_gemma_fused_gated_mlp_plugin_creator_v2_trt{};

//! 供 ONNX Parser（``opset_import`` 版本 ``kGemmaFusedGatedMlpOnnxOpsetVersion``）查找；在 ``createPlugin`` 中分派到 v1/v2。
class GemmaFusedGatedMlpPluginCreatorOnnx19 final : public nvinfer1::IPluginCreator {
public:
    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return "GemmaFusedGatedMlp"; }

    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return kGemmaFusedGatedMlpOnnxOpsetVersion; }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override {
        static nvinfer1::PluginField fields[] = {
            {"act_id", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
            {"hidden_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
            {"inter_dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
            {"plugin_version", nullptr, nvinfer1::PluginFieldType::kINT32, 1},
            {"gate_up_weight", nullptr, nvinfer1::PluginFieldType::kFLOAT16, 0},
            {"down_weight", nullptr, nvinfer1::PluginFieldType::kFLOAT16, 0},
        };
        static nvinfer1::PluginFieldCollection fc{6, fields};
        return &fc;
    }

    nvinfer1::IPluginV2* createPlugin(
        nvinfer1::AsciiChar const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override {
        if (onnx_plugin_field_collection_has_baked_weights(fc)) {
            return g_gemma_fused_gated_mlp_plugin_creator_v2_default.createPlugin(name, fc);
        }
        return g_gemma_fused_gated_mlp_plugin_creator_v1_default.createPlugin(name, fc);
    }

    nvinfer1::IPluginV2* deserializePlugin(
        nvinfer1::AsciiChar const* name, void const* serialData, size_t serialLength) noexcept override {
        if (serialLength >= sizeof(uint32_t) && serialData != nullptr) {
            uint32_t magic = 0;
            std::memcpy(&magic, serialData, sizeof(uint32_t));
            if (magic == kSerialMagicV2) {
                return g_gemma_fused_gated_mlp_plugin_creator_v2_default.deserializePlugin(name, serialData, serialLength);
            }
        }
        return g_gemma_fused_gated_mlp_plugin_creator_v1_default.deserializePlugin(name, serialData, serialLength);
    }

    void setPluginNamespace(nvinfer1::AsciiChar const* pluginNamespace) noexcept override {
        m_creator_ns = pluginNamespace != nullptr ? pluginNamespace : "";
    }

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return m_creator_ns.c_str(); }

private:
    std::string m_creator_ns{};
};

static GemmaFusedGatedMlpPluginCreatorOnnx19 g_gemma_fused_gated_mlp_plugin_creator_onnx19_default{};
static GemmaFusedGatedMlpPluginCreatorOnnx19 g_gemma_fused_gated_mlp_plugin_creator_onnx19_trt{};

struct GemmaFusedGatedMlpPluginRegisterOnce {
    GemmaFusedGatedMlpPluginRegisterOnce() noexcept {
        nvinfer1::IPluginRegistry* reg = ::getPluginRegistry();
        if (reg == nullptr) {
            return;
        }
        auto try_register = [&](nvinfer1::IPluginCreator& creator, char const* ns, char const* ver) noexcept {
            if (reg->getCreator("GemmaFusedGatedMlp", ver, ns) != nullptr) {
                return;
            }
            creator.setPluginNamespace(ns);
            (void) reg->registerCreator(static_cast<nvinfer1::IPluginCreatorInterface&>(creator), ns);
        };
        try_register(g_gemma_fused_gated_mlp_plugin_creator_v1_default, "", "1");
        try_register(g_gemma_fused_gated_mlp_plugin_creator_v1_trt, "trt", "1");
        try_register(g_gemma_fused_gated_mlp_plugin_creator_v2_default, "", "2");
        try_register(g_gemma_fused_gated_mlp_plugin_creator_v2_trt, "trt", "2");
        try_register(g_gemma_fused_gated_mlp_plugin_creator_onnx19_default, "", kGemmaFusedGatedMlpOnnxOpsetVersion);
        try_register(g_gemma_fused_gated_mlp_plugin_creator_onnx19_trt, "trt", kGemmaFusedGatedMlpOnnxOpsetVersion);
    }
};

static GemmaFusedGatedMlpPluginRegisterOnce g_gemma_fused_gated_mlp_plugin_register_once{};

} // namespace mopt_trt

extern "C" TENSORRTAPI void setLoggerFinder(nvinfer1::ILoggerFinder* finder) noexcept {
    (void) finder;
}

extern "C" TENSORRTAPI nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& nbCreators) noexcept {
    nbCreators = 6;
    static nvinfer1::IPluginCreatorInterface* const kCreators[] = {
        &mopt_trt::g_gemma_fused_gated_mlp_plugin_creator_v1_default,
        &mopt_trt::g_gemma_fused_gated_mlp_plugin_creator_v1_trt,
        &mopt_trt::g_gemma_fused_gated_mlp_plugin_creator_v2_default,
        &mopt_trt::g_gemma_fused_gated_mlp_plugin_creator_v2_trt,
        &mopt_trt::g_gemma_fused_gated_mlp_plugin_creator_onnx19_default,
        &mopt_trt::g_gemma_fused_gated_mlp_plugin_creator_onnx19_trt,
    };
    return kCreators;
}
