// SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
#include "common/dtype.h"

#include "common/check.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <sstream>

namespace pi05
{

std::size_t dtypeSize(nvinfer1::DataType dtype)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBF16: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kUINT8: return 1;
    case nvinfer1::DataType::kBOOL: return 1;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kINT64: return 8;
    case nvinfer1::DataType::kFP8: return 1;
    default: EDGE_CHECK(false, "Unsupported dtype " << static_cast<int>(dtype)); return 0;
    }
}

std::string dtypeName(nvinfer1::DataType dtype)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT: return "fp32";
    case nvinfer1::DataType::kHALF: return "fp16";
    case nvinfer1::DataType::kBF16: return "bf16";
    case nvinfer1::DataType::kINT8: return "int8";
    case nvinfer1::DataType::kUINT8: return "uint8";
    case nvinfer1::DataType::kBOOL: return "bool";
    case nvinfer1::DataType::kINT32: return "int32";
    case nvinfer1::DataType::kINT64: return "int64";
    case nvinfer1::DataType::kFP8: return "fp8";
    default: return "unknown(" + std::to_string(static_cast<int>(dtype)) + ")";
    }
}

nvinfer1::DataType dtypeFromNpyDescr(std::string const& descr)
{
    // 去掉字节序前缀（'<' '|' '=' '>'，仅支持小端/无序）。
    std::string d = descr;
    if (!d.empty() && (d[0] == '<' || d[0] == '|' || d[0] == '='))
    {
        d = d.substr(1);
    }
    EDGE_CHECK(!descr.empty() && descr[0] != '>', "big-endian npy not supported: " << descr);
    if (d == "f4")
    {
        return nvinfer1::DataType::kFLOAT;
    }
    if (d == "f2")
    {
        return nvinfer1::DataType::kHALF;
    }
    if (d == "i8")
    {
        return nvinfer1::DataType::kINT64;
    }
    if (d == "i4")
    {
        return nvinfer1::DataType::kINT32;
    }
    if (d == "b1")
    {
        return nvinfer1::DataType::kBOOL;
    }
    if (d == "u1")
    {
        return nvinfer1::DataType::kUINT8;
    }
    EDGE_CHECK(false, "Unsupported npy descr: " << descr);
    return nvinfer1::DataType::kFLOAT;
}

std::string npyDescrFromDtype(nvinfer1::DataType dtype)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT: return "<f4";
    case nvinfer1::DataType::kHALF: return "<f2";
    case nvinfer1::DataType::kINT64: return "<i8";
    case nvinfer1::DataType::kINT32: return "<i4";
    case nvinfer1::DataType::kBOOL: return "|b1";
    case nvinfer1::DataType::kUINT8: return "|u1";
    default:
        EDGE_CHECK(false, "No npy descr for dtype " << dtypeName(dtype));
        return "";
    }
}

void convertToFloatHost(
    void const* src, nvinfer1::DataType srcType, float* dst, std::size_t count)
{
    switch (srcType)
    {
    case nvinfer1::DataType::kFLOAT:
    {
        auto const* p = static_cast<float const*>(src);
        for (std::size_t i = 0; i < count; ++i)
        {
            dst[i] = p[i];
        }
        break;
    }
    case nvinfer1::DataType::kHALF:
    {
        auto const* p = static_cast<__half const*>(src);
        for (std::size_t i = 0; i < count; ++i)
        {
            dst[i] = __half2float(p[i]);
        }
        break;
    }
    case nvinfer1::DataType::kBF16:
    {
        auto const* p = static_cast<__nv_bfloat16 const*>(src);
        for (std::size_t i = 0; i < count; ++i)
        {
            dst[i] = __bfloat162float(p[i]);
        }
        break;
    }
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
    {
        auto const* p = static_cast<uint8_t const*>(src);
        for (std::size_t i = 0; i < count; ++i)
        {
            dst[i] = static_cast<float>(p[i]);
        }
        break;
    }
    case nvinfer1::DataType::kINT32:
    {
        auto const* p = static_cast<int32_t const*>(src);
        for (std::size_t i = 0; i < count; ++i)
        {
            dst[i] = static_cast<float>(p[i]);
        }
        break;
    }
    case nvinfer1::DataType::kINT64:
    {
        auto const* p = static_cast<int64_t const*>(src);
        for (std::size_t i = 0; i < count; ++i)
        {
            dst[i] = static_cast<float>(p[i]);
        }
        break;
    }
    default: EDGE_CHECK(false, "convertToFloatHost: unsupported dtype " << dtypeName(srcType));
    }
}

void convertFromFloatHost(
    float const* src, void* dst, nvinfer1::DataType dstType, std::size_t count)
{
    switch (dstType)
    {
    case nvinfer1::DataType::kFLOAT:
    {
        auto* p = static_cast<float*>(dst);
        for (std::size_t i = 0; i < count; ++i)
        {
            p[i] = src[i];
        }
        break;
    }
    case nvinfer1::DataType::kHALF:
    {
        auto* p = static_cast<__half*>(dst);
        for (std::size_t i = 0; i < count; ++i)
        {
            p[i] = __float2half(src[i]);
        }
        break;
    }
    case nvinfer1::DataType::kBF16:
    {
        auto* p = static_cast<__nv_bfloat16*>(dst);
        for (std::size_t i = 0; i < count; ++i)
        {
            p[i] = __float2bfloat16(src[i]);
        }
        break;
    }
    default:
        EDGE_CHECK(false, "convertFromFloatHost: unsupported dtype " << dtypeName(dstType));
    }
}

std::string shapeToString(std::vector<int64_t> const& shape)
{
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
        {
            oss << ",";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

} // namespace pi05
