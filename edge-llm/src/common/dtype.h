// SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <NvInfer.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pi05
{

//! dtype 元信息（覆盖 pi05 管线用到的类型即可）。
std::size_t dtypeSize(nvinfer1::DataType dtype);
std::string dtypeName(nvinfer1::DataType dtype);

//! numpy descr（如 "<f4"）↔ TRT dtype。bf16 没有标准 npy 表示，npy 侧用 fp32 承载。
nvinfer1::DataType dtypeFromNpyDescr(std::string const& descr);
std::string npyDescrFromDtype(nvinfer1::DataType dtype);

//! host 侧任意 dtype → fp32（用于对拍比较与跨引擎 dtype 适配）。
void convertToFloatHost(
    void const* src, nvinfer1::DataType srcType, float* dst, std::size_t count);

//! host 侧 fp32 → 任意 dtype。
void convertFromFloatHost(
    float const* src, void* dst, nvinfer1::DataType dstType, std::size_t count);

inline int64_t volume(std::vector<int64_t> const& shape)
{
    int64_t v = 1;
    for (int64_t const d : shape)
    {
        v *= d;
    }
    return v;
}

std::string shapeToString(std::vector<int64_t> const& shape);

} // namespace pi05
