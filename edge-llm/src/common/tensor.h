// SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common/dtype.h"

#include <NvInfer.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pi05
{

enum class DeviceType
{
    kHOST,
    kDEVICE,
};

//! RAII Tensor（参考 TensorRT-Edge-LLM cpp/common/tensor，按 Phase 1 需求精简）。
//!
//! - 自有内存：构造时按 shape×dtype 分配（device 用 cudaMalloc，host 用 pinned）。
//! - 禁止拷贝，允许 move。
class Tensor
{
public:
    Tensor() = default;
    Tensor(std::vector<int64_t> shape, nvinfer1::DataType dtype, DeviceType device,
        std::string name = "");
    ~Tensor();

    Tensor(Tensor const&) = delete;
    Tensor& operator=(Tensor const&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    void* data() const
    {
        return mData;
    }

    template <typename T>
    T* dataAs() const
    {
        return static_cast<T*>(mData);
    }

    std::vector<int64_t> const& shape() const
    {
        return mShape;
    }

    nvinfer1::DataType dtype() const
    {
        return mDtype;
    }

    DeviceType device() const
    {
        return mDevice;
    }

    std::string const& name() const
    {
        return mName;
    }

    int64_t numel() const
    {
        return volume(mShape);
    }

    std::size_t numBytes() const
    {
        return static_cast<std::size_t>(numel()) * dtypeSize(mDtype);
    }

    bool valid() const
    {
        return mData != nullptr;
    }

    nvinfer1::Dims trtDims() const;

    //! 仅修改 shape 视图，要求 numel 不超过已分配容量。
    void reshape(std::vector<int64_t> shape);

    // ---- 拷贝便捷接口（同步） ----
    void copyFromHost(void const* src, std::size_t bytes);
    void copyToHost(void* dst, std::size_t bytes) const;
    void copyFromDevice(void const* src, std::size_t bytes);

    //! device tensor → host fp32 vector（自动做 dtype 转换，用于对拍）。
    std::vector<float> toFloatVector() const;

private:
    void release();

    void* mData = nullptr;
    std::vector<int64_t> mShape;
    nvinfer1::DataType mDtype = nvinfer1::DataType::kFLOAT;
    DeviceType mDevice = DeviceType::kHOST;
    std::string mName;
    std::size_t mCapacity = 0;
};

} // namespace pi05
