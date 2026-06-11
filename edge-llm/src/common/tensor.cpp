// SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
#include "common/tensor.h"

#include "common/check.h"

#include <utility>

namespace pi05
{

Tensor::Tensor(
    std::vector<int64_t> shape, nvinfer1::DataType dtype, DeviceType device, std::string name)
    : mShape(std::move(shape))
    , mDtype(dtype)
    , mDevice(device)
    , mName(std::move(name))
{
    mCapacity = numBytes();
    if (mCapacity == 0)
    {
        return;
    }
    if (mDevice == DeviceType::kDEVICE)
    {
        CUDA_CHECK(cudaMalloc(&mData, mCapacity));
    }
    else
    {
        CUDA_CHECK(cudaMallocHost(&mData, mCapacity));
    }
}

Tensor::~Tensor()
{
    release();
}

Tensor::Tensor(Tensor&& other) noexcept
{
    *this = std::move(other);
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if (this != &other)
    {
        release();
        mData = other.mData;
        mShape = std::move(other.mShape);
        mDtype = other.mDtype;
        mDevice = other.mDevice;
        mName = std::move(other.mName);
        mCapacity = other.mCapacity;
        other.mData = nullptr;
        other.mCapacity = 0;
    }
    return *this;
}

void Tensor::release()
{
    if (mData == nullptr)
    {
        return;
    }
    if (mDevice == DeviceType::kDEVICE)
    {
        cudaFree(mData);
    }
    else
    {
        cudaFreeHost(mData);
    }
    mData = nullptr;
    mCapacity = 0;
}

nvinfer1::Dims Tensor::trtDims() const
{
    nvinfer1::Dims dims{};
    EDGE_CHECK(mShape.size() <= static_cast<std::size_t>(nvinfer1::Dims::MAX_DIMS),
        "too many dims: " << mShape.size());
    dims.nbDims = static_cast<int32_t>(mShape.size());
    for (std::size_t i = 0; i < mShape.size(); ++i)
    {
        dims.d[i] = mShape[i];
    }
    return dims;
}

void Tensor::reshape(std::vector<int64_t> shape)
{
    std::size_t const newBytes = static_cast<std::size_t>(volume(shape)) * dtypeSize(mDtype);
    EDGE_CHECK(newBytes <= mCapacity,
        "reshape exceeds capacity: " << newBytes << " > " << mCapacity << " (" << mName << ")");
    mShape = std::move(shape);
}

void Tensor::copyFromHost(void const* src, std::size_t bytes)
{
    EDGE_CHECK(bytes <= mCapacity, "copyFromHost overflow on " << mName);
    if (mDevice == DeviceType::kDEVICE)
    {
        CUDA_CHECK(cudaMemcpy(mData, src, bytes, cudaMemcpyHostToDevice));
    }
    else
    {
        CUDA_CHECK(cudaMemcpy(mData, src, bytes, cudaMemcpyHostToHost));
    }
}

void Tensor::copyToHost(void* dst, std::size_t bytes) const
{
    EDGE_CHECK(bytes <= mCapacity, "copyToHost overflow on " << mName);
    if (mDevice == DeviceType::kDEVICE)
    {
        CUDA_CHECK(cudaMemcpy(dst, mData, bytes, cudaMemcpyDeviceToHost));
    }
    else
    {
        CUDA_CHECK(cudaMemcpy(dst, mData, bytes, cudaMemcpyHostToHost));
    }
}

void Tensor::copyFromDevice(void const* src, std::size_t bytes)
{
    EDGE_CHECK(bytes <= mCapacity, "copyFromDevice overflow on " << mName);
    EDGE_CHECK(mDevice == DeviceType::kDEVICE, "copyFromDevice target must be device tensor");
    CUDA_CHECK(cudaMemcpy(mData, src, bytes, cudaMemcpyDeviceToDevice));
}

std::vector<float> Tensor::toFloatVector() const
{
    std::size_t const count = static_cast<std::size_t>(numel());
    std::vector<uint8_t> raw(numBytes());
    copyToHost(raw.data(), raw.size());
    std::vector<float> out(count);
    convertToFloatHost(raw.data(), mDtype, out.data(), count);
    return out;
}

} // namespace pi05
