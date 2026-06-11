// SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
#include "runtime/trtEngine.h"

#include "common/check.h"
#include "common/logger.h"

#include <fstream>

namespace pi05
{

namespace
{

std::vector<int64_t> dimsToShape(nvinfer1::Dims const& dims)
{
    std::vector<int64_t> shape(static_cast<std::size_t>(dims.nbDims));
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        shape[static_cast<std::size_t>(i)] = dims.d[i];
    }
    return shape;
}

} // namespace

TrtEngine::TrtEngine(std::string const& enginePath)
    : mPath(enginePath)
{
    std::ifstream f(enginePath, std::ios::binary | std::ios::ate);
    EDGE_CHECK(f.good(), "cannot open engine: " << enginePath);
    auto const size = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    mBlob.resize(size);
    f.read(mBlob.data(), static_cast<std::streamsize>(size));

    mRuntime.reset(nvinfer1::createInferRuntime(TrtLogger::instance()));
    EDGE_CHECK(mRuntime != nullptr, "createInferRuntime failed");
    mEngine.reset(mRuntime->deserializeCudaEngine(mBlob.data(), mBlob.size()));
    EDGE_CHECK(mEngine != nullptr, "deserializeCudaEngine failed: " << enginePath);
    mContext.reset(mEngine->createExecutionContext());
    EDGE_CHECK(mContext != nullptr, "createExecutionContext failed: " << enginePath);

    int32_t const nbIO = mEngine->getNbIOTensors();
    mBindings.reserve(static_cast<std::size_t>(nbIO));
    for (int32_t i = 0; i < nbIO; ++i)
    {
        char const* name = mEngine->getIOTensorName(i);
        BindingInfo info;
        info.name = name;
        info.dtype = mEngine->getTensorDataType(name);
        info.isInput = mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        info.dims = mEngine->getTensorShape(name);
        mBindings.push_back(info);
    }
}

bool TrtEngine::hasBinding(std::string const& name) const
{
    for (auto const& b : mBindings)
    {
        if (b.name == name)
        {
            return true;
        }
    }
    return false;
}

BindingInfo const& TrtEngine::binding(std::string const& name) const
{
    for (auto const& b : mBindings)
    {
        if (b.name == name)
        {
            return b;
        }
    }
    EDGE_CHECK(false, "binding not found: " << name << " in " << mPath);
    return mBindings.front();
}

std::vector<std::string> TrtEngine::inputNames() const
{
    std::vector<std::string> names;
    for (auto const& b : mBindings)
    {
        if (b.isInput)
        {
            names.push_back(b.name);
        }
    }
    return names;
}

std::vector<std::string> TrtEngine::outputNames() const
{
    std::vector<std::string> names;
    for (auto const& b : mBindings)
    {
        if (!b.isInput)
        {
            names.push_back(b.name);
        }
    }
    return names;
}

void TrtEngine::setInput(
    std::string const& name, void* devicePtr, std::vector<int64_t> const& shape)
{
    auto const& info = binding(name);
    EDGE_CHECK(info.isInput, name << " is not an input of " << mPath);

    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int32_t>(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        dims.d[i] = shape[i];
    }
    EDGE_CHECK(mContext->setInputShape(name.c_str(), dims),
        "setInputShape failed for " << name << " shape=" << shapeToString(shape) << " engine="
                                    << mPath);
    EDGE_CHECK(mContext->setTensorAddress(name.c_str(), devicePtr),
        "setTensorAddress failed for " << name);
}

void TrtEngine::allocateOutputs()
{
    EDGE_CHECK(mContext->allInputDimensionsSpecified(),
        "not all input dims specified for " << mPath);
    for (auto const& b : mBindings)
    {
        if (b.isInput)
        {
            continue;
        }
        auto const dims = mContext->getTensorShape(b.name.c_str());
        auto shape = dimsToShape(dims);
        for (int64_t const d : shape)
        {
            EDGE_CHECK(d >= 0, "unresolved output dim for " << b.name << " in " << mPath);
        }
        auto it = mOutputs.find(b.name);
        std::size_t const needBytes
            = static_cast<std::size_t>(volume(shape)) * dtypeSize(b.dtype);
        if (it == mOutputs.end() || it->second.numBytes() < needBytes)
        {
            mOutputs[b.name] = Tensor(shape, b.dtype, DeviceType::kDEVICE, b.name);
        }
        else
        {
            it->second.reshape(shape);
        }
        EDGE_CHECK(mContext->setTensorAddress(b.name.c_str(), mOutputs[b.name].data()),
            "setTensorAddress failed for output " << b.name);
    }
}

Tensor& TrtEngine::output(std::string const& name)
{
    auto it = mOutputs.find(name);
    EDGE_CHECK(it != mOutputs.end(), "output not allocated: " << name << " in " << mPath);
    return it->second;
}

void TrtEngine::enqueue(cudaStream_t stream)
{
    EDGE_CHECK(mContext->enqueueV3(stream), "enqueueV3 failed for " << mPath);
}

} // namespace pi05
