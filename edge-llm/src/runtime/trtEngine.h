// SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common/tensor.h"

#include <NvInfer.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace pi05
{

//! 单个 binding 的元信息。
struct BindingInfo
{
    std::string name;
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
    bool isInput = false;
    nvinfer1::Dims dims{}; // engine 声明 shape（动态维为 -1）
};

//! TRT engine 薄封装（参考 TensorRT-Edge-LLM cpp/runtime/exec/engineExecutor，
//! Phase 1 简化：默认 context memory、无 CUDA Graph）。
//!
//! 用法：
//!   TrtEngine e(path);
//!   e.setInput("x", devPtr, {1, 3, 224, 224});
//!   e.allocateOutputs();           // 依据 context 推断输出 shape 并分配
//!   e.enqueue(stream);
//!   Tensor& y = e.output("y");
class TrtEngine
{
public:
    explicit TrtEngine(std::string const& enginePath);
    ~TrtEngine() = default;

    TrtEngine(TrtEngine const&) = delete;
    TrtEngine& operator=(TrtEngine const&) = delete;

    std::vector<BindingInfo> const& bindings() const
    {
        return mBindings;
    }

    bool hasBinding(std::string const& name) const;
    BindingInfo const& binding(std::string const& name) const;

    std::vector<std::string> inputNames() const;
    std::vector<std::string> outputNames() const;

    //! 绑定外部 device 内存为输入并设置 shape。
    void setInput(std::string const& name, void* devicePtr, std::vector<int64_t> const& shape);

    //! 所有输入 setInput 后调用：推断输出 shape，分配（或复用）输出 device tensor 并绑定。
    void allocateOutputs();

    //! 推断后的输出 shape（须在 allocateOutputs 之后）。
    Tensor& output(std::string const& name);

    void enqueue(cudaStream_t stream);

    std::string const& path() const
    {
        return mPath;
    }

private:
    std::string mPath;
    std::vector<char> mBlob;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
    std::vector<BindingInfo> mBindings;
    std::map<std::string, Tensor> mOutputs;
};

} // namespace pi05
