// SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common/npy.h"
#include "common/tensor.h"
#include "runtime/trtEngine.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace pi05
{

struct Pi05RuntimeConfig
{
    std::string engineDir;
    std::string embedPrefixEngine = "embed_prefix.engine";
    std::string llmEngine = "llm.engine";
    std::string denoiseEngine = "denoise.engine";
    int numSteps = 10;
    //! 加性 attention mask 的无效位填充值；与 Python TRT 路径
    //! ``trt_attention_mask_neg_cap``（默认 -1e4）一致。
    float attnMaskFill = -1e4F;
};

//! 管线外部输入（host 数据，由 dump 脚本生成的 npy 加载）。
struct Pi05Inputs
{
    std::vector<NpyArray> images;     //!< image_{i}: fp32 [B,3,224,224]
    std::vector<NpyArray> imageMasks; //!< image_mask_{i}: bool
    NpyArray langTokens;              //!< int64 [B,T]
    NpyArray langMasks;               //!< bool [B,T]
    NpyArray noise;                   //!< fp32 [B,H,D]
    //! AdaRMS 预计算模式（denoise 引擎输入为 adarms_mod 而非 timestep）时，
    //! 每步 modulation：fp32 [num_norms,B,3*dim]，长度 == numSteps。
    std::vector<NpyArray> adarmsModSteps;
};

//! 中间/最终结果记录（fp32 host，用于对拍）。
struct StageRecord
{
    std::vector<int64_t> shape;
    std::vector<float> data;
};

//! pi05 推理管线：embed_prefix → (host mask/pos) → llm → denoise×N (Euler)。
//!
//! 等价于 Python 侧 ``PI0Pytorch.sample_actions`` 的 TRT 全挂载路径
//! （pi0_pytorch.py L381-424 + pi05_trt_engine_setup.py）。
class Pi05Runtime
{
public:
    explicit Pi05Runtime(Pi05RuntimeConfig config);
    ~Pi05Runtime();

    //! 执行全管线，中间结果存入 records()。
    void run(Pi05Inputs const& inputs);

    std::map<std::string, StageRecord> const& records() const
    {
        return mRecords;
    }

    //! denoise 引擎是否为 AdaRMS 预计算变体（输入 adarms_mod）。
    bool denoiseUsesAdarmsMod() const;

    int numImageViews() const;

private:
    //! 把 npy host 数据按 binding dtype 转换后上传至 device tensor。
    Tensor uploadForBinding(
        TrtEngine& engine, std::string const& name, NpyArray const& array);

    void recordDeviceTensor(std::string const& key, Tensor const& tensor);

    Pi05RuntimeConfig mConfig;
    std::unique_ptr<TrtEngine> mEmbedPrefix;
    std::unique_ptr<TrtEngine> mLlm;
    std::unique_ptr<TrtEngine> mDenoise;
    cudaStream_t mStream = nullptr;
    std::map<std::string, StageRecord> mRecords;
};

} // namespace pi05
