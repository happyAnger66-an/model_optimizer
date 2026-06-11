// SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
#include "pi05/pi05Runtime.h"

#include "common/check.h"
#include "common/logger.h"

#include <algorithm>
#include <cstring>

namespace pi05
{

namespace
{

std::string joinPath(std::string const& dir, std::string const& name)
{
    if (dir.empty() || dir.back() == '/')
    {
        return dir + name;
    }
    return dir + "/" + name;
}

//! 与 binding 声明 dims（动态维 -1）做兼容性校验后返回实际 shape。
void checkShapeCompatible(BindingInfo const& info, std::vector<int64_t> const& shape)
{
    EDGE_CHECK(static_cast<std::size_t>(info.dims.nbDims) == shape.size(),
        "rank mismatch for " << info.name << ": engine=" << info.dims.nbDims
                             << " npy=" << shape.size());
    for (int32_t i = 0; i < info.dims.nbDims; ++i)
    {
        if (info.dims.d[i] >= 0)
        {
            EDGE_CHECK(info.dims.d[i] == shape[static_cast<std::size_t>(i)],
                "dim " << i << " mismatch for " << info.name << ": engine=" << info.dims.d[i]
                       << " npy=" << shape[static_cast<std::size_t>(i)]);
        }
    }
}

bool isFloatFamily(nvinfer1::DataType t)
{
    return t == nvinfer1::DataType::kFLOAT || t == nvinfer1::DataType::kHALF
        || t == nvinfer1::DataType::kBF16;
}

} // namespace

Pi05Runtime::Pi05Runtime(Pi05RuntimeConfig config)
    : mConfig(std::move(config))
{
    CUDA_CHECK(cudaStreamCreate(&mStream));
    mEmbedPrefix = std::make_unique<TrtEngine>(
        joinPath(mConfig.engineDir, mConfig.embedPrefixEngine));
    mLlm = std::make_unique<TrtEngine>(joinPath(mConfig.engineDir, mConfig.llmEngine));
    mDenoise = std::make_unique<TrtEngine>(joinPath(mConfig.engineDir, mConfig.denoiseEngine));

    LOG_INFO("engines loaded: embed_prefix=%s llm=%s denoise=%s (adarms_mod=%d, views=%d)",
        mConfig.embedPrefixEngine.c_str(), mConfig.llmEngine.c_str(),
        mConfig.denoiseEngine.c_str(), static_cast<int>(denoiseUsesAdarmsMod()),
        numImageViews());
}

Pi05Runtime::~Pi05Runtime()
{
    if (mStream != nullptr)
    {
        cudaStreamDestroy(mStream);
    }
}

bool Pi05Runtime::denoiseUsesAdarmsMod() const
{
    return mDenoise->hasBinding("adarms_mod");
}

int Pi05Runtime::numImageViews() const
{
    int views = 0;
    while (mEmbedPrefix->hasBinding("image_" + std::to_string(views)))
    {
        ++views;
    }
    return views;
}

Tensor Pi05Runtime::uploadForBinding(
    TrtEngine& engine, std::string const& name, NpyArray const& array)
{
    auto const& info = engine.binding(name);
    checkShapeCompatible(info, array.shape);

    Tensor dev(array.shape, info.dtype, DeviceType::kDEVICE, name);
    if (info.dtype == array.dtype)
    {
        dev.copyFromHost(array.data.data(), array.data.size());
    }
    else if (isFloatFamily(info.dtype))
    {
        // 任意 npy dtype → fp32 → binding dtype。
        auto const f32 = array.toFloat();
        std::vector<uint8_t> converted(dev.numBytes());
        convertFromFloatHost(f32.data(), converted.data(), info.dtype, f32.size());
        dev.copyFromHost(converted.data(), converted.size());
    }
    else if (info.dtype == nvinfer1::DataType::kINT32
        && array.dtype == nvinfer1::DataType::kINT64)
    {
        auto const* src = reinterpret_cast<int64_t const*>(array.data.data());
        std::vector<int32_t> narrowed(static_cast<std::size_t>(array.numel()));
        for (std::size_t i = 0; i < narrowed.size(); ++i)
        {
            narrowed[i] = static_cast<int32_t>(src[i]);
        }
        dev.copyFromHost(narrowed.data(), narrowed.size() * sizeof(int32_t));
    }
    else
    {
        EDGE_CHECK(false,
            "no conversion from npy dtype " << dtypeName(array.dtype) << " to binding dtype "
                                            << dtypeName(info.dtype) << " for " << name);
    }
    engine.setInput(name, dev.data(), array.shape);
    return dev;
}

void Pi05Runtime::recordDeviceTensor(std::string const& key, Tensor const& tensor)
{
    StageRecord rec;
    rec.shape = tensor.shape();
    rec.data = tensor.toFloatVector();
    mRecords[key] = std::move(rec);
}

void Pi05Runtime::run(Pi05Inputs const& inputs)
{
    mRecords.clear();
    int const views = numImageViews();
    EDGE_CHECK(static_cast<int>(inputs.images.size()) == views,
        "engine expects " << views << " image views, got " << inputs.images.size());
    EDGE_CHECK(inputs.images.size() == inputs.imageMasks.size(),
        "images / imageMasks count mismatch");

    // ---- Stage 1: embed_prefix ----------------------------------------------------------
    std::vector<Tensor> embedInputs;
    for (int i = 0; i < views; ++i)
    {
        embedInputs.push_back(uploadForBinding(
            *mEmbedPrefix, "image_" + std::to_string(i), inputs.images[static_cast<std::size_t>(i)]));
        embedInputs.push_back(uploadForBinding(*mEmbedPrefix,
            "image_mask_" + std::to_string(i), inputs.imageMasks[static_cast<std::size_t>(i)]));
    }
    embedInputs.push_back(uploadForBinding(*mEmbedPrefix, "lang_tokens", inputs.langTokens));
    embedInputs.push_back(uploadForBinding(*mEmbedPrefix, "lang_masks", inputs.langMasks));

    mEmbedPrefix->allocateOutputs();
    mEmbedPrefix->enqueue(mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    Tensor& prefixEmbs = mEmbedPrefix->output("prefix_embs");
    Tensor& prefixPadMasks = mEmbedPrefix->output("prefix_pad_masks");
    Tensor& prefixAttMasks = mEmbedPrefix->output("prefix_att_masks");
    recordDeviceTensor("prefix_embs", prefixEmbs);
    recordDeviceTensor("prefix_pad_masks", prefixPadMasks);

    auto const& embShape = prefixEmbs.shape(); // [B, S, hidden]
    int64_t const batch = embShape[0];
    int64_t const seqLen = embShape[1];
    EDGE_CHECK(batch == 1, "Phase 1 only supports batch=1, got " << batch);
    LOG_INFO("embed_prefix done: prefix_embs=%s", shapeToString(embShape).c_str());

    // ---- Host: 构造 4D 加性 attention mask 与 position_ids -------------------------------
    // 等价 make_att_2d_masks + _prepare_attention_masks_4d + neg-cap clamp
    // （pi0_pytorch.py L52-81/L157-160, pi05_trt_engine_setup.py L176-187）。
    std::size_t const s = static_cast<std::size_t>(seqLen);
    std::vector<uint8_t> padHost(s);
    std::vector<uint8_t> attHost(s);
    prefixPadMasks.copyToHost(padHost.data(), padHost.size());
    prefixAttMasks.copyToHost(attHost.data(), attHost.size());

    std::vector<int64_t> attCumsum(s);
    {
        int64_t acc = 0;
        for (std::size_t j = 0; j < s; ++j)
        {
            acc += attHost[j] != 0 ? 1 : 0;
            attCumsum[j] = acc;
        }
    }
    std::vector<float> mask4d(s * s);
    for (std::size_t i = 0; i < s; ++i)
    {
        bool const padI = padHost[i] != 0;
        for (std::size_t j = 0; j < s; ++j)
        {
            bool const visible
                = (attCumsum[j] <= attCumsum[i]) && padI && (padHost[j] != 0);
            mask4d[i * s + j] = visible ? 0.0F : mConfig.attnMaskFill;
        }
    }
    std::vector<int64_t> positionIds(s);
    {
        int64_t acc = 0;
        for (std::size_t j = 0; j < s; ++j)
        {
            acc += padHost[j] != 0 ? 1 : 0;
            positionIds[j] = acc - 1;
        }
    }

    // ---- Stage 2: llm（prefix KV） -------------------------------------------------------
    NpyArray maskArr;
    maskArr.shape = {1, 1, seqLen, seqLen};
    maskArr.dtype = nvinfer1::DataType::kFLOAT;
    maskArr.data.resize(mask4d.size() * sizeof(float));
    std::memcpy(maskArr.data.data(), mask4d.data(), maskArr.data.size());

    NpyArray posArr;
    posArr.shape = {1, seqLen};
    posArr.dtype = nvinfer1::DataType::kINT64;
    posArr.data.resize(positionIds.size() * sizeof(int64_t));
    std::memcpy(posArr.data.data(), positionIds.data(), posArr.data.size());

    Tensor llmMask = uploadForBinding(*mLlm, "attention_mask", maskArr);
    Tensor llmPos = uploadForBinding(*mLlm, "position_ids", posArr);

    // inputs_embeds 直接复用 embed_prefix 输出 device buffer（要求 dtype 一致）。
    auto const& llmEmbedsInfo = mLlm->binding("inputs_embeds");
    Tensor llmEmbedsConverted;
    if (llmEmbedsInfo.dtype == prefixEmbs.dtype())
    {
        mLlm->setInput("inputs_embeds", prefixEmbs.data(), prefixEmbs.shape());
    }
    else
    {
        LOG_WARN("inputs_embeds dtype mismatch (%s -> %s), converting via host",
            dtypeName(prefixEmbs.dtype()).c_str(), dtypeName(llmEmbedsInfo.dtype).c_str());
        auto const f32 = prefixEmbs.toFloatVector();
        std::vector<uint8_t> converted(
            f32.size() * dtypeSize(llmEmbedsInfo.dtype));
        convertFromFloatHost(f32.data(), converted.data(), llmEmbedsInfo.dtype, f32.size());
        llmEmbedsConverted
            = Tensor(prefixEmbs.shape(), llmEmbedsInfo.dtype, DeviceType::kDEVICE, "inputs_embeds");
        llmEmbedsConverted.copyFromHost(converted.data(), converted.size());
        mLlm->setInput("inputs_embeds", llmEmbedsConverted.data(), llmEmbedsConverted.shape());
    }

    mLlm->allocateOutputs();
    mLlm->enqueue(mStream);
    CUDA_CHECK(cudaStreamSynchronize(mStream));

    EDGE_CHECK(mLlm->hasBinding("past_keys") && mLlm->hasBinding("past_values"),
        "Phase 1 requires llm engine with stacked past_keys/past_values outputs "
        "(present_key_values.{i} variant not supported yet)");
    Tensor& pastKeys = mLlm->output("past_keys");
    Tensor& pastValues = mLlm->output("past_values");
    recordDeviceTensor("past_keys", pastKeys);
    recordDeviceTensor("past_values", pastValues);
    LOG_INFO("llm done: past_keys=%s", shapeToString(pastKeys.shape()).c_str());

    // ---- Stage 3: denoise ×numSteps（Euler: x_t += dt * v_t, dt = -1/numSteps） ----------
    bool const useAdarms = denoiseUsesAdarmsMod();
    if (useAdarms)
    {
        EDGE_CHECK(static_cast<int>(inputs.adarmsModSteps.size()) == mConfig.numSteps,
            "denoise engine needs adarms_mod; expected " << mConfig.numSteps
                << " adarms_mod_step{k}.npy files, got " << inputs.adarmsModSteps.size());
    }

    auto const& xtInfo = mDenoise->binding("x_t");
    EDGE_CHECK(xtInfo.dtype == nvinfer1::DataType::kFLOAT, "x_t binding must be fp32");
    EDGE_CHECK(inputs.noise.dtype == nvinfer1::DataType::kFLOAT, "noise npy must be fp32");

    std::size_t const actionNumel = static_cast<std::size_t>(inputs.noise.numel());
    std::vector<float> xtHost(actionNumel);
    std::memcpy(xtHost.data(), inputs.noise.data.data(), actionNumel * sizeof(float));

    Tensor xtDev(inputs.noise.shape, nvinfer1::DataType::kFLOAT, DeviceType::kDEVICE, "x_t");
    std::vector<float> vtHost(actionNumel);

    // prefix_pad_masks / past_keys / past_values 直接零拷贝绑定上游输出。
    auto bindUpstream = [this](std::string const& name, Tensor& upstream)
    {
        auto const& info = mDenoise->binding(name);
        EDGE_CHECK(info.dtype == upstream.dtype(),
            "denoise input " << name << " dtype " << dtypeName(info.dtype)
                             << " != upstream " << dtypeName(upstream.dtype()));
        mDenoise->setInput(name, upstream.data(), upstream.shape());
    };

    float const dt = -1.0F / static_cast<float>(mConfig.numSteps);
    Tensor timestepDev;
    Tensor adarmsDev;
    if (!useAdarms)
    {
        timestepDev = Tensor({batch}, nvinfer1::DataType::kFLOAT, DeviceType::kDEVICE, "timestep");
    }

    for (int step = 0; step < mConfig.numSteps; ++step)
    {
        float const t = 1.0F + dt * static_cast<float>(step);

        bindUpstream("prefix_pad_masks", prefixPadMasks);
        bindUpstream("past_keys", pastKeys);
        bindUpstream("past_values", pastValues);

        xtDev.copyFromHost(xtHost.data(), actionNumel * sizeof(float));
        mDenoise->setInput("x_t", xtDev.data(), xtDev.shape());

        if (useAdarms)
        {
            adarmsDev = uploadForBinding(
                *mDenoise, "adarms_mod", inputs.adarmsModSteps[static_cast<std::size_t>(step)]);
        }
        else
        {
            std::vector<float> tHost(static_cast<std::size_t>(batch), t);
            timestepDev.copyFromHost(tHost.data(), tHost.size() * sizeof(float));
            mDenoise->setInput("timestep", timestepDev.data(), timestepDev.shape());
        }

        mDenoise->allocateOutputs();
        mDenoise->enqueue(mStream);
        CUDA_CHECK(cudaStreamSynchronize(mStream));

        Tensor& vt = mDenoise->output("v_t");
        EDGE_CHECK(static_cast<std::size_t>(vt.numel()) == actionNumel,
            "v_t numel mismatch: " << vt.numel() << " vs " << actionNumel);
        auto const vtF32 = vt.toFloatVector();
        std::memcpy(vtHost.data(), vtF32.data(), actionNumel * sizeof(float));

        StageRecord rec;
        rec.shape = vt.shape();
        rec.data = vtF32;
        mRecords["v_t_step" + std::to_string(step)] = std::move(rec);

        for (std::size_t i = 0; i < actionNumel; ++i)
        {
            xtHost[i] += dt * vtHost[i];
        }
    }

    StageRecord actions;
    actions.shape = inputs.noise.shape;
    actions.data = xtHost;
    mRecords["actions"] = std::move(actions);
    LOG_INFO("denoise done: %d steps, actions=%s", mConfig.numSteps,
        shapeToString(inputs.noise.shape).c_str());
}

} // namespace pi05
