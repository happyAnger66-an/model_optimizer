// SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
#include "pi05/preprocess.h"

#include "common/check.h"
#include "common/logger.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace pi05
{

namespace
{

//! bilinear resize（half-pixel centers，uint8 HWC），等价 letterbox 的 resize 部分。
std::vector<uint8_t> resizeBilinearU8(
    uint8_t const* src, int srcH, int srcW, int dstH, int dstW)
{
    std::vector<uint8_t> dst(static_cast<std::size_t>(dstH) * dstW * 3);
    float const scaleY = static_cast<float>(srcH) / static_cast<float>(dstH);
    float const scaleX = static_cast<float>(srcW) / static_cast<float>(dstW);
    for (int y = 0; y < dstH; ++y)
    {
        float const fy = (static_cast<float>(y) + 0.5F) * scaleY - 0.5F;
        int const y0 = std::max(0, static_cast<int>(std::floor(fy)));
        int const y1 = std::min(srcH - 1, y0 + 1);
        float const wy = std::min(1.0F, std::max(0.0F, fy - static_cast<float>(y0)));
        for (int x = 0; x < dstW; ++x)
        {
            float const fx = (static_cast<float>(x) + 0.5F) * scaleX - 0.5F;
            int const x0 = std::max(0, static_cast<int>(std::floor(fx)));
            int const x1 = std::min(srcW - 1, x0 + 1);
            float const wx = std::min(1.0F, std::max(0.0F, fx - static_cast<float>(x0)));
            for (int c = 0; c < 3; ++c)
            {
                float const v00 = src[(static_cast<std::size_t>(y0) * srcW + x0) * 3 + c];
                float const v01 = src[(static_cast<std::size_t>(y0) * srcW + x1) * 3 + c];
                float const v10 = src[(static_cast<std::size_t>(y1) * srcW + x0) * 3 + c];
                float const v11 = src[(static_cast<std::size_t>(y1) * srcW + x1) * 3 + c];
                float const v = v00 * (1 - wy) * (1 - wx) + v01 * (1 - wy) * wx
                    + v10 * wy * (1 - wx) + v11 * wy * wx;
                dst[(static_cast<std::size_t>(y) * dstW + x) * 3 + c]
                    = static_cast<uint8_t>(std::min(255.0F, std::max(0.0F, std::round(v))));
            }
        }
    }
    return dst;
}

//! resize_with_pad（uint8 HWC）：等比缩放 + 黑边居中 pad。
std::vector<uint8_t> resizeWithPadU8(
    uint8_t const* src, int srcH, int srcW, int dstH, int dstW)
{
    float const ratio = std::max(static_cast<float>(srcW) / static_cast<float>(dstW),
        static_cast<float>(srcH) / static_cast<float>(dstH));
    int const resizedH = static_cast<int>(static_cast<float>(srcH) / ratio);
    int const resizedW = static_cast<int>(static_cast<float>(srcW) / ratio);
    auto const resized = resizeBilinearU8(src, srcH, srcW, resizedH, resizedW);

    std::vector<uint8_t> padded(static_cast<std::size_t>(dstH) * dstW * 3, 0);
    int const padH0 = (dstH - resizedH) / 2;
    int const padW0 = (dstW - resizedW) / 2;
    for (int y = 0; y < resizedH; ++y)
    {
        std::memcpy(&padded[((static_cast<std::size_t>(y) + padH0) * dstW + padW0) * 3],
            &resized[static_cast<std::size_t>(y) * resizedW * 3],
            static_cast<std::size_t>(resizedW) * 3);
    }
    return padded;
}

} // namespace

NpyArray imageToModelInput(NpyArray const& image, int targetH, int targetW)
{
    EDGE_CHECK(image.dtype == nvinfer1::DataType::kUINT8,
        "image must be uint8, got " << dtypeName(image.dtype));
    EDGE_CHECK(image.shape.size() == 3 && image.shape[2] == 3,
        "image must be HWC [H,W,3], got " << shapeToString(image.shape));

    int const srcH = static_cast<int>(image.shape[0]);
    int const srcW = static_cast<int>(image.shape[1]);
    uint8_t const* hwc = image.data.data();
    std::vector<uint8_t> resized;
    if (srcH != targetH || srcW != targetW)
    {
        LOG_INFO("resize_with_pad %dx%d -> %dx%d (bilinear, 与 jax antialias 有微小差异)",
            srcH, srcW, targetH, targetW);
        resized = resizeWithPadU8(hwc, srcH, srcW, targetH, targetW);
        hwc = resized.data();
    }

    // uint8 HWC → fp32 CHW，/255*2-1（Observation.from_dict torch 路径）。
    NpyArray out;
    out.shape = {1, 3, targetH, targetW};
    out.dtype = nvinfer1::DataType::kFLOAT;
    out.data.resize(static_cast<std::size_t>(3) * targetH * targetW * sizeof(float));
    auto* chw = reinterpret_cast<float*>(out.data.data());
    std::size_t const plane = static_cast<std::size_t>(targetH) * targetW;
    for (int y = 0; y < targetH; ++y)
    {
        for (int x = 0; x < targetW; ++x)
        {
            std::size_t const p = static_cast<std::size_t>(y) * targetW + x;
            for (int c = 0; c < 3; ++c)
            {
                chw[static_cast<std::size_t>(c) * plane + p]
                    = static_cast<float>(hwc[p * 3 + static_cast<std::size_t>(c)]) / 255.0F
                        * 2.0F
                    - 1.0F;
            }
        }
    }
    return out;
}

NpyArray zeroImageInput(int targetH, int targetW)
{
    NpyArray out;
    out.shape = {1, 3, targetH, targetW};
    out.dtype = nvinfer1::DataType::kFLOAT;
    std::size_t const count = static_cast<std::size_t>(3) * targetH * targetW;
    out.data.resize(count * sizeof(float));
    auto* p = reinterpret_cast<float*>(out.data.data());
    std::fill(p, p + count, -1.0F); // uint8 0 → -1.0
    return out;
}

Pi05Inputs buildModelInputs(
    RawObservation const& obs, Pi05Assets const& assets, PaligemmaTokenizer const& tokenizer)
{
    Pi05Inputs inputs;

    // ---- LiberoInputs：base / left_wrist / right_wrist(zeros)，mask=[T,T,F] ----
    inputs.images.push_back(imageToModelInput(obs.baseImage));
    inputs.images.push_back(imageToModelInput(obs.wristImage));
    inputs.images.push_back(zeroImageInput());
    bool const maskValues[3] = {true, true, false};
    for (bool const m : maskValues)
    {
        NpyArray mask;
        mask.shape = {1};
        mask.dtype = nvinfer1::DataType::kBOOL;
        mask.data = {static_cast<uint8_t>(m ? 1 : 0)};
        inputs.imageMasks.push_back(std::move(mask));
    }

    // ---- Normalize(state)（pi05: quantile） ----
    std::vector<float> stateNorm = assets.useQuantileNorm
        ? normalizeQuantile(obs.state, assets.state)
        : normalizeZScore(obs.state, assets.state);

    // ---- TokenizePrompt ----
    TokenizedPrompt const tok = assets.discreteStateInput
        ? tokenizer.tokenizeWithState(obs.prompt, stateNorm)
        : tokenizer.tokenize(obs.prompt);

    inputs.langTokens.shape = {1, assets.maxTokenLen};
    inputs.langTokens.dtype = nvinfer1::DataType::kINT64;
    inputs.langTokens.data.resize(tok.tokens.size() * sizeof(int64_t));
    std::memcpy(inputs.langTokens.data.data(), tok.tokens.data(), inputs.langTokens.data.size());

    inputs.langMasks.shape = {1, assets.maxTokenLen};
    inputs.langMasks.dtype = nvinfer1::DataType::kBOOL;
    inputs.langMasks.data = tok.mask;

    return inputs;
}

std::vector<float> postprocessActions(std::vector<float> const& actions,
    std::vector<int64_t> const& shape, Pi05Assets const& assets)
{
    EDGE_CHECK(shape.size() == 3 && shape[0] == 1, "actions must be [1,H,D], got "
            << shapeToString(shape));
    auto const horizon = static_cast<std::size_t>(shape[1]);
    auto const dim = static_cast<std::size_t>(shape[2]);

    std::vector<float> work = actions;
    if (assets.useQuantileNorm)
    {
        unnormalizeQuantileInplace(work.data(), dim, horizon, assets.actions);
    }
    else
    {
        unnormalizeZScoreInplace(work.data(), dim, horizon, assets.actions);
    }

    // LiberoOutputs: actions[:, :outputActionDim]
    auto const outDim = static_cast<std::size_t>(assets.outputActionDim);
    std::vector<float> out(horizon * outDim);
    for (std::size_t h = 0; h < horizon; ++h)
    {
        std::memcpy(&out[h * outDim], &work[h * dim], outDim * sizeof(float));
    }
    return out;
}

} // namespace pi05
