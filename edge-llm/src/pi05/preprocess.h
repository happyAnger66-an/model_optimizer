// SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common/npy.h"
#include "pi05/assets.h"
#include "pi05/pi05Runtime.h"
#include "pi05/tokenizer.h"

#include <string>
#include <vector>

namespace pi05
{

//! 原始观测（LiberoInputs 之前的格式）。
struct RawObservation
{
    NpyArray baseImage;  //!< uint8 HWC [H,W,3]
    NpyArray wristImage; //!< uint8 HWC [H,W,3]
    std::vector<float> state;
    std::string prompt;
};

//! uint8 HWC [H,W,3] → fp32 CHW [1,3,224,224]，范围 [-1,1]。
//! 非 224 输入先 resize_with_pad（bilinear，黑边 letterbox；与 jax 实现存在
//! antialias 数值差异，224 输入恒等、完全精确）。
NpyArray imageToModelInput(NpyArray const& image, int targetH = 224, int targetW = 224);

//! 全黑 padding 视图（uint8 全 0 → 变换后为 -1.0）。
NpyArray zeroImageInput(int targetH = 224, int targetW = 224);

//! 复刻 Python 前处理链，组装 Pi05Runtime 输入：
//! LiberoInputs(3 views, mask=[T,T,F]) → Normalize(state) → TokenizePrompt
//! → PadStatesAndActions → Observation.from_dict(图像 [-1,1] CHW)。
//!
//! noise 为空时由调用方负责填充 inputs.noise。
Pi05Inputs buildModelInputs(RawObservation const& obs, Pi05Assets const& assets,
    PaligemmaTokenizer const& tokenizer);

//! 后处理：actions [B,H,actionDim] fp32 → Unnormalize(quantile/zscore) →
//! LiberoOutputs 截取前 outputActionDim 维，返回 [H, outputActionDim]。
std::vector<float> postprocessActions(
    std::vector<float> const& actions, std::vector<int64_t> const& shape,
    Pi05Assets const& assets);

} // namespace pi05
