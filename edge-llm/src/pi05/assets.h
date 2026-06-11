// SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <vector>

namespace pi05
{

//! 单个量的归一化统计（openpi NormStats）。
struct NormStats
{
    std::vector<float> mean;
    std::vector<float> std;
    std::vector<float> q01;
    std::vector<float> q99;
};

//! 部署资产（scripts/prepare_pi05_assets.py 的输出）。
struct Pi05Assets
{
    int maxTokenLen = 200;
    int actionHorizon = 10;
    int actionDim = 32;
    int outputActionDim = 7;
    int numViews = 3;
    bool useQuantileNorm = true;
    bool discreteStateInput = false;
    std::string tokenizerModelPath;

    NormStats state;
    NormStats actions;

    static Pi05Assets load(std::string const& assetsDir);
};

//! quantile 归一化：(x - q01) / (q99 - q01 + 1e-6) * 2 - 1（截断到 stats 维度）。
std::vector<float> normalizeQuantile(std::vector<float> const& x, NormStats const& stats);

//! z-score 归一化：(x - mean) / (std + 1e-6)。
std::vector<float> normalizeZScore(std::vector<float> const& x, NormStats const& stats);

//! quantile 逆归一化（等价 Unnormalize._unnormalize_quantile）：
//! 前 dim(q01) 维 (x+1)/2*(q99-q01+1e-6)+q01，其余维原样保留。
void unnormalizeQuantileInplace(float* x, std::size_t lastDim, std::size_t rows,
    NormStats const& stats);

//! z-score 逆归一化：x*(std+1e-6)+mean（mean/std 不足处 pad 0/1）。
void unnormalizeZScoreInplace(float* x, std::size_t lastDim, std::size_t rows,
    NormStats const& stats);

} // namespace pi05
