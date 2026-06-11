// SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
#include "pi05/assets.h"

#include "common/check.h"
#include "common/npy.h"

#include <fstream>
#include <map>

namespace pi05
{

namespace
{

std::vector<float> loadFloatNpy(std::string const& path)
{
    auto const arr = loadNpy(path);
    return arr.toFloat();
}

NormStats loadNormStats(std::string const& dir, std::string const& key)
{
    NormStats s;
    s.mean = loadFloatNpy(dir + "/" + key + "_mean.npy");
    s.std = loadFloatNpy(dir + "/" + key + "_std.npy");
    s.q01 = loadFloatNpy(dir + "/" + key + "_q01.npy");
    s.q99 = loadFloatNpy(dir + "/" + key + "_q99.npy");
    EDGE_CHECK(s.q01.size() == s.q99.size(), "q01/q99 size mismatch for " << key);
    return s;
}

} // namespace

Pi05Assets Pi05Assets::load(std::string const& assetsDir)
{
    Pi05Assets assets;

    std::ifstream meta(assetsDir + "/meta.txt");
    EDGE_CHECK(meta.good(), "cannot open " << assetsDir << "/meta.txt");
    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(meta, line))
    {
        auto const eq = line.find('=');
        if (eq == std::string::npos)
        {
            continue;
        }
        kv[line.substr(0, eq)] = line.substr(eq + 1);
    }
    auto getInt = [&kv](std::string const& key, int fallback)
    {
        auto const it = kv.find(key);
        return it == kv.end() ? fallback : std::stoi(it->second);
    };
    assets.maxTokenLen = getInt("max_token_len", assets.maxTokenLen);
    assets.actionHorizon = getInt("action_horizon", assets.actionHorizon);
    assets.actionDim = getInt("action_dim", assets.actionDim);
    assets.outputActionDim = getInt("output_action_dim", assets.outputActionDim);
    assets.numViews = getInt("num_views", assets.numViews);
    assets.useQuantileNorm = getInt("use_quantile_norm", 1) != 0;
    assets.discreteStateInput = getInt("discrete_state_input", 0) != 0;
    std::string const tokModel
        = kv.count("tokenizer_model") ? kv["tokenizer_model"] : "paligemma_tokenizer.model";
    assets.tokenizerModelPath = assetsDir + "/" + tokModel;

    assets.state = loadNormStats(assetsDir, "state");
    assets.actions = loadNormStats(assetsDir, "actions");
    return assets;
}

std::vector<float> normalizeQuantile(std::vector<float> const& x, NormStats const& stats)
{
    // 与 Normalize._normalize_quantile 一致：stats 按 x 的末维截断。
    std::size_t const dim = std::min(x.size(), stats.q01.size());
    std::vector<float> out = x;
    for (std::size_t i = 0; i < dim; ++i)
    {
        out[i] = (x[i] - stats.q01[i]) / (stats.q99[i] - stats.q01[i] + 1e-6F) * 2.0F - 1.0F;
    }
    return out;
}

std::vector<float> normalizeZScore(std::vector<float> const& x, NormStats const& stats)
{
    std::size_t const dim = std::min(x.size(), stats.mean.size());
    std::vector<float> out = x;
    for (std::size_t i = 0; i < dim; ++i)
    {
        out[i] = (x[i] - stats.mean[i]) / (stats.std[i] + 1e-6F);
    }
    return out;
}

void unnormalizeQuantileInplace(
    float* x, std::size_t lastDim, std::size_t rows, NormStats const& stats)
{
    std::size_t const dim = std::min(lastDim, stats.q01.size());
    for (std::size_t r = 0; r < rows; ++r)
    {
        float* row = x + r * lastDim;
        for (std::size_t i = 0; i < dim; ++i)
        {
            row[i] = (row[i] + 1.0F) / 2.0F * (stats.q99[i] - stats.q01[i] + 1e-6F)
                + stats.q01[i];
        }
        // 超出 stats 维度的部分原样保留（pad 维）。
    }
}

void unnormalizeZScoreInplace(
    float* x, std::size_t lastDim, std::size_t rows, NormStats const& stats)
{
    for (std::size_t r = 0; r < rows; ++r)
    {
        float* row = x + r * lastDim;
        for (std::size_t i = 0; i < lastDim; ++i)
        {
            float const mean = i < stats.mean.size() ? stats.mean[i] : 0.0F;
            float const std = i < stats.std.size() ? stats.std[i] : 1.0F;
            row[i] = row[i] * (std + 1e-6F) + mean;
        }
    }
}

} // namespace pi05
