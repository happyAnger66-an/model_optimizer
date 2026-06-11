// SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
//! pi05_compare: 跑 C++ pi05 管线并与 Python golden npy 逐 stage 对拍。
//!
//! 用法：
//!   pi05_compare --engine-dir DIR --io-dir DIR [--num-steps 10] [--tol 1e-2]
//!                [--embed-prefix embed_prefix.engine] [--llm llm.engine]
//!                [--denoise denoise.engine] [--neg-cap -1e4] [--dump-out DIR]
//!
//! io-dir 文件约定（由 scripts/dump_pi05_io.py 生成）：
//!   输入:  image_{i}.npy image_mask_{i}.npy lang_tokens.npy lang_masks.npy noise.npy
//!          [adarms_mod_step{k}.npy]  (AdaRMS 预计算引擎)
//!   golden: prefix_embs.npy prefix_pad_masks.npy past_keys.npy past_values.npy
//!           v_t_step{k}.npy actions.npy  （均为可选，存在即比对）

#include "common/check.h"
#include "common/logger.h"
#include "common/npy.h"
#include "pi05/pi05Runtime.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace
{

using pi05::NpyArray;

struct Args
{
    std::string engineDir;
    std::string ioDir;
    std::string dumpOut;
    std::string embedPrefixEngine = "embed_prefix.engine";
    std::string llmEngine = "llm.engine";
    std::string denoiseEngine = "denoise.engine";
    int numSteps = 10;
    float negCap = -1e4F;
    double tol = 1e-2;
};

Args parseArgs(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i)
    {
        std::string const key = argv[i];
        auto next = [&]() -> std::string
        {
            EDGE_CHECK(i + 1 < argc, "missing value for " << key);
            return argv[++i];
        };
        if (key == "--engine-dir")
        {
            args.engineDir = next();
        }
        else if (key == "--io-dir")
        {
            args.ioDir = next();
        }
        else if (key == "--dump-out")
        {
            args.dumpOut = next();
        }
        else if (key == "--embed-prefix")
        {
            args.embedPrefixEngine = next();
        }
        else if (key == "--llm")
        {
            args.llmEngine = next();
        }
        else if (key == "--denoise")
        {
            args.denoiseEngine = next();
        }
        else if (key == "--num-steps")
        {
            args.numSteps = std::stoi(next());
        }
        else if (key == "--neg-cap")
        {
            args.negCap = std::stof(next());
        }
        else if (key == "--tol")
        {
            args.tol = std::stod(next());
        }
        else
        {
            EDGE_CHECK(false, "unknown argument: " << key);
        }
    }
    EDGE_CHECK(!args.engineDir.empty(), "--engine-dir is required");
    EDGE_CHECK(!args.ioDir.empty(), "--io-dir is required");
    return args;
}

std::string ioPath(Args const& args, std::string const& name)
{
    return args.ioDir + "/" + name + ".npy";
}

//! 比对结果；max_abs <= tol 视为 PASS。
bool compareRecord(std::string const& key, pi05::StageRecord const& got,
    NpyArray const& golden, double tol)
{
    auto const ref = golden.toFloat();
    if (ref.size() != got.data.size())
    {
        std::printf("  [FAIL] %-18s numel mismatch: cpp=%zu golden=%zu\n", key.c_str(),
            got.data.size(), ref.size());
        return false;
    }
    double maxAbs = 0.0;
    double sumAbs = 0.0;
    std::size_t maxIdx = 0;
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        double const diff = std::abs(static_cast<double>(got.data[i]) - ref[i]);
        sumAbs += diff;
        if (diff > maxAbs)
        {
            maxAbs = diff;
            maxIdx = i;
        }
    }
    double const meanAbs = sumAbs / static_cast<double>(ref.size());
    bool const pass = maxAbs <= tol;
    std::printf("  [%s] %-18s shape=%-24s max_abs=%.6g mean_abs=%.6g (idx=%zu cpp=%.6g ref=%.6g)\n",
        pass ? "PASS" : "FAIL", key.c_str(), pi05::shapeToString(got.shape).c_str(), maxAbs,
        meanAbs, maxIdx, static_cast<double>(got.data[maxIdx]), ref[maxIdx]);
    return pass;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        Args const args = parseArgs(argc, argv);

        pi05::Pi05RuntimeConfig config;
        config.engineDir = args.engineDir;
        config.embedPrefixEngine = args.embedPrefixEngine;
        config.llmEngine = args.llmEngine;
        config.denoiseEngine = args.denoiseEngine;
        config.numSteps = args.numSteps;
        config.attnMaskFill = args.negCap;

        pi05::Pi05Runtime runtime(config);

        // ---- 加载输入 ----
        pi05::Pi05Inputs inputs;
        int const views = runtime.numImageViews();
        for (int i = 0; i < views; ++i)
        {
            inputs.images.push_back(
                pi05::loadNpy(ioPath(args, "image_" + std::to_string(i))));
            inputs.imageMasks.push_back(
                pi05::loadNpy(ioPath(args, "image_mask_" + std::to_string(i))));
        }
        inputs.langTokens = pi05::loadNpy(ioPath(args, "lang_tokens"));
        inputs.langMasks = pi05::loadNpy(ioPath(args, "lang_masks"));
        inputs.noise = pi05::loadNpy(ioPath(args, "noise"));
        if (runtime.denoiseUsesAdarmsMod())
        {
            for (int k = 0; k < args.numSteps; ++k)
            {
                inputs.adarmsModSteps.push_back(
                    pi05::loadNpy(ioPath(args, "adarms_mod_step" + std::to_string(k))));
            }
        }

        // ---- 执行 ----
        runtime.run(inputs);

        // ---- golden 对拍 ----
        std::vector<std::string> keys = {
            "prefix_embs", "prefix_pad_masks", "past_keys", "past_values"};
        for (int k = 0; k < args.numSteps; ++k)
        {
            keys.push_back("v_t_step" + std::to_string(k));
        }
        keys.push_back("actions");

        std::printf("\n== compare against golden (tol=%g) ==\n", args.tol);
        int compared = 0;
        int failed = 0;
        for (auto const& key : keys)
        {
            auto const it = runtime.records().find(key);
            if (it == runtime.records().end())
            {
                continue;
            }
            std::string const path = ioPath(args, key);
            if (!pi05::fileExists(path))
            {
                std::printf("  [SKIP] %-18s (no golden %s)\n", key.c_str(), path.c_str());
                continue;
            }
            ++compared;
            if (!compareRecord(key, it->second, pi05::loadNpy(path), args.tol))
            {
                ++failed;
            }
        }

        // ---- 可选：dump C++ 侧结果为 npy ----
        if (!args.dumpOut.empty())
        {
            for (auto const& [key, rec] : runtime.records())
            {
                NpyArray arr;
                arr.shape = rec.shape;
                arr.dtype = nvinfer1::DataType::kFLOAT;
                arr.data.resize(rec.data.size() * sizeof(float));
                std::memcpy(arr.data.data(), rec.data.data(), arr.data.size());
                pi05::saveNpy(args.dumpOut + "/" + key + ".npy", arr);
            }
            std::printf("dumped %zu records to %s\n", runtime.records().size(),
                args.dumpOut.c_str());
        }

        std::printf("\n== summary: %d compared, %d failed ==\n", compared, failed);
        return failed == 0 ? 0 : 1;
    }
    catch (std::exception const& e)
    {
        std::fprintf(stderr, "[pi05_compare] error: %s\n", e.what());
        return 2;
    }
}
