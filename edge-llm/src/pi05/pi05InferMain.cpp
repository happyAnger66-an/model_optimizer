// SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
//! pi05_infer: 端到端推理 CLI —— 原始观测（uint8 图像 + state + prompt）→ 7 维 actions。
//!
//! 复刻 Python ``Policy.infer`` 全链路：
//!   LiberoInputs → Normalize → TokenizePrompt → Observation
//!   → embed_prefix/llm/denoise (TRT) → Unnormalize → LiberoOutputs
//!
//! 用法：
//!   pi05_infer --engine-dir D --assets-dir A --obs-dir O
//!              [--noise noise.npy | --seed 0] [--num-steps 10]
//!              [--adarms-dir DIR] [--out actions.npy]
//!              [--golden actions_final.npy] [--tol 1e-3] [--prompt "..."]
//!
//! obs-dir 文件：base_image.npy(uint8 HWC) wrist_image.npy state.npy prompt.txt

#include "common/check.h"
#include "common/npy.h"
#include "pi05/assets.h"
#include "pi05/pi05Runtime.h"
#include "pi05/preprocess.h"
#include "pi05/tokenizer.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>

namespace
{

using pi05::NpyArray;

struct Args
{
    std::string engineDir;
    std::string assetsDir;
    std::string obsDir;
    std::string noisePath;
    std::string adarmsDir;
    std::string outPath;
    std::string goldenPath;
    std::string prompt;
    std::string embedPrefixEngine = "embed_prefix.engine";
    std::string llmEngine = "llm.engine";
    std::string denoiseEngine = "denoise.engine";
    int numSteps = 10;
    unsigned seed = 0;
    float negCap = -1e4F;
    double tol = 1e-3;
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
        else if (key == "--assets-dir")
        {
            args.assetsDir = next();
        }
        else if (key == "--obs-dir")
        {
            args.obsDir = next();
        }
        else if (key == "--noise")
        {
            args.noisePath = next();
        }
        else if (key == "--adarms-dir")
        {
            args.adarmsDir = next();
        }
        else if (key == "--out")
        {
            args.outPath = next();
        }
        else if (key == "--golden")
        {
            args.goldenPath = next();
        }
        else if (key == "--prompt")
        {
            args.prompt = next();
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
        else if (key == "--seed")
        {
            args.seed = static_cast<unsigned>(std::stoul(next()));
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
    EDGE_CHECK(!args.assetsDir.empty(), "--assets-dir is required");
    EDGE_CHECK(!args.obsDir.empty(), "--obs-dir is required");
    return args;
}

std::string readTextFile(std::string const& path)
{
    std::ifstream f(path);
    EDGE_CHECK(f.good(), "cannot open " << path);
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return content;
}

NpyArray makeNoise(pi05::Pi05Assets const& assets, Args const& args)
{
    if (!args.noisePath.empty())
    {
        auto noise = pi05::loadNpy(args.noisePath);
        if (noise.shape.size() == 2)
        {
            noise.shape = {1, noise.shape[0], noise.shape[1]};
        }
        return noise;
    }
    NpyArray noise;
    noise.shape = {1, assets.actionHorizon, assets.actionDim};
    noise.dtype = nvinfer1::DataType::kFLOAT;
    noise.data.resize(static_cast<std::size_t>(noise.numel()) * sizeof(float));
    std::mt19937 gen(args.seed);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    auto* p = reinterpret_cast<float*>(noise.data.data());
    for (int64_t i = 0; i < noise.numel(); ++i)
    {
        p[i] = dist(gen);
    }
    std::printf("[pi05_infer] noise: generated (mt19937 seed=%u; 与 Python numpy 不可比，"
                "对拍请传 --noise)\n",
        args.seed);
    return noise;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        Args const args = parseArgs(argc, argv);

        auto const assets = pi05::Pi05Assets::load(args.assetsDir);
        pi05::PaligemmaTokenizer const tokenizer(assets.tokenizerModelPath, assets.maxTokenLen);

        // ---- 原始观测 ----
        pi05::RawObservation obs;
        obs.baseImage = pi05::loadNpy(args.obsDir + "/base_image.npy");
        obs.wristImage = pi05::loadNpy(args.obsDir + "/wrist_image.npy");
        obs.state = pi05::loadNpy(args.obsDir + "/state.npy").toFloat();
        obs.prompt
            = !args.prompt.empty() ? args.prompt : readTextFile(args.obsDir + "/prompt.txt");

        // ---- 前处理 ----
        auto inputs = pi05::buildModelInputs(obs, assets, tokenizer);
        inputs.noise = makeNoise(assets, args);

        // ---- 引擎管线 ----
        pi05::Pi05RuntimeConfig config;
        config.engineDir = args.engineDir;
        config.embedPrefixEngine = args.embedPrefixEngine;
        config.llmEngine = args.llmEngine;
        config.denoiseEngine = args.denoiseEngine;
        config.numSteps = args.numSteps;
        config.attnMaskFill = args.negCap;
        pi05::Pi05Runtime runtime(config);

        if (runtime.denoiseUsesAdarmsMod())
        {
            std::string const dir = !args.adarmsDir.empty() ? args.adarmsDir : args.obsDir;
            for (int k = 0; k < args.numSteps; ++k)
            {
                inputs.adarmsModSteps.push_back(
                    pi05::loadNpy(dir + "/adarms_mod_step" + std::to_string(k) + ".npy"));
            }
        }

        runtime.run(inputs);

        // ---- 后处理 ----
        auto const& actionsRec = runtime.records().at("actions");
        auto const finalActions
            = pi05::postprocessActions(actionsRec.data, actionsRec.shape, assets);
        auto const horizon = static_cast<std::size_t>(assets.actionHorizon);
        auto const outDim = static_cast<std::size_t>(assets.outputActionDim);

        std::printf("\n== actions [%zu, %zu] ==\n", horizon, outDim);
        for (std::size_t h = 0; h < horizon; ++h)
        {
            std::printf("  step %2zu:", h);
            for (std::size_t d = 0; d < outDim; ++d)
            {
                std::printf(" % .5f", static_cast<double>(finalActions[h * outDim + d]));
            }
            std::printf("\n");
        }

        if (!args.outPath.empty())
        {
            NpyArray out;
            out.shape = {static_cast<int64_t>(horizon), static_cast<int64_t>(outDim)};
            out.dtype = nvinfer1::DataType::kFLOAT;
            out.data.resize(finalActions.size() * sizeof(float));
            std::memcpy(out.data.data(), finalActions.data(), out.data.size());
            pi05::saveNpy(args.outPath, out);
            std::printf("saved actions to %s\n", args.outPath.c_str());
        }

        if (!args.goldenPath.empty())
        {
            auto const golden = pi05::loadNpy(args.goldenPath).toFloat();
            EDGE_CHECK(golden.size() == finalActions.size(),
                "golden numel mismatch: " << golden.size() << " vs " << finalActions.size());
            double maxAbs = 0.0;
            double sumAbs = 0.0;
            for (std::size_t i = 0; i < golden.size(); ++i)
            {
                double const diff
                    = std::abs(static_cast<double>(finalActions[i]) - golden[i]);
                maxAbs = std::max(maxAbs, diff);
                sumAbs += diff;
            }
            bool const pass = maxAbs <= args.tol;
            std::printf("\n== golden compare: [%s] max_abs=%.6g mean_abs=%.6g (tol=%g) ==\n",
                pass ? "PASS" : "FAIL", maxAbs, sumAbs / static_cast<double>(golden.size()),
                args.tol);
            return pass ? 0 : 1;
        }
        return 0;
    }
    catch (std::exception const& e)
    {
        std::fprintf(stderr, "[pi05_infer] error: %s\n", e.what());
        return 2;
    }
}
