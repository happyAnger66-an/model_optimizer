// SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
#include "pi05/tokenizer.h"

#include "common/check.h"
#include "common/logger.h"

#include <sentencepiece_processor.h>

#include <algorithm>
#include <cmath>
#include <sstream>

namespace pi05
{

PaligemmaTokenizer::PaligemmaTokenizer(std::string const& modelPath, int maxLen)
    : mProcessor(std::make_unique<sentencepiece::SentencePieceProcessor>())
    , mMaxLen(maxLen)
{
    auto const status = mProcessor->Load(modelPath);
    EDGE_CHECK(status.ok(), "sentencepiece load failed: " << status.ToString());
}

PaligemmaTokenizer::~PaligemmaTokenizer() = default;

std::string PaligemmaTokenizer::cleanText(std::string const& prompt)
{
    // strip 两端空白
    auto const first = prompt.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
    {
        return "";
    }
    auto const last = prompt.find_last_not_of(" \t\r\n");
    std::string s = prompt.substr(first, last - first + 1);
    // replace("_", " ").replace("\n", " ")
    for (char& c : s)
    {
        if (c == '_' || c == '\n')
        {
            c = ' ';
        }
    }
    return s;
}

TokenizedPrompt PaligemmaTokenizer::padTruncate(std::vector<int> tokens) const
{
    auto const len = static_cast<int>(tokens.size());
    if (len > mMaxLen)
    {
        LOG_WARN("token length (%d) exceeds max length (%d), truncating", len, mMaxLen);
    }
    TokenizedPrompt out;
    out.tokens.resize(static_cast<std::size_t>(mMaxLen), 0);
    out.mask.resize(static_cast<std::size_t>(mMaxLen), 0);
    int const used = std::min(len, mMaxLen);
    for (int i = 0; i < used; ++i)
    {
        out.tokens[static_cast<std::size_t>(i)] = tokens[static_cast<std::size_t>(i)];
        out.mask[static_cast<std::size_t>(i)] = 1;
    }
    // 与 Python 一致：超长时 mask 全 True
    if (len >= mMaxLen)
    {
        std::fill(out.mask.begin(), out.mask.end(), static_cast<uint8_t>(1));
    }
    return out;
}

TokenizedPrompt PaligemmaTokenizer::tokenize(std::string const& prompt) const
{
    std::string const cleaned = cleanText(prompt);
    std::vector<int> ids;
    auto status = mProcessor->Encode(cleaned, &ids);
    EDGE_CHECK(status.ok(), "sentencepiece encode failed: " << status.ToString());
    std::vector<int> tokens;
    tokens.push_back(mProcessor->bos_id()); // add_bos=True
    tokens.insert(tokens.end(), ids.begin(), ids.end());
    // "\n" 单独 encode 作为 "start of answer" token
    std::vector<int> newlineIds;
    status = mProcessor->Encode("\n", &newlineIds);
    EDGE_CHECK(status.ok(), "sentencepiece encode failed: " << status.ToString());
    tokens.insert(tokens.end(), newlineIds.begin(), newlineIds.end());
    return padTruncate(std::move(tokens));
}

TokenizedPrompt PaligemmaTokenizer::tokenizeWithState(
    std::string const& prompt, std::vector<float> const& state) const
{
    // np.digitize(state, bins=linspace(-1,1,257)[:-1]) - 1
    // bins[k] = -1 + 2k/256, k=0..255；digitize 返回插入点（右开），减 1。
    std::ostringstream stateStr;
    for (std::size_t i = 0; i < state.size(); ++i)
    {
        float const x = state[i];
        int idx = static_cast<int>(std::floor((x + 1.0F) / 2.0F * 256.0F));
        // 与 np.digitize 边界一致：x < -1 → -1+? digitize 给 0 → -1 后为 -1；
        // 实际 state 已归一化到 [-1,1]，clamp 到 [0,255] 再减按 python 行为偏移。
        idx = std::max(0, std::min(255, idx));
        if (i > 0)
        {
            stateStr << " ";
        }
        stateStr << idx;
    }
    std::string const cleaned = cleanText(prompt);
    std::string const full = "Task: " + cleaned + ", State: " + stateStr.str() + ";\nAction: ";
    std::vector<int> ids;
    auto const status = mProcessor->Encode(full, &ids);
    EDGE_CHECK(status.ok(), "sentencepiece encode failed: " << status.ToString());
    std::vector<int> tokens;
    tokens.push_back(mProcessor->bos_id());
    tokens.insert(tokens.end(), ids.begin(), ids.end());
    return padTruncate(std::move(tokens));
}

} // namespace pi05
