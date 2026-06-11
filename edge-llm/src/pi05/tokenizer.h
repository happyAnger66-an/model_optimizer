// SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sentencepiece
{
class SentencePieceProcessor;
}

namespace pi05
{

struct TokenizedPrompt
{
    std::vector<int64_t> tokens; //!< [maxLen]，pad 0
    std::vector<uint8_t> mask;   //!< [maxLen]，bool
};

//! PaliGemma tokenizer（等价 openpi PaligemmaTokenizer，pi0 格式 / 离散 state 格式）。
class PaligemmaTokenizer
{
public:
    explicit PaligemmaTokenizer(std::string const& modelPath, int maxLen);
    ~PaligemmaTokenizer();

    //! pi0/pi05(discrete_state_input=False) 格式：
    //! [bos] + encode(strip(prompt).replace('_',' ').replace('\n',' ')) + encode("\n")
    TokenizedPrompt tokenize(std::string const& prompt) const;

    //! pi05 discrete_state_input=True 格式：
    //! [bos] + encode("Task: {prompt}, State: {bins};\nAction: ")
    //! state 须已归一化到 [-1,1]，digitize 到 256 bins。
    TokenizedPrompt tokenizeWithState(
        std::string const& prompt, std::vector<float> const& state) const;

private:
    TokenizedPrompt padTruncate(std::vector<int> tokens) const;
    static std::string cleanText(std::string const& prompt);

    std::unique_ptr<sentencepiece::SentencePieceProcessor> mProcessor;
    int mMaxLen;
};

} // namespace pi05
