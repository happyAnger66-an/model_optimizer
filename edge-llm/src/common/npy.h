// SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <NvInfer.h>

#include <cstdint>
#include <string>
#include <vector>

namespace pi05
{

//! 极简 npy v1.0 读写（C-order，小端），覆盖 f4/f2/i4/i8/b1/u1。
struct NpyArray
{
    std::vector<int64_t> shape;
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
    std::vector<uint8_t> data;

    int64_t numel() const;
    std::vector<float> toFloat() const;
};

NpyArray loadNpy(std::string const& path);
void saveNpy(std::string const& path, NpyArray const& array);

bool fileExists(std::string const& path);

} // namespace pi05
