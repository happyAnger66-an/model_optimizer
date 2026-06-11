// SPDX-FileCopyrightText: Copyright (c) 2026 the model_optimizer team.
// SPDX-License-Identifier: Apache-2.0
#include "common/npy.h"

#include "common/check.h"
#include "common/dtype.h"

#include <cstring>
#include <fstream>
#include <sstream>

namespace pi05
{

namespace
{

constexpr char kMagic[] = "\x93NUMPY";

std::string extractField(std::string const& header, std::string const& key)
{
    auto const pos = header.find("'" + key + "'");
    EDGE_CHECK(pos != std::string::npos, "npy header missing key: " << key);
    auto const colon = header.find(':', pos);
    EDGE_CHECK(colon != std::string::npos, "bad npy header near key: " << key);
    return header.substr(colon + 1);
}

} // namespace

int64_t NpyArray::numel() const
{
    return volume(shape);
}

std::vector<float> NpyArray::toFloat() const
{
    std::vector<float> out(static_cast<std::size_t>(numel()));
    convertToFloatHost(data.data(), dtype, out.data(), out.size());
    return out;
}

bool fileExists(std::string const& path)
{
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

NpyArray loadNpy(std::string const& path)
{
    std::ifstream f(path, std::ios::binary);
    EDGE_CHECK(f.good(), "cannot open npy file: " << path);

    char magic[6];
    f.read(magic, 6);
    EDGE_CHECK(std::memcmp(magic, kMagic, 6) == 0, "bad npy magic: " << path);

    uint8_t version[2];
    f.read(reinterpret_cast<char*>(version), 2);

    uint32_t headerLen = 0;
    if (version[0] == 1)
    {
        uint16_t len16 = 0;
        f.read(reinterpret_cast<char*>(&len16), 2);
        headerLen = len16;
    }
    else
    {
        f.read(reinterpret_cast<char*>(&headerLen), 4);
    }

    std::string header(headerLen, '\0');
    f.read(header.data(), headerLen);

    // descr
    auto descrPart = extractField(header, "descr");
    auto const q1 = descrPart.find('\'');
    auto const q2 = descrPart.find('\'', q1 + 1);
    EDGE_CHECK(q1 != std::string::npos && q2 != std::string::npos, "bad descr in " << path);
    std::string const descr = descrPart.substr(q1 + 1, q2 - q1 - 1);

    // fortran_order
    auto fortranPart = extractField(header, "fortran_order");
    EDGE_CHECK(fortranPart.find("False") != std::string::npos,
        "fortran-order npy not supported: " << path);

    // shape
    auto shapePart = extractField(header, "shape");
    auto const p1 = shapePart.find('(');
    auto const p2 = shapePart.find(')', p1);
    EDGE_CHECK(p1 != std::string::npos && p2 != std::string::npos, "bad shape in " << path);
    std::string const shapeStr = shapePart.substr(p1 + 1, p2 - p1 - 1);

    NpyArray arr;
    arr.dtype = dtypeFromNpyDescr(descr);
    std::stringstream ss(shapeStr);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        // 去空白
        std::string trimmed;
        for (char const c : item)
        {
            if (!std::isspace(static_cast<unsigned char>(c)))
            {
                trimmed += c;
            }
        }
        if (!trimmed.empty())
        {
            arr.shape.push_back(std::stoll(trimmed));
        }
    }
    // 标量 () → shape [1]
    if (arr.shape.empty())
    {
        arr.shape.push_back(1);
    }

    std::size_t const bytes = static_cast<std::size_t>(arr.numel()) * dtypeSize(arr.dtype);
    arr.data.resize(bytes);
    f.read(reinterpret_cast<char*>(arr.data.data()), static_cast<std::streamsize>(bytes));
    EDGE_CHECK(static_cast<std::size_t>(f.gcount()) == bytes, "truncated npy data: " << path);
    return arr;
}

void saveNpy(std::string const& path, NpyArray const& array)
{
    std::ostringstream shapeSs;
    shapeSs << "(";
    for (std::size_t i = 0; i < array.shape.size(); ++i)
    {
        shapeSs << array.shape[i] << (array.shape.size() == 1 ? "," : "");
        if (i + 1 < array.shape.size())
        {
            shapeSs << ", ";
        }
    }
    shapeSs << ")";

    std::string header = "{'descr': '" + npyDescrFromDtype(array.dtype)
        + "', 'fortran_order': False, 'shape': " + shapeSs.str() + ", }";
    // 头部总长（magic6 + ver2 + len2 + header）按 64 对齐，补空格 + '\n' 结尾。
    std::size_t const base = 6 + 2 + 2;
    std::size_t total = base + header.size() + 1;
    std::size_t const pad = (64 - total % 64) % 64;
    header.append(pad, ' ');
    header += '\n';

    std::ofstream f(path, std::ios::binary);
    EDGE_CHECK(f.good(), "cannot write npy file: " << path);
    f.write(kMagic, 6);
    uint8_t const version[2] = {1, 0};
    f.write(reinterpret_cast<char const*>(version), 2);
    auto const len16 = static_cast<uint16_t>(header.size());
    f.write(reinterpret_cast<char const*>(&len16), 2);
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    f.write(reinterpret_cast<char const*>(array.data.data()),
        static_cast<std::streamsize>(array.data.size()));
}

} // namespace pi05
