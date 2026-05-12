#!/usr/bin/env python3
# Copyright 2026 the model_optimizer team.
#
# SPDX-License-Identifier: Apache-2.0
#
"""将 ``trt::GemmaFusedGatedMlp`` 三输入 ONNX 后处理为 **单输入 + 权重张量属性**（TensorRT 插件 v2，用于消除权重边 ``Move``）。

用法（在仓库根目录执行，或任意目录指定绝对路径）::

    python3 scripts/embed_gemma_fused_gated_mlp_trt_onnx.py /path/to/model.onnx
    python3 scripts/embed_gemma_fused_gated_mlp_trt_onnx.py /path/to/model.onnx -o /path/to/model_static.onnx
    python3 scripts/embed_gemma_fused_gated_mlp_trt_onnx.py model.onnx --keep-initializers
    python3 scripts/embed_gemma_fused_gated_mlp_trt_onnx.py model.onnx --inline-weights  # 单文件（大模型易超 protobuf 解析上限，慎用）

默认把 **节点属性里的大张量** 与图中其它大权重一并 **外置** 到 ``<onnx 文件名>.data``，主 ``.onnx`` 保持较小，便于 ``onnx.load`` / ``model-opt build`` 预检解析。

依赖：``onnx``；需将仓库 ``src`` 加入 ``PYTHONPATH``（本脚本已自动插入与 ``scripts`` 同级的 ``src``）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_src_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)
    return repo_root


def main() -> int:
    _ensure_src_on_path()

    import onnx

    from model_optimizer.ops.gemma_fused_gated_mlp_onnx_embed import embed_gemma_fused_gated_mlp_trt_static_weights

    p = argparse.ArgumentParser(
        description="Fold GemmaFusedGatedMlp ONNX weights into tensor attributes (TRT plugin v2, single input x)."
    )
    p.add_argument(
        "onnx_in",
        type=Path,
        help="输入 ONNX 路径（须含 domain=trt 的 GemmaFusedGatedMlp 三输入节点，且权重为独立 initializer）",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出 ONNX 路径；默认在输入同目录下写入 ``<stem>_gemma_trt_static.onnx``",
    )
    p.add_argument(
        "--keep-initializers",
        action="store_true",
        help="折叠后仍保留原 ``gate_up_weight`` / ``down_weight`` initializer（默认会删除已嵌入且无其它引用的 initializer）",
    )
    p.add_argument(
        "--inline-weights",
        action="store_true",
        help="不把权重外置：写出单个巨大 .onnx（数 GB 时 onnx.load / protobuf 常 DecodeError；默认会外置）",
    )
    args = p.parse_args()

    onnx_in: Path = args.onnx_in.expanduser().resolve()
    if not onnx_in.is_file():
        print(f"error: input ONNX not found: {onnx_in}", file=sys.stderr)
        return 2

    out: Path
    if args.output is not None:
        out = args.output.expanduser().resolve()
    else:
        out = onnx_in.with_name(f"{onnx_in.stem}_gemma_trt_static{onnx_in.suffix}")

    model = onnx.load(str(onnx_in))
    embed_gemma_fused_gated_mlp_trt_static_weights(
        model,
        remove_embedded_initializers=not bool(args.keep_initializers),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.inline_weights:
        onnx.save(model, str(out))
        print(f"wrote {out}")
    else:
        # 权重进节点属性后若仍全部挤在一个 ModelProto 里，多 GB 单文件常触发 protobuf DecodeError。
        # save_as_external_data + convert_attribute 把属性/initializer 中的 raw 权重拆到 .data 文件。
        ext_name = f"{out.name}.data"
        onnx.save(
            model,
            str(out),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=ext_name,
            size_threshold=0,
            convert_attribute=True,
        )
        data_path = out.parent / ext_name
        print(f"wrote {out} (+ external data {data_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
