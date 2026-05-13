#!/usr/bin/env python3
# Copyright 2026 the model_optimizer team.
#
# SPDX-License-Identifier: Apache-2.0
#
"""对比两份 TensorRT ``trtexec`` 导出的 **layer 信息** 与 **profile**，按 Metadata 中的 ONNX 名筛选 attention 相关层并汇总耗时。

典型采集（TensorRT 10.x，示例）::

    trtexec --loadEngine=baseline.engine \\
      --exportLayerInfo=baseline_layer.json \\
      --exportProfile=baseline_profile.json \\
      --dumpProfile --warmUp=200 --iterations=500

再对 fused MLP 引擎重复一遍，得到 ``fused_*``。本脚本读取两套 JSON，按 **TRT 层名** 将 profile 中的耗时
与 ``--exportLayerInfo`` / ``--dumpLayerInfo`` 中的 ``Metadata`` 对齐，筛选 attention 段后做 diff。

说明：

- ``Metadata`` 常为 ``[ONNX Layer: /foo]\\x1f[ONNX Layer: /bar]``（``\\x1f`` 为单元分隔符），脚本会拆出 ONNX 名。
- 不同 TRT 版本 profile JSON 字段名可能不同；脚本会尝试多种常见键；若对不上请把样例顶层键名发维护者扩展。
- 若 profile 条目不含 ``Metadata``，必须用 ``--layer-*`` 提供 layer json 做 **按层名 join**。

用法::

    python3 scripts/diff_trt_attention_profile.py \\
      --layer-a /path/baseline_layer.json \\
      --layer-b /path/fused_layer.json \\
      --profile-a /path/baseline_profile.json \\
      --profile-b /path/fused_profile.json

依赖：标准库 ``json`` / ``re`` / ``argparse``；无需 TensorRT Python 包。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_layer_rows(root: Any) -> Iterable[dict[str, Any]]:
    """从 layer info / 部分 profile 包装中取出层字典列表。"""
    if isinstance(root, list):
        for x in root:
            if isinstance(x, dict):
                yield x
        return
    if not isinstance(root, dict):
        return
    for key in ("Layers", "layers", "layer", "Layer"):
        v = root.get(key)
        if isinstance(v, list):
            for x in v:
                if isinstance(x, dict):
                    yield x
            return
    # 有些导出把整个数组放在无名根；或嵌套 ``{"engine": {"Layers": ...}}``
    for v in root.values():
        if isinstance(v, dict):
            yield from _iter_layer_rows(v)
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            yield from _iter_layer_rows(v)


def _parse_onnx_names_from_metadata(meta: Any) -> list[str]:
    if meta is None:
        return []
    if not isinstance(meta, str) or not meta.strip():
        return []
    parts = re.split(r"[\x1f\x1e\n]+", meta)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.search(r"\[ONNX Layer:\s*([^\]]+)\]", p)
        if m:
            out.append(m.group(1).strip())
        else:
            out.append(p)
    return out


def _pick_time_ms(row: dict[str, Any]) -> float | None:
    """从单条 profile / layer 行里抠毫秒级耗时（尽力而为）。"""
    # 显式常见键（不同 trtexec / 版本）
    for k in (
        "Average time (ms)",
        "average time (ms)",
        "avg_time_ms",
        "averageMs",
        "AverageMs",
        "timeMs",
        "Time (ms)",
        "time_ms",
        "latency_ms",
        "Latency (ms)",
        "Median(ms)",
        "median_ms",
        "gpu_ms",
        "GpuMs",
    ):
        if k in row and isinstance(row[k], (int, float)):
            return float(row[k])
    # 任意包含 time 且单位为 ms 的键
    for k, v in row.items():
        if not isinstance(v, (int, float)):
            continue
        lk = str(k).lower()
        if "time" in lk and ("ms" in lk or lk.endswith("_ms")):
            return float(v)
    return None


def _layer_name(row: dict[str, Any]) -> str | None:
    for k in ("Name", "name", "layerName", "LayerName"):
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def _build_name_to_metadata(layer_root: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in _iter_layer_rows(layer_root):
        nm = _layer_name(row)
        if nm is None:
            continue
        meta = row.get("Metadata", row.get("metadata", ""))
        if isinstance(meta, str):
            out[nm] = meta
    return out


def _build_name_to_time(profile_root: Any) -> dict[str, float]:
    times: dict[str, float] = {}
    for row in _iter_layer_rows(profile_root):
        nm = _layer_name(row)
        if nm is None:
            continue
        t = _pick_time_ms(row)
        if t is None:
            continue
        # 若重名层出现多次，累加（少见）
        times[nm] = times.get(nm, 0.0) + t
    return times


def _row_metadata(row: dict[str, Any], meta_map: dict[str, str]) -> str:
    nm = _layer_name(row)
    if nm is None:
        return ""
    if isinstance(row.get("Metadata"), str):
        return str(row["Metadata"])
    if isinstance(row.get("metadata"), str):
        return str(row["metadata"])
    return meta_map.get(nm, "")


def _matches(meta: str, trt_name: str, include: re.Pattern[str], exclude: re.Pattern[str] | None) -> bool:
    blob = f"{meta}\n{trt_name}"
    if not include.search(blob):
        return False
    if exclude is not None and exclude.search(blob):
        return False
    return True


def _aggregate(
    *,
    layer_path: Path | None,
    profile_path: Path,
    include: re.Pattern[str],
    exclude: re.Pattern[str] | None,
    topk: int,
) -> tuple[float, list[tuple[str, float, str]]]:
    prof = _read_json(profile_path)
    meta_map: dict[str, str] = {}
    if layer_path is not None:
        meta_map = _build_name_to_metadata(_read_json(layer_path))
    times = _build_name_to_time(prof)

    matched: list[tuple[str, float, str]] = []
    total = 0.0
    for nm, ms in sorted(times.items(), key=lambda kv: kv[1], reverse=True):
        meta = meta_map.get(nm, "")
        if not _matches(meta, nm, include, exclude):
            continue
        total += ms
        matched.append((nm, ms, meta))

    matched.sort(key=lambda x: x[1], reverse=True)
    if topk > 0:
        matched = matched[:topk]
    return total, matched


def main() -> int:
    p = argparse.ArgumentParser(
        description="Diff attention-related layer times between two trtexec profile/layer JSON bundles."
    )
    p.add_argument("--layer-a", type=Path, default=None, help="Engine A layer info JSON (exportLayerInfo / dumpLayerInfo)")
    p.add_argument("--layer-b", type=Path, default=None, help="Engine B layer info JSON")
    p.add_argument("--profile-a", type=Path, required=True, help="Engine A profile JSON (--exportProfile / dumpProfile)")
    p.add_argument("--profile-b", type=Path, required=True, help="Engine B profile JSON")
    p.add_argument("--label-a", type=str, default="A", help="打印标签")
    p.add_argument("--label-b", type=str, default="B", help="打印标签")
    p.add_argument(
        "--include-regex",
        type=str,
        default=r"(?i)(Softmax|attention_mask|q_proj|k_proj|v_proj|o_proj|rotary|/MatMul|/MatMul_1|/Add_2|/Add_3)",
        help="匹配 Metadata 或 TRT 层名；命中则计入 attention 汇总（默认偏 Gemma 解码器 attention）",
    )
    p.add_argument(
        "--exclude-regex",
        type=str,
        default=r"(?i)(GemmaFusedGatedMlp|/mlp/|mlp/)",
        help="命中则从 attention 汇总中排除（默认去掉 MLP 插件/子串）",
    )
    p.add_argument("--no-exclude", action="store_true", help="不使用排除正则")
    p.add_argument("--topk", type=int, default=25, help="各侧打印贡献最大的前 K 条匹配层（0 表示不打印明细）")
    args = p.parse_args()

    include = re.compile(args.include_regex)
    exclude = None if args.no_exclude else re.compile(args.exclude_regex)

    if not args.profile_a.is_file() or not args.profile_b.is_file():
        print("error: --profile-a / --profile-b must exist", file=sys.stderr)
        return 2
    if args.layer_a is not None and not args.layer_a.is_file():
        print(f"error: --layer-a not found: {args.layer_a}", file=sys.stderr)
        return 2
    if args.layer_b is not None and not args.layer_b.is_file():
        print(f"error: --layer-b not found: {args.layer_b}", file=sys.stderr)
        return 2

    total_a, rows_a = _aggregate(
        layer_path=args.layer_a,
        profile_path=args.profile_a,
        include=include,
        exclude=exclude,
        topk=int(args.topk),
    )
    total_b, rows_b = _aggregate(
        layer_path=args.layer_b,
        profile_path=args.profile_b,
        include=include,
        exclude=exclude,
        topk=int(args.topk),
    )

    delta = total_b - total_a
    pct = (delta / total_a * 100.0) if total_a > 0 else float("nan")

    print("=== Attention bucket (regex on Metadata ∪ TRT Name) ===")
    print(f"include: {args.include_regex!r}")
    print(f"exclude: {None if exclude is None else args.exclude_regex!r}")
    print()
    print(f"{args.label_a:16} sum_ms = {total_a:.6f}  (n matched layers with timing: inferred from profile keys)")
    print(f"{args.label_b:16} sum_ms = {total_b:.6f}")
    print(f"{'delta (B-A)':16} ms     = {delta:+.6f}  ({pct:+.2f}% vs A)" if total_a > 0 else f"{'delta (B-A)':16} ms     = {delta:+.6f}")
    print()

    if args.topk > 0:
        print(f"--- Top {args.topk} contributors: {args.label_a} ---")
        for nm, ms, meta in rows_a:
            onnx = "; ".join(_parse_onnx_names_from_metadata(meta)[:6])
            tail = " ..." if len(meta) > 200 else ""
            print(f"  {ms:10.4f} ms  {nm}")
            if onnx:
                print(f"            onnx: {onnx}")
        print()
        print(f"--- Top {args.topk} contributors: {args.label_b} ---")
        for nm, ms, meta in rows_b:
            onnx = "; ".join(_parse_onnx_names_from_metadata(meta)[:6])
            print(f"  {ms:10.4f} ms  {nm}")
            if onnx:
                print(f"            onnx: {onnx}")

    print()
    print("提示：若 profile 里几乎抽不到耗时，请确认 trtexec 使用了 ``--dumpProfile`` / ``--exportProfile``，")
    print("且 JSON 中每层是否有毫秒字段；可把某一层的原始 dict 键名发出来以便扩展 ``_pick_time_ms``。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
