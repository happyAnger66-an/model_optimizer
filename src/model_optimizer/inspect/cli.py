# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import fnmatch
import json
import os
from typing import Any


def _load_engine_layer_json(engine_path: str) -> dict[str, Any]:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f:
        blob = f.read()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(blob)
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
    insp = engine.create_engine_inspector()
    s = insp.get_engine_information(trt.LayerInformationFormat.JSON)
    try:
        return json.loads(s)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "EngineInspector returned non-JSON output; try upgrading TensorRT or use ONELINE format."
        ) from exc


def _iter_layers(obj: dict[str, Any]) -> list[dict[str, Any]]:
    # TRT has changed JSON field names across versions; handle a few common cases.
    for key in ("Layers", "layers", "LayerInformation", "layerInformation"):
        v = obj.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    # Some versions wrap as {"engine": {"layers": [...]}}
    eng = obj.get("engine")
    if isinstance(eng, dict):
        for key in ("Layers", "layers"):
            v = eng.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def _get_str(d: dict[str, Any], *keys: str) -> str:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        if isinstance(v, str):
            return v
        return str(v)
    return ""


def _layer_name(layer: dict[str, Any]) -> str:
    return _get_str(layer, "Name", "name", "layerName", "LayerName")


def _layer_type(layer: dict[str, Any]) -> str:
    return _get_str(layer, "LayerType", "layerType", "Type", "type")


def _layer_metadata(layer: dict[str, Any]) -> str:
    md = layer.get("Metadata", layer.get("metadata"))
    if md is None:
        return ""
    if isinstance(md, str):
        return md
    if isinstance(md, list):
        return " ".join(str(x) for x in md)
    return str(md)


def _match_any(value: str, *, glob: str | None, contains: str | None) -> bool:
    if glob:
        if not fnmatch.fnmatch(value, glob):
            return False
    if contains:
        if contains not in value:
            return False
    return True


def inspect_engine(args: argparse.Namespace) -> int:
    engine_path = os.path.expanduser(args.engine)
    info = _load_engine_layer_json(engine_path)
    layers = _iter_layers(info)
    if not layers:
        raise RuntimeError("No layers found in engine inspector JSON output.")

    name_glob = args.name_glob
    name_contains = args.name_contains
    md_glob = args.metadata_glob
    md_contains = args.metadata_contains
    type_glob = args.type_glob
    type_contains = args.type_contains

    shown = 0
    for i, layer in enumerate(layers):
        name = _layer_name(layer)
        ltype = _layer_type(layer)
        meta = _layer_metadata(layer)

        if not _match_any(name, glob=name_glob, contains=name_contains):
            continue
        if not _match_any(meta, glob=md_glob, contains=md_contains):
            continue
        if not _match_any(ltype, glob=type_glob, contains=type_contains):
            continue

        if args.format == "jsonl":
            out = {"index": i, "name": name, "type": ltype, "metadata": meta}
            print(json.dumps(out, ensure_ascii=False))
        else:
            # oneline/tsv-ish for easy copy/paste into config patterns
            print(f"{i}\t{name}\t{ltype}\t{meta}")
        shown += 1
        if args.limit is not None and shown >= args.limit:
            break

    if shown == 0:
        print("No matching layers.")
        return 2
    return 0


def inspect_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="model-opt inspect",
        description="Inspect compiled artifacts (TensorRT engines, etc.).",
    )
    sub = parser.add_subparsers(dest="subcmd", required=True)

    p_engine = sub.add_parser("model", help="Inspect TensorRT .engine layers")
    p_engine.add_argument("engine", type=str, help="Path to TensorRT engine (.engine)")
    p_engine.add_argument("--format", choices=("oneline", "jsonl"), default="oneline")
    p_engine.add_argument("--limit", type=int, default=None)

    p_engine.add_argument("--name-glob", type=str, default=None, help="Filter by layer name glob (*, ?)")
    p_engine.add_argument("--name-contains", type=str, default=None, help="Filter by substring in layer name")
    p_engine.add_argument("--metadata-glob", type=str, default=None, help="Filter by metadata glob (*, ?)")
    p_engine.add_argument(
        "--metadata-contains", type=str, default=None, help="Filter by substring in metadata"
    )
    p_engine.add_argument("--type-glob", type=str, default=None, help="Filter by layer type glob")
    p_engine.add_argument("--type-contains", type=str, default=None, help="Filter by substring in layer type")

    args = parser.parse_args(argv[1:])
    if args.subcmd == "model":
        raise SystemExit(inspect_engine(args))
    raise SystemExit(1)

