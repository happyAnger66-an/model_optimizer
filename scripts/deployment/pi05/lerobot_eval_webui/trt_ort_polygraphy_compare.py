"""TensorRT vs ONNX Runtime：基于 Polygraphy 的子图逐张量对比（类 layer 输出对比）。

思路对齐 NVIDIA Polygraphy / Model-Optimizer ``ReferenceRunner``：

- 使用 ``ModifyOutputs(..., outputs=MARK_ALL)`` 将 ONNX 中间张量提升为 graph output，再用
  ``OnnxrtRunner`` 与 ``TrtRunner`` 在同一组输入上推理，对齐输出名后统计 ``max_abs`` /
  ``mean_abs`` 等（见 ``polygraphy.comparator.Comparator``）。

模式：

1. **默认**（``mark_all=False``）：在**原始 ONNX** 的声明输出上，对比 ``CUDA EP 的 ORT`` 与
   ``反序列化的 .engine``（需与 ONNX 来自同一导出，绑定名一致）。
2. **mark_all + rebuild_trt**：在 MARK_ALL 后的 ONNX 上，ORT 与 **当场 TensorRT builder 编译**
   的引擎对比，可看到大量中间张量；**不会**加载用户现成的 ``.engine``（二者图输出集合不一致）。

依赖：``pip install polygraphy onnx onnxruntime-gpu``（及本机 TensorRT Python 包，与现有 TRT 推理一致）。
"""

from __future__ import annotations

import copy
import io
import logging
import sys
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .config import Args

logger = logging.getLogger(__name__)

_PART_KEYS = ("vit", "embed_prefix", "llm", "expert", "denoise")


def _trt_ort_engine_pairs(args: Args) -> list[tuple[str, str, str]]:
    """返回 (标签, TRT 引擎相对路径, ORT ONNX 相对路径)。"""
    pairs: list[tuple[str, str, str]] = []
    mapping: list[tuple[str, str, str]] = [
        ("vit", args.vit_engine, getattr(args, "ort_vit_engine", "") or ""),
        ("embed_prefix", args.embed_prefix_engine, getattr(args, "ort_embed_prefix_engine", "") or ""),
        ("llm", args.llm_engine, getattr(args, "ort_llm_engine", "") or ""),
        ("expert", args.expert_engine, getattr(args, "ort_expert_engine", "") or ""),
        ("denoise", args.denoise_engine, getattr(args, "ort_denoise_engine", "") or ""),
    ]
    for label, trt_n, ort_n in mapping:
        if trt_n and ort_n:
            pairs.append((label, trt_n, ort_n))
    return pairs


def _try_import_polygraphy() -> tuple[Any, str | None]:
    try:
        import polygraphy  # noqa: F401

        from polygraphy import constants
        from polygraphy.backend.common import bytes_from_path
        from polygraphy.backend.onnx import BytesFromOnnx
        from polygraphy.backend.onnx import ModifyOutputs as ModifyOnnxOutputs
        from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
        from polygraphy.backend.trt import CreateConfig, EngineFromBytes, EngineFromNetwork, TrtRunner
        from polygraphy.backend.trt import NetworkFromOnnxBytes
        from polygraphy.comparator import Comparator

        pg = {
            "constants": constants,
            "bytes_from_path": bytes_from_path,
            "BytesFromOnnx": BytesFromOnnx,
            "ModifyOnnxOutputs": ModifyOnnxOutputs,
            "OnnxrtRunner": OnnxrtRunner,
            "SessionFromOnnx": SessionFromOnnx,
            "CreateConfig": CreateConfig,
            "EngineFromBytes": EngineFromBytes,
            "EngineFromNetwork": EngineFromNetwork,
            "NetworkFromOnnxBytes": NetworkFromOnnxBytes,
            "TrtRunner": TrtRunner,
            "Comparator": Comparator,
        }
        return pg, None
    except ImportError as exc:
        return {}, str(exc)


def _normalize_shape_for_feed(shape: Iterable[Any]) -> tuple[int, ...]:
    out: list[int] = []
    for d in shape:
        if isinstance(d, int):
            out.append(d if d > 0 else 1)
        else:
            out.append(1)
    return tuple(out)


def _synthetic_feed_dict(input_metadata: Any, *, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    feeds: dict[str, np.ndarray] = {}
    for name, tmeta in input_metadata.items():
        shape = _normalize_shape_for_feed(tmeta.shape)
        dtype = getattr(tmeta, "dtype", None)
        np_dtype = np.float32
        if dtype is not None:
            s = str(dtype).lower()
            if "float16" in s or "fp16" in s:
                np_dtype = np.float16
            elif "int64" in s:
                np_dtype = np.int64
            elif "int32" in s:
                np_dtype = np.int32
        if np.issubdtype(np_dtype, np.integer):
            feeds[name] = rng.integers(-2, 3, size=shape, dtype=np_dtype)
        else:
            feeds[name] = rng.normal(0.0, 0.05, size=shape).astype(np_dtype)
    return feeds


def _summarize_diff(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    max_tensors: int,
) -> tuple[list[dict[str, Any]], int]:
    """对齐同名输出，返回按 max_abs 降序的摘要列表与总输出数。"""
    names = sorted(set(left.keys()) & set(right.keys()))
    rows: list[dict[str, Any]] = []
    for name in names:
        a = np.asarray(left[name], dtype=np.float64)
        b = np.asarray(right[name], dtype=np.float64)
        if a.shape != b.shape:
            rows.append(
                {
                    "name": name,
                    "shape_left": list(a.shape),
                    "shape_right": list(b.shape),
                    "error": "shape_mismatch",
                }
            )
            continue
        diff = a - b
        absd = np.abs(diff)
        rows.append(
            {
                "name": name,
                "shape": list(a.shape),
                "max_abs": float(np.max(absd)) if absd.size else 0.0,
                "mean_abs": float(np.mean(absd)) if absd.size else 0.0,
                "rmse": float(np.sqrt(np.mean(diff**2))) if diff.size else 0.0,
            }
        )
    rows.sort(key=lambda r: r.get("max_abs", 0.0), reverse=True)
    total = len(names)
    return rows[:max_tensors], total


def _run_pair(
    *,
    onnx_path: Path,
    engine_path: Path | None,
    pg: dict[str, Any],
    providers: list[str] | str,
    mark_all: bool,
    rebuild_trt: bool,
    max_report_tensors: int,
    seed: int,
) -> dict[str, Any]:
    import onnx

    from polygraphy.logger import G_LOGGER, LogMode

    try:
        G_LOGGER.verbosity = LogMode.ERROR
        if hasattr(G_LOGGER, "module_severity"):
            G_LOGGER.module_severity = "ERROR"
    except Exception:
        pass

    model = onnx.load(str(onnx_path))
    onnx_bytes_src: Any
    if mark_all:
        if not rebuild_trt:
            return {
                "ok": False,
                "error": "mark_all=True 时必须同时 rebuild_trt=True（从 ONNX 现场编译 TRT 以匹配 MARK_ALL 图输出）。",
            }
        modified = pg["ModifyOnnxOutputs"](copy.deepcopy(model), outputs=pg["constants"].MARK_ALL)
        onnx_bytes_src = pg["BytesFromOnnx"](modified)
        ort_runner = pg["OnnxrtRunner"](pg["SessionFromOnnx"](onnx_bytes_src, providers=providers))
        trt_runner = pg["TrtRunner"](
            pg["EngineFromNetwork"](
                pg["NetworkFromOnnxBytes"](onnx_bytes_src),
                pg["CreateConfig"](),
            )
        )
    else:
        onnx_bytes_src = pg["BytesFromOnnx"](str(onnx_path))
        ort_runner = pg["OnnxrtRunner"](pg["SessionFromOnnx"](onnx_bytes_src, providers=providers))
        if engine_path is None or not engine_path.is_file():
            return {"ok": False, "error": f"缺少 TensorRT 引擎文件: {engine_path}"}
        trt_runner = pg["TrtRunner"](pg["EngineFromBytes"](pg["bytes_from_path"](str(engine_path))))

    with ort_runner as r0:
        feeds = _synthetic_feed_dict(r0.get_input_metadata(use_numpy_dtypes=False), seed=seed)

    stdout_bak = sys.stdout
    sys.stdout = io.StringIO()
    try:
        run_results = pg["Comparator"].run([ort_runner, trt_runner], data_loader=[feeds])
    except Exception as exc:
        return {"ok": False, "error": f"Comparator.run 失败: {exc!s}"}
    finally:
        sys.stdout = stdout_bak

    try:
        o0 = OrderedDict(run_results[0][1][0])
        o1 = OrderedDict(run_results[1][1][0])
    except Exception as exc:
        return {"ok": False, "error": f"解析 Polygraphy 输出失败: {exc!s}"}

    summary, n_total = _summarize_diff(o0, o1, max_tensors=max_report_tensors)
    return {
        "ok": True,
        "mark_all": mark_all,
        "rebuild_trt": rebuild_trt,
        "outputs_reported": len(summary),
        "outputs_compared_total": n_total,
        "tensors": summary,
    }


def build_trt_ort_polygraphy_report(args: Args) -> dict[str, Any]:
    """在 ``trt_ort_compare`` 已启用且路径有效时调用；返回可嵌入 meta 的 dict。"""
    if not getattr(args, "trt_ort_compare", False):
        return {"enabled": False, "reason": "trt_ort_compare=False"}
    if not getattr(args, "trt_ort_polygraphy_compare", False):
        return {"enabled": False, "reason": "trt_ort_polygraphy_compare=False"}

    mark_all = bool(getattr(args, "trt_ort_polygraphy_mark_all", False))
    rebuild = bool(getattr(args, "trt_ort_polygraphy_rebuild_trt", False))
    if mark_all and not rebuild:
        return {
            "enabled": True,
            "ok": False,
            "error": "trt_ort_polygraphy_mark_all=True 时必须同时 trt_ort_polygraphy_rebuild_trt=True。",
        }

    pg, imp_err = _try_import_polygraphy()
    if imp_err:
        return {
            "enabled": True,
            "ok": False,
            "import_error": imp_err,
            "hint": "请安装: pip install polygraphy onnx onnxruntime-gpu",
        }

    trt_root = Path(args.engine_path).expanduser()
    ort_root = Path(getattr(args, "ort_engine_path", "") or "").expanduser()
    if not trt_root.is_dir() or not ort_root.is_dir():
        return {"enabled": True, "ok": False, "error": "engine_path 或 ort_engine_path 不是有效目录"}

    parts_filter = frozenset(getattr(args, "trt_ort_polygraphy_parts", ()) or ())
    if parts_filter and not parts_filter.issubset(_PART_KEYS):
        bad = parts_filter - frozenset(_PART_KEYS)
        return {"enabled": True, "ok": False, "error": f"非法 trt_ort_polygraphy_parts: {sorted(bad)}"}

    prov = list(
        getattr(args, "trt_ort_polygraphy_ort_providers", None)
        or ("CUDAExecutionProvider", "CPUExecutionProvider")
    )
    max_tensors = int(getattr(args, "trt_ort_polygraphy_max_report_tensors", 256))
    base_seed = int(getattr(args, "trt_ort_polygraphy_seed", 0))

    subgraphs: dict[str, Any] = {}
    for label, trt_name, ort_name in _trt_ort_engine_pairs(args):
        if parts_filter and label not in parts_filter:
            continue
        onnx_p = ort_root / ort_name
        eng_p = trt_root / trt_name
        if not onnx_p.is_file():
            subgraphs[label] = {"ok": False, "error": f"缺少 ONNX: {onnx_p}"}
            continue
        need_engine_file = not (mark_all and rebuild)
        if need_engine_file and not eng_p.is_file():
            subgraphs[label] = {"ok": False, "error": f"缺少 TRT engine: {eng_p}"}
            continue
        logger.info(
            "Polygraphy TRT vs ORT：%s onnx=%s engine=%s mark_all=%s rebuild=%s",
            label,
            onnx_p,
            eng_p if not mark_all else "(rebuild from onnx)",
            mark_all,
            rebuild,
        )
        try:
            sub_seed = base_seed + sum(ord(c) for c in label) % 10_007
            subgraphs[label] = {
                "onnx": str(onnx_p.resolve()),
                "engine": None if mark_all and rebuild else str(eng_p.resolve()),
                **_run_pair(
                    onnx_path=onnx_p,
                    engine_path=eng_p,
                    pg=pg,
                    providers=prov,
                    mark_all=mark_all,
                    rebuild_trt=rebuild,
                    max_report_tensors=max_tensors,
                    seed=sub_seed,
                ),
            }
        except Exception as exc:  # pragma: no cover
            logger.exception("Polygraphy 子图 %s 失败", label)
            subgraphs[label] = {"ok": False, "error": str(exc)}

    ok_all = bool(subgraphs) and all(
        isinstance(v, dict) and v.get("ok") for v in subgraphs.values()
    )
    return {
        "enabled": True,
        "ok": ok_all,
        "mark_all": mark_all,
        "rebuild_trt": rebuild,
        "ort_providers": prov,
        "subgraphs": subgraphs,
    }
