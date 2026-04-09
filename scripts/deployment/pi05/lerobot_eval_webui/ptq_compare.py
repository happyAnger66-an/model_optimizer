"""Pi0.5 选择性 PTQ（Vit / LLM / Expert / Denoise）与分层输出误差分析（WebUI / eval 管线）。"""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from termcolor import colored

from model_optimizer.models.pi05.dit import Pi05DenoiseStep
from model_optimizer.models.pi05.expert import Expert
from model_optimizer.models.pi05.llm import LLM
from model_optimizer.models.pi05.vit import Vit
from model_optimizer.quantization.quantization_utils import quantize_model
from model_optimizer.utils.utils import normalize_quant_cfg, set_dynamic_quant

from .calib import unwrap_pytorch_pi05_model
from .dataset import tree_to_numpy

logger = logging.getLogger(__name__)

_PART_PREFIXES: dict[str, tuple[str, ...]] = {
    "vit": (
        "paligemma_with_expert.paligemma.model.vision_tower",
        "paligemma_with_expert.paligemma.model.multi_modal_projector",
    ),
    "llm": ("paligemma_with_expert.paligemma.model.language_model",),
    "expert": ("paligemma_with_expert.gemma_expert.model",),
    # Pi05 单次 denoise：expert 主干 + 根上 action/time 投影（与 openpi PI0Pytorch.pi05 一致）
    "denoise": (
        "paligemma_with_expert.gemma_expert.model",
        "action_in_proj",
        "time_mlp_in",
        "time_mlp_out",
        "action_out_proj",
    ),
}


def load_ptq_quant_cfg(path: Path) -> dict[str, Any]:
    """加载量化配置：``.json`` 为 JSON；``.py`` / ``.pyw`` 需定义 ``QUANT_CFG``（dict）。

    兼容 ``llm_quant_nvfp4_cfg.py`` 等写法：``QUANT_CFG = { ... }``，内含 ``quant_mode``、``quant_cfg`` 等字段。
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ptq_quant_cfg: 找不到量化配置文件或路径不是文件: {path}")
    suffix = path.suffix.lower()
    if suffix in (".py", ".pyw"):
        mod_name = f"_ptq_quant_cfg_{path.stem}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"ptq_quant_cfg: 无法从文件建立模块: {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        if not hasattr(mod, "QUANT_CFG"):
            raise AttributeError(f"{path} 中未定义 QUANT_CFG（应为 ModelOpt 量化 dict）")
        raw = getattr(mod, "QUANT_CFG")
        if not isinstance(raw, dict):
            raise TypeError(f"{path} 的 QUANT_CFG 必须是 dict，实际为 {type(raw).__name__}")
        cfg = copy.deepcopy(raw)
    else:
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise TypeError(f"{path} 的 JSON 根节点必须是 object/dict")
        cfg = copy.deepcopy(loaded)
    if "quant_mode" in cfg:
        _, cfg = normalize_quant_cfg(cfg)
    return cfg


def _vit_from_pi05(pi05: nn.Module) -> Vit:
    return Vit(
        pi05.paligemma_with_expert.paligemma.config,
        pi05.paligemma_with_expert.paligemma.model.vision_tower,
        pi05.paligemma_with_expert.paligemma.model.multi_modal_projector,
    )


def _llm_from_pi05(pi05: nn.Module) -> LLM:
    pal = pi05.paligemma_with_expert.paligemma
    return LLM(pal.config.text_config, pal.get_decoder())


def _expert_from_pi05(pi05: nn.Module) -> Expert:
    ge = pi05.paligemma_with_expert.gemma_expert
    return Expert(ge.config, ge.model)


def _denoise_from_pi05(pi05: nn.Module) -> Pi05DenoiseStep:
    return Pi05DenoiseStep.construct_model(pi05)


def apply_selective_ptq(
    policy_ptq: Any,
    calib_dir: Path,
    quant_cfg: dict[str, Any],
    parts: tuple[str, ...],
) -> None:
    """对第二份 policy 内的 Pi0.5 子模块就地 PTQ（与 ``Vit/LLM/Expert.quantize`` 一致，跳过 ONNX）。"""
    m = unwrap_pytorch_pi05_model(policy_ptq)
    if m is None:
        raise RuntimeError("ptq_compare：无法从 policy_ptq 解析 _model。")
    calib_s = str(calib_dir.expanduser().resolve())
    for part in parts:
        sub_cfg = copy.deepcopy(quant_cfg)
        if part == "vit":
            wrap = _vit_from_pi05(m)
            dl = wrap.get_calibrate_dataset(calib_s)
            print(colored(f"[ptq] quantize vit …", "cyan"), flush=True)
            quantize_model(wrap, sub_cfg, dl)
            set_dynamic_quant(wrap, "bf16")
        elif part == "llm":
            wrap = _llm_from_pi05(m)
            dl = wrap.get_calibrate_dataset(calib_s)
            print(colored(f"[ptq] quantize llm …", "cyan"), flush=True)
            quantize_model(wrap, sub_cfg, dl)
            set_dynamic_quant(wrap, "bf16")
        elif part == "expert":
            wrap = _expert_from_pi05(m)
            dl = wrap.get_calibrate_dataset(calib_s)
            print(colored(f"[ptq] quantize expert …", "cyan"), flush=True)
            quantize_model(wrap, sub_cfg, dl)
            set_dynamic_quant(wrap, "bf16")
        elif part == "denoise":
            wrap = _denoise_from_pi05(m)
            dl = wrap.get_calibrate_dataset(calib_s)
            print(colored(f"[ptq] quantize denoise (Pi05DenoiseStep) …", "cyan"), flush=True)
            quantize_model(wrap, sub_cfg, dl)
            # 与 Pi05DenoiseStep.quantize 一致（export/TRT 侧常用 fp16）
            set_dynamic_quant(wrap, "fp16")
        else:
            raise ValueError(f"unknown ptq part: {part!r}")
    print(colored("[ptq] 选择性量化完成", "green"), flush=True)


def _try_is_quantized_linear(mod: nn.Module) -> bool:
    try:
        from modelopt.torch.quantization.utils import is_quantized_linear

        return bool(is_quantized_linear(mod))
    except Exception:
        n = type(mod).__name__
        return "Quant" in n and "Linear" in n


def collect_quant_linear_names(root: nn.Module, parts: tuple[str, ...]) -> list[str]:
    prefixes: list[str] = []
    for p in parts:
        if p not in _PART_PREFIXES:
            continue
        prefixes.extend(_PART_PREFIXES[p])
    names: list[str] = []
    for name, mod in root.named_modules():
        if not name or not _try_is_quantized_linear(mod):
            continue
        if any(name.startswith(pref) for pref in prefixes):
            names.append(name)
    return sorted(names)


def get_submodule(mod: nn.Module, target: str) -> nn.Module:
    if hasattr(mod, "get_submodule"):
        return mod.get_submodule(target)
    cur: nn.Module = mod
    for part in target.split("."):
        cur = getattr(cur, part)
    return cur


def _output_hook(storage: dict[str, torch.Tensor], key: str):
    def hook(_module: nn.Module, _inp: Any, out: Any) -> None:
        x = out[0] if isinstance(out, tuple) else out
        if not isinstance(x, torch.Tensor):
            return
        storage[key] = x.detach().float().cpu()

    return hook


def _subsample_fp_flat(t: torch.Tensor, max_elems: int) -> np.ndarray:
    x = t.detach().float().cpu().reshape(-1)
    n = int(x.numel())
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if n <= max_elems:
        return x.numpy().astype(np.float64, copy=False)
    idx = (torch.arange(max_elems) * float(n - 1) / float(max_elems - 1)).long().clamp(0, n - 1)
    return x[idx].numpy().astype(np.float64, copy=False)


def _merge_welford(
    n_a: int, mean_a: float, m2_a: float, n_b: int, mean_b: float, m2_b: float
) -> tuple[int, float, float]:
    if n_b <= 0:
        return n_a, mean_a, m2_a
    if n_a <= 0:
        return n_b, mean_b, m2_b
    n = n_a + n_b
    delta = mean_b - mean_a
    mean = mean_a + delta * (n_b / n)
    m2 = m2_a + m2_b + (delta * delta) * (n_a * n_b / n)
    return n, mean, m2


def _init_fp_hist_state(bins: int, max_elems: int) -> dict[str, Any]:
    return {
        "bins": int(bins),
        "max_elems": int(max_elems),
        "edges": None,
        "counts": None,
        "under": 0,
        "over": 0,
        "n_stat": 0,
        "mean": 0.0,
        "m2": 0.0,
        "vmin": float("inf"),
        "vmax": float("-inf"),
    }


def _fp_hist_add_tensor(st: dict[str, Any], t: torch.Tensor) -> None:
    x_np = _subsample_fp_flat(t, st["max_elems"])
    if x_np.size == 0:
        return
    st["vmin"] = min(st["vmin"], float(x_np.min()))
    st["vmax"] = max(st["vmax"], float(x_np.max()))
    bn = int(x_np.size)
    bm = float(x_np.mean())
    b_m2 = float(np.square(x_np - bm).sum())
    st["n_stat"], st["mean"], st["m2"] = _merge_welford(
        int(st["n_stat"]), float(st["mean"]), float(st["m2"]), bn, bm, b_m2
    )

    if st["edges"] is None:
        lo, hi = np.percentile(x_np, [0.1, 99.9])
        lo_f, hi_f = float(lo), float(hi)
        if not np.isfinite(lo_f) or not np.isfinite(hi_f):
            lo_f, hi_f = float(np.nanmin(x_np)), float(np.nanmax(x_np))
        if hi_f <= lo_f:
            lo_f -= 1.0
            hi_f += 1.0
        span = hi_f - lo_f
        pad = max(span * 0.05, 1e-6)
        lo_f -= pad
        hi_f += pad
        st["edges"] = np.linspace(lo_f, hi_f, st["bins"] + 1)
        st["counts"] = np.zeros(st["bins"], dtype=np.int64)

    edges = st["edges"]
    e0, e1 = float(edges[0]), float(edges[-1])
    st["under"] += int((x_np < e0).sum())
    st["over"] += int((x_np > e1).sum())
    m = (x_np >= e0) & (x_np <= e1)
    if np.any(m):
        c = np.histogram(x_np[m], bins=edges)[0]
        st["counts"] = st["counts"] + c.astype(np.int64, copy=False)


def _fp_hist_to_json(st: dict[str, Any]) -> dict[str, Any]:
    if st.get("edges") is None or st.get("counts") is None:
        return {}
    n_s = int(st["n_stat"])
    var = float(st["m2"] / n_s) if n_s > 0 else 0.0
    std = float(np.sqrt(var)) if var > 0 else 0.0
    edges = st["edges"]
    counts = st["counts"]
    return {
        "bin_edges": [float(x) for x in edges],
        "counts": [int(x) for x in counts],
        "underflow": int(st["under"]),
        "overflow": int(st["over"]),
        "subsample_max_elems": int(st["max_elems"]),
        "stats": {
            "n": n_s,
            "min": float(st["vmin"]) if st["vmin"] != float("inf") else 0.0,
            "max": float(st["vmax"]) if st["vmax"] != float("-inf") else 0.0,
            "mean": float(st["mean"]),
            "std": std,
        },
    }


def write_ptq_layer_report(
    policy_fp: Any,
    policy_ptq: Any,
    parts: tuple[str, ...],
    *,
    dataset: Any,
    repack_fn: Any,
    start_index: int,
    num_samples: int,
    report_path: Path,
    include_activation_histogram: bool = True,
    hist_bins: int = 40,
    hist_max_elems: int = 100_000,
) -> None:
    """在若干条样本上对比 FP 与 PTQ 对应量化层的输出（MSE/MAE/max_abs）。

    可选附带每层 **FP 路径**上该模块输出的 subsample 直方图：bin 边界由该层**首次**见到的 subsample
    的 0.1%–99.9% 分位扩展后固定，跨样本累加 counts；underflow/overflow 单独计数便于看长尾。
    """
    fp_m = unwrap_pytorch_pi05_model(policy_fp)
    q_m = unwrap_pytorch_pi05_model(policy_ptq)
    if fp_m is None or q_m is None:
        raise RuntimeError("layer report: 无法解析 policy 底层模型")

    names = collect_quant_linear_names(q_m, parts)
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not names:
        payload = {
            "parts": list(parts),
            "layers": [],
            "note": "未在 PTQ 模型中发现 QuantLinear（检查 quant_cfg 与 parts）",
        }
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(colored(f"[ptq] layer report（空）→ {report_path}", "yellow"), flush=True)
        return

    acc = {n: {"mse": 0.0, "mae": 0.0, "max_abs": 0.0, "n": 0} for n in names}
    hist_states: dict[str, dict[str, Any]] | None = None
    if include_activation_histogram:
        hb = max(4, int(hist_bins))
        hm = max(1024, int(hist_max_elems))
        hist_states = {n: _init_fp_hist_state(hb, hm) for n in names}
    n_ds = len(dataset)
    end_i = min(start_index + max(0, int(num_samples)), n_ds)
    used = 0

    for idx in range(start_index, end_i):
        raw = tree_to_numpy(dataset[idx])
        packed = repack_fn(dict(raw))
        if "actions" not in packed:
            continue
        obs = {k: v for k, v in packed.items() if k != "actions"}

        storage_fp: dict[str, torch.Tensor] = {}
        storage_q: dict[str, torch.Tensor] = {}
        hooks_fp: list[Any] = []
        hooks_q: list[Any] = []
        for name in names:
            try:
                mf = get_submodule(fp_m, name)
                mq = get_submodule(q_m, name)
            except Exception as exc:
                logger.warning("layer report skip %s: %s", name, exc)
                continue
            hooks_fp.append(mf.register_forward_hook(_output_hook(storage_fp, name)))
            hooks_q.append(mq.register_forward_hook(_output_hook(storage_q, name)))

        try:
            policy_fp.infer(obs)
            snap_fp = {k: v.clone() for k, v in storage_fp.items()}
            storage_q.clear()
            policy_ptq.infer(obs)
            snap_q = {k: v.clone() for k, v in storage_q.items()}
        finally:
            for h in hooks_fp:
                h.remove()
            for h in hooks_q:
                h.remove()

        for name in names:
            if name not in snap_fp or name not in snap_q:
                continue
            a, b = snap_fp[name], snap_q[name]
            if a.shape != b.shape:
                logger.warning("layer report shape mismatch %s %s vs %s", name, a.shape, b.shape)
                continue
            d = a - b
            acc[name]["mse"] += float(d.pow(2).mean())
            acc[name]["mae"] += float(d.abs().mean())
            acc[name]["max_abs"] += float(d.abs().max())
            acc[name]["n"] += 1
            if hist_states is not None and name in hist_states:
                try:
                    _fp_hist_add_tensor(hist_states[name], a)
                except Exception as exc:  # pragma: no cover
                    logger.warning("layer report histogram skip %s: %s", name, exc)
        used += 1

    rows: list[dict[str, Any]] = []
    for name, s in acc.items():
        c = int(s["n"])
        if c <= 0:
            continue
        row: dict[str, Any] = {
            "module": name,
            "mse_mean": s["mse"] / c,
            "mae_mean": s["mae"] / c,
            "max_abs_mean": s["max_abs"] / c,
            "samples": c,
        }
        if hist_states is not None and name in hist_states:
            hj = _fp_hist_to_json(hist_states[name])
            if hj:
                row["fp_activation_histogram"] = hj
        rows.append(row)
    rows.sort(key=lambda r: float(r["mse_mean"]), reverse=True)

    payload: dict[str, Any] = {
        "parts": list(parts),
        "indices_start": int(start_index),
        "indices_used": int(used),
        "layer_count": len(rows),
        "layers": rows,
    }
    if include_activation_histogram:
        payload["histogram_note"] = (
            "fp_activation_histogram：每层 FP 输出的 subsample 线性直方图；"
            "bin 边界由该层首次 subsample 的 0.1%–99.9% 分位扩展后固定；"
            "underflow/overflow 统计落在边界外的值，便于看异常幅值。"
        )
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(colored(f"[ptq] layer report → {report_path}（{len(rows)} layers）", "green"), flush=True)
