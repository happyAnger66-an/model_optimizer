"""Pi0.5 选择性 PTQ（Vit / LLM / Expert）与分层输出误差分析（WebUI / eval 管线）。"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from termcolor import colored

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
}


def load_ptq_quant_cfg(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    with open(path, encoding="utf-8") as f:
        cfg: dict[str, Any] = json.load(f)
    cfg = copy.deepcopy(cfg)
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
) -> None:
    """在若干条样本上对比 FP 与 PTQ 对应量化层的输出（MSE/MAE/max_abs）。"""
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
        used += 1

    rows: list[dict[str, Any]] = []
    for name, s in acc.items():
        c = int(s["n"])
        if c <= 0:
            continue
        rows.append(
            {
                "module": name,
                "mse_mean": s["mse"] / c,
                "mae_mean": s["mae"] / c,
                "max_abs_mean": s["max_abs"] / c,
                "samples": c,
            }
        )
    rows.sort(key=lambda r: float(r["mse_mean"]), reverse=True)

    payload = {
        "parts": list(parts),
        "indices_start": int(start_index),
        "indices_used": int(used),
        "layer_count": len(rows),
        "layers": rows,
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(colored(f"[ptq] layer report → {report_path}（{len(rows)} layers）", "green"), flush=True)
