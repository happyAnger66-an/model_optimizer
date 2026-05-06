# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Quantization utilities for TensorRT Edge-LLM.

This module provides core quantization functionality using NVIDIA ModelOpt.
"""

from typing import Any, Callable, Dict, Optional

import modelopt.torch.quantization as mtq
import torch
from modelopt.torch.quantization.nn import TensorQuantizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def _calib_batch_to_model_device_dtype(
    model: torch.nn.Module, batch: Dict[str, Any]
) -> Dict[str, Any]:
    """将校准 batch 移到 ``model`` 所在设备，并把 **浮点** 张量对齐到模型参数 dtype。

    ``Pi05`` 的 ``LLM.quantize`` 把 **内部** ``GemmaModel``（``self.model``）交给 ``mtq.quantize``，
    校准循环里调用的是 ``model(**data)``，**不会**经过 ``LLM.forward`` 里对 ``inputs_embeds`` 的
    dtype 对齐。AWQ 族标定会把解码器暂转 ``float32``，而磁盘上的 ``inputs_embeds`` 常为
    ``bfloat16``，否则会在 ``F.linear`` 等处出现 ``mat1`` 与 ``mat2`` dtype 不一致。
    """
    try:
        device = model.device  # type: ignore[attr-defined]
    except Exception:
        device = next(model.parameters()).device
    try:
        param_dtype = next(model.parameters()).dtype
    except StopIteration:
        param_dtype = None

    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor = value.to(device)
            if (
                param_dtype is not None
                and value.is_floating_point()
                and param_dtype.is_floating_point
            ):
                tensor = tensor.to(param_dtype)
            out[key] = tensor
        else:
            out[key] = value
    return out


def _calib_batch_to_device_only(model: torch.nn.Module, batch: Dict[str, Any]) -> Dict[str, Any]:
    """将校准 batch 中张量移到 ``model`` 所在设备，**不改变 dtype**。

    用于 ``forward_context`` + 子模块量化的路径：``Pi05DenoiseStep`` 上 ``action_in_proj`` 等常为
    ``float32``，而 ``gemma_expert`` 为 ``bfloat16``。若用 :func:`_calib_batch_to_model_device_dtype`
    按 ``next(model.parameters())`` 统一 cast，易把 ``x_t`` 转成 bf16 导致与 fp32 权重 ``mat1/mat2``
    dtype 不一致。
    """
    try:
        device = model.device  # type: ignore[attr-defined]
    except Exception:
        device = next(model.parameters()).device
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _tensor_qdq_error_analysis(
    model: torch.nn.Module,
    calib_dataloader: DataLoader,
    forward_batch: Callable[[torch.nn.Module, Any], None],
    *,
    desc: str = "QDQ error analysis",
) -> Dict[str, Dict[str, float]]:
    """Run the same calibration data through the model and aggregate tensor-level QDQ error per TensorQuantizer.

    For each :class:`TensorQuantizer` forward, compares the module output to its input (first argument).
    If ``pre_quant_scale`` / Hadamard rotation / block reshape is used inside the quantizer, this
    measures the full effect of that submodule (not only the inner ``_fake_quantize`` kernel).

    Args:
        model: Quantized model (after ``mtq.quantize``).
        calib_dataloader: Same loader as calibration.
        forward_batch: ``(model, batch) -> None`` — one forward pass for one batch.
        desc: Progress bar description.

    Returns:
        Mapping from quantizer module name to metrics:
        ``rmse``, ``mae``, ``global_rel_l2``, ``max_batch_rel_l2``, ``numel``, ``num_forwards``.
    """
    stats: Dict[str, Dict[str, float]] = {}
    hooks = []

    def _ensure(name: str) -> Dict[str, float]:
        if name not in stats:
            stats[name] = {
                "numel": 0.0,
                "sum_sq": 0.0,
                "sum_abs": 0.0,
                "sum_x_norm_sq": 0.0,
                "max_batch_rel_l2": 0.0,
                "num_forwards": 0.0,
            }
        return stats[name]

    @torch.no_grad()
    def _hook(name: str):
        def fn(module: TensorQuantizer, inp: tuple, out: torch.Tensor) -> None:
            if module._disabled or not module._if_quant:
                return
            x = inp[0]
            if not isinstance(x, torch.Tensor) or not isinstance(out, torch.Tensor):
                return
            if x.numel() == 0 or out.numel() == 0:
                return
            if x.shape != out.shape:
                return
            s = _ensure(name)
            x_f = x.detach().float()
            out_f = out.detach().float()
            diff = out_f - x_f
            n = float(x_f.numel())
            s["numel"] += n
            s["sum_sq"] += float((diff * diff).sum().item())
            s["sum_abs"] += float(diff.abs().sum().item())
            s["sum_x_norm_sq"] += float((x_f * x_f).sum().item())
            x_norm = float(x_f.norm().item())
            if x_norm > 0.0:
                s["max_batch_rel_l2"] = max(
                    s["max_batch_rel_l2"], float(diff.norm().item()) / x_norm
                )
            s["num_forwards"] += 1.0

        return fn

    for qname, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            hooks.append(module.register_forward_hook(_hook(qname)))

    try:
        try:
            n = len(calib_dataloader)
        except TypeError:
            n = None
        pbar = tqdm(calib_dataloader, total=n, desc=desc, unit="num_samples")
        for data in pbar:
            forward_batch(model, data)
    finally:
        for h in hooks:
            h.remove()

    report: Dict[str, Dict[str, float]] = {}
    for name, s in stats.items():
        numel = s["numel"]
        if numel <= 0.0:
            continue
        sum_sq = s["sum_sq"]
        sum_x2 = s["sum_x_norm_sq"]
        report[name] = {
            "rmse": (sum_sq / numel) ** 0.5,
            "mae": s["sum_abs"] / numel,
            "global_rel_l2": (sum_sq**0.5) / (sum_x2**0.5 + 1e-12),
            "max_batch_rel_l2": s["max_batch_rel_l2"],
            "numel": numel,
            "num_forwards": s["num_forwards"],
        }
    return report


def _print_qdq_report(report: Dict[str, Dict[str, float]], *, top_k: int = 32) -> None:
    if not report:
        print("QDQ error analysis: no TensorQuantizer hooks collected statistics (empty report).")
        return
    ranked = sorted(report.items(), key=lambda kv: kv[1]["rmse"], reverse=True)
    print(
        f"QDQ error analysis: {len(report)} quantizer(s); "
        f"showing top {min(top_k, len(ranked))} by RMSE (tensor I/O, post-calibration fake quant)."
    )
    print(
        f"{'quantizer':<72} {'rmse':>12} {'mae':>12} {'g_rel_L2':>10} {'max_b_rel':>10}"
    )
    for name, m in ranked[:top_k]:
        short = name if len(name) <= 72 else name[:33] + "..." + name[-34:]
        print(
            f"{short:<72} {m['rmse']:12.6g} {m['mae']:12.6g} "
            f"{m['global_rel_l2']:10.6g} {m['max_batch_rel_l2']:10.6g}"
        )


def quant_config_targets_hf_bmm_kv(quant_cfg: Dict[str, Any]) -> bool:
    """True if ``quant_cfg`` requests HF attention BMM quantizers (e.g. :class:`FP8_KV_CFG`).

    ModelOpt registers ``*_bmm_quantizer`` hooks when the **root** passed to ``mtq.quantize``
    is a HuggingFace ``PreTrainedModel``. Wrappers such as :class:`Pi05DenoiseStep` must then
    pass ``gemma_expert`` as the quantize root and run calibration via the wrapper ``forward``.
    """
    qc = quant_cfg.get("quant_cfg")
    if not isinstance(qc, dict):
        return False
    return any("bmm_quantizer" in str(k) for k in qc)


def enable_huggingface_checkpointing_patch() -> None:
    from modelopt.torch.opt.plugins.huggingface import (
        _LIBRARY_CLASSES_FOR_PATCHING, _PATCHED_CLASSES,
        patch_pretrained_methods)
    """Enables automatic save/restore of ModelOpt state with HuggingFace checkpointing APIs.
    This is adapted from TensorRT Model Optimizer: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/0.37.0/modelopt/torch/opt/plugins/huggingface.py#L127
    Edge-LLM finds that _from_config() should not be patched.

    """
    for name, (classes, methods_list) in _LIBRARY_CLASSES_FOR_PATCHING.items():
        for cls, patch_methods in zip(classes, methods_list):
            if cls in _PATCHED_CLASSES:
                continue
            patch_methods = [
                method for method in patch_methods
                if method[0] != "_from_config"
            ]  # Edge-LLM finds that _from_config() should not be patched.
            patch_pretrained_methods(cls, patch_methods)
            _PATCHED_CLASSES.add(cls)
        print(f"ModelOpt save/restore enabled for `{name}` library.")


def quantize_model(
    model: torch.nn.Module,
    quant_config: Dict[str, Any],
    calib_dataloader: DataLoader,
    *,
    forward_context: Optional[torch.nn.Module] = None,
    measure_quant_error: bool = False,
    qdq_error_report: Optional[Dict[str, Dict[str, float]]] = None,
) -> torch.nn.Module:
    """
    Quantize a PyTorch model using the specified configuration and calibration data.
    
    Args:
        model: PyTorch module passed to ``mtq.quantize`` (quantization root).
        quant_config: Quantization configuration dictionary
        calib_dataloader: DataLoader for calibration data
        forward_context: If set, calibration and QDQ analysis move tensors with
            ``_calib_batch_to_model_device_dtype(forward_context, ...)`` and invoke
            ``forward_context`` instead of ``model(**data)``. Use this when ``model`` is an
            HF decoder subtree (e.g. ``gemma_expert``) for ``FP8_KV_CFG``, while the runnable
            forward lives on a wrapper (e.g. :class:`Pi05DenoiseStep`).
        measure_quant_error: If True, after ``mtq.quantize`` runs an extra pass over
            ``calib_dataloader`` and prints per-``TensorQuantizer`` tensor I/O QDQ metrics.
        qdq_error_report: If not ``None``, filled with the same metrics returned by
            :func:`_tensor_qdq_error_analysis` (only when ``measure_quant_error`` is True).

    Returns:
        Quantized PyTorch model
    """

    def _phi4mm_kwargs(m: torch.nn.Module) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if hasattr(m, "config") and "phi4mm" in getattr(
                m.config, "model_type", "").lower():
            kwargs["input_mode"] = 0
            kwargs["use_cache"] = False
        return kwargs

    ctx = forward_context if forward_context is not None else model

    def _run_calib_forward(qm: torch.nn.Module, batch: Any) -> None:
        """Run one calibration forward; ``qm`` is the quantize root (may equal ``ctx``)."""
        kwargs = _phi4mm_kwargs(qm)
        if isinstance(batch, dict):
            if forward_context is not None:
                batch = _calib_batch_to_device_only(ctx, batch)
            else:
                batch = _calib_batch_to_model_device_dtype(ctx, batch)
            if forward_context is not None:
                # Pi05DenoiseStep.forward: positional only (matches calib collector keys).
                forward_context(
                    batch["prefix_pad_masks"],
                    batch["past_keys"],
                    batch["past_values"],
                    batch["x_t"],
                    batch["timestep"],
                )
            else:
                qm(**batch, **kwargs)
        else:
            batch = batch.to(ctx.device)  # type: ignore[attr-defined]
            qm(batch, **kwargs)

    # Define calibration loop
    def calibrate_loop(qmodel: torch.nn.Module) -> None:
        """
        Calibration loop that adjusts weights and scaling factors.
        
        Args:
            qmodel: Model to calibrate
        """
        try:
            n = len(calib_dataloader)
        except TypeError:
            n = None
        if n is not None:
            print(f"Calibrating model on {n} samples...")
        else:
            print("Calibrating model (streaming calibration data, total unknown)...")
        pbar = tqdm(calib_dataloader, total=n, desc="Calibrating", unit="num_samples")

        for data in pbar:
            _run_calib_forward(qmodel, data)

    def analysis_forward_batch(m: torch.nn.Module, data: Any) -> None:
        _run_calib_forward(m, data)

    # Get quantization config and perform quantization
    mtq.quantize(model, quant_config, forward_loop=calibrate_loop)
    mtq.print_quant_summary(model)
    if measure_quant_error:
        rep = _tensor_qdq_error_analysis(
            model,
            calib_dataloader,
            analysis_forward_batch,
            desc="QDQ error (analysis)",
        )
        _print_qdq_report(rep)
        if qdq_error_report is not None:
            qdq_error_report.clear()
            qdq_error_report.update(rep)
    return model


def quantize_draft_model(
    base_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    quant_config: Dict[str, Any],
    calib_dataloader: DataLoader,
    *,
    measure_quant_error: bool = False,
    qdq_error_report: Optional[Dict[str, Dict[str, float]]] = None,
) -> torch.nn.Module:
    """
    Quantize a PyTorch model using the specified configuration and calibration data.
    
    Args:
        base_model: Base model which is used to generate inputs for the draft model.
        draft_model: The draft model to quantize
        quant_config: Quantization configuration dictionary
        calib_dataloader: DataLoader for calibration data
        measure_quant_error: If True, after ``mtq.quantize`` runs an extra analysis pass
            (same as :func:`quantize_model`).
        qdq_error_report: Optional dict filled when ``measure_quant_error`` is True.

    Returns:
        Quantized PyTorch model
    """

    def _draft_forward_batch(dm: torch.nn.Module, data: Any) -> None:
        assert base_model.device == dm.device, "Base model and draft model must be on the same device"
        if isinstance(data, dict):
            batch = {k: v.to(dm.device) for k, v in data.items()}
            outputs = base_model(**batch, output_hidden_states=True)
            ids_for_draft = batch
        else:
            batch_tensor = data.to(base_model.device)
            outputs = base_model(batch_tensor, output_hidden_states=True)
            ids_for_draft = batch_tensor
        all_hidden_states = outputs["hidden_states"]
        idx = [
            2,
            ((len(all_hidden_states) - 1) // 2),
            len(all_hidden_states) - 4,
        ]
        hidden_states_0 = all_hidden_states[idx[0]]
        hidden_states_1 = all_hidden_states[idx[1]]
        hidden_states_2 = all_hidden_states[idx[2]]
        hidden_states = torch.cat(
            [hidden_states_0, hidden_states_1, hidden_states_2], dim=-1
        )
        hidden_states_from_draft = torch.zeros_like(hidden_states_0)
        dm.quant_forward(hidden_states, hidden_states_from_draft, input_ids=ids_for_draft)

    # Define calibration loop
    def calibrate_loop(draft_model: torch.nn.Module) -> None:
        """
        Calibration loop that adjusts weights and scaling factors.
        
        Args:
            draft_model: Model to calibrate
        """
        try:
            n = len(calib_dataloader)
        except TypeError:
            n = None
        if n is not None:
            print(f"Calibrating model on {n} samples...")
        else:
            print("Calibrating model (streaming calibration data, total unknown)...")
        pbar = tqdm(calib_dataloader, total=n, desc="Calibrating", unit="num_samples")
        assert base_model.device == draft_model.device, "Base model and draft model must be on the same device"

        for data in pbar:
            _draft_forward_batch(draft_model, data)

    # Get quantization config and perform quantization
    mtq.quantize(draft_model, quant_config, forward_loop=calibrate_loop)
    mtq.print_quant_summary(draft_model)
    if measure_quant_error:
        rep = _tensor_qdq_error_analysis(
            draft_model,
            calib_dataloader,
            _draft_forward_batch,
            desc="QDQ error (analysis)",
        )
        _print_qdq_report(rep)
        if qdq_error_report is not None:
            qdq_error_report.clear()
            qdq_error_report.update(rep)
    return draft_model
