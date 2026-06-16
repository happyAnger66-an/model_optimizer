"""分阶段 wall-time 统计：可嵌套 key、按 step 展开、统一汇总行格式。

Key 约定（点分路径，便于分组打印）::

    policy.infer          # 整段 ``Policy.infer`` wall time（≈ predict_ms − policy.align）
    policy.preprocess     # input_transform + tensorize + H2D + Observation.from_dict + noise
    policy.preprocess.input_transform
    policy.preprocess.numpy_to_torch
    policy.preprocess.h2d
    policy.preprocess.noise_h2d
    policy.preprocess.observation_from_dict
    policy.preprocess.observation_image
    policy.preprocess.observation_pack
    policy.postprocess    # actions/state D2H + output_transform
    policy.postprocess.actions_d2h
    policy.postprocess.state_d2h
    policy.postprocess.state_cpu_reuse
    policy.postprocess.output_transform
    policy.align          # webui ``align_action_dim``（在 backend.predict 内）
    sample_actions        # 整段 ``PI0Pytorch.sample_actions`` wall time（模型纯推理）
    policy.sample_actions.cuda_sync
    embed_prefix          # 单次 sample 的 embed_prefix wall time
    embed_prefix.vision   # SigLIP / TRT ViT / FlashRT vision 段
    embed_prefix.lang_embedding  # 语言 token lookup（FP8/FP16）
    embed_prefix.lang_scale      # × sqrt(H) 缩放
    prefix_llm            # prefix KV 前向
    flashrt.setup         # 首次 backend / setup_prompt
    denoise.total         # 整段 backend.run()（10 步扩散一次跑完）
    denoise.step.0        # pipeline 内第 0 步（flow-matching step）
    denoise.step.9        # 第 9 步

汇总输出与 webui ``[summary:engine]`` 对齐::

    [summary:engine] denoise.total   n=38 mean=...
    [summary:engine] denoise.step    n=380 mean=...   # 所有 step 样本池化
    [summary:engine] denoise.step.0  n=38 mean=...
"""

from __future__ import annotations

import re
import time
import types
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# 默认打印顺序：Policy 外壳 → 模型整段 → 子阶段 → denoise。
KEY_POLICY_INFER = "policy.infer"
KEY_POLICY_PREPROCESS = "policy.preprocess"
KEY_POLICY_PREPROCESS_INPUT_TRANSFORM = "policy.preprocess.input_transform"
KEY_POLICY_PREPROCESS_NUMPY_TO_TORCH = "policy.preprocess.numpy_to_torch"
KEY_POLICY_PREPROCESS_H2D = "policy.preprocess.h2d"
KEY_POLICY_PREPROCESS_NOISE_H2D = "policy.preprocess.noise_h2d"
KEY_POLICY_PREPROCESS_OBSERVATION = "policy.preprocess.observation_from_dict"
KEY_POLICY_PREPROCESS_OBSERVATION_IMAGE = "policy.preprocess.observation_image"
KEY_POLICY_PREPROCESS_OBSERVATION_PACK = "policy.preprocess.observation_pack"
KEY_POLICY_POSTPROCESS = "policy.postprocess"
KEY_POLICY_POSTPROCESS_ACTIONS_D2H = "policy.postprocess.actions_d2h"
KEY_POLICY_POSTPROCESS_STATE_D2H = "policy.postprocess.state_d2h"
KEY_POLICY_POSTPROCESS_STATE_CPU_REUSE = "policy.postprocess.state_cpu_reuse"
KEY_POLICY_POSTPROCESS_OUTPUT_TRANSFORM = "policy.postprocess.output_transform"
KEY_POLICY_ALIGN = "policy.align"
KEY_SAMPLE_ACTIONS = "sample_actions"
KEY_POLICY_SAMPLE_ACTIONS_CUDA_SYNC = "policy.sample_actions.cuda_sync"
_POLICY_SUMMARY_PREFIX = "[summary:policy]"
_DEFAULT_SUMMARY_ORDER: tuple[str, ...] = (
    KEY_POLICY_INFER,
    KEY_POLICY_PREPROCESS,
    KEY_POLICY_PREPROCESS_INPUT_TRANSFORM,
    KEY_POLICY_PREPROCESS_NUMPY_TO_TORCH,
    KEY_POLICY_PREPROCESS_H2D,
    KEY_POLICY_PREPROCESS_NOISE_H2D,
    KEY_POLICY_PREPROCESS_OBSERVATION,
    KEY_POLICY_PREPROCESS_OBSERVATION_IMAGE,
    KEY_POLICY_PREPROCESS_OBSERVATION_PACK,
    KEY_POLICY_POSTPROCESS,
    KEY_POLICY_POSTPROCESS_ACTIONS_D2H,
    KEY_POLICY_POSTPROCESS_STATE_D2H,
    KEY_POLICY_POSTPROCESS_STATE_CPU_REUSE,
    KEY_POLICY_POSTPROCESS_OUTPUT_TRANSFORM,
    KEY_POLICY_ALIGN,
    KEY_SAMPLE_ACTIONS,
    KEY_POLICY_SAMPLE_ACTIONS_CUDA_SYNC,
    "embed_prefix",
    "embed_prefix.vision",
    "embed_prefix.lang_embedding",
    "embed_prefix.lang_scale",
    "prefix_llm",
    "flashrt.setup",
    "denoise.calibrate",
    "denoise.total",
    "denoise.step",
)


def perf_line_ms(values: Sequence[float]) -> str:
    """毫秒列表 → ``n=.. mean=.. p50=.. p90=.. p99=.. ms``。"""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return "n=0"
    return (
        f"n={int(arr.size)} mean={float(np.mean(arr)):.2f} "
        f"p50={float(np.percentile(arr, 50)):.2f} "
        f"p90={float(np.percentile(arr, 90)):.2f} "
        f"p99={float(np.percentile(arr, 99)):.2f} ms"
    )


def _sync_cuda_value(value: Any) -> float | None:
    """Synchronize the CUDA device that owns ``value`` and return sync time in ms."""

    try:
        import torch
    except ImportError:
        return None
    device = None
    if hasattr(value, "is_cuda") and bool(value.is_cuda):
        device = value.device
    elif isinstance(value, dict):
        for item in value.values():
            dt = _sync_cuda_value(item)
            if dt is not None:
                return dt
        return None
    elif isinstance(value, (list, tuple)):
        for item in value:
            dt = _sync_cuda_value(item)
            if dt is not None:
                return dt
        return None
    if device is None or not torch.cuda.is_available():
        return None
    t0 = time.perf_counter()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) * 1000.0


def _prompt_cache_for_policy(policy: Any) -> dict[Any, dict[str, Any]]:
    cache = getattr(policy, "_mopt_prompt_token_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(policy, "_mopt_prompt_token_cache", cache)
    return cache


def _prompt_to_cache_key(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    try:
        return str(prompt.item())
    except Exception:
        return str(prompt)


def _clone_cached_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clone_cached_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_cached_tree(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _apply_input_transform_with_prompt_cache(policy: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    """Apply ``policy._input_transform`` while caching prompt tokenization when safe."""

    transform = getattr(policy, "_input_transform", None)
    transforms = getattr(transform, "transforms", None)
    if not transforms:
        return transform(inputs)

    data = inputs
    cache = _prompt_cache_for_policy(policy)
    for tr in transforms:
        # TokenizePrompt depends only on prompt unless discrete_state_input=True.
        if (
            tr.__class__.__name__ == "TokenizePrompt"
            and not bool(getattr(tr, "discrete_state_input", False))
            and "prompt" in data
        ):
            prompt_key = _prompt_to_cache_key(data["prompt"])
            cache_key = (id(tr), prompt_key)
            prompt = data.pop("prompt")
            cached = cache.get(cache_key)
            if cached is None:
                tokenized = tr({"prompt": prompt})
                cached = {
                    "tokenized_prompt": np.asarray(tokenized["tokenized_prompt"]).copy(),
                    "tokenized_prompt_mask": np.asarray(tokenized["tokenized_prompt_mask"]).copy(),
                }
                cache[cache_key] = cached
            data = {**data, **_clone_cached_tree(cached)}
            continue
        data = tr(data)
    return data


def _observation_from_torch_dict_fast(data: dict[str, Any], observation_cls: Any, collector: StagePerfCollector) -> Any:
    t_img0 = time.perf_counter()
    for key in data["image"]:
        image = data["image"][key]
        if image.dtype == np.uint8:
            data["image"][key] = image.astype(np.float32) / 255.0 * 2.0 - 1.0
        elif hasattr(image, "dtype"):
            try:
                import torch
            except ImportError:
                torch = None  # type: ignore[assignment]
            if torch is not None and image.dtype == torch.uint8:
                data["image"][key] = (
                    image.to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
                )
    collector.record(
        KEY_POLICY_PREPROCESS_OBSERVATION_IMAGE,
        (time.perf_counter() - t_img0) * 1000.0,
    )

    t_pack0 = time.perf_counter()
    observation = observation_cls(
        images=data["image"],
        image_masks=data["image_mask"],
        state=data["state"],
        tokenized_prompt=data.get("tokenized_prompt"),
        tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
        token_ar_mask=data.get("token_ar_mask"),
        token_loss_mask=data.get("token_loss_mask"),
    )
    collector.record(
        KEY_POLICY_PREPROCESS_OBSERVATION_PACK,
        (time.perf_counter() - t_pack0) * 1000.0,
    )
    return observation


def wrap_sample_actions_with_stage_perf(
    model: Any,
    collector: StagePerfCollector,
    *,
    warmup_skips: int = 0,
) -> None:
    """在 ``model.sample_actions`` 最外层打补丁，统计整段 wall time（不修改 openpi 源码）。

    须在 TRT / native / FlashRT 等所有对 ``sample_actions`` 的替换**之后**调用，并配合
    ``Policy._sample_actions`` 刷新（见各 executor 的 ``_sync_policy_sample_actions_ref``）。

    可重复调用：若已有本模块包装且中间层被整体替换（如 FlashRT），会对**当前**
    ``sample_actions`` 重新包装，避免计时落在已失效的旧 callable 上。
    """
    _orig = model.sample_actions
    if getattr(model, "_mopt_sample_actions_stage_wrapped", False):
        inner = getattr(model, "_mopt_sample_actions_stage_inner", None)
        # 外层实现已被替换（例如 FlashRT 覆盖）时，对新的 callable 重新包装。
        if inner is not None and _orig is inner:
            return
    skips = max(0, int(warmup_skips))
    state = {"invocation": 0}

    def _sa(self, device, observation, noise=None, num_steps=10):
        state["invocation"] += 1
        record = collector.enabled and state["invocation"] > skips
        t0 = time.perf_counter()
        try:
            result = _orig(device, observation, noise=noise, num_steps=num_steps)
            if record:
                sync_ms = _sync_cuda_value(result)
                if sync_ms is not None:
                    collector.record(KEY_POLICY_SAMPLE_ACTIONS_CUDA_SYNC, sync_ms)
            return result
        finally:
            if record:
                collector.record(
                    KEY_SAMPLE_ACTIONS,
                    (time.perf_counter() - t0) * 1000.0,
                )

    model.sample_actions = types.MethodType(_sa, model)
    model._mopt_sample_actions_stage_wrapped = True
    model._mopt_sample_actions_stage_inner = _orig


def wrap_policy_infer_with_stage_perf(
    policy: Any,
    collector: StagePerfCollector,
    *,
    warmup_skips: int = 0,
) -> None:
    """在 ``Policy.infer`` 最外层打补丁，拆分 preprocess / sample_actions / postprocess（不改 openpi）。

    仅对 ``_is_pytorch_model=True`` 走显式分 phase；JAX 策略回退原始 ``infer`` 并只记 ``policy.infer`` 总时长。
    逻辑与 ``openpi.policies.policy.Policy.infer`` 保持同步（PyTorch 路径）。
    """
    if not callable(getattr(policy, "infer", None)):
        return
    if getattr(policy, "_mopt_policy_infer_stage_wrapped", False):
        inner = getattr(policy, "_mopt_policy_infer_stage_inner", None)
        if inner is not None:
            policy.infer = inner
    _orig_infer = policy.infer
    skips = max(0, int(warmup_skips))
    state = {"invocation": 0}

    def _infer(self, obs: dict, *, noise=None):
        state["invocation"] += 1
        record = collector.enabled and state["invocation"] > skips
        if not record:
            return _orig_infer(obs, noise=noise)

        import jax
        import numpy as np
        import torch
        from openpi.models import model as _openpi_model

        t_infer0 = time.perf_counter()
        try:
            if not getattr(self, "_is_pytorch_model", False):
                out = _orig_infer(obs, noise=noise)
                return out

            t_pre0 = time.perf_counter()
            t_sub0 = time.perf_counter()
            inputs = jax.tree.map(lambda x: x, obs)
            inputs = _apply_input_transform_with_prompt_cache(self, inputs)
            state_cpu = np.asarray(inputs["state"]) if "state" in inputs else None
            collector.record(
                KEY_POLICY_PREPROCESS_INPUT_TRANSFORM,
                (time.perf_counter() - t_sub0) * 1000.0,
            )

            t_sub0 = time.perf_counter()
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x))[None, ...], inputs)
            collector.record(
                KEY_POLICY_PREPROCESS_NUMPY_TO_TORCH,
                (time.perf_counter() - t_sub0) * 1000.0,
            )

            t_sub0 = time.perf_counter()
            inputs = jax.tree.map(lambda x: x.to(self._pytorch_device), inputs)
            collector.record(
                KEY_POLICY_PREPROCESS_H2D,
                (time.perf_counter() - t_sub0) * 1000.0,
            )
            sample_device = self._pytorch_device
            sample_kwargs = dict(self._sample_kwargs)
            if noise is not None:
                t_sub0 = time.perf_counter()
                noise_t = torch.from_numpy(noise).to(self._pytorch_device)
                if noise_t.ndim == 2:
                    noise_t = noise_t[None, ...]
                sample_kwargs["noise"] = noise_t
                collector.record(
                    KEY_POLICY_PREPROCESS_NOISE_H2D,
                    (time.perf_counter() - t_sub0) * 1000.0,
                )

            t_sub0 = time.perf_counter()
            # 与 openpi ``Observation.from_dict`` 完全一致；勿用仅做微优化的 fast path，
            # 否则图像 dtype/layout 与 TRT/FlashRT 组合时可能导致 actions 为 NaN。
            observation = _openpi_model.Observation.from_dict(inputs)
            collector.record(
                KEY_POLICY_PREPROCESS_OBSERVATION,
                (time.perf_counter() - t_sub0) * 1000.0,
            )
            collector.record(
                KEY_POLICY_PREPROCESS,
                (time.perf_counter() - t_pre0) * 1000.0,
            )

            t_sa0 = time.perf_counter()
            sample_count_before = len(collector.values_for_key(KEY_SAMPLE_ACTIONS))
            outputs = {
                "state": inputs["state"],
                "actions": self._sample_actions(
                    sample_device, observation, **sample_kwargs
                ),
            }
            sample_count_after = len(collector.values_for_key(KEY_SAMPLE_ACTIONS))
            sync_ms = (
                _sync_cuda_value(outputs["actions"])
                if sample_count_after == sample_count_before
                else None
            )
            if sync_ms is not None:
                collector.record(
                    KEY_POLICY_SAMPLE_ACTIONS_CUDA_SYNC,
                    sync_ms,
                )
            sa_ms = (time.perf_counter() - t_sa0) * 1000.0

            t_post0 = time.perf_counter()
            outputs_cpu = {}
            for key, value in outputs.items():
                t_sub0 = time.perf_counter()
                if key == "state" and state_cpu is not None:
                    outputs_cpu[key] = state_cpu
                    collector.record(
                        KEY_POLICY_POSTPROCESS_STATE_CPU_REUSE,
                        (time.perf_counter() - t_sub0) * 1000.0,
                    )
                else:
                    tensor = value.detach()
                    if key == "actions":
                        _sync_cuda_value(tensor)
                        # ``sample_actions`` 可能返回 (T, D) 或 (1, T, D)；避免 ``[0]`` 误取单步。
                        if tensor.ndim >= 3:
                            tensor = tensor[0]
                    outputs_cpu[key] = np.asarray(tensor.cpu())
                if key == "actions":
                    collector.record(
                        KEY_POLICY_POSTPROCESS_ACTIONS_D2H,
                        (time.perf_counter() - t_sub0) * 1000.0,
                    )
                elif key == "state" and state_cpu is None:
                    collector.record(
                        KEY_POLICY_POSTPROCESS_STATE_D2H,
                        (time.perf_counter() - t_sub0) * 1000.0,
                    )
            outputs = outputs_cpu
            t_sub0 = time.perf_counter()
            outputs = self._output_transform(outputs)
            collector.record(
                KEY_POLICY_POSTPROCESS_OUTPUT_TRANSFORM,
                (time.perf_counter() - t_sub0) * 1000.0,
            )
            collector.record(
                KEY_POLICY_POSTPROCESS,
                (time.perf_counter() - t_post0) * 1000.0,
            )

            outputs["policy_timing"] = {"infer_ms": sa_ms}
            return outputs
        finally:
            collector.record(
                KEY_POLICY_INFER,
                (time.perf_counter() - t_infer0) * 1000.0,
            )

    policy.infer = types.MethodType(_infer, policy)
    policy._mopt_policy_infer_stage_wrapped = True
    policy._mopt_policy_infer_stage_inner = _orig_infer


def install_infer_stage_perf(
    policy: Any,
    model: Any,
    collector: StagePerfCollector,
    *,
    warmup_skips: int = 0,
) -> None:
    """安装 ``Policy.infer`` 与 ``model.sample_actions`` 分阶段计时（须在 TRT/FlashRT 替换之后）。"""
    if not collector.enabled:
        return
    wrap_policy_infer_with_stage_perf(policy, collector, warmup_skips=warmup_skips)
    wrap_sample_actions_with_stage_perf(model, collector, warmup_skips=warmup_skips)


def stage_perf_from_policy(policy: Any) -> StagePerfCollector | None:
    """返回 policy 上已挂载且启用的 :class:`StagePerfCollector`，否则 ``None``。"""
    sp = getattr(policy, "_stage_perf", None)
    if isinstance(sp, StagePerfCollector) and sp.enabled:
        return sp
    return None


def _summary_prefix_for_key(key: str, engine_prefix: str) -> str:
    if key.startswith("policy."):
        return _POLICY_SUMMARY_PREFIX
    return engine_prefix


def _step_index(key: str) -> int:
    """从 ``denoise.step.3`` 解析 step 下标；非 step key 返回 -1。"""
    m = re.fullmatch(r"denoise\.step\.(\d+)", key)
    return int(m.group(1)) if m else -1


@dataclass
class StagePerfCollector:
    """可复用的阶段耗时采集器（线程内单实例即可；非线程安全）。"""

    enabled: bool = True
    summary_prefix: str = "[summary:engine]"
    meta_summary_prefix: str = "[summary:emb]"
    _samples: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _meta: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def record(self, stage: str, dt_ms: float) -> None:
        if not self.enabled:
            return
        self._samples[stage].append(float(dt_ms))

    def record_meta(self, key: str, value: float | int) -> None:
        """Record scalar metadata (token counts, bytes read, etc.) per inference step."""
        if not self.enabled:
            return
        self._meta[key].append(float(value))

    @contextmanager
    def timed(self, stage: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, (time.perf_counter() - t0) * 1000.0)

    def record_step(self, step: int, dt_ms: float, *, prefix: str = "denoise") -> None:
        """记录扩散单步耗时，key 为 ``{prefix}.step.{step}``。"""
        self.record(f"{prefix}.step.{int(step)}", dt_ms)

    def pooled(self, pattern_prefix: str) -> list[float]:
        """合并所有以 ``pattern_prefix`` 开头的 key 的样本（用于 step 池化统计）。"""
        out: list[float] = []
        for k, vals in self._samples.items():
            if k == pattern_prefix or k.startswith(pattern_prefix + "."):
                out.extend(vals)
        return out

    def keys_for_summary(self, extra_order: Sequence[str] | None = None) -> list[str]:
        """生成汇总打印用的 key 列表（去重、有序）。"""
        order = list(extra_order or _DEFAULT_SUMMARY_ORDER)
        seen: set[str] = set()
        keys: list[str] = []

        def _add(k: str) -> None:
            if k not in seen and self._samples.get(k):
                seen.add(k)
                keys.append(k)

        for k in order:
            if k == "denoise.step":
                # 池化行：有任意 denoise.step.N 时出现
                if any(_step_index(x) >= 0 for x in self._samples):
                    _add("denoise.step")
                continue
            _add(k)

        # 其余未列出的 key 按字母序追加
        for k in sorted(self._samples.keys()):
            if k not in seen and _step_index(k) < 0:
                _add(k)

        # denoise.step.N 按 step 编号排序
        step_keys = sorted(
            (k for k in self._samples if _step_index(k) >= 0),
            key=_step_index,
        )
        for k in step_keys:
            _add(k)

        return keys

    def values_for_key(self, key: str) -> list[float]:
        if key == "denoise.step":
            return self.pooled("denoise.step")
        return list(self._samples.get(key, []))

    def meta_values_for_key(self, key: str) -> list[float]:
        return list(self._meta.get(key, []))

    def format_meta_summary_lines(
        self,
        *,
        prefix: str | None = None,
    ) -> list[str]:
        tag = prefix if prefix is not None else self.meta_summary_prefix
        lines: list[str] = []
        for key in sorted(self._meta.keys()):
            vals = self.meta_values_for_key(key)
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            if key.endswith(".bytes_read"):
                mean_b = float(np.mean(arr))
                if mean_b >= 1024 * 1024:
                    val_str = f"mean={mean_b / (1024 * 1024):.3f} MiB"
                elif mean_b >= 1024:
                    val_str = f"mean={mean_b / 1024:.2f} KiB"
                else:
                    val_str = f"mean={mean_b:.0f} B"
                lines.append(f"{tag} {key:<40} n={int(arr.size)} {val_str}")
            elif key.endswith(".valid_tokens") or key.endswith("_tokens"):
                lines.append(
                    f"{tag} {key:<40} n={int(arr.size)} mean={float(np.mean(arr)):.2f}"
                )
            else:
                lines.append(f"{tag} {key:<40} {perf_line_ms(vals)}")
        return lines

    def format_summary_lines(
        self,
        *,
        prefix: str | None = None,
        extra_order: Sequence[str] | None = None,
    ) -> list[str]:
        """返回可打印的汇总行（不含换色）。"""
        tag = prefix if prefix is not None else self.summary_prefix
        lines: list[str] = []
        for key in self.keys_for_summary(extra_order):
            vals = self.values_for_key(key)
            if not vals:
                continue
            # 打印名：denoise.total → denoise.total；denoise.step.3 → denoise.step.3
            label = key if key != "denoise.step" else "denoise.step"
            row_tag = _summary_prefix_for_key(key, tag)
            lines.append(f"{row_tag} {label:<16} {perf_line_ms(vals)}")
        lines.extend(self.format_meta_summary_lines())
        return lines

    def merge_from(self, other: StagePerfCollector | None) -> None:
        if other is None or not other.enabled:
            return
        for k, vals in other._samples.items():
            self._samples[k].extend(vals)
        for k, vals in other._meta.items():
            self._meta[k].extend(vals)

    def clear(self) -> None:
        self._samples.clear()
        self._meta.clear()

    def to_dict(self) -> dict[str, list[float]]:
        return {k: list(v) for k, v in self._samples.items()}

    @classmethod
    def from_dict(cls, data: dict[str, list[float]], *, enabled: bool = True) -> StagePerfCollector:
        c = cls(enabled=enabled)
        for k, vals in data.items():
            c._samples[k].extend(float(x) for x in vals)
        return c


def _lines_from_perf_holder(obj: Any) -> list[str]:
    if obj is None:
        return []
    if isinstance(obj, StagePerfCollector):
        return obj.format_summary_lines()
    fn = getattr(obj, "format_perf_summary_lines", None)
    if callable(fn):
        return fn()
    sp = getattr(obj, "stage_perf", None)
    if sp is not None and sp is not obj:
        return _lines_from_perf_holder(sp)
    return []


def format_collector_from_policy(policy: Any) -> list[str]:
    """从 policy / model / native_executor 上挂载的 :class:`StagePerfCollector` 取汇总行。"""
    if policy is None:
        return []
    seen: set[int] = set()
    candidates: list[Any] = [policy]
    model = getattr(policy, "_model", None)
    if model is not None:
        candidates.append(model)
    native_ex = getattr(policy, "_native_executor", None)
    if native_ex is not None:
        candidates.append(native_ex)
    for obj in candidates:
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        for attr in ("_stage_perf", "stage_perf"):
            lines = _lines_from_perf_holder(getattr(obj, attr, None))
            if lines:
                return lines
        lines = _lines_from_perf_holder(obj)
        if lines:
            return lines
    return []


def format_perf_from_bundle(bundle: dict[str, Any] | None) -> list[str]:
    """webui bundle 汇总：优先 ``bundle['native_executor']``，再 TRT executor，再 policy。"""
    if not bundle:
        return []
    native_ex = bundle.get("native_executor")
    lines = _lines_from_perf_holder(native_ex)
    if lines:
        return lines
    policy = bundle.get("policy")
    if policy is not None:
        trt_ex = getattr(policy, "_trt_executor", None)
        lines = _lines_from_perf_holder(trt_ex)
        if lines:
            return lines
    return format_collector_from_policy(policy)
