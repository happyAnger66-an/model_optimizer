# Copyright 2025 the model_optimizer team.
# SPDX-License-Identifier: Apache-2.0
"""π0.5 Gemma 解码器的 **fused MLP**（方案 A：纯 ONNX 图重写，零 plugin）。

把 HuggingFace / openpi 风格的 ``GemmaMLP``::

    h = down_proj(act_fn(gate_proj(x)) * up_proj(x))

合并为单次 FC1 GEMM 的等价实现::

    [g; u] = gate_up_proj(x)        # gate / up 权重在 N 维 concat，一次 GEMM
    h = down_proj(act_fn(g) * u)

数学等价，无精度损失，省一次 weight HBM load 并提升 FC1 的算术强度。后续的
``Split / Gelu / Mul`` 仍由 ONNX 原生 op 表示，建引擎时由 TensorRT MyelinGraph
自行融合到 down_proj 之前。

详见 ``kernelSrc/docs/fused_mlp.md`` § 3 方案 A 与 § 1 L1。

使用方式::

    from model_optimizer.models.pi05.fused_mlp import install_fused_mlp
    install_fused_mlp(hf_gemma_decoder)   # in-place 替换所有层的 ``.mlp``

幂等：已经替换过的层不会被重复替换。可由环境变量
``MODEL_OPT_PI05_FUSED_MLP=0`` 关闭（用于精度回归调试）。
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_ENV_FLAG = "MODEL_OPT_PI05_FUSED_MLP"
# 与统一特性体系（feature_config）的环境变量命名对齐；两者均可用，旧名优先。
_ENV_FLAG_ALIAS = "MODEL_OPT_FEATURE_FUSED_MLP"


def is_fused_mlp_enabled(default: bool = True) -> bool:
    """读取 ``MODEL_OPT_PI05_FUSED_MLP``（或别名 ``MODEL_OPT_FEATURE_FUSED_MLP``）。

    - 未设置：返回 ``default``（默认开启）。
    - ``0`` / ``false`` / ``no`` / ``off``：关闭。
    - 其他非空值：开启。

    注：经 CLI ``--feature_config`` 走 ``apply_features`` 时，启停已由
    ``FeatureConfig.is_enabled`` 判定，本函数仅用于直接构造的兼容路径。
    """
    raw = os.environ.get(_ENV_FLAG)
    if raw is None:
        raw = os.environ.get(_ENV_FLAG_ALIAS)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


class _LinearMetaShim:
    """轻量只读视图，用于让下游 ``module.<gate|up>_proj.weight.dtype/device``
    这样的探测代码在合并后继续可用。

    `FusedGemmaMLP` 把 ``gate_proj`` / ``up_proj`` 暴露为返回该 shim 的属性，
    但 shim 本身既不是 ``nn.Module`` 也不是 ``nn.Linear``：ModelOpt 的
    ``mtq.quantize`` 只会发现合并后的 ``gate_up_proj`` 和原始 ``down_proj``
    两个 ``nn.Linear``，不会重复插入量化器。
    """

    __slots__ = ("weight",)

    def __init__(self, weight: torch.Tensor) -> None:
        # 持有 ``gate_up_proj.weight`` 的引用而非拷贝，dtype/device 始终与
        # 合并后的真实权重保持同步（即便后续 ``.to()`` 切换 dtype）。
        self.weight = weight


class FusedGemmaMLP(nn.Module):
    """合并 gate / up 的 ``GemmaMLP`` 替身。

    构造时把传入 ``orig_mlp`` 的 ``gate_proj.weight`` 与 ``up_proj.weight``
    在 N 维（dim=0）concat 成 ``gate_up_proj.weight``，保留 ``down_proj``
    与 ``act_fn``。前向：

    .. code-block:: text

        gu = self.gate_up_proj(x)                      # [..., 2I]
        gate, up = gu.chunk(2, dim=-1)                 # 逻辑 split，不落盘
        h = self.act_fn(gate) * up                     # [..., I]
        y = self.down_proj(h)                          # [..., H]

    与原 ``GemmaMLP.forward`` 数学等价（``x[..., :I]`` ↔ ``gate``，
    ``x[..., I:]`` ↔ ``up``）。
    """

    def __init__(self, orig_mlp: nn.Module) -> None:
        super().__init__()

        gate = orig_mlp.gate_proj
        up = orig_mlp.up_proj
        down = orig_mlp.down_proj

        if not (isinstance(gate, nn.Linear) and isinstance(up, nn.Linear)
                and isinstance(down, nn.Linear)):
            raise TypeError(
                "FusedGemmaMLP requires gate/up/down_proj to be nn.Linear; got "
                f"gate={type(gate).__name__}, up={type(up).__name__}, "
                f"down={type(down).__name__}"
            )
        if gate.weight.shape != up.weight.shape:
            raise ValueError(
                f"gate_proj.weight {tuple(gate.weight.shape)} != "
                f"up_proj.weight {tuple(up.weight.shape)}; cannot concat."
            )
        if gate.bias is not None or up.bias is not None:
            raise NotImplementedError(
                "FusedGemmaMLP currently assumes bias=False on gate/up "
                "(GemmaMLP uses bias=False)."
            )

        # 透传元数据，保持下游 ``mlp.hidden_size`` 等查询。
        self.hidden_size = orig_mlp.hidden_size
        self.intermediate_size = orig_mlp.intermediate_size
        self.config = orig_mlp.config
        self.act_fn = orig_mlp.act_fn

        H, I = self.hidden_size, self.intermediate_size

        # FC1（合并）：``hidden -> 2 * intermediate``。
        self.gate_up_proj = nn.Linear(
            H, 2 * I, bias=False,
            device=gate.weight.device, dtype=gate.weight.dtype,
        )
        with torch.no_grad():
            # 顺序约定：前 I 列为 gate，后 I 列为 up。前向用 ``chunk(2, dim=-1)``
            # 对应同样的拆分（``chunk`` 在 dim=-1 上对半切；
            # ``[..., :I] = gate``，``[..., I:] = up``）。
            self.gate_up_proj.weight.copy_(
                torch.cat([gate.weight.detach(), up.weight.detach()], dim=0)
            )

        # FC2：直接复用原 ``down_proj``，保留参数 / 设备 / dtype / 量化状态。
        self.down_proj = down

        # 兼容 shim：保持 ``mlp.gate_proj.weight.dtype/device`` 这类探测可用。
        # 不会被 ``mtq.quantize`` 识别为 ``nn.Linear``。
        self._gate_proj_shim = _LinearMetaShim(self.gate_up_proj.weight)
        self._up_proj_shim = _LinearMetaShim(self.gate_up_proj.weight)

    # ── 兼容属性（read-only） ────────────────────────────────────────────────
    # NOTE: 用 ``@property`` 而非 ``self.xxx = shim``，避免被
    # ``nn.Module.__setattr__`` 误判为子模块；同时让 ``mlp.up_proj`` 总是返回
    # 反映 ``gate_up_proj.weight`` 当前 dtype/device 的最新视图。

    @property
    def gate_proj(self) -> _LinearMetaShim:  # type: ignore[override]
        return self._gate_proj_shim

    @property
    def up_proj(self) -> _LinearMetaShim:  # type: ignore[override]
        return self._up_proj_shim

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"merged_gate_up=True, act_fn={type(self.act_fn).__name__}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gu = self.gate_up_proj(x)
        gate, up = gu.chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


def install_fused_mlp(
    gemma_model: nn.Module,
    *,
    enabled: bool | None = None,
    strict: bool = False,
) -> int:
    """把 ``gemma_model.layers[*].mlp`` in-place 替换为 :class:`FusedGemmaMLP`。

    Args:
        gemma_model: 必须暴露 ``.layers``（迭代得到每层 decoder block）。
        enabled: 显式开关；``None`` 时读取 :func:`is_fused_mlp_enabled`。
        strict: ``True`` 时遇到无法替换的层抛异常，默认仅打 warning。

    Returns:
        实际替换的层数（已替换或被跳过的不算）。
    """
    if enabled is None:
        enabled = is_fused_mlp_enabled()
    if not enabled:
        logger.info("install_fused_mlp: disabled via %s, skipping.", _ENV_FLAG)
        return 0

    layers = getattr(gemma_model, "layers", None)
    if layers is None:
        msg = (
            f"{type(gemma_model).__name__} has no `.layers`; "
            "expected an HF Gemma decoder."
        )
        if strict:
            raise AttributeError(msg)
        logger.warning("install_fused_mlp: %s", msg)
        return 0

    replaced = 0
    skipped: list[tuple[int, str]] = []
    for idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            skipped.append((idx, "missing .mlp"))
            continue
        if isinstance(mlp, FusedGemmaMLP):
            continue
        if not all(hasattr(mlp, k) for k in ("gate_proj", "up_proj", "down_proj")):
            skipped.append(
                (idx, f"{type(mlp).__name__} lacks gate/up/down_proj")
            )
            continue
        try:
            layer.mlp = FusedGemmaMLP(mlp)
        except Exception as e:  # noqa: BLE001
            skipped.append((idx, f"build failed: {e!r}"))
            if strict:
                raise
            continue
        replaced += 1

    if skipped:
        if strict:
            raise RuntimeError(f"install_fused_mlp: cannot fuse {skipped}")
        for idx, reason in skipped:
            logger.warning(
                "install_fused_mlp: skip layer %d (%s)", idx, reason
            )

    if replaced > 0:
        logger.info(
            "install_fused_mlp: merged gate/up into gate_up_proj on %d layer(s).",
            replaced,
        )
    return replaced


def fused_mlp_state_dict_remap(state_dict: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """把旧 checkpoint（含 ``mlp.gate_proj.weight`` / ``mlp.up_proj.weight``）
    重写成 ``mlp.gate_up_proj.weight``，便于 ``model.load_state_dict`` 在
    :func:`install_fused_mlp` 之后直接吃旧 ckpt。

    现阶段 ``LLM.quantize`` 走的是 ModelOpt 在线 PTQ，不会触发该路径；保留作为
    后续离线 ckpt 加载的备用工具。
    """
    if prefix and not prefix.endswith("."):
        prefix = prefix + "."

    out: dict[str, Any] = {}
    gate_keys: dict[str, str] = {}
    up_keys: dict[str, str] = {}
    for k, v in state_dict.items():
        if k.endswith("mlp.gate_proj.weight") and k.startswith(prefix):
            gate_keys[k[: -len("gate_proj.weight")]] = k
        elif k.endswith("mlp.up_proj.weight") and k.startswith(prefix):
            up_keys[k[: -len("up_proj.weight")]] = k
        else:
            out[k] = v

    for stem, gk in gate_keys.items():
        uk = up_keys.pop(stem, None)
        if uk is None:
            # gate 没有匹配的 up，保留原 key（让 load_state_dict 报错给用户看）。
            out[gk] = state_dict[gk]
            continue
        out[stem + "gate_up_proj.weight"] = torch.cat(
            [state_dict[gk], state_dict[uk]], dim=0
        )
    # 残余的 up（无对应 gate）原样保留。
    for uk in up_keys.values():
        out[uk] = state_dict[uk]
    return out


__all__ = [
    "FusedGemmaMLP",
    "install_fused_mlp",
    "is_fused_mlp_enabled",
    "fused_mlp_state_dict_remap",
]
