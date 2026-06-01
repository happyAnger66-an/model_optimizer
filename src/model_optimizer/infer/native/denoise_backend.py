"""去噪阶段后端抽象（与 TRT denoise engine 接口对齐）。

设计目标（见 docs/optimizer/ddup/native_decoder_implementation_todo.md §7）：

- pi05 推理是分阶段后端矩阵：``vit / embed_prefix / llm / expert / denoise`` 可各自独立选后端。
- ``denoise`` 阶段无论走 TensorRT engine 还是 native 自实现，都遵循**同一调用接口**，
  这样 ``vit/llm`` 留在 TensorRT、``denoise`` 切到 native 时无需改动上层 ``denoise_step`` hook。

规范接口（与 ``denoise.onnx`` / :class:`Pi05TensorRTExecutor` 的 ``denoise_step_trt`` 一致）::

    backend(prefix_pad_masks, past_keys, past_values, x_t, timestep) -> v_t

其中：

- ``prefix_pad_masks``: ``[batch, prefix_len]``
- ``past_keys`` / ``past_values``: **堆叠张量** ``[num_layers, batch, prefix_len, head_dim]``
  （与 LLM 导出、:meth:`Pi05TensorRTExecutor._stack_past_key_value_tensors` 堆叠方式一致）
- ``x_t``: ``[batch, action_horizon, action_dim]``
- ``timestep``: 默认时间标量 ``[batch]``；AdaRMS 预计算模式下承载打包 modulation
- 返回 ``v_t``: ``[batch, action_horizon, action_dim]``

``NativeDenoiseBackend`` 在 ``model_optimizer`` 内自实现，**不依赖 flash_rt 库**；其运行时设计借鉴
FlashRT（静态 buffer、AdaRMS modulation 预计算、可选整循环 CUDA Graph、静态 FP8 scale），但计算图
复用本仓库的 ``Pi05DenoiseStep`` / expert 前向。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch

from .decoder_runner import NativeDenoiseLoopRunner
from .quant_runtime import NativeQuantRuntime

logger = logging.getLogger(__name__)


class DenoiseBackend(ABC):
    """去噪阶段后端统一接口（与 TRT denoise engine 调用对齐）。"""

    @abstractmethod
    def __call__(
        self,
        prefix_pad_masks: torch.Tensor,
        past_keys: torch.Tensor,
        past_values: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """给定 prefix KV、噪声动作 x_t、时间步，返回速度场 v_t。"""
        raise NotImplementedError

    def calibrate(self, samples: list[dict[str, Any]]) -> None:  # noqa: ARG002
        """可选：用数据做静态量化校准（默认 no-op）。"""
        return None

    def dump_summary(self) -> str:
        return ""


def _stacked_to_dynamic_cache(past_keys: torch.Tensor, past_values: torch.Tensor) -> Any:
    """把堆叠 KV ``[num_layers, batch, prefix_len, head_dim]`` 还原为 ``DynamicCache``。

    与 :meth:`Pi05TensorRTExecutor._wrap_past_key_values` 等价，使 native denoise 计算
    （复用 HF expert 前向，期望 ``DynamicCache``）可消费 TRT 风格的堆叠张量输入。
    """
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    num_layers = int(past_keys.shape[0])
    for i in range(num_layers):
        cache.update(past_keys[i : i + 1], past_values[i : i + 1], i)
    return cache


class NativeDenoiseBackend(DenoiseBackend):
    """``model_optimizer`` 内自实现的去噪后端（不依赖 flash_rt）。

    - 接口与 TRT denoise engine 对齐（堆叠 KV）。
    - 计算复用宿主模型原始 ``denoise_step``（``Pi05DenoiseStep`` 路径）。
    - 运行时优化：可选整步 CUDA Graph replay（:class:`NativeDenoiseLoopRunner`）。
    - 量化：可选 :class:`NativeQuantRuntime`，对输入做静态 scale + 可选在线重标定。
    """

    def __init__(
        self,
        raw_denoise_step: Callable[..., torch.Tensor],
        *,
        use_cuda_graph: bool = True,
        graph_warmup: int = 3,
        perf: bool = True,
        quant_runtime: NativeQuantRuntime | None = None,
    ) -> None:
        self._raw = raw_denoise_step
        self._runner = NativeDenoiseLoopRunner(
            raw_denoise_step,
            use_cuda_graph=use_cuda_graph,
            graph_warmup=graph_warmup,
            perf=perf,
        )
        self._quant = quant_runtime

    def __call__(
        self,
        prefix_pad_masks: torch.Tensor,
        past_keys: torch.Tensor,
        past_values: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if self._quant is not None:
            self._quant.observe_sample(
                {
                    "prefix_pad_masks": prefix_pad_masks,
                    "x_t": x_t,
                    "timestep": timestep,
                }
            )
            prefix_pad_masks, x_t, timestep = self._quant.apply_quantized_inputs(
                prefix_pad_masks=prefix_pad_masks,
                x_t=x_t,
                timestep=timestep,
            )
        past_key_values = _stacked_to_dynamic_cache(past_keys, past_values)
        # state=None：pi05 embed_suffix 不使用 state（与 TRT denoise_step_trt 的 ``del state`` 一致）。
        return self._runner.run(None, prefix_pad_masks, past_key_values, x_t, timestep)

    def dump_summary(self) -> str:
        parts = [self._runner.dump_summary()]
        if self._quant is not None:
            parts.append(self._quant.dump_summary())
        return "\n".join(parts)
