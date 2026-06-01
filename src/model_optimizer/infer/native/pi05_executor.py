from __future__ import annotations

import atexit
import logging
import types
from collections.abc import Mapping
from typing import Any

import torch

from ..executor import Executor
from ...models.pi05.model_pi05 import Pi05Model
from .decoder_runner import NativeDenoiseLoopRunner, NativeDenoiseLoopRunnerV2
from .quant_runtime import NativeQuantRuntime

logger = logging.getLogger(__name__)


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


class Pi05NativeExecutor(Executor):
    """Pi0.5 Native 执行器（Phase A MVP）。

    当前只覆盖 ``expert`` / ``denoise_step`` 阶段：
    - expert: 可选 ``torch.compile(reduce-overhead)``
    - denoise: 可选 CUDA Graph capture/replay（按输入 signature 缓存）
    """

    def __init__(self, policy, precision=torch.bfloat16, config=None):
        super().__init__(policy)
        pi05_model = Pi05Model(policy)
        self.pi05_model = pi05_model.model
        self.precision = precision
        self.config = config
        self._denoise_runner: NativeDenoiseLoopRunner | None = None
        self._denoise_runner_v2: NativeDenoiseLoopRunnerV2 | None = None
        self._quant_runtime: NativeQuantRuntime | None = None
        self._orig_denoise = None
        self._orig_expert_forward = None
        self._orig_sample_actions = None
        try:
            setattr(self.policy, "_native_executor", self)
        except Exception:
            pass

    def load_model(self, config=None):
        if config is None:
            return
        self.config = config

        enable_expert = bool(_cfg_get(config, "enable_expert", True))
        enable_denoise = bool(_cfg_get(config, "enable_denoise", True))
        use_cuda_graph = bool(_cfg_get(config, "use_cuda_graph", True))
        graph_warmup = int(_cfg_get(config, "graph_warmup", 3) or 3)
        compile_expert = bool(_cfg_get(config, "compile_expert", False))
        perf = bool(_cfg_get(config, "perf", True))
        full_loop_graph = bool(_cfg_get(config, "full_loop_graph", False))
        quant_spec_path = str(_cfg_get(config, "quant_spec_path", "") or "").strip()
        recalib_enable = bool(_cfg_get(config, "recalib_enable", False))
        recalib_max_samples = int(_cfg_get(config, "recalib_max_samples", 0) or 0)
        recalib_percentile = float(_cfg_get(config, "recalib_percentile", 99.9) or 99.9)

        if quant_spec_path:
            try:
                self._quant_runtime = NativeQuantRuntime.from_path(
                    quant_spec_path,
                    recalib_enable=recalib_enable,
                    recalib_max_samples=recalib_max_samples,
                    recalib_percentile=recalib_percentile,
                )
            except Exception as exc:
                logger.warning("[native-quant] failed to load spec, continue without it: %s", exc)

        if enable_expert:
            self._install_expert_runtime(compile_expert=compile_expert)
            logger.info("[native] expert stage enabled (compile=%s)", compile_expert)

        if enable_denoise:
            self._install_denoise_runtime(
                use_cuda_graph=use_cuda_graph,
                graph_warmup=graph_warmup,
                perf=perf,
            )
            logger.info(
                "[native] denoise stage enabled (cuda_graph=%s warmup=%s)",
                use_cuda_graph,
                graph_warmup,
            )
            if full_loop_graph:
                self._install_full_loop_runtime(
                    use_cuda_graph=use_cuda_graph,
                    graph_warmup=graph_warmup,
                    perf=perf,
                )
                logger.info(
                    "[native-v2] full-loop graph enabled (cuda_graph=%s warmup=%s)",
                    use_cuda_graph,
                    graph_warmup,
                )

        self._sync_policy_sample_actions_ref()
        atexit.register(self._dump_summary_atexit)

    def _install_expert_runtime(self, *, compile_expert: bool) -> None:
        expert_model = self.pi05_model.paligemma_with_expert.gemma_expert.model
        self._orig_expert_forward = expert_model.forward
        if not compile_expert:
            return
        try:
            expert_model.forward = torch.compile(
                expert_model.forward, mode="reduce-overhead"
            )
        except Exception as exc:
            logger.warning("[native] expert torch.compile failed, keep eager: %s", exc)

    def _install_denoise_runtime(
        self,
        *,
        use_cuda_graph: bool,
        graph_warmup: int,
        perf: bool,
    ) -> None:
        self._orig_denoise = self.pi05_model.denoise_step
        self._denoise_runner = NativeDenoiseLoopRunner(
            self._orig_denoise,
            use_cuda_graph=use_cuda_graph,
            graph_warmup=graph_warmup,
            perf=perf,
        )

        def denoise_step_native(
            self_m,
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            timestep,
        ):
            assert self._denoise_runner is not None
            if self._quant_runtime is not None:
                self._quant_runtime.observe_sample(
                    {
                        "prefix_pad_masks": prefix_pad_masks,
                        "x_t": x_t,
                        "timestep": timestep,
                    }
                )
                prefix_pad_masks, x_t, timestep = self._quant_runtime.apply_quantized_inputs(
                    prefix_pad_masks=prefix_pad_masks,
                    x_t=x_t,
                    timestep=timestep,
                )
            return self._denoise_runner.run(
                state, prefix_pad_masks, past_key_values, x_t, timestep
            )

        self.pi05_model.denoise_step = types.MethodType(
            denoise_step_native, self.pi05_model
        )

    def _install_full_loop_runtime(
        self,
        *,
        use_cuda_graph: bool,
        graph_warmup: int,
        perf: bool,
    ) -> None:
        self._orig_sample_actions = self.pi05_model.sample_actions
        self._denoise_runner_v2 = NativeDenoiseLoopRunnerV2(
            self._orig_sample_actions,
            use_cuda_graph=use_cuda_graph,
            graph_warmup=graph_warmup,
            default_num_steps=10,
            perf=perf,
        )

        def sample_actions_native_full_loop(
            self_m,
            device,
            observation,
            noise=None,
            num_steps=10,
        ):
            assert self._denoise_runner_v2 is not None
            if noise is None:
                # full-loop graph 路径要求显式 noise；无 noise 时回退原始实现，保持行为兼容。
                return self._orig_sample_actions(
                    device,
                    observation,
                    noise=noise,
                    num_steps=num_steps,
                )
            return self._denoise_runner_v2.run(
                device=device,
                observation=observation,
                noise=noise,
                num_steps=num_steps,
            )

        self.pi05_model.sample_actions = types.MethodType(
            sample_actions_native_full_loop,
            self.pi05_model,
        )

    def _sync_policy_sample_actions_ref(self) -> None:
        pol = self.policy
        if hasattr(pol, "_sample_actions"):
            pol._sample_actions = self.pi05_model.sample_actions

    def _dump_summary_atexit(self) -> None:
        try:
            if self._denoise_runner is not None:
                logger.info("%s", self._denoise_runner.dump_summary())
            if self._denoise_runner_v2 is not None:
                logger.info("%s", self._denoise_runner_v2.dump_summary())
            if self._quant_runtime is not None:
                logger.info("%s", self._quant_runtime.dump_summary())
        except Exception as exc:
            logger.warning("[native] dump summary failed: %s", exc)

