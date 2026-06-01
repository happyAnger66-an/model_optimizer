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
            if full_loop_graph:
                # full-loop 模式下，避免安装单步 denoise runner（含额外包装逻辑），
                # 直接对原始 denoise_step 循环做 graph capture/replay，降低 capture 冲突风险。
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
            else:
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
        if self._orig_denoise is None:
            self._orig_denoise = self.pi05_model.denoise_step
        self._orig_sample_actions = self.pi05_model.sample_actions

        def denoise_loop_capture_safe(
            device,
            observation,
            noise=None,
            num_steps=10,
        ):
            """仅 denoise num_steps 循环（capture-safe for-loop）。"""
            probe = observation.get("__capture_probe") if isinstance(observation, dict) else None

            def _set_probe(stage: str) -> None:
                if isinstance(probe, dict):
                    probe["stage"] = stage

            def _denoise_step_with_probe(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                timestep,
            ):
                _set_probe("denoise_embed_suffix_before")
                suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
                    self.pi05_model.embed_suffix(state, x_t, timestep)
                )
                _set_probe("denoise_embed_suffix_after")

                suffix_len = suffix_pad_masks.shape[1]
                batch_size = prefix_pad_masks.shape[0]
                prefix_len = prefix_pad_masks.shape[1]

                _set_probe("denoise_build_masks")
                prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                    batch_size, suffix_len, prefix_len
                )
                suffix_cumsum = torch.cumsum(suffix_att_masks, dim=1)
                suffix_att_2d_masks = suffix_cumsum[:, None, :] <= suffix_cumsum[:, :, None]
                suffix_pad_2d_masks = suffix_pad_masks[:, None, :] * suffix_pad_masks[:, :, None]
                suffix_att_2d_masks = suffix_att_2d_masks & suffix_pad_2d_masks
                full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

                prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
                position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
                full_att_2d_masks_4d = self.pi05_model._prepare_attention_masks_4d(full_att_2d_masks)

                _set_probe("denoise_forward_before")
                self.pi05_model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
                outputs_embeds, _ = self.pi05_model.paligemma_with_expert.forward(
                    attention_mask=full_att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=[None, suffix_embs],
                    use_cache=False,
                    adarms_cond=[None, adarms_cond],
                )
                _set_probe("denoise_forward_after")

                _set_probe("denoise_proj_before")
                suffix_out = outputs_embeds[1]
                suffix_out = suffix_out[:, -self.pi05_model.config.action_horizon :]
                suffix_out = suffix_out.to(dtype=torch.float32)
                out = self.pi05_model.action_out_proj(suffix_out)
                _set_probe("denoise_proj_after")
                return out

            state = observation["state"]
            prefix_pad_masks = observation["prefix_pad_masks"]
            past_key_values = observation["past_key_values"]
            time_buf = observation["time_buf"]
            bsize = int(prefix_pad_masks.shape[0])

            _set_probe("loop_setup")
            n_steps = max(int(num_steps), 1)
            dt = -1.0 / float(n_steps)
            x_t = noise
            for s in range(n_steps):
                _set_probe(f"denoise_step_{s}_before")
                t_scalar = 1.0 - (float(s) / float(n_steps))
                time_buf.fill_(t_scalar)
                v_t = _denoise_step_with_probe(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    time_buf,
                )
                _set_probe(f"denoise_step_{s}_after")
                x_t = x_t + v_t * dt
            _set_probe("loop_done")
            return x_t

        self._denoise_runner_v2 = NativeDenoiseLoopRunnerV2(
            denoise_loop_capture_safe,
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
            # 保持与原始 sample_actions 一致：未显式传 noise 时在此处采样。
            if noise is None:
                try:
                    bsize = int(observation.state.shape[0])
                    cfg = getattr(self_m, "config", None)
                    action_horizon = int(
                        getattr(cfg, "action_horizon", getattr(self_m, "action_horizon"))
                    )
                    action_dim = int(
                        getattr(cfg, "action_dim", getattr(self_m, "action_dim"))
                    )
                    if not hasattr(self_m, "sample_noise"):
                        raise AttributeError("model has no sample_noise")
                    noise = self_m.sample_noise(
                        (bsize, action_horizon, action_dim),
                        device,
                    )
                except Exception as exc:
                    logger.warning(
                        "[native-v2] failed to synthesize noise for full-loop graph, fallback original sample_actions: %s",
                        exc,
                    )
                    return self._orig_sample_actions(
                        device,
                        observation,
                        noise=None,
                        num_steps=num_steps,
                    )

            # 目标边界：仅 denoise loop capture。
            # preprocess + prefix kv cache 维持 eager（与 FlashRT “整段 denoise loop 单次 replay” 目标一致）。
            images, img_masks, lang_tokens, lang_masks, state = self_m._preprocess_observation(
                observation,
                train=False,
            )
            prefix_embs, prefix_pad_masks, prefix_att_masks = self_m.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
            )
            prefix_cumsum = torch.cumsum(prefix_att_masks, dim=1)
            prefix_att_2d_masks = prefix_cumsum[:, None, :] <= prefix_cumsum[:, :, None]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :] * prefix_pad_masks[:, :, None]
            prefix_att_2d_masks = prefix_att_2d_masks & prefix_pad_2d_masks
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self_m._prepare_attention_masks_4d(prefix_att_2d_masks)
            self_m.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
            _, past_key_values = self_m.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            denoise_inputs = {
                "state": state,
                "prefix_pad_masks": prefix_pad_masks,
                "past_key_values": past_key_values,
                "time_buf": torch.empty(
                    (int(prefix_pad_masks.shape[0]),),
                    dtype=torch.float32,
                    device=device,
                ),
                "__capture_probe": {"stage": "prepared_inputs"},
            }
            return self._denoise_runner_v2.run(
                device=device,
                observation=denoise_inputs,
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

