from __future__ import annotations

import atexit
import logging
import math
import types
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from ..executor import Executor
from ...models.pi05.model_pi05 import Pi05Model
from .decoder_runner import NativeDenoiseLoopRunner, NativeDenoiseLoopRunnerV2
from .denoise_backend import NativeDenoiseBackend
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
        self._denoise_backend: NativeDenoiseBackend | None = None
        self._quant_runtime: NativeQuantRuntime | None = None
        self._orig_denoise = None
        self._orig_expert_forward = None
        self._orig_sample_actions = None
        self._flashrt_backend = None
        self._flashrt_calibrated = False
        self._flashrt_calib_count = 0
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
        # FlashRT 仓内移植 decoder（整循环替代 denoise loop）
        flashrt_decoder = bool(_cfg_get(config, "flashrt_decoder", False))
        flashrt_build_dir = str(_cfg_get(config, "flashrt_build_dir", "") or "").strip() or None
        flashrt_fmha_so = str(_cfg_get(config, "flashrt_fmha_so", "") or "").strip() or None
        flashrt_use_fp8 = bool(_cfg_get(config, "flashrt_use_fp8", True))
        flashrt_act_scales_path = str(_cfg_get(config, "flashrt_act_scales_path", "") or "").strip()
        flashrt_calibrate = bool(_cfg_get(config, "flashrt_calibrate", False))
        flashrt_calib_samples = int(_cfg_get(config, "flashrt_calib_samples", 8) or 8)
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

        if enable_denoise and flashrt_decoder:
            self._install_flashrt_loop_runtime(
                build_dir=flashrt_build_dir,
                fmha_so=flashrt_fmha_so,
                use_fp8=flashrt_use_fp8,
                act_scales_path=flashrt_act_scales_path,
                calibrate=flashrt_calibrate,
                calib_samples=flashrt_calib_samples,
                perf=perf,
            )
            logger.info(
                "[native-flashrt] decoder full-loop enabled (fp8=%s build_dir=%s calibrate=%s)",
                flashrt_use_fp8,
                flashrt_build_dir,
                flashrt_calibrate,
            )
        elif enable_denoise:
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

    @staticmethod
    def _stack_past_key_values(past_key_values: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """``DynamicCache`` / ``list[(k,v)]`` → 堆叠 ``[num_layers, ...]``（与 TRT 一致）。"""
        if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
            keys = list(past_key_values.key_cache)
            vals = list(past_key_values.value_cache)
        else:
            n = len(past_key_values)
            keys = [past_key_values[i][0] for i in range(n)]
            vals = [past_key_values[i][1] for i in range(n)]
        return torch.cat(keys, dim=0), torch.cat(vals, dim=0)

    def _install_denoise_runtime(
        self,
        *,
        use_cuda_graph: bool,
        graph_warmup: int,
        perf: bool,
    ) -> None:
        # denoise 阶段后端：与 TRT denoise engine 接口对齐（堆叠 KV），可与 vit/llm=TRT 自由组合。
        self._orig_denoise = self.pi05_model.denoise_step
        self._denoise_backend = NativeDenoiseBackend(
            self._orig_denoise,
            use_cuda_graph=use_cuda_graph,
            graph_warmup=graph_warmup,
            perf=perf,
            quant_runtime=self._quant_runtime,
        )

        backend = self._denoise_backend
        stack_kv = self._stack_past_key_values

        def denoise_step_native(
            self_m,
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            timestep,
        ):
            del state  # pi05 embed_suffix 不使用 state（与 TRT denoise_step_trt 的 ``del state`` 一致）
            past_keys, past_values = stack_kv(past_key_values)
            return backend(prefix_pad_masks, past_keys, past_values, x_t, timestep)

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
                # capture-safe suffix embedding（pi05 路径）
                time_freq = observation["time_freq"]
                suffix_pad_mask_template = observation["suffix_pad_mask_template"]
                suffix_att_mask_template = observation["suffix_att_mask_template"]

                _set_probe("denoise_time_emb")
                sin_input = time_freq[None, :] * timestep[:, None]
                time_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
                time_emb = time_emb.to(dtype=timestep.dtype)

                _set_probe("denoise_action_proj")
                action_emb = self.pi05_model.action_in_proj(x_t)

                _set_probe("denoise_time_mlp")
                x = self.pi05_model.time_mlp_in(time_emb)
                x = F.silu(x)
                x = self.pi05_model.time_mlp_out(x)
                adarms_cond = F.silu(x)

                suffix_embs = action_emb
                bsize_local, action_time_dim = suffix_embs.shape[:2]
                suffix_pad_masks = suffix_pad_mask_template.expand(bsize_local, action_time_dim)
                suffix_att_masks = suffix_att_mask_template.expand(bsize_local, action_time_dim)
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
                "time_freq": (
                    (2.0 * math.pi)
                    / (
                        4e-3
                        * (4.0 / 4e-3)
                        ** torch.linspace(
                            0.0,
                            1.0,
                            int(self_m.action_in_proj.out_features // 2),
                            dtype=torch.float32,
                            device=device,
                        )
                    )
                ),
                "suffix_pad_mask_template": torch.ones(
                    (1, int(self_m.config.action_horizon)),
                    dtype=torch.bool,
                    device=device,
                ),
                "suffix_att_mask_template": torch.cat(
                    [
                        torch.ones((1, 1), dtype=torch.float32, device=device),
                        torch.zeros(
                            (1, max(int(self_m.config.action_horizon) - 1, 0)),
                            dtype=torch.float32,
                            device=device,
                        ),
                    ],
                    dim=1,
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

    def _install_flashrt_loop_runtime(
        self,
        *,
        build_dir: str | None,
        fmha_so: str | None,
        use_fp8: bool,
        act_scales_path: str,
        calibrate: bool,
        calib_samples: int,
        perf: bool,
    ) -> None:
        """在 ``sample_actions`` 整循环层用仓内 FlashRT decoder 替代 denoise loop。

        - prefix（vit+llm）维持原路径（可与 TRT 组合），产出 prefix KV；
        - decoder 整 10 步循环交给 :class:`FlashRtDecoderBackend`（FP8 静态 + 离线量化）；
        - 任意失败安全回退到原始 ``sample_actions``。
        """
        self._flashrt_backend = None
        self._flashrt_calibrated = False
        self._flashrt_calib_count = 0
        self._orig_sample_actions = self.pi05_model.sample_actions
        m = self.pi05_model
        calib_samples = max(int(calib_samples), 1)

        def _build_backend(enc_seq: int):
            from .flashrt_decoder import FlashRtDecoderBackend, state_dict_getter

            sd = dict(m.state_dict())
            getter = state_dict_getter(sd)
            Da = int(m.action_in_proj.out_features)
            layers = m.paligemma_with_expert.gemma_expert.model.layers
            Ha = int(layers[0].mlp.gate_proj.out_features)
            num_layers = len(layers)
            Sa = int(m.config.action_horizon)
            backend = FlashRtDecoderBackend(
                getter,
                Sa=Sa,
                Da=Da,
                Ha=Ha,
                num_layers=num_layers,
                num_q_heads=8,
                steps=10,
                use_fp8=use_fp8,
                build_dir=build_dir,
                fmha_so=fmha_so,
            )
            backend.setup_prompt(enc_seq)
            if act_scales_path and not calibrate:
                try:
                    backend.load_act_scales(act_scales_path)
                except FileNotFoundError:
                    logger.warning(
                        "[native-flashrt] act scales not found (%s)，用全 0（首跑可加 flashrt_calibrate=true 导出）",
                        act_scales_path,
                    )
            return backend

        def sample_actions_flashrt(self_m, device, observation, noise=None, num_steps=10):
            try:
                images, img_masks, lang_tokens, lang_masks, state = self_m._preprocess_observation(
                    observation, train=False
                )
                prefix_embs, prefix_pad_masks, prefix_att_masks = self_m.embed_prefix(
                    images, img_masks, lang_tokens, lang_masks
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
                past_keys, past_values = self._stack_past_key_values(past_key_values)
                enc_seq = int(past_keys.shape[-2])

                if noise is None:
                    bsize = int(prefix_pad_masks.shape[0])
                    noise = self_m.sample_noise(
                        (bsize, int(self_m.config.action_horizon), int(m.action_in_proj.in_features)),
                        device,
                    )

                if self._flashrt_backend is None:
                    self._flashrt_backend = _build_backend(enc_seq)
                backend = self._flashrt_backend
                backend.setup_prompt(enc_seq)

                noise_2d = noise.reshape(-1, noise.shape[-1])
                if calibrate and not self._flashrt_calibrated:
                    # 多样本标定：跨 N 个 observation（KV/noise 各异）累计取 max。
                    if self._flashrt_calib_count == 0:
                        backend.reset_act_scales()
                    backend.accumulate_calibration(past_keys, past_values, noise_2d)
                    self._flashrt_calib_count += 1
                    logger.info(
                        "[native-flashrt] calibrate sample %d/%d",
                        self._flashrt_calib_count,
                        calib_samples,
                    )
                    if self._flashrt_calib_count >= calib_samples:
                        self._flashrt_calibrated = True
                        if act_scales_path:
                            backend.save_act_scales(act_scales_path)

                action = backend.run(past_keys, past_values, noise_2d)
                return action.reshape(noise.shape).to(dtype=torch.float32)
            except Exception as exc:
                logger.warning(
                    "[native-flashrt] full-loop failed, fallback original sample_actions: %s",
                    exc,
                )
                return self._orig_sample_actions(
                    device, observation, noise=noise, num_steps=num_steps
                )

        self.pi05_model.sample_actions = types.MethodType(sample_actions_flashrt, self.pi05_model)

    def _sync_policy_sample_actions_ref(self) -> None:
        pol = self.policy
        if hasattr(pol, "_sample_actions"):
            pol._sample_actions = self.pi05_model.sample_actions

    def _dump_summary_atexit(self) -> None:
        try:
            if self._denoise_backend is not None:
                logger.info("%s", self._denoise_backend.dump_summary())
            if self._denoise_runner is not None:
                logger.info("%s", self._denoise_runner.dump_summary())
            if self._denoise_runner_v2 is not None:
                logger.info("%s", self._denoise_runner_v2.dump_summary())
            if self._quant_runtime is not None and self._denoise_backend is None:
                logger.info("%s", self._quant_runtime.dump_summary())
        except Exception as exc:
            logger.warning("[native] dump summary failed: %s", exc)

