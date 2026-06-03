from __future__ import annotations

import atexit
import logging
import math
import time
import types
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from ..executor import Executor
from ..perf import StagePerfCollector, install_infer_stage_perf
from ...models.pi05.model_pi05 import Pi05Model
from .decoder_runner import NativeDenoiseLoopRunner, NativeDenoiseLoopRunnerV2
from .denoise_backend import NativeDenoiseBackend
from .graph_capture import resolve_eager_denoise_step, restore_eager_ops_for_cuda_graph
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
        # 分阶段耗时（见 ``model_optimizer.infer.perf.StagePerfCollector``）。
        self._stage_perf = StagePerfCollector(enabled=False)
        try:
            setattr(self.policy, "_native_executor", self)
            setattr(self.policy, "_stage_perf", self._stage_perf)
            setattr(self.pi05_model, "_stage_perf", self._stage_perf)
            setattr(self.pi05_model, "_native_executor", self)
        except Exception as exc:
            logger.warning("[native] failed to attach perf hooks on policy/model: %s", exc)

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

        self._stage_perf.enabled = bool(perf)
        if enable_denoise and flashrt_decoder:
            if use_cuda_graph or full_loop_graph:
                logger.info(
                    "[native-flashrt] native_use_cuda_graph / native_full_loop_graph 对 FlashRT 无效 "
                    "（denoise 由 FVK decoder_forward 整 10 步循环完成，非 PyTorch denoise_step CUDA Graph）"
                )
            self._install_flashrt_loop_runtime(
                build_dir=flashrt_build_dir,
                fmha_so=flashrt_fmha_so,
                use_fp8=flashrt_use_fp8,
                act_scales_path=flashrt_act_scales_path,
                calibrate=flashrt_calibrate,
                calib_samples=flashrt_calib_samples,
                perf=self._stage_perf.enabled,
            )
            logger.info(
                "[native-flashrt] decoder full-loop enabled (fp8=%s build_dir=%s calibrate=%s)",
                flashrt_use_fp8,
                flashrt_build_dir,
                flashrt_calibrate,
            )
        elif enable_denoise:
            # 单步 ``denoise_step``（含 embed_suffix + create_causal_mask）在 capture 区会动态分配张量，
            # 不能稳定做 CUDA Graph；``use_cuda_graph=True`` 时改走 full-loop（仅 capture 10 步 denoise）。
            prefer_full_loop = bool(full_loop_graph) or bool(use_cuda_graph)
            if prefer_full_loop:
                if use_cuda_graph and not full_loop_graph:
                    logger.info(
                        "[native] per-step denoise CUDA graph 不可用（HF forward/mask 动态分配）；"
                        "自动切换为 full-loop denoise graph（prefix TRT 仍 eager）"
                    )
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
                    use_cuda_graph=False,
                    graph_warmup=graph_warmup,
                    perf=perf,
                )
                logger.info(
                    "[native] denoise stage enabled (per-step eager, cuda_graph=false)",
                )

        self._install_sample_actions_stage_timer(config)
        self._sync_policy_sample_actions_ref()
        if enable_denoise and flashrt_decoder:
            logger.info(
                "[native-flashrt] perf=%s sample_actions=%s",
                self._stage_perf.enabled,
                getattr(self.pi05_model.sample_actions, "__name__", type(self.pi05_model.sample_actions)),
            )
        atexit.register(self._dump_summary_atexit)

    def _install_sample_actions_stage_timer(self, config: Any) -> None:
        """``Policy.infer`` + ``sample_actions`` 分阶段计时（须在 FlashRT/TRT 等替换之后）。"""
        if not self._stage_perf.enabled:
            return
        warmup = int(_cfg_get(config, "sample_actions_warmup_skips", 10) or 10)
        install_infer_stage_perf(
            self.policy,
            self.pi05_model,
            self._stage_perf,
            warmup_skips=warmup,
        )

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

    def _prepare_cuda_graph_eager_ops(self, *, use_cuda_graph: bool) -> None:
        if not use_cuda_graph:
            return
        restored = restore_eager_ops_for_cuda_graph(self.pi05_model)
        if restored:
            logger.info(
                "[native] CUDA graph: restored eager ops (removed profiler wrap on %s)",
                ", ".join(restored),
            )

    def _install_denoise_runtime(
        self,
        *,
        use_cuda_graph: bool,
        graph_warmup: int,
        perf: bool,
    ) -> None:
        # denoise 阶段后端：与 TRT denoise engine 接口对齐（堆叠 KV），可与 vit/llm=TRT 自由组合。
        self._prepare_cuda_graph_eager_ops(use_cuda_graph=use_cuda_graph)
        eager_denoise = resolve_eager_denoise_step(self.pi05_model)
        self._orig_denoise = eager_denoise
        if use_cuda_graph and eager_denoise is not self.pi05_model.denoise_step:
            logger.info(
                "[native] CUDA graph will capture eager denoise_step "
                "(bypass Pi0 stage profiler wrapper)"
            )
        self._denoise_backend = NativeDenoiseBackend(
            eager_denoise,
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
        self._prepare_cuda_graph_eager_ops(use_cuda_graph=use_cuda_graph)
        if self._orig_denoise is None:
            self._orig_denoise = resolve_eager_denoise_step(self.pi05_model)
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
                prefix_pad_masks,
                past_key_values,
                x_t,
                timestep,
            ):
                _set_probe("denoise_embed_suffix_before")
                time_freq = observation["time_freq"]
                sin_input_buf = observation["sin_input_buf"]
                time_emb_buf = observation["time_emb_buf"]
                half = sin_input_buf.shape[1]

                _set_probe("denoise_time_emb")
                sin_input_buf.copy_(time_freq.unsqueeze(0))
                sin_input_buf.mul_(timestep.unsqueeze(1))
                torch.sin(sin_input_buf, out=time_emb_buf[:, :half])
                torch.cos(sin_input_buf, out=time_emb_buf[:, half:])

                _set_probe("denoise_action_proj")
                suffix_embs = self.pi05_model.action_in_proj(x_t)

                _set_probe("denoise_time_mlp")
                x = self.pi05_model.time_mlp_in(time_emb_buf)
                x = F.silu(x)
                x = self.pi05_model.time_mlp_out(x)
                adarms_cond = F.silu(x)
                _set_probe("denoise_embed_suffix_after")

                _set_probe("denoise_build_masks")
                full_att_2d_masks_4d = observation["full_att_2d_masks_4d"]
                position_ids = observation["position_ids"]

                _set_probe("denoise_forward_before")
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

            prefix_pad_masks = observation["prefix_pad_masks"]
            past_key_values = observation["past_key_values"]
            time_buf = observation["time_buf"]

            _set_probe("loop_setup")
            n_steps = max(int(num_steps), 1)
            dt = -1.0 / float(n_steps)
            x_t = noise
            for s in range(n_steps):
                _set_probe(f"denoise_step_{s}_before")
                t_scalar = 1.0 - (float(s) / float(n_steps))
                time_buf.fill_(t_scalar)
                v_t = _denoise_step_with_probe(
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    time_buf,
                )
                _set_probe(f"denoise_step_{s}_after")
                x_t.add_(v_t, alpha=dt)
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
            images, img_masks, lang_tokens, lang_masks, _state = self_m._preprocess_observation(
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
            batch = int(prefix_pad_masks.shape[0])
            action_horizon = int(self_m.config.action_horizon)
            prefix_len = int(prefix_pad_masks.shape[1])
            suffix_len = action_horizon
            time_emb_dim = int(self_m.action_in_proj.out_features)
            mlp_dtype = self_m.time_mlp_in.weight.dtype

            suffix_pad_masks = torch.ones(
                (batch, action_horizon), dtype=torch.bool, device=device
            )
            suffix_att_masks = torch.cat(
                [
                    torch.ones((batch, 1), dtype=torch.float32, device=device),
                    torch.zeros(
                        (batch, max(action_horizon - 1, 0)),
                        dtype=torch.float32,
                        device=device,
                    ),
                ],
                dim=1,
            )
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
                batch, suffix_len, prefix_len
            )
            suffix_cumsum = torch.cumsum(suffix_att_masks, dim=1)
            suffix_att_2d_masks = suffix_cumsum[:, None, :] <= suffix_cumsum[:, :, None]
            suffix_pad_2d_masks = suffix_pad_masks[:, None, :] * suffix_pad_masks[:, :, None]
            suffix_att_2d_masks = suffix_att_2d_masks & suffix_pad_2d_masks
            full_att_2d_masks = torch.cat(
                [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
            )
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
            full_att_2d_masks_4d = self_m._prepare_attention_masks_4d(full_att_2d_masks)

            denoise_inputs = {
                "prefix_pad_masks": prefix_pad_masks,
                "past_key_values": past_key_values,
                "time_buf": torch.empty((batch,), dtype=torch.float32, device=device),
                "time_freq": (
                    (2.0 * math.pi)
                    / (
                        4e-3
                        * (4.0 / 4e-3)
                        ** torch.linspace(
                            0.0,
                            1.0,
                            time_emb_dim // 2,
                            dtype=torch.float32,
                            device=device,
                        )
                    )
                ),
                "sin_input_buf": torch.empty(
                    (batch, time_emb_dim // 2), dtype=mlp_dtype, device=device
                ),
                "time_emb_buf": torch.empty(
                    (batch, time_emb_dim), dtype=mlp_dtype, device=device
                ),
                "full_att_2d_masks_4d": full_att_2d_masks_4d,
                "position_ids": position_ids,
                "__capture_probe": {"stage": "prepared_inputs"},
            }
            self_m.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
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
        stage_perf = self._stage_perf

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
                stage_perf=stage_perf,
            )
            backend.setup_prompt(enc_seq)
            if use_fp8 and act_scales_path and not calibrate:
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
                with stage_perf.timed("embed_prefix"):
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
                with stage_perf.timed("prefix_llm"):
                    _, past_key_values = self_m.paligemma_with_expert.forward(
                        attention_mask=prefix_att_2d_masks_4d,
                        position_ids=prefix_position_ids,
                        past_key_values=None,
                        inputs_embeds=[prefix_embs, None],
                        use_cache=True,
                    )
                past_keys, past_values = self._stack_past_key_values(past_key_values)
                if int(prefix_pad_masks.shape[0]) != 1:
                    raise ValueError(
                        "FlashRT decoder hybrid currently supports batch=1; "
                        f"got batch={int(prefix_pad_masks.shape[0])}"
                    )
                valid_prefix = prefix_pad_masks[0].to(dtype=torch.bool)
                enc_seq = int(valid_prefix.sum().item())
                if past_keys.dim() == 4:
                    past_keys = past_keys[:, :, valid_prefix, :].contiguous()
                    past_values = past_values[:, :, valid_prefix, :].contiguous()
                else:
                    past_keys = past_keys[:, valid_prefix, :].contiguous()
                    past_values = past_values[:, valid_prefix, :].contiguous()

                if noise is None:
                    bsize = int(prefix_pad_masks.shape[0])
                    noise = self_m.sample_noise(
                        (bsize, int(self_m.config.action_horizon), int(m.action_in_proj.in_features)),
                        device,
                    )

                with stage_perf.timed("flashrt.setup"):
                    if self._flashrt_backend is None:
                        self._flashrt_backend = _build_backend(enc_seq)
                    backend = self._flashrt_backend
                    backend.setup_prompt(enc_seq)

                noise_2d = noise.reshape(-1, noise.shape[-1])
                # fp16 模式无需激活标定（act scales 不被读取）；仅 FP8 模式标定。
                if use_fp8 and calibrate and not self._flashrt_calibrated:
                    # 多样本标定：跨 N 个 observation（KV/noise 各异）累计取 max。
                    if self._flashrt_calib_count == 0:
                        backend.reset_act_scales()
                    with stage_perf.timed("denoise.calibrate"):
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

                with stage_perf.timed("denoise.total"):
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

    def format_perf_summary_lines(self) -> list[str]:
        """返回 native 阶段耗时行（委托 :class:`StagePerfCollector`）。"""
        return self._stage_perf.format_summary_lines()

    @property
    def stage_perf(self) -> StagePerfCollector:
        return self._stage_perf

    def _dump_summary_atexit(self) -> None:
        try:
            if self._stage_perf.enabled:
                for line in self.format_perf_summary_lines():
                    logger.info("%s", line)
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

