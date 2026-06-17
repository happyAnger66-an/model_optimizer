import types

from termcolor import colored
import torch
import torch.nn as nn

from ..executor import Executor
from ..perf import StagePerfCollector, install_infer_stage_perf
from ...models.pi05.model_pi05 import Pi05Model
from .pi05_trt_engine_setup import cfg_get
from .trt_torch import Engine

from transformers.cache_utils import DynamicCache


class _TrtLanguageModelStub(nn.Module):
    """TRT LLM 挂载后保留 ``embed_tokens`` + ``config`` + TRT ``forward``，释放 Gemma 层权重。"""

    def __init__(self, *, config, embed_tokens, forward_fn):
        super().__init__()
        self.config = config
        self.embed_tokens = embed_tokens
        self._forward_fn = forward_fn

    def forward(self, *args, **kwargs):
        return self._forward_fn(*args, **kwargs)


class _TrtExpertModelStub(nn.Module):
    """TRT expert 挂载后仅保留 ``config`` + TRT ``forward``。"""

    def __init__(self, *, config, forward_fn):
        super().__init__()
        self.config = config
        self._forward_fn = forward_fn

    def forward(self, *args, **kwargs):
        return self._forward_fn(*args, **kwargs)


class Pi05TensorRTExecutor(Executor):
    def __init__(self, policy, precision=torch.bfloat16, config=None):
        super().__init__(policy)
        pi05_model = Pi05Model(policy)
        self.pi05_model = pi05_model.model
#        self.pi05_model.to(precision)
        self.config = config
        self._trt_engines: dict[str, Engine] = {}
        self._trt_lang_model_stub: _TrtLanguageModelStub | None = None
        self._trt_expert_model_stub: _TrtExpertModelStub | None = None
        self._stage_perf = StagePerfCollector(enabled=False)
        # 暴露给上层（webui 汇总）读取 engine 级统计
        try:
            setattr(self.policy, "_trt_executor", self)
            setattr(self.policy, "_stage_perf", self._stage_perf)
            setattr(self.pi05_model, "_stage_perf", self._stage_perf)
        except Exception:
            pass

    def load_model(self, config=None):
        if config is None:
            return

        self.config = config
        self._setup_trt_engine()
        # ``PI0Pytorch`` 构造里会把 ``sample_actions`` 包成 ``torch.compile``；TRT 在
        # 运行时替换 ``embed_*`` / ``forward`` / ``denoise_step`` 后，再对整条推理
        # 图做 dynamo/inductor 易与 TRT 子图不兼容。每次换 engine 后恢复为类上原始实现。
        self._restore_eager_sample_actions()
        from model_optimizer.infer.tensorrt.pi0_stage_profiler import (
            maybe_install_pi0_stage_profiler,
        )

        maybe_install_pi0_stage_profiler(self.pi05_model)
        self._install_sample_actions_stage_timer()
        # Policy 在构造时缓存了 ``_sample_actions = model.sample_actions``（见
        # openpi ``policies/policy.py``），此后只改 ``model.sample_actions`` 不会
        # 影响 ``infer()``；不刷新则仍走 torch.compile 包装，profiler / TRT 路径均可能 hook 不到。
        self._sync_policy_sample_actions_ref()
        if bool(cfg_get(self.config, "release_pytorch_weights", True)):
            self._release_pytorch_model()
        from model_optimizer.infer.perf.gpu_memory import gpu_mem_report

        gpu_mem_report("after_trt_engines")

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def _wrap_past_key_values(self, input_keys, input_values):
        k_v_cache = DynamicCache()
        num_layers = input_keys.shape[0]
        for i in range(num_layers):
            k_v_cache.update(input_keys[i:i+1], input_values[i:i+1], i)

        return k_v_cache

    @staticmethod
    def _stack_past_key_value_tensors(past_key_values):
        """将 LLM 输出的 KV（与 expert TRT 包装一致）堆成 ``[num_layers, ...]`` 张量。"""
        if past_key_values is None:
            raise ValueError("past_key_values is None")
        n = len(past_key_values)
        keys = []
        vals = []
        for i in range(n):
            entry = past_key_values[i]
            if isinstance(entry, (tuple, list)):
                k, v = entry[0], entry[1]
            else:
                raise TypeError(
                    f"Unexpected past_key_values[{i}] type: {type(entry)}"
                )
            keys.append(k)
            vals.append(v)
        return torch.cat(keys, dim=0), torch.cat(vals, dim=0)

    def _setup_trt_engine(self) -> None:
        """挂载 TRT 子图（vit / embed_prefix / llm / expert / denoise）。"""
        if not self.config.engine_path:
            return
        self._trt_engines = {}
        from .pi05_trt_engine_setup import Pi05TrtEngineInstaller

        Pi05TrtEngineInstaller(self).install_all()

    def _install_sample_actions_stage_timer(self) -> None:
        """整段 ``sample_actions`` wall time（与 ``StagePerfCollector`` / webui 汇总对齐）。"""
        from .pi0_stage_profiler import env_profile_enabled, env_warmup_skips

        enabled = bool(cfg_get(self.config, "stage_perf", env_profile_enabled()))
        self._stage_perf.enabled = enabled
        if not enabled:
            return
        warmup = int(
            cfg_get(
                self.config,
                "sample_actions_warmup_skips",
                env_warmup_skips(),
            )
            or 0
        )
        install_infer_stage_perf(
            self.policy,
            self.pi05_model,
            self._stage_perf,
            warmup_skips=warmup,
        )

    def format_perf_summary_lines(self) -> list[str]:
        return self._stage_perf.format_summary_lines()

    @property
    def stage_perf(self) -> StagePerfCollector:
        return self._stage_perf

    def _sync_policy_sample_actions_ref(self) -> None:
        """让 ``Policy.infer`` 使用的 ``_sample_actions`` 与当前 ``model.sample_actions`` 一致。"""
        pol = self.policy
        if hasattr(pol, "_sample_actions"):
            pol._sample_actions = self.pi05_model.sample_actions

    def _restore_eager_sample_actions(self) -> None:
        """将 ``sample_actions`` 从实例上的 ``torch.compile`` 恢复为类定义的 Python 方法。

        ``PI0Pytorch.__init__`` 中 ``self.sample_actions = torch.compile(...)`` 只写在实例
        ``__dict__`` 里，类属性仍是原始 ``def sample_actions``。用 ``MethodType`` 绑定到
        当前 ``pi05_model`` 后，调用链会走已挂好的 TRT 包装，且不再触发整图编译。

        注意：还须调用 :meth:`_sync_policy_sample_actions_ref`，否则 ``Policy`` 仍持有
        构造时缓存的 ``torch.compile`` 可调用对象。
        """
        model = self.pi05_model
        raw_fn = type(model).__dict__.get("sample_actions")
        if not isinstance(raw_fn, types.FunctionType):
            return
        model.sample_actions = types.MethodType(raw_fn, model)

    def _release_pytorch_model(self) -> None:
        """释放已被 TRT engine 接管的 PyTorch 子模块权重（保留 lang embedding 等仍被调用的部分）。"""
        pwe = self.pi05_model.paligemma_with_expert
        paligemma_model = pwe.paligemma.model

        if cfg_get(self.config, "vit_engine", None):
            if hasattr(paligemma_model, "vision_tower"):
                print(colored("release PyTorch vision_tower (TRT vit 已挂载)", "green"))
                del paligemma_model.vision_tower

        if cfg_get(self.config, "llm_engine", None):
            self._release_language_model_for_trt(pwe, paligemma_model)

        if cfg_get(self.config, "expert_engine", None):
            self._release_gemma_expert_for_trt("TRT expert 已挂载")
        elif cfg_get(self.config, "denoise_engine", None) and not cfg_get(
            self.config, "expert_engine", None
        ):
            # denoise 整图 TRT 时 inference 不再走 gemma_expert PyTorch forward
            self._release_gemma_expert_module("denoise TRT 已挂载")

        torch.cuda.empty_cache()

    def _release_language_model_for_trt(self, pwe, paligemma_model) -> None:
        if not hasattr(paligemma_model, "language_model"):
            return
        old_lm = paligemma_model.language_model
        if old_lm is None or isinstance(old_lm, _TrtLanguageModelStub):
            return

        forward_fn = old_lm.forward
        config = old_lm.config
        embed_tokens = old_lm.embed_tokens
        old_lm.embed_tokens = None

        stub = _TrtLanguageModelStub(
            config=config,
            embed_tokens=embed_tokens,
            forward_fn=forward_fn,
        )
        paligemma_model.language_model = stub
        self._trt_lang_model_stub = stub

        if not getattr(pwe, "_mopt_fp8_lang_embedding_installed", False):
            pwe.embed_language_tokens = lambda tokens: embed_tokens(tokens)

        del old_lm
        print(
            colored(
                "release PyTorch language_model layers (保留 embed_tokens + TRT forward)",
                "green",
            )
        )

    def _release_gemma_expert_for_trt(self, reason: str) -> None:
        ge = self.pi05_model.paligemma_with_expert.gemma_expert
        if not hasattr(ge, "model") or ge.model is None:
            return
        old_model = ge.model
        if isinstance(old_model, _TrtExpertModelStub):
            return

        stub = _TrtExpertModelStub(config=old_model.config, forward_fn=old_model.forward)
        ge.model = stub
        self._trt_expert_model_stub = stub
        if hasattr(ge, "lm_head") and ge.lm_head is not None:
            del ge.lm_head
        del old_model
        print(colored(f"release PyTorch gemma_expert layers ({reason})", "green"))
        torch.cuda.empty_cache()

    def _release_gemma_expert_module(self, reason: str) -> None:
        ge = self.pi05_model.paligemma_with_expert.gemma_expert
        released = False
        if hasattr(ge, "model") and ge.model is not None:
            print(colored(f"release PyTorch gemma_expert.model ({reason})", "green"))
            del ge.model
            ge.model = None
            released = True
        if hasattr(ge, "lm_head") and ge.lm_head is not None:
            print(colored(f"release PyTorch gemma_expert.lm_head ({reason})", "green"))
            del ge.lm_head
            released = True
        if released:
            torch.cuda.empty_cache()


class Pi05PyTorchExecutor(Executor):
    def __init__(self, policy):
        super().__init__(policy)
        self.pi05_model = Pi05Model(policy)

    def load_model(self):
        self.pi05_model.model.action_head.model.forward = torch.compile(
            self.pi05_model.model.action_head.model.forward, mode="max-autotune"
        )
