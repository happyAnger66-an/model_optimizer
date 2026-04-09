import json
import os
from abc import ABC, abstractmethod

import torch
from termcolor import colored

from model_optimizer.calibrate.pi05_calib_load import MANIFEST_FORMAT

import logging
logger = logging.getLogger(__name__)


class Pi05CalibCollector(ABC):
    name = "pi05"

    def __init__(
        self,
        pytorch_model,
        save_dir,
        input_keys,
        *,
        flush_every: int = 2,
        max_samples: int = 0,
    ):
        self.model = pytorch_model
        self.save_dir = save_dir
        self.input_keys = set(input_keys)
        self.flush_every = max(1, int(flush_every))
        self.max_samples = max(0, int(max_samples))
        self.old_forward = None
        self._pending: list = []
        self._shard_idx = 0
        self._shard_dir: str | None = None
        self._total_samples = 0
        self.hooks = []
        self.register_hooks()

    def _ensure_shard_dir(self) -> str:
        if self._shard_dir is None:
            self._shard_dir = os.path.join(self.save_dir, f"{self.name}_calib_shards")
            os.makedirs(self._shard_dir, exist_ok=True)
        return self._shard_dir

    def _save_shard(self, chunk: list) -> None:
        if not chunk:
            return
        shard_dir = self._ensure_shard_dir()
        path = os.path.join(shard_dir, f"shard_{self._shard_idx:05d}.pt")
        torch.save(chunk, path)
        self._total_samples += len(chunk)
        self._shard_idx += 1

    def _flush_if_needed(self) -> None:
        while len(self._pending) >= self.flush_every:
            chunk = self._pending[: self.flush_every]
            self._pending = self._pending[self.flush_every :]
            self._save_shard(chunk)

    def _append_sample(self, sample) -> None:
        if self.max_samples > 0 and self._total_samples + len(self._pending) >= self.max_samples:
            return
        self._pending.append(sample)
        self._flush_if_needed()

    def hook_forward_input(self, *args, **kwargs):
        one_input = {}
        for key, value in kwargs.items():
            if key in self.input_keys:
                one_data = value.clone().cpu()
                one_input[key] = one_data
        self._append_sample(one_input)
        return self.old_forward(*args, **kwargs)

    @abstractmethod
    def register_hooks(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement register_hooks")

    @abstractmethod
    def unregister_hooks(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement unregister_hooks")

    def stop_collect(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if self._pending:
            self._save_shard(self._pending)
            self._pending.clear()

        manifest_path = os.path.join(self.save_dir, f"{self.name}_calib_manifest.json")
        shard_relpaths: list[str] = []
        if self._shard_dir is not None and self._shard_idx > 0:
            base = os.path.basename(self._shard_dir)
            shard_relpaths = [f"{base}/shard_{i:05d}.pt" for i in range(self._shard_idx)]

        meta = {
            "format": MANIFEST_FORMAT,
            "component": self.name,
            "shard_relpaths": shard_relpaths,
            "total_samples": self._total_samples,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            f.write("\n")

        print(
            colored(
                f"collectd {self._total_samples} datas in {self._shard_idx} shard(s); "
                f"manifest -> {manifest_path}",
                "green",
            ),
            flush=True,
        )
        self.unregister_hooks()


class Pi05LLMCalibCollector(Pi05CalibCollector):
    name = "pi05_llm"
    def __init__(
        self,
        pi05_model,
        save_dir,
        input_keys=("inputs_embeds", "attention_mask", "position_ids"),
        *,
        max_samples: int = 0,
    ):
        super().__init__(pi05_model, save_dir, input_keys, max_samples=max_samples)

    def register_hooks(self):
        self.old_forward = self.model.paligemma_with_expert.paligemma.model.language_model.forward
        print(colored(f'hook llm forward', "green"))
        self.model.paligemma_with_expert.paligemma.model.language_model.forward = self.hook_forward_input

    def unregister_hooks(self):
        self.model.paligemma_with_expert.paligemma.model.language_model.forward = self.old_forward

class Pi05ExpertCalibCollector(Pi05CalibCollector):
    name = "pi05_expert"
    def __init__(
        self,
        pi05_model,
        save_dir,
        input_keys=("attention_mask", "position_ids", "inputs_embeds", "adarms_cond", "past_key_values"),
        *,
        max_samples: int = 0,
    ):
        super().__init__(pi05_model, save_dir, input_keys, max_samples=max_samples)

    def _do_past_key_values(self, past_key_values):
        past_keys, past_values = [], []
        for i in range(len(past_key_values)):
            past_keys.append(past_key_values[i][0].clone().cpu())
            past_values.append(past_key_values[i][1].clone().cpu())

        past_keys_tensor = torch.cat(past_keys, dim=0)
        past_values_tensor = torch.cat(past_values, dim=0)
        return past_keys_tensor, past_values_tensor

    def hook_forward_input(self, *args, **kwargs):
        one_input = {}
        for key, value in kwargs.items():
            if key in self.input_keys:
                if key == 'past_key_values':
                    past_keys_tensor, past_values_tensor = self._do_past_key_values(value)
                    one_input['input_keys'] = past_keys_tensor
                    one_input['input_values'] = past_values_tensor
                else:
                    one_data = value.clone().cpu()
                    one_input[key] = one_data
        self._append_sample(one_input)
        return self.old_forward(*args, **kwargs)

    def register_hooks(self):
        self.old_forward = self.model.paligemma_with_expert.gemma_expert.model.forward
        print(colored(f'hook expert forward', "green"))
        self.model.paligemma_with_expert.gemma_expert.model.forward = self.hook_forward_input

    def unregister_hooks(self):
        self.model.paligemma_with_expert.gemma_expert.model.forward = self.old_forward

class Pi05VitCalibCollector(Pi05CalibCollector):
    name = "pi05_vit"
    def __init__(self, pi05_model, save_dir, input_keys=("pixel_values",), *, max_samples: int = 0):
        super().__init__(pi05_model, save_dir, input_keys, max_samples=max_samples)

    def hook_forward_input(self, *args, **kwargs):
        one_input = {}
#        print(f'arg: {args} kwargs: {kwargs}')
        one_data = args[0].clone().cpu()
        self._append_sample(one_data)
        return self.old_forward(*args, **kwargs)

    def register_hooks(self):
        self.old_forward = self.model.paligemma_with_expert.paligemma.model.vision_tower.forward
        print(colored(f'hook vit forward', "green"))
        self.model.paligemma_with_expert.paligemma.model.vision_tower.forward = self.hook_forward_input

    def unregister_hooks(self):
        self.model.paligemma_with_expert.paligemma.model.vision_tower.forward = self.old_forward


class Pi05DenoiseCalibCollector(Pi05CalibCollector):
    """收集 Pi0.5 denoise_step（一次去噪步）校准输入。

    目标：为 `Pi05DenoiseStep.export()` / TensorRT build 提供与运行时一致的输入张量：
    - prefix_pad_masks: [B, prefix_len] (bool)
    - past_keys/past_values: cat 后 [num_layers, B, num_kv_heads, prefix_len, head_dim]
    - x_t: [B, action_horizon, action_dim] (float32)
    - timestep: [B] (float32)
    """

    name = "pi05_denoise"

    def __init__(
        self,
        pi05_model,
        save_dir,
        input_keys=("prefix_pad_masks", "past_keys", "past_values", "x_t", "timestep"),
        *,
        max_samples: int = 0,
    ):
        super().__init__(pi05_model, save_dir, input_keys, max_samples=max_samples)

    def _stack_past_key_values(self, past_key_values):
        # past_key_values: list/tuple of (k,v) per layer
        past_keys, past_values = [], []
        for i in range(len(past_key_values)):
            past_keys.append(past_key_values[i][0].clone().cpu())
            past_values.append(past_key_values[i][1].clone().cpu())
        past_keys_tensor = torch.cat(past_keys, dim=0)
        past_values_tensor = torch.cat(past_values, dim=0)
        return past_keys_tensor, past_values_tensor

    def hook_forward_input(self, *args, **kwargs):
        # 兼容签名：denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep)
        one_input = {}

        prefix_pad_masks = kwargs.get("prefix_pad_masks", None)
        past_key_values = kwargs.get("past_key_values", None)
        x_t = kwargs.get("x_t", None)
        timestep = kwargs.get("timestep", None)

        if prefix_pad_masks is None and len(args) >= 3:
            prefix_pad_masks = args[2]
        if past_key_values is None and len(args) >= 4:
            past_key_values = args[3]
        if x_t is None and len(args) >= 5:
            x_t = args[4]
        if timestep is None and len(args) >= 6:
            timestep = args[5]

        if prefix_pad_masks is not None and "prefix_pad_masks" in self.input_keys:
            one_input["prefix_pad_masks"] = prefix_pad_masks.clone().cpu()
        if x_t is not None and "x_t" in self.input_keys:
            one_input["x_t"] = x_t.clone().cpu()
        if timestep is not None and "timestep" in self.input_keys:
            one_input["timestep"] = timestep.clone().cpu()
        if past_key_values is not None and ("past_keys" in self.input_keys or "past_values" in self.input_keys):
            pk, pv = self._stack_past_key_values(past_key_values)
            if "past_keys" in self.input_keys:
                one_input["past_keys"] = pk
            if "past_values" in self.input_keys:
                one_input["past_values"] = pv

        self._append_sample(one_input)
        return self.old_forward(*args, **kwargs)

    def register_hooks(self):
        if not hasattr(self.model, "denoise_step"):
            raise AttributeError("pi05_model has no attribute denoise_step; cannot collect denoise calib.")
        self.old_forward = self.model.denoise_step
        print(colored("hook denoise_step", "green"))
        self.model.denoise_step = self.hook_forward_input

    def unregister_hooks(self):
        if self.old_forward is not None:
            self.model.denoise_step = self.old_forward