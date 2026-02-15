import os
from abc import ABC, abstractmethod
from functools import partial

import torch
import numpy as np
from termcolor import colored

from model_optimizer.utils.torch_hooks.hooks import hook_module_inputs

from transformers.models.gemma.modeling_gemma import GemmaModel

import logging
logger = logging.getLogger(__name__)


class Pi05CalibCollector(ABC):
    name = "pi05"
    def __init__(self, pytorch_model, save_dir, input_keys):
        self.model = pytorch_model
        self.save_dir = save_dir
        self.input_keys = set(input_keys)
        self.old_forward = None
        self._datas = []
        self.hooks = []
        self.register_hooks()

    def hook_forward_input(self, *args, **kwargs):
        one_input = {}
        for key, value in kwargs.items():
            if key in self.input_keys:
                one_data = value.clone().cpu()
                one_input[key] = one_data
        self._datas.append(one_input)
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
        print(colored(f'collectd {len(self._datas)} datas', 'green'))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(self._datas, f'{self.save_dir}/{self.name}_calib_datas.pt')
        self.unregister_hooks()


class Pi05LLMCalibCollector(Pi05CalibCollector):
    name = "pi05_llm"
    def __init__(self, pi05_model, save_dir, input_keys=['inputs_embeds', 'attention_mask', 'position_ids']):
        super().__init__(pi05_model, save_dir, input_keys)

    def register_hooks(self):
        self.old_forward = self.model.paligemma_with_expert.paligemma.model.language_model.forward
        print(colored(f'hook llm forward', "green"))
        self.model.paligemma_with_expert.paligemma.model.language_model.forward = self.hook_forward_input

    def unregister_hooks(self):
        self.model.paligemma_with_expert.paligemma.model.language_model.forward = self.old_forward

class Pi05ExpertCalibCollector(Pi05CalibCollector):
    name = "pi05_expert"
    def __init__(self, pi05_model, save_dir, input_keys=['attention_mask', 
    'position_ids', 'inputs_embeds', 'adarms_cond', 'past_key_values']):
        super().__init__(pi05_model, save_dir, input_keys)

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
        self._datas.append(one_input)
        return self.old_forward(*args, **kwargs)

    def register_hooks(self):
        self.old_forward = self.model.paligemma_with_expert.gemma_expert.model.forward
        print(colored(f'hook expert forward', "green"))
        self.model.paligemma_with_expert.gemma_expert.model.forward = self.hook_forward_input

    def unregister_hooks(self):
        self.model.paligemma_with_expert.gemma_expert.model.forward = self.old_forward
