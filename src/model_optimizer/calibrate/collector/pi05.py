import os
from functools import partial

import torch
import numpy as np
from termcolor import colored

from model_optimizer.utils.torch_hooks.hooks import hook_module_inputs

from transformers.models.gemma.modeling_gemma import GemmaModel

import logging
logger = logging.getLogger(__name__)


class Pi05CalibCollector:
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

    def register_hooks(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement register_hooks")

    def unregister_hooks(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement unregister_hooks")

    def stop_collect(self):
        print(colored(f'collectd {len(self._datas)} datas', 'green'))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(self._datas, f'{self.save_dir}/pi05_llm_calib_data.pt')
        self.unregister_hooks()


class Pi05LLMCalibCollector(Pi05CalibCollector):
    def __init__(self, pi05_model, save_dir, input_keys=['inputs_embeds', 'attention_mask', 'position_ids']):
        super().__init__(pi05_model, save_dir, input_keys)

    def register_hooks(self):
        self.old_forward = self.pi05_model.paligemma_with_expert.paligemma.model.language_model.forward
        print(colored(f'hook llm forward', "green"))
        self.pi05_model.paligemma_with_expert.paligemma.model.language_model.forward = self.hook_forward_input

    def unregister_hooks(self):
        self.pi05_model.paligemma_with_expert.paligemma.model.language_model.forward = self.old_forward
