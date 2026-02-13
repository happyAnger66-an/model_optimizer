import os

import torch
import numpy as np
from termcolor import colored

from model_optimizer.utils.torch_hooks.hooks import hook_module_inputs

from transformers.models.gemma.modeling_gemma import GemmaModel

import logging
logger = logging.getLogger(__name__)

class Pi05LLMCalibCollector:
    def __init__(self, pi05_model, save_dir):
        self.pi05_model = pi05_model
        self.calib_dict = {}
        self._datas = []
        self.hooks = []
        self.keys = set(['inputs_embeds', 'attention_mask', 'position_ids'])
        self.old_forward = None
        self.save_dir = save_dir
        self.register_hooks()

    def register_hooks(self):
        self.old_forward = self.pi05_model.paligemma_with_expert.paligemma.model.language_model.forward

        def hook_forward_input(*args, **kwargs):
            one_input = {}
            for key, value in kwargs.items():
                if key in self.keys:
                    one_data = value.clone().cpu()
                    one_input[key] = one_data
            self._datas.append(one_input)
            return self.old_forward(*args, **kwargs)
        print(colored(f'hook llm forward', "green"))
        self.pi05_model.paligemma_with_expert.paligemma.model.language_model.forward = hook_forward_input

    def stop_collect(self):
        print(colored(f'collectd {len(self._datas)} datas', 'green'))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(self._datas, f'{self.save_dir}/pi05_llm_calib_data.pt')
        self.pi05_model.paligemma_with_expert.paligemma.model.language_model.forward = self.old_forward
