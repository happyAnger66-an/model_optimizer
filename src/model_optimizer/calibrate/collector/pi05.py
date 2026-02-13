import numpy as np
from termcolor import colored

from model_optimizer.torch_hooks.hooks import hook_module_inputs

from transformers.models.gemma.modeling_gemma import GemmaForCausalLM


class Pi05LLMCalibCollector:
    def __init__(self, pi05_model):
        self.calib_dict = {}
        self._datas = []
        self.pi05_model = pi05_model
        self.hooks = [self.register_hooks(GemmaForCausalLM,
            self.pi05_model.paligemma_with_expert.paligemma.model.language_model)]

    def register_hooks(self, target_cls, target_model):
        def hook_input(m, args, kwargs):
            print(
                colored(f'hook module input: {type(m)} args:{len(args)} ', "green"))
            for arg in args:
                one_input = arg.clone().cpu().numpy()
                print(colored(f'one_input shape: {one_input.shape}', "green"))

        self.hooks = hook_module_inputs(target_model,
                                        hook_input, target_cls)

    def stop_collect(self):
        for hook in self.hooks:
            hook.remove()
