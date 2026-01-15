import time
import os
import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training import config as _config
from openpi.policies import policy_config

from ..model import Model, register_model
from .vit import Vit
from .llm import LLM
from .expert import Expert


class Pi05Model(Model):
    def __init__(self, model_name, model_path):
        super().__init__(model_name, model_path)

    def load(self, config):
        self.pi05_model = self._get_pi0_model(self.model_name, self.model_path)

    def _get_pi0_model(self):
        config = _config.get_config(self.model_name)
        policy = policy_config.create_trained_policy(config, self.model_path)
        pi_model = policy._model

        return pi_model

    @classmethod
    def construct_from_name_path(cls, model_name, model_path):
        return cls(model_name, model_path)
    
    @property
    def model(self):
        return self.pi05_model

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def quantize(self, *args, **kwargs):
        quant_cfg = args[0]

    def quantize_sub_model(self, *args, **kwargs):
        sub_model_name, model_dir, quant_cfg, calib_data, calib_method = args
        if sub_model_name == "vit":
            sub_model = Vit.construct_model(self.pi05_model)
        elif sub_model_name == "llm":
            sub_model = LLM.construct_model(self.pi05_model)
        elif sub_model_name == "expert":
            sub_model = Expert.construct_model(self.pi05_model)
        else:
            raise ValueError(f"Invalid sub model name: {sub_model_name}")
        sub_model.quantize(model_dir, quant_cfg, calib_data, calib_method)

    def export_onnx(self, *args, **kwargs):
        export_dir = args[0]
        vit_model = Vit.export_onnx(self.pi05_model, export_dir)
        llm_model = LLM.export_onnx(self.pi05_model, export_dir)
        expert_model = Expert.export_onnx(self.pi05_model, export_dir)
        return vit_model, llm_model, expert_model