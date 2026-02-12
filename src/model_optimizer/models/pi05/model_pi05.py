import time
import os
import torch

import dataclasses

import jax

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.training import data_loader as _data_loader

from ..model import Model
from .vit import Vit
from .llm import LLM
from .expert import Expert
from termcolor import colored

from model_optimizer.infer.tensorrt.trt_torch import Engine


class Pi05Model(Model):
    def __init__(self, model_name_or_policy, model_path=None, pi05_model=None):
        self.pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_path is None:
            policy = model_name_or_policy
            if hasattr(model_name_or_policy, "_policy"):
                policy = model_name_or_policy._policy
            self.pi05_model = policy._model
            self.embedding_layer = None
            return

        super().__init__(model_name_or_policy, model_path)
        self.pi05_model = pi05_model
        self.embedding_layer = None
        self.policy = None
        print(colored(f"Start Pi05Model load...", "green"))
        self.load()
        print(colored(f"Pi05Model load done.", "green"))

    def __getattr__(self, name):
        return getattr(self.pi05_model, name)

    @property
    def model(self):
        return self.pi05_model

    def val(self, dataset, batch_size, max_data=100, output_dir=None):
#        import pdb;pdb.set_trace()
        config = _config.get_config(dataset)
        config = dataclasses.replace(config, batch_size=1)
#        data_config = config.data.create(config.assets_dirs, config.model)
        data_loader = _data_loader.create_data_loader(
            config,
            framework='pytorch')

        for idx, data in enumerate(data_loader):
            print(colored(f'val idx {idx}', "green"))
#            import pdb;pdb.set_trace()
            obs, action = data
            observation = jax.tree.map(lambda x: x.to(self.pytorch_device, \
                dtype=torch.float32), obs)
            loss = self.policy._model(observation, action)
            print(colored(f'val idx {idx} loss: {loss}', "green"))

    @property
    def config(self):
        return self.pi05_model.config

    def load(self):
        if self.pi05_model is not None:
            print(colored("pi05_model is already set", "red"))
            return
        self.pi05_model = self._get_pi0_model()

    def _get_pi0_model(self):
        config = _config.get_config(self.model_name)
        print(colored(f'pi05 model config: {config}', "dark_grey"))
        policy = policy_config.create_trained_policy(config, self.model_path)
        pi05_model = policy._model
        self.policy = policy
        return pi05_model

    def _release_pytorch_model(self):
        if hasattr(self.pi05_model.paligemma_with_expert.paligemma.model, "vision_tower"):
            del self.pi05_model.paligemma_with_expert.paligemma.model.vision_tower

        self.embedding_layer = self.pi05_model.paligemma_with_expert.paligemma.get_input_embeddings()
        if hasattr(self.pi05_model.paligemma_with_expert.paligemma.model, "language_model"):
            del self.pi05_model.paligemma_with_expert.paligemma.model.language_model

        if hasattr(self.pi05_model.paligemma_with_expert.gemma_expert, "model"):
            del self.pi05_model.paligemma_with_expert.gemma_expert.model
        if hasattr(self.pi05_model.paligemma_with_expert.gemma_expert, "lm_head"):
            del self.pi05_model.paligemma_with_expert.gemma_expert.lm_head
        torch.cuda.empty_cache()

    def setup_tensorrt(self, engine_path, device="cuda"):
        self._release_pytorch_model()
        vit_engine = Engine(os.path.join(engine_path, "vit.engine"))
        llm_engine = Engine(os.path.join(engine_path, "llm.engine"))
        expert_engine = Engine(os.path.join(engine_path, "expert.engine"))
        self.pi05_model.paligemma_with_expert.paligemma.model.vision_tower = vit_engine
        self.pi05_model.paligemma_with_expert.embed_language_tokens = self.embedding_layer
        self.pi05_model.paligemma_with_expert.paligemma.model.language_model = llm_engine

        self.pi05_model.paligemma_with_expert.gemma_expert.model = expert_engine
        return self

    @classmethod
    def construct_from_name_path(cls, model_name, model_path):
        real_name = model_name.split("/")[0]
        print(f'pi05 model name: {real_name}')
        return cls(real_name, model_path)

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

    def export(self, output_dir):
        vit_model = Vit.construct_model(self.pi05_model)
        vit_model.export(output_dir)

        llm_model = LLM.construct_model(self.pi05_model)
        llm_model.export(output_dir)

        expert_model = Expert.construct_model(self.pi05_model)
        expert_model.export(output_dir)
        return vit_model, llm_model, expert_model
