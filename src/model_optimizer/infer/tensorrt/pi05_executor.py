import os
from functools import partial

from termcolor import colored
import torch

from ..executor import Executor
from ...models.pi05.model_pi05 import Pi05Model
from .trt_torch import Engine


# def embed_image(self, pixel_values):
#    self.get_image_features(pixel_values)

class Pi05TensorRTExecutor(Executor):
    def __init__(self, policy, precision=torch.bfloat16, config=None):
        super().__init__(policy)
        pi05_model = Pi05Model(policy)
        self.pi05_model = pi05_model.model
#        self.pi05_model.to(precision)
        self.config = config

    def load_model(self, config=None):
        if config is None:
            return

        self.config = config
        self._release_pytorch_model()
        self._setup_trt_engine()
      #  self.pi05_model.paligemma_with_expert.embed_image = partial(
      #      embed_image, self.pi05_model.paligemma_with_expert.paligemma.model)
        self.pi05_model.paligemma_with_expert.embed_language_tokens = self.embedding_layer

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def _setup_trt_engine(self):
        if self.config.engine_path:
            if self.config.vit_engine:
                print(
                    colored(f"replace vision_tower with {self.config.vit_engine}", "green"))
                vit_engine = Engine(os.path.join(
                    self.config.engine_path, self.config.vit_engine))
                self.pi05_model.paligemma_with_expert.paligemma.model.vision_tower = vit_engine
            if self.config.llm_engine:
                print(
                    colored(f"replace language_model with {self.config.llm_engine}", "green"))
                llm_engine = Engine(os.path.join(
                    self.config.engine_path, "llm.engine"))
                self.pi05_model.paligemma_with_expert.paligemma.model.language_model = llm_engine
            if self.config.expert_engine:
                print(
                    colored(f"replace expert with {self.config.expert_engine}", "green"))
                expert_engine = Engine(os.path.join(
                    self.config.engine_path, "expert.engine"))
                self.pi05_model.paligemma_with_expert.gemma_expert.model = expert_engine

    def _release_pytorch_model(self):
        if self.config.vit_engine:
            print(colored(f"release vision_tower engine", "green"))
            if hasattr(self.pi05_model.paligemma_with_expert.paligemma.model, "vision_tower"):
                del self.pi05_model.paligemma_with_expert.paligemma.model.vision_tower

        if self.config.llm_engine:
            print(colored(f"release language_model engine", "green"))
            if hasattr(self.pi05_model.paligemma_with_expert.paligemma.model, "language_model"):
                del self.pi05_model.paligemma_with_expert.paligemma.model.language_model

        self.embedding_layer = self.pi05_model.paligemma_with_expert.paligemma.get_input_embeddings()

        if self.config.expert_engine:
            print(colored(f"release expert engine", "green"))
            if hasattr(self.pi05_model.paligemma_with_expert.gemma_expert, "model"):
                del self.pi05_model.paligemma_with_expert.gemma_expert.model

            if hasattr(self.pi05_model.paligemma_with_expert.gemma_expert, "lm_head"):
                del self.pi05_model.paligemma_with_expert.gemma_expert.lm_head
        torch.cuda.empty_cache()


class Pi05PyTorchExecutor(Executor):
    def __init__(self, policy):
        super().__init__(policy)
        self.pi05_model = Pi05Model(policy)

    def load_model(self):
        self.pi05_model.model.action_head.model.forward = torch.compile(
            self.pi05_model.model.action_head.model.forward, mode="max-autotune"
        )
