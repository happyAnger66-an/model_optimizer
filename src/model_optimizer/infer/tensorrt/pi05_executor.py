import os
from functools import partial

import torch

from ..executor import Executor
from ...models.pi05.model_pi05 import Pi05Model
from .trt_torch import Engine


#def embed_image(self, pixel_values):
#    self.get_image_features(pixel_values)

class Pi05TensorRTExecutor(Executor):
    def __init__(self, config, pytorch_model_path, engine_path):
        super().__init__(config)
        self.pytorch_model_path = pytorch_model_path
        self.engine_path = engine_path
        pi05_model = Pi05Model(config.model_name, pytorch_model_path)
        pi05_model.load()
        self.pi05_model = pi05_model.model

    def load_model(self):
        self._release_pytorch_model()
        self._setup_trt_engine()
      #  self.pi05_model.paligemma_with_expert.embed_image = partial(
      #      embed_image, self.pi05_model.paligemma_with_expert.paligemma.model)
        self.pi05_model.paligemma_with_expert.embed_language_tokens = self.embedding_layer

    def infer(self, input_data):
        self.pi05_model.sample_actions(input_data)

    def _setup_trt_engine(self):
        vit_engine = Engine(os.path.join(self.engine_path, "vit.engine"))
        llm_engine = Engine(os.path.join(self.engine_path, "llm.engine"))
        self.pi05_model.paligemma_with_expert.paligemma.model.vision_tower = vit_engine
        self.pi05_model.paligemma_with_expert.paligemma.model.language_model = llm_engine

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
