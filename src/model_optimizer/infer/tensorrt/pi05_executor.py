import os
from functools import partial

from termcolor import colored
import torch

from ..executor import Executor
from ...models.pi05.model_pi05 import Pi05Model
from .trt_torch import Engine

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.cache_utils import DynamicCache

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
        self._setup_trt_engine()
      #  self._release_pytorch_model()
      #  self.pi05_model.paligemma_with_expert.embed_image = partial(
      #      embed_image, self.pi05_model.paligemma_with_expert.paligemma.model)
      #  self.pi05_model.paligemma_with_expert.embed_language_tokens = self.embedding_layer

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def _wrap_past_key_values(self, input_keys, input_values):
        k_v_cache = DynamicCache()
        num_layers = input_keys.shape[0]
        for i in range(num_layers):
            k_v_cache.update(input_keys[i:i+1], input_values[i:i+1], i)

        return k_v_cache

    def _setup_trt_engine(self):
        if self.config.engine_path:
            if self.config.vit_engine:
                print(
                    colored(f"replace vision_tower with {self.config.vit_engine}", "green"))

                def vit_return_wrap(output):
                    return output['last_hidden_state']

                vit_engine = Engine(os.path.join(
                    self.config.engine_path, self.config.vit_engine), return_wrap=vit_return_wrap,
                    perf=True)

                def get_image_features(pixel_values):
                    return vit_engine(pixel_values)

                self.pi05_model.paligemma_with_expert.paligemma.model.get_image_features = get_image_features

            if self.config.llm_engine:
                print(
                    colored(f"replace language_model with {self.config.llm_engine}", "green"))
                llm_engine = Engine(os.path.join(
                    self.config.engine_path, "llm.engine"))

                def llm_forward(input_ids=None,
                                attention_mask=None,
                                position_ids=None,
                                past_key_values=None,
                                inputs_embeds=None,
                                labels=None,
                                use_cache=False,
                                output_attentions=False,
                                output_hidden_states=False,
                                cache_position=None,
                                logits_to_keep=None,
                                adarms_cond=None, **kwargs):
                    outputs = llm_engine(
                        inputs_embeds, attention_mask, position_ids)
                    k_v_cache = self._wrap_past_key_values(
                        outputs['past_keys'],
                        outputs['past_values'])

                    output = BaseModelOutputWithPooling(
                        last_hidden_state=outputs['last_hidden_state'],
                        past_key_values=k_v_cache
                    )
                    return output

                self.pi05_model.paligemma_with_expert.paligemma.model.language_model.forward = llm_forward

            if self.config.expert_engine:
                print(
                    colored(f"replace expert with {self.config.expert_engine}", "green"))

                def expert_return_wrap(output):
                    #                    print(
                    #                        colored(f"expert_return_warp output: {output}", "green"))
                    output = BaseModelOutputWithPooling(
                        last_hidden_state=output['last_hidden_state'],
                    )
                    return output

                expert_engine = Engine(os.path.join(
                    self.config.engine_path, "expert.engine"), return_wrap=expert_return_wrap, perf=True)

                def expert_forward(inputs_ids=None, attention_mask=None,
                                   position_ids=None,
                                   past_key_values=None,
                                   inputs_embeds=None,
                                   use_cache=False,
                                   output_attentions=False,
                                   output_hidden_states=False,
                                   cache_position=None,
                                   adarms_cond=None,
                                   **kwargs):
                    input_keys = torch.cat(
                        [past_key_values[i][0] for i in range(len(past_key_values))], dim=0)
                    input_values = torch.cat(
                        [past_key_values[i][1] for i in range(len(past_key_values))], dim=0)
#                    for i in range(len(past_key_values)):
#                        input_key_values.append(
#                            (past_key_values[i][0], past_key_values[i][1]))
                    return expert_engine(attention_mask, position_ids, inputs_embeds, adarms_cond, input_keys, input_values)

                self.pi05_model.paligemma_with_expert.gemma_expert.model.forward = expert_forward

    def _release_pytorch_model(self):
        if self.config.vit_engine:
            print(colored(f"release vision_tower engine", "green"))
            if hasattr(self.pi05_model.paligemma_with_expert.paligemma.model, "vision_tower"):
                del self.pi05_model.paligemma_with_expert.paligemma.model.vision_tower

        if self.config.llm_engine:
            print(colored(f"release language_model engine", "green"))
            self.embedding_layer = self.pi05_model.paligemma_with_expert.paligemma.get_input_embeddings()

            if hasattr(self.pi05_model.paligemma_with_expert.paligemma.model, "language_model"):
                del self.pi05_model.paligemma_with_expert.paligemma.model.language_model

        if self.config.expert_engine:
            print(colored(f"release expert engine", "green"))
#            if hasattr(self.pi05_model.paligemma_with_expert.gemma_expert, "model"):
#                del self.pi05_model.paligemma_with_expert.gemma_expert.model

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
