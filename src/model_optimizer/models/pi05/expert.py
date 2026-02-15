from transformers.cache_utils import DynamicCache
import os
import time
import torch

import logging
from ..model import Model
from termcolor import colored

from model_optimizer.quantization.quantization_utils import quantize_model

logger = logging.getLogger(__name__)


class Expert(torch.nn.Module, Model):
    def __init__(self, config, gemma_expert, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.gemma_expert = gemma_expert # this is gemma_expert.model
        self.device = self.gemma_expert.device
        self.gemma_expert.config._attn_implementation = "eager"

    def _wrap_past_key_values(self, input_keys, input_values):
        k_v_cache = DynamicCache()
#        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        num_layers = input_keys.shape[0]
        for i in range(num_layers):
            k_v_cache.update(input_keys[i:i+1], input_values[i:i+1], i)

        return k_v_cache

    def forward(self, attention_mask, position_ids, inputs_embeds, adarms_cond=None, input_keys=None, input_values=None):
        logger.info(
            f'Pi05Expert input attention_mask: {attention_mask.shape} position_ids: {position_ids.shape} inputs_embeds: {inputs_embeds.shape}')
        k_v_cache = None
        if input_keys is not None and input_values is not None:
            k_v_cache = self._wrap_past_key_values(input_keys, input_values)

        output = self.gemma_expert(attention_mask=attention_mask, position_ids=position_ids,
                                   inputs_embeds=inputs_embeds, adarms_cond=adarms_cond,
                                   past_key_values=k_v_cache)
        logger.info(f'Pi05Expert output: {output.last_hidden_state.shape}')
        return output.last_hidden_state

    @classmethod
    def construct_from_name_path(cls, model_name, model_path):
        from .model_pi05 import Pi05Model
        pi05_model = Pi05Model.construct_from_name_path(model_name, model_path)
        return cls.construct_model(pi05_model, dtype=torch.bfloat16)

    @classmethod
    def construct_model(cls, pi05_model, dtype=torch.bfloat16):
        gemma_expert_model = pi05_model.paligemma_with_expert.gemma_expert.model
        config = pi05_model.paligemma_with_expert.gemma_expert.config
        expert_model = cls(config, gemma_expert_model)
        return expert_model
    
    def quantize(self, quant_cfg, calib_data, export_dir):
        calib_dataloader = self.get_calibrate_dataset(calib_data)
        quantize_model(self, quant_cfg, calib_dataloader)
        self.export(export_dir, dynamo=False)

    def export(self, export_dir, export_dtype=torch.bfloat16, dynamo=True):
        self.eval().cuda()

#        old_dense_forward = self.gemma_expert.norm.dense.forward
#        old_input_layernorm_dense_forward = self.gemma_expert.input_layernorm.dense.forward
#
#        def norm_dense_forward(old_model, cond):
#            cond = cond.to(torch.float32)
#            return old_dense_forward(old_model, cond)

#        def input_layernorm_dense_forward(old_model, cond):
#            cond = cond.to(torch.float32)
#            return old_input_layernorm_dense_forward(old_model, cond)

#        self.gemma_expert.norm.dense.forward = norm_dense_forward
#        self.gemma_expert.input_layernorm.dense.forward = input_layernorm_dense_forward

        output_dir = export_dir
        os.makedirs(output_dir, exist_ok=True)
        start = time.time()
        logger.info("Start export onnx ...")
        print(colored(f"Start Expert export onnx...", "green"))

#        logger.info(f'gemma_expert_model {self.gemma_expert}')
        logger.info(f'config {self.config}')
        print(colored(f'gemma_expert model config {self.config}', "dark_grey"))

        # time embeds
        adarms_cond = torch.zeros(1, 1024, dtype=export_dtype, device="cuda")

        # attention mask
        attention_mask = torch.randn((1, 1, 10, 978),
                                     dtype=export_dtype,
                                     device="cuda")

        # position ids
        position_ids = torch.randint(1, self.config.vocab_size, (1, 10),
                                     dtype=torch.int64,
                                     device="cuda")

        # action embeds
        inputs_embeds = torch.randn((1, 10, 1024),
                                    dtype=export_dtype,
                                    device="cuda")

        # past key values
        past_keys = []
        past_values = []
        for _ in range(18):
            past_keys.append(torch.randn((1, 1, 968, 256),
                             dtype=export_dtype, device="cuda"))
            past_values.append(torch.randn(
                (1, 1, 968, 256), dtype=export_dtype, device="cuda"))

        past_keys_tensor = torch.cat(past_keys, dim=0)
        past_values_tensor = torch.cat(past_values, dim=0)

        output_path = f"{output_dir}/expert.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                self,
                # Include position_ids in ONNX export
                (attention_mask, position_ids, inputs_embeds,
                 adarms_cond, past_keys_tensor, past_values_tensor),
                output_path,
                input_names=["attention_mask", "position_ids",
                             # Add position_ids to input names
                             "inputs_embeds", "adarms_cond", "past_keys", "past_values"],
                output_names=["last_hidden_state"],
                opset_version=19,
                dynamo=dynamo,
                do_constant_folding=True,
                dynamic_axes={
                    "attention_mask": {0: "batch_size"},
                    "position_ids": {0: "batch_size"},
                    "inputs_embeds": {0: "batch_size"},
                    "adarms_cond": {0: "batch_size"},
                    "past_keys": {2: "llm_seq_len"},
                    "past_values": {2: "llm_seq_len"},
                },
            )
        end = time.time()
        logger.info(f"export onnx to {output_dir} done cost:{end - start}s")
        print(
            colored(f"Expert export onnx done to {output_dir} dtype:{export_dtype} cost:{end - start}s", "green"))
        return self

    @classmethod
    def export_onnx(cls, pi05_model, export_dir):
        expert_model = cls.construct_model(pi05_model, dtype=torch.float16)
        expert_model.eval().cuda()

        logger.info(f'gemma_expert_model {expert_model.gemma_expert}')
        logger.info(f'config {expert_model.config}')

        attention_mask = torch.randn((1, 1, 10, 978),
                                     dtype=torch.float16,
                                     device="cuda")
        position_ids = torch.randint(1, expert_model.config.vocab_size, (1, 10),
                                     dtype=torch.int64,
                                     device="cuda")
        inputs_embeds = torch.randn((1, 10, 1024),
                                    dtype=torch.float16,
                                    device="cuda")

        output_dir = export_dir
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/expert.onnx"
        start = time.time()
        logger.info(f"Start export onnx ...")
        with torch.inference_mode():
            torch.onnx.export(
                expert_model,
                # Include position_ids in ONNX export
                (attention_mask, position_ids, inputs_embeds),
                output_path,
                input_names=["attention_mask", "position_ids",
                             "inputs_embeds"],  # Add position_ids to input names
                output_names=["hidden_states"],
                opset_version=19,
                dynamo=False,
                do_constant_folding=True,
                dynamic_axes={
                    "attention_mask": {0: "batch_size", 2: "action_seq_len", 3: "llm_seq_len"},
                    "position_ids": {0: "batch_size", 1: "seq_len"},
                    "inputs_embeds": {0: "batch_size", 1: "seq_len"},
                },
            )
        end = time.time()
        logger.info(f"export onnx to {output_dir} done cost:{end - start}s")
        return expert_model, output_path
