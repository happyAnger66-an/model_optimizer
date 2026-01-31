import os
import time
import torch

import logging
from ..model import Model
from termcolor import colored

logger = logging.getLogger(__name__)


class Expert(torch.nn.Module, Model):
    def __init__(self, config, gemma_expert, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.gemma_expert = gemma_expert

    def forward(self, attention_mask, position_ids, inputs_embeds, past_key_values=None):
        logger.info(
            f'Pi05Expert input attention_mask: {attention_mask.shape} position_ids: {position_ids.shape} inputs_embeds: {inputs_embeds.shape}')
        time_emb = torch.zeros(1, 1024, dtype=torch.float16, device="cuda")
        output = self.gemma_expert(attention_mask=attention_mask, position_ids=position_ids,
                                   inputs_embeds=inputs_embeds, adarms_cond=time_emb)
        logger.info(f'Pi05Expert output: {output.last_hidden_state.shape}')
        return output.last_hidden_state

    @classmethod
    def construct_from_name_path(cls, model_name, model_path):
        from .model_pi05 import Pi05Model
        pi05_model = Pi05Model.construct_from_name_path(model_name, model_path)
        return cls.construct_model(pi05_model)

    @classmethod
    def construct_model(cls, pi05_model, dtype=torch.float16):
        gemma_expert_model = pi05_model.paligemma_with_expert.gemma_expert.model
        config = pi05_model.paligemma_with_expert.gemma_expert.config
        expert_model = cls(config, gemma_expert_model).to(dtype)
        return expert_model

    def export(self, export_dir):
        self.eval().cuda()
#        self.to(torch.float16)

        output_dir = export_dir
        os.makedirs(output_dir, exist_ok=True)
        start = time.time()
        logger.info("Start export onnx ...")
        print(colored(f"Start Expert export onnx...", "green"))

        logger.info(f'gemma_expert_model {self.gemma_expert}')
        logger.info(f'config {self.config}')

        attention_mask = torch.randn((1, 1, 10, 978),
                                     dtype=torch.float16,
                                     device="cuda")
        position_ids = torch.randint(1, self.config.vocab_size, (1, 10),
                                     dtype=torch.int64,
                                     device="cuda")
        inputs_embeds = torch.randn((1, 10, 1024),
                                    dtype=torch.float16,
                                    device="cuda")

        output_path = f"{output_dir}/expert.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                self,
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
        print(
            colored(f"Expert export onnx done to {output_dir} cost:{end - start}s", "green"))
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
