import os
import time
import torch

from ..model import Model
from ..token import get_tokenizer
import logging

from transformers.cache_utils import DynamicCache

from termcolor import colored

logger = logging.getLogger(__name__)


class LLM(torch.nn.Module, Model):
    def __init__(self, config, llm, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.config = config
        self.llm.config._attn_implementation = "eager"

    def _wrap_past_key_values(self, input_keys, input_values):
        k_v_cache = DynamicCache()
        num_layers = input_keys.shape[0]
        for i in range(num_layers):
            k_v_cache.update(input_keys[i:i+1], input_values[i:i+1], i)

        return k_v_cache

    def forward(self, inputs_embeds, attention_mask, position_ids):
        prefix_output = self.llm(inputs_embeds=inputs_embeds,
                                 attention_mask=attention_mask,
                                 position_ids=position_ids,
                                 use_cache=True)
        past_key_caches = prefix_output.past_key_values
        past_keys = []
        past_values = []
        for i in range(self.config.num_hidden_layers):
            past_keys.append(past_key_caches[i][0])
            past_values.append(past_key_caches[i][1])

        past_keys_tensor = torch.cat(past_keys, dim=0)
        past_values_tensor = torch.cat(past_values, dim=0)

        return past_keys_tensor, past_values_tensor, prefix_output.last_hidden_state

    def quantize(self, model_dir, quant_cfg, calib_data, calib_method):
        tokenizer = get_tokenizer(model_dir)

        from model_optimizer.quantization.llm_quantization import quantize_llm
        quantize_llm(self, tokenizer, quant_cfg, calib_data, calib_method)

    @classmethod
    def construct_from_name_path(cls, model_name, model_path):
        from .model_pi05 import Pi05Model
        pi05_model = Pi05Model.construct_from_name_path(model_name, model_path)
        return cls.construct_model(pi05_model)

    @classmethod
    def construct_model(cls, pi05_model, dtype=torch.bfloat16):
        paligemma = pi05_model.paligemma_with_expert.paligemma
        llm_model = cls(pi05_model.paligemma_with_expert.paligemma.config.text_config,
                        paligemma.get_decoder())
        return llm_model

    def export(self, export_dir):
        self.eval().cuda()

        output_dir = export_dir
        os.makedirs(output_dir, exist_ok=True)
        start = time.time()
        logger.info("Start export onnx ...")
        print(colored(f"Start LLM export onnx...", "green"))
        inputs_embeds = torch.randn((1, 968, 2048),
                                    dtype=torch.bfloat16,
                                    device="cuda",
                                    )
        attention_mask = torch.randn((1, 1, 968, 968),
                                     dtype=torch.float32,
                                     device="cuda",
                                     )
        position_ids = torch.randint(1, 1000, (1, 968),
                                     dtype=torch.int64,
                                     device="cuda",
                                     )
    # am: torch.Size([1, 1, 968, 968]) - torch.float32, pi: torch.Size([1, 968])-torch.int64 prefix: torch.Size([1, 968, 2048])-torch.bfloat16
        with torch.inference_mode():
            torch.onnx.export(
                self,
                # Include position_ids in ONNX export
                (inputs_embeds, attention_mask, position_ids),
                f"{output_dir}/llm.onnx",
                input_names=["inputs_embeds", "attention_mask",
                             "position_ids"],  # Add position_ids to input names
                output_names=["past_keys", "past_values", "last_hidden_state"],
                opset_version=19,
                dynamo=True,
                do_constant_folding=True,
                dynamic_axes={
                    "inputs_embeds": {0: "batch_size", 1: "seq_len"},
                    "attention_mask": {0: "batch_size", 1: "num_heads", 2: "seq_len", 3: "seq_len"},
                    "position_ids": {0: "batch_size", 1: "seq_len"},
                },
            )
        end = time.time()
        logger.info(f"export onnx to {output_dir} done cost:{end - start}s")
        print(
            colored(f"LLM export onnx done to {output_dir} cost:{end - start}s", "green"))
        return self

    @classmethod
    def export_onnx(cls, pi_model, export_dir):
        del pi_model.paligemma_with_expert.gemma_expert

        llm_model = cls.construct_model(pi_model, dtype=torch.float16)
        llm_model.eval().cuda()

        output_dir = export_dir
        os.makedirs(output_dir, exist_ok=True)
        start = time.time()
        logger.info("Start export onnx ...")
        inputs_embeds = torch.randn((1, 968, 2048),
                                    dtype=torch.float16,
                                    device="cuda",
                                    )
        attention_mask = torch.randn((1, 1, 968, 968),
                                     dtype=torch.float32,
                                     device="cuda",
                                     )
        position_ids = torch.randint(1, 1000, (1, 968),
                                     dtype=torch.int64,
                                     device="cuda",
                                     )
    # am: torch.Size([1, 1, 968, 968]) - torch.float32, pi: torch.Size([1, 968])-torch.int64 prefix: torch.Size([1, 968, 2048])-torch.bfloat16
        with torch.inference_mode():
            torch.onnx.export(
                llm_model,
                # Include position_ids in ONNX export
                (inputs_embeds, attention_mask, position_ids),
                f"{output_dir}/llm.onnx",
                input_names=["inputs_embeds", "attention_mask",
                             "position_ids"],  # Add position_ids to input names
                output_names=["last_hidden_state"],
                opset_version=19,
                dynamo=False,
                do_constant_folding=True,
                dynamic_axes={
                    "inputs_embeds": {0: "batch_size"},
                    "attention_mask": {0: "batch_size"},
                    "position_ids": {0: "batch_size"},
                },
            )
        end = time.time()
        logger.info(f"export onnx to {output_dir} done cost:{end - start}s")
        return llm_model
