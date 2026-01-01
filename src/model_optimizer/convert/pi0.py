import os
import time
import torch

from openpi.training import config as _config
from openpi.policies import policy_config

from ..quantization.models.pi0 import Pi05Vit, Pi0LLM

def get_pi0_model(model_name, model_path):
    config = _config.get_config(model_name)
    policy = policy_config.create_trained_policy(config, model_path)
    pi_model = policy._model

    return pi_model

def convert_llm(model_name, model_path, export_dir):
    pi_model = get_pi0_model(model_name, model_path)
    del pi_model.paligemma_with_expert.gemma_expert
#    del pi_model.paligemma_with_expert.paligemma.model.vision_tower

    paligemma = pi_model.paligemma_with_expert.paligemma.model
    llm_model = Pi0LLM(paligemma.get_decoder()).to(torch.float16)
    llm_model.eval().cuda()

    output_dir = export_dir
    os.makedirs(output_dir, exist_ok=True)
    start = time.time()
    print(f"Start export onnx ...")
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
#am: torch.Size([1, 1, 968, 968]) - torch.float32, pi: torch.Size([1, 968])-torch.int64 prefix: torch.Size([1, 968, 2048])-torch.bfloat16
    with torch.inference_mode():
        torch.onnx.export(
            llm_model,
            (inputs_embeds, attention_mask, position_ids),  # Include position_ids in ONNX export
            f"{output_dir}/llm.onnx",
            input_names=["inputs_embeds", "attention_mask", "position_ids"],  # Add position_ids to input names
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
    print(f"export onnx to {output_dir} done cost:{end - start}s")
    return llm_model

def convert_vit(model_name, model_path, export_dir):
    pi_model = get_pi0_model(model_name, model_path)

    paligemma = pi_model.paligemma_with_expert.paligemma.model
    config = pi_model.paligemma_with_expert.paligemma.config
    vit_model = Pi05Vit(config, paligemma.vision_tower, paligemma.multi_modal_projector).to(torch.float16)
    vit_model.eval().cuda()

    pixel_values = torch.randn((1, 3, 224, 224),
        dtype=torch.float16,
        device="cuda",
    )

    output_dir = export_dir
    os.makedirs(output_dir, exist_ok=True)
    start = time.time()
    print(f"Start export onnx ...")
    with torch.inference_mode():
        torch.onnx.export(
            vit_model,
            (pixel_values),  # Include position_ids in ONNX export
            f"{output_dir}/vit.onnx",
            input_names=["pixel_values"],  # Add position_ids to input names
            output_names=["vit_embeds"],
            opset_version=19,
            dynamo=False,
            do_constant_folding=True,
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "vit_embeds": {0: "batch_size"},
            },
        )
    end = time.time()
    print(f"export onnx to {output_dir} done cost:{end - start}s")
    return vit_model

def convert_pi05_model(model_name, model_path ,export_dir):
    convert_llm(model_name, model_path, export_dir)
#    convert_vit(model_name, model_path, export_dir)
