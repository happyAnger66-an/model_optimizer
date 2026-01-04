import os
import torch

from openpi.training import config as _config
from openpi.policies import policy_config

from ..models.pi0 import Pi05Vit
from .cfg import build_quant_cfg, QUANT_CFG_CHOICES
from ..calibrate.pi0 import ViTCalibrationDataset, get_data_loader

import modelopt.torch.quantization as mtq

def load_pi0_model(config_name, checkpoint_dir):
    model_config = _config.get_config(config_name)
    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    print(f"Loading pi0 model {weight_path} ...")
    if is_pytorch:
        print(f'load pytorch model....')
        model = model_config.model.load_pytorch(model_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        raise RuntimeError(f"pi0 only support pytorch model.")

    model.eval().cuda() 
    print(f"Loading pi0 model {weight_path} Done.")
    return model

def quantize_pi05_vit(args):
    config_name = args.config_name
    config = _config.get_config(config_name)
    policy = policy_config.create_trained_policy(config, args.model_path)
    pi_model = policy._model

    paligemma = pi_model.paligemma_with_expert.paligemma.model
    vit_model = Pi05Vit(paligemma.vision_tower, paligemma.multi_modal_projector)
    vit_model.eval().cuda()

    dataloader = get_data_loader(config_name)

    def calibrate_loop(model):
        """Adjusts weights and scaling factors based on selected algorithms."""
        for idx, data in enumerate(dataloader):
#            import pdb
#            pdb.set_trace()
            input_data, label = data
            input_data = pi_model._preprocess_observation(input_data)
            if idx % 10 == 0:
                print(f"Calibrating batch {idx}...")
            images = input_data[0]
            for i, img in enumerate(images):
                cuda_img = img.cuda()
                print(f"{idx}th img {idx}:  {img.shape}")
                model(cuda_img)

    quant_cfg = QUANT_CFG_CHOICES[args.qformat]
    quant_cfg["algorithm"] = args.calibrate_method
    print(f"Starting quantization... quant_cfg: {quant_cfg}")

    import time
    start_time = time.time()
    mtq.quantize(vit_model, quant_cfg, forward_loop=calibrate_loop)
    end_time = time.time()
    print(f"Quantization finishes in {end_time - start_time}s.")

    mtq.print_quant_summary(vit_model)


    pixel_values = torch.randn((1, 3, 224, 224),
        dtype=torch.float16,
        device="cuda",
    )

    output_dir = args.export_dir
    vit_dtype = args.qformat
    os.makedirs(output_dir, exist_ok=True)
    start = time.time()
    print(f"Start export onnx ...")
    with torch.inference_mode():
        torch.onnx.export(
            vit_model,
            (pixel_values),  # Include position_ids in ONNX export
            f"{output_dir}/vit_{vit_dtype}.onnx",
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
