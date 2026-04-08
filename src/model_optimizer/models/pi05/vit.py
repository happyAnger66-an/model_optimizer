from ..model import Model
import os
import time
import torch

from termcolor import colored

import logging

from model_optimizer.calibrate.pi05_calib_load import open_pi05_calib_for_quantize
from model_optimizer.quantization.quantization_utils import quantize_model
from model_optimizer.utils.utils import is_nvfp4_quantized, set_dynamic_quant

logger = logging.getLogger(__name__)


class Vit(torch.nn.Module, Model):
    def __init__(self, config, vision_tower, multi_modal_projector, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.vision_tower = vision_tower
        self.device = self.vision_tower.device
        self.multi_modal_projector = multi_modal_projector

    def get_calibrate_dataset(self, calib_data):
        return open_pi05_calib_for_quantize(calib_data, component="pi05_vit")

    def forward(self, pixel_values):
        #        logger.info(f'Pi05Vit input: {pixel_values.shape}')
        image_outputs = self.vision_tower(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = image_features / \
            (self.config.text_config.hidden_size ** 0.5)
#        logger.info(f'Pi05Vit output: {image_features.shape}')
        return image_features

    def export(self, export_dir, dynamo=True, mode=None):
        self.eval().cuda()

        pixel_values = torch.randn(
            (1, 3, 224, 224), dtype=torch.bfloat16, device="cuda")

        output_dir = export_dir
        os.makedirs(output_dir, exist_ok=True)
        start = time.time()
        logger.info("Start export onnx ...")
        print(colored(f"Start Vit export onnx...", "green"))
        with torch.inference_mode():
            torch.onnx.export(
                self,
                (pixel_values),  # Include position_ids in ONNX export
                f"{output_dir}/vit.onnx",
                # Add position_ids to input names
                input_names=["pixel_values"],
                output_names=["image_features"],
                opset_version=19,
                dynamo=dynamo,
                do_constant_folding=True,
                #                dynamic_axes={
                #                    "pixel_values": {0: "batch_size"},
                #                    "vit_embeds": {0: "batch_size"},
                #                },
            )
        end = time.time()
        logger.info(f"export onnx to {output_dir} done cost:{end - start}s")
        print(
            colored(f"Vit export onnx done to {output_dir} cost:{end - start}s", "green"))
        return self

    @classmethod
    def construct_from_name_path(cls, model_name, model_path):
        from .model_pi05 import Pi05Model
        pi05_model = Pi05Model.construct_from_name_path(model_name, model_path)
        return cls.construct_model(pi05_model)

    @classmethod
    def construct_model(cls, pi05_model, dtype=torch.float16):
        vit_model = cls(pi05_model.paligemma_with_expert.paligemma.config,
                        pi05_model.paligemma_with_expert.paligemma.model.vision_tower,
                        pi05_model.paligemma_with_expert.paligemma.model.multi_modal_projector).to(dtype)
        # pi05_model.paligemma_with_expert.paligemma.model.multi_modal_projector).to(dtype)
        return vit_model

    @classmethod
    def export_onnx(cls, pi05_model, export_dir):
        vit_model = cls.construct_model(pi05_model, dtype=torch.float16)
        vit_model.eval().cuda()

        pixel_values = torch.randn(
            (1, 3, 224, 224), dtype=torch.float16, device="cuda")

        output_dir = export_dir
        os.makedirs(output_dir, exist_ok=True)
        start = time.time()
        logger.info("Start export onnx ...")
        with torch.inference_mode():
            torch.onnx.export(
                vit_model,
                (pixel_values),  # Include position_ids in ONNX export
                f"{output_dir}/vit.onnx",
                # Add position_ids to input names
                input_names=["pixel_values"],
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
        logger.info(f"export onnx to {output_dir} done cost:{end - start}s")
        return vit_model

    def quantize(self, quant_cfg, calib_data, export_dir):
        # tokenizer = get_tokenizer(model_dir)
        calib_dataloader = self.get_calibrate_dataset(calib_data)
        quantize_model(self, quant_cfg, calib_dataloader)
        self.is_quantized = True
        set_dynamic_quant(self, "fp16")

        self.export(export_dir, dynamo=False)
        if is_nvfp4_quantized(quant_cfg):
            print(colored("nvfp4 quantization detected, post processing...", "green"))
            self._nvfp4_post_processing(export_dir)