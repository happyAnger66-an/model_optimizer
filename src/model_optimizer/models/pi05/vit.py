import os
import time
import torch

import logging

logger = logging.getLogger(__name__)

class Vit(torch.nn.Module):
    def __init__(self, config, vision_tower, multi_modal_projector, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.vision_tower = vision_tower
        self.multi_modal_projector = multi_modal_projector

    def forward(self, pixel_values):
        logger.info(f'Pi05Vit input: {pixel_values.shape}')
        image_outputs = self.vision_tower(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = image_features / \
            (self.config.text_config.hidden_size ** 0.5)
        logger.info(f'Pi05Vit output: {image_features.shape}')
        return image_features

    @classmethod
    def export_onnx(cls, pi05_model, export_dir):
        vit_model = cls(pi05_model.config, pi05_model.vision_tower,
                        pi05_model.multi_modal_projector).to(torch.float16)
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
