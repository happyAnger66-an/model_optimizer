import torch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


class Pi05Vit(torch.nn.Module):
    def __init__(self, config, vision_tower, multi_modal_projector, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.vision_tower = vision_tower
        self.multi_modal_projector = multi_modal_projector

    def forward(self, pixel_values):
        print(f'Pi05Vit input: {pixel_values.shape}')
        image_outputs = self.vision_tower(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = image_features / (self.config.text_config.hidden_size ** 0.5)
        print(f'Pi05Vit output: {image_features.shape}')
        return image_features


class Pi0LLM(torch.nn.Module):
    def __init__(self, llm, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.llm.config._attn_implementation = "eager"

    def forward(self, inputs_embeds, attention_mask, position_ids):
        prefix_output = self.llm(inputs_embeds=inputs_embeds,
                                 attention_mask=attention_mask,
                                 position_ids=position_ids)
#        return prefix_output.last_hidden_state
        return prefix_output.past_key_values, prefix_output.last_hidden_state
