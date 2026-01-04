import torch

class Pi05Expert(torch.nn.Module):
    def __init__(self, config, gemma_expert, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.gemma_expoert = gemma_expert

    def forward(self, past_key_values=None):
        print(f'Pi05Expert input: {pixel_values.shape}')
        image_outputs = self.vision_tower(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = image_features / (self.config.text_config.hidden_size ** 0.5)
        print(f'Pi05Expert output: {image_features.shape}')
        return image_features