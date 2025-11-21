import torch
from datasets import load_dataset

def load_calibration_data(model_name, data_size, batch_size, device):
    """Load and prepare calibration data."""
    dataset = load_dataset("zh-plus/tiny-imagenet")

    images = dataset["train"][:data_size]["image"]
    calib_tensor = [transforms(img) for img in images]
    calib_tensor = [t.to(device) for t in calib_tensor]
    return torch.utils.data.DataLoader(
        calib_tensor, batch_size=batch_size, shuffle=True, num_workers=4
    )