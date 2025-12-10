from torch.utils.data import Dataset
import dataclasses

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

def get_data_loader(config_name):
#    config = _config.get_config("pi0_aloha_sim")
    config = _config.get_config(config_name)
    config = dataclasses.replace(config, batch_size=1, num_workers=1)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=1,
        shuffle=True,
        framework='pytorch'  # must specify.
    )

    return loader

class ViTCalibrationDataset(Dataset):
    """
    A dataset that uses LeRobotSingleDataset for ViT calibration data.
    This provides realistic calibration data for the vision transformer.
    """

    def __init__(
        self,
        dataset_path,
        config_name,
        policy,
        calib_size = 100,
    ):
        """
        Initialize the ViT calibration dataset.

        Args:
            dataset_path: Path to the LeRobot dataset
            modality_configs: Modality configuration for the dataset
            embodiment_tag: Embodiment tag for the dataset
            policy: Gr00tPolicy instance for using apply_transforms()
            calib_size: Number of calibration samples to use
            video_backend: Video backend for loading videos
        """
        self.calib_size = calib_size
        self.policy = policy

        # Initialize the LeRobot dataset
        self.data_loader = get_data_loader(config_name)

        # Use sequential indices for calibration
        self.dataset_size = len(self.data_loader)
        self.calib_size = min(calib_size, self.dataset_size)
        print(f"ViT Dataset size: {self.dataset_size} calib_size: {self.calib_size}")

    def __len__(self):
        return self.calib_size

    def __getitem__(self, idx):
        # Use sequential indices directly
        data = self.data_loader[idx]

        # Process the data to get pixel_values and position_ids for ViT
        processed_data = self._process_vit_data(data)
        return processed_data

    def _process_vit_data(self, data):
        """
        Process LeRobot data to extract pixel_values and position_ids for ViT calibration.
        """
        try:
#            # Ensure data is in the correct format for apply_transforms
#            is_batch = self.policy._check_state_is_batched(data)
#            if not is_batch:
#                data = unsqueeze_dict_values(data)

            # Apply the same transforms as used in training/inference
            transformed_data = self.policy._input_transformers(data)

            # Check if we have eagle pixel values
            if "eagle_pixel_values" in transformed_data:
                pixel_values = transformed_data["eagle_pixel_values"]
                batch_size = pixel_values.shape[0]
                # Generate position_ids for the patches
                num_patches = (
                    self.policy.model.backbone.eagle_model.vision_model.vision_model.embeddings.num_patches
                )
                position_ids = torch.arange(
                    num_patches, dtype=torch.long, device=pixel_values.device
                ).expand((batch_size, -1))
                return {
                    "pixel_values": pixel_values,
                    "position_ids": position_ids,
                }
            else:
                raise RuntimeError(
                    "eagle data not found in transformed_data. This indicates an issue with apply_transforms()."
                )
        except Exception as e:
            print(f"Warning: ViT data processing failed: {e}, using dummy data")
            raise RuntimeError(f"apply_transforms() failed: {e}")