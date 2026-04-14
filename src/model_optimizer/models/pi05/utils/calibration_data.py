#!/usr/bin/env python3
"""Calibration data loading utilities for FP8/NVFP4 quantization.

This is vendored from `third_party/openpi_on_thor/calibration_data.py` to avoid
runtime dependency on `openpi_on_thor`.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from openpi.policies import policy_config


class CalibrationDataset(Dataset):
    """Dataset for FP8/NVFP4 calibration using real data samples."""

    def __init__(
        self,
        config_obj,
        checkpoint_dir: str,
        num_samples: int = 32,
        device: str = "cuda",
        compute_dtype=torch.float16,
    ):
        print("  Initializing calibration dataset...")

        policy = policy_config.create_trained_policy(config_obj, checkpoint_dir)
        self.input_transform = policy._input_transform
        self.policy = policy

        self.device = device
        self._pytorch_device = device
        self.compute_dtype = compute_dtype
        self.action_horizon = config_obj.model.action_horizon
        self.action_dim = config_obj.model.action_dim

        print(f"  Loading {num_samples} calibration samples from dataset...")
        repo_id = config_obj.data.repo_id

        # LeRobot is optional; fall back to dummy calibration if unavailable.
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

        self.lerobot_dataset = LeRobotDataset(repo_id)
        step_size = max(len(self.lerobot_dataset) // num_samples, 1)
        self.sample_indices = list(
            range(0, min(len(self.lerobot_dataset), num_samples * step_size), step_size)
        )
        self.num_samples = len(self.sample_indices)

        print(f"  Calibration dataset ready with {self.num_samples} samples (dtype: {compute_dtype})")

    def _process_data(self, data):
        import jax
        import numpy as np
        from openpi.models.model import Observation

        inputs = jax.tree.map(lambda x: x, data)
        inputs = self.input_transform(inputs)

        def convert_to_torch(x):
            tensor = torch.from_numpy(np.array(x))
            if tensor.dtype in [torch.float32, torch.float64]:
                tensor = tensor.to(dtype=self.compute_dtype)
            return tensor.to(self._pytorch_device)[None, ...]

        inputs = jax.tree.map(convert_to_torch, inputs)
        observation = Observation.from_dict(inputs)

        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=(1, self.action_horizon, self.action_dim),
            dtype=self.compute_dtype,
            device=self.device,
        )
        return observation, noise

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data_idx = self.sample_indices[idx]
        data = self.lerobot_dataset[data_idx]

        sample = {
            "observation/image": data["image"],
            "observation/wrist_image": data["wrist_image"],
            "observation/state": data["state"],
            "prompt": data["task"],
        }
        return self._process_data(sample)


def no_batch_collate_fn(batch):
    """Return the sample without batching (batch_size should be 1)."""

    return batch[0]


def load_calibration_data(
    config_obj,
    checkpoint_dir: str,
    num_samples: int = 32,
    device: str = "cuda",
    batch_size: int = 1,
    compute_dtype=torch.float16,
):
    """Load calibration data from dataset for FP8/NVFP4 quantization.

    Returns:
        DataLoader for calibration data, or None if loading fails.
    """

    try:
        dataset = CalibrationDataset(
            config_obj=config_obj,
            checkpoint_dir=checkpoint_dir,
            num_samples=num_samples,
            device=device,
            compute_dtype=compute_dtype,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=no_batch_collate_fn,
        )
    except Exception as e:
        print(f"  Warning: Failed to load dataset: {e}")
        print("  Falling back to dummy inputs for calibration")
        return None

