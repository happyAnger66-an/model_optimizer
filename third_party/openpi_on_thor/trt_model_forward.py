#!/usr/bin/env python3
from functools import partial

import torch
import openpi_on_thor.trt_torch as trt


def pi0_tensorrt_sample_actions(self, device, observation, noise=None, num_steps=None):
    """
    TensorRT-accelerated sample_actions for π₀.5 models.

    This replaces the PyTorch model's sample_actions method with TensorRT inference.

    Args:
        device: CUDA device (e.g., "cuda:0")
        observation: Observation object with images, image_masks, tokenized_prompt, etc.
        noise: Optional noise tensor [batch, action_horizon, action_dim] (if None, generates random noise)
        num_steps: Denoising steps (not used, TensorRT engine uses compiled steps)

    Returns:
        actions: [batch, action_horizon, action_dim] float32 tensor
    """
    # Prepare inputs from observation
    # Convert images dict to concatenated tensor [batch, 9, 224, 224]
    image_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    images_list = []
    img_masks_list = []

    for key in image_keys:
        if key in observation.images:
            img = observation.images[key]
            # Ensure correct shape [batch, C, H, W]
            if img.dim() == 3:  # [C, H, W]
                img = img.unsqueeze(0)  # [1, C, H, W]
            images_list.append(img)

            # Get mask for this image
            mask = observation.image_masks.get(key, torch.ones(img.shape[0], dtype=torch.bool, device=device))
            if mask.dim() == 0:  # scalar
                mask = mask.unsqueeze(0)
            img_masks_list.append(mask)

    # Concatenate all images: [batch, 9, 224, 224]
    images = torch.cat(images_list, dim=1)
    # Stack masks: [batch, 3]
    img_masks = torch.stack(img_masks_list, dim=1)

    # Get language tokens and masks
    lang_tokens = observation.tokenized_prompt
    lang_masks = observation.tokenized_prompt_mask

    # Get state
    state = observation.state

    # Get batch size from images
    batch_size = images.shape[0]

    target_dtype = torch.float16

    # Handle noise input - generate if not provided
    if noise is None:
        # Get action shape from the model config (stored during setup)
        # Default to common values if not available
        action_horizon = getattr(self, "action_horizon", 10)
        action_dim = getattr(self, "action_dim", 32)
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=(batch_size, action_horizon, action_dim),
            dtype=target_dtype,
            device=device,
        )
    else:
        # Ensure noise is a tensor
        if not isinstance(noise, torch.Tensor):
            noise = torch.from_numpy(noise)
        # Ensure correct batch dimension
        if noise.dim() == 2:  # [action_horizon, action_dim]
            noise = noise.unsqueeze(0)  # [1, action_horizon, action_dim]
        # Ensure correct dtype and device
        if noise.dtype != target_dtype:
            noise = noise.to(target_dtype)
        if not noise.is_cuda:
            noise = noise.cuda()
        noise = noise.contiguous()

    # Convert all tensors to the target dtype that matches the TensorRT engine
    if images.dtype != target_dtype:
        images = images.to(target_dtype)
    if state.dtype != target_dtype:
        state = state.to(target_dtype)

    # Ensure tensors are on CUDA
    images = images.cuda().contiguous()
    img_masks = img_masks.cuda().contiguous()
    lang_tokens = lang_tokens.cuda().contiguous()
    lang_masks = lang_masks.cuda().contiguous()
    state = state.cuda().contiguous()

    # Set runtime shapes for dynamic inputs
    self.trt_engine.set_runtime_tensor_shape("images", images.shape)
    self.trt_engine.set_runtime_tensor_shape("img_masks", img_masks.shape)
    self.trt_engine.set_runtime_tensor_shape("lang_tokens", lang_tokens.shape)
    self.trt_engine.set_runtime_tensor_shape("lang_masks", lang_masks.shape)
    self.trt_engine.set_runtime_tensor_shape("state", state.shape)
    self.trt_engine.set_runtime_tensor_shape("noise", noise.shape)

    # Run TensorRT inference
    # The engine expects inputs in order: images, img_masks, lang_tokens, lang_masks, state, noise
    outputs = self.trt_engine(images, img_masks, lang_tokens, lang_masks, state, noise)

    # Extract actions from output dict
    actions = outputs["actions"]

    return actions


def setup_pi0_tensorrt_engine(policy, engine_path):
    """
    Setup TensorRT engine for π₀.5 model inference.

    This function loads a TensorRT engine and hooks it to replace the PyTorch
    sample_actions method, providing significant inference speedup.

    Args:
        policy: π₀.5 policy instance from policy_config.create_trained_policy()
        engine_path: Path to the .engine file (e.g., "model_fp32.engine")

    Returns:
        policy: Modified policy with TensorRT engine attached

    Example:
        >>> from openpi.training import config as _config
        >>> from openpi.policies import policy_config
        >>> config = _config.get_config("pi05_droid")
        >>> policy = policy_config.create_trained_policy(config, checkpoint_dir)
        >>> policy = setup_pi0_tensorrt_engine(
        ...     policy,
        ...     os.path.join(checkpoint_dir, "model_fp16.engine")
        ... )
        >>> # Now policy.infer() uses TensorRT automatically
        >>> actions = policy.infer(observation)["actions"]
    """
    print(f"Setting up π₀.5 TensorRT engine from {engine_path}...")

    # Get the model object (use _model for Policy class)
    model = policy._model if hasattr(policy, "_model") else policy.model

    # Load TensorRT engine
    model.trt_engine = trt.Engine(engine_path)

    # Store action dimensions for noise generation
    if hasattr(model, "config"):
        model.action_horizon = model.config.action_horizon
        model.action_dim = model.config.action_dim
        print(f"  Action dimensions: horizon={model.action_horizon}, dim={model.action_dim}")

    # Save the original sample_actions method (optional, for fallback)
    if not hasattr(model, "_original_sample_actions"):
        model._original_sample_actions = model.sample_actions

    # Replace sample_actions with TensorRT version
    trt_sample_actions = partial(pi0_tensorrt_sample_actions, model)
    model.sample_actions = trt_sample_actions

    # IMPORTANT: Also update policy._sample_actions if it exists
    # The Policy class caches a reference to model.sample_actions during __init__
    if hasattr(policy, "_sample_actions"):
        policy._sample_actions = trt_sample_actions

    print("TensorRT engine hooked to policy._model.sample_actions")

    # Delete PyTorch model components to save memory
    print("Deleting PyTorch model components to save memory...")

    # Delete PaliGemma components
    if hasattr(model, "paligemma_with_expert"):
        if hasattr(model.paligemma_with_expert, "paligemma"):
            del model.paligemma_with_expert.paligemma
        if hasattr(model.paligemma_with_expert, "gemma_expert"):
            del model.paligemma_with_expert.gemma_expert

    # Delete diffusion components (if present)
    if hasattr(model, "time_mlp_in"):
        del model.time_mlp_in
    if hasattr(model, "time_mlp_out"):
        del model.time_mlp_out
    if hasattr(model, "action_in_proj"):
        del model.action_in_proj
    if hasattr(model, "action_out_proj"):
        del model.action_out_proj

    torch.cuda.empty_cache()
    print("PyTorch components deleted, memory freed")

    return policy
