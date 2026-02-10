from openpi.policies.libero_policy import make_libero_example
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Literal
import warnings
# from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
# from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
# from gr00t.data.embodiment_tags import EmbodimentTag
# from gr00t.policy.gr00t_policy import Gr00tPolicy
# from gr00t.policy.policy import BasePolicy
from openpi_client.base_policy import BasePolicy
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training.data_loader import DataLoader
from openpi.policies import policy_config

import dataclasses
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import tyro

import addict
from termcolor import colored

warnings.simplefilter("ignore", category=FutureWarning)

"""
Combined inference script supporting both PyTorch and TensorRT modes.

Example commands:

# PyTorch mode (default):
python groot/scripts/deployment/standalone_inference_script.py \
  --model_path /path/to/checkpoint \
  --dataset_path /path/to/dataset \
  --embodiment_tag GR1 \
  --traj-ids 0 1 2 \
  --inference-mode pytorch

# TensorRT mode:
python groot/scripts/deployment/standalone_inference_script.py \
  --model_path /path/to/checkpoint \
  --dataset_path /path/to/dataset \
  --embodiment_tag GR1 \
  --traj-ids 0 1 2 \
  --inference-mode tensorrt \
  --trt_engine_path ./groot_n1d6_onnx/dit_model_bf16.trt
"""

###############################################################################
# TENSORRT Module Wrappers
###############################################################################


def set_seed(seed: int = 0):
    """
    Set seed for all random number generators.
    """
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch CPU & CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic CUDA ops
    torch.use_deterministic_algorithms(True, warn_only=True)

    # For cuDNN deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch requires this to be set for some CUDA kernels
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def parse_action(action: dict[str, Any]) -> dict[str, Any]:
    # Unbatch and add prefix
    return {f"action.{key}": action[key][0] for key in action}


def save_data(data_list: list[dict[str, Any]], save_file_path: str, save_file_name: str):
    save_dict = {}
    for i, data_dict in enumerate(data_list):
        for key, value in data_dict.items():
            # 创建唯一的键名：索引_键名
            save_key = f"item_{i}_{key}"
            save_dict[save_key] = value

    save_dict['metadata'] = np.array({
        'num_items': len(data_list),
        'item_keys': list(data_list[0].keys()) if data_list else []
    }, dtype=object)

    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    np.savez(f'{save_file_path}/{save_file_name}.npz', **save_dict)


def load_input_data(input_data_file: str):
    """
    从npz文件加载字典列表
    """
    loaded = np.load(input_data_file, allow_pickle=True)

    # 提取元数据
    meta = loaded['metadata'].item()
    num_items = meta['num_items']

    # 重建字典列表
    result = []
    for i in range(num_items):
        item_dict = {}
        # 查找属于当前item的所有键
        for key in loaded.files:
            if key.startswith(f"item_{i}_"):
                original_key = key.replace(f"item_{i}_", "", 1)
                item_dict[original_key] = loaded[key]

        if item_dict:  # 只添加非空字典
            result.append(item_dict)

    loaded.close()
    return result


def get_input_data(input_data_file, max_nums=40):
    if input_data_file is None:
        for i in range(max_nums):
            obs = make_libero_example()
            print(f"make_libero_example {i}")
            yield obs
    else:
        input_data_list = load_input_data(input_data_file)
        for i in range(min(max_nums, len(input_data_list))):
            print(f"load_input_data {i}")
            yield input_data_list[i]


def run_single_trajectory(
    policy: BasePolicy,
    loader: DataLoader,
    traj_id: int,
    embodiment_tag,
    steps=100,
    action_horizon=10,
    skip_timing_steps=1,
    perf=False,
    args=None,
):
    """
    Run inference on a single trajectory.

    Args:
        skip_timing_steps: Number of initial inference steps to skip when calculating timing statistics

    Returns: tuple: (
        state_keys,
        action_keys,
        pred_action_across_time,
        traj,
        actual_steps,
        timing_dict,
    )
    """
    logging.info("\n" + "=" * 80)
    logging.info(f"=== Running Trajectory {traj_id} ===")
    logging.info("=" * 80)

    import numpy as np
    time_results = []
    input_data_list = []
    output_data_list = []

    i = 0
    for obs in get_input_data(args.input_data_path, 40):
        if args.save_input_path:
            input_data_list.append(obs)

        inference_start = time.time()
    #    import pdb; pdb.set_trace()
        _action_chunk = policy.infer(obs)

        if args.save_output_path:
            output_data_list.append(_action_chunk)

        if perf:
            model = policy._model
            model.perf = True
        inference_time = time.time() - inference_start
        print(
            colored(f"Inference time: {inference_time:.4f} seconds", "green"))
        i += 1
        if i > 10 and perf:
            time_results.append(inference_time)
    if perf:
        print(colored(
            f"e2e {np.mean(time_results)*1000:.2f} ± {np.std(time_results)*1000:.2f} ms (shared)", "green"))
        print(colored(
            f"action {np.mean(model.time_results['action'])*1000:.2f} ± {np.std(model.time_results['action'])*1000:.2f} ms (shared)", "green"))
        print(colored(
            f"vit {np.mean(model.time_results['vit'])*1000:.2f} ± {np.std(model.time_results['vit'])*1000:.2f} ms (shared)", "green"))
        print(colored(
            f"llm {np.mean(model.time_results['llm'])*1000:.2f} ± {np.std(model.time_results['llm'])*1000:.2f} ms (shared)", "green"))

    if args.save_input_path:
        print(colored(f"save input datas to {args.save_input_path}", "green"))
        save_data(input_data_list, args.save_input_path, "inputs")

    if args.save_output_path:
        print(
            colored(f"save output datas to {args.save_output_path}", "green"))
        save_data(output_data_list, args.save_output_path, "outputs")
#    print(colored(f"Time results: {time_results}", "green"))
#    print(f"Average time: {sum(time_results) / len(time_results):.4f} seconds")
#    print(colored(f"Min time: {min(time_results):.4f} seconds", "green"))
#    print(colored(f"Max time: {max(time_results):.4f} seconds", "green"))
#    print(colored(f"Median time: {np.median(time_results):.4f} seconds", "green"))
#    print(colored(f"Std time: {np.std(time_results):.4f} seconds", "green"))
#    print(colored(f"90th percentile time: {np.percentile(time_results, 90):.4f} seconds", "green"))
#    print(colored(f"95th percentile time: {np.percentile(time_results, 95):.4f} seconds", "green"))
#    print(colored(f"99th percentile time: {np.percentile(time_results, 99):.4f} seconds", "green"))


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""
    config_name: str = "pi05_libero"
    """Config name to use."""

    host: str = "127.0.0.1"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    steps: int = 200
    """Maximum number of steps to evaluate (will be capped by trajectory length)."""

    precision: str = "fp16"
    """Precision to use."""

    traj_ids: list[int] = field(default_factory=lambda: [0])
    """List of trajectory IDs to evaluate."""

    action_horizon: int = 16
    """Action horizon to evaluate."""

    video_backend: Literal["decord",
                           "torchvision_av", "torchcodec"] = "torchcodec"
    """Video backend to use for various codec options. h264: decord or av: torchvision_av"""

    dataset_path: str = "demo_data/robot_sim.PickNPlace/"
    """Path to the dataset."""

    embodiment_tag: str = "pi05"
    """Embodiment tag to use."""

    model_path: str | None = None
    """Path to the model checkpoint."""

    inference_mode: Literal["pytorch", "tensorrt"] = "pytorch"
    """Inference mode: 'pytorch' (default) or 'tensorrt'."""

    trt_engine_path: str = ""
    """Path to TensorRT engine file (.trt). Used only when inference_mode='tensorrt'."""

    vit_engine: str = ""
    """Path to TensorRT vision engine file (.trt). Used only when inference_mode='tensorrt'."""

    llm_engine: str = ""
    """Path to TensorRT language model engine file (.trt). Used only when inference_mode='tensorrt'."""

    expert_engine: str = ""
    """Path to TensorRT expert engine file (.trt). Used only when inference_mode='tensorrt'."""

    denoising_steps: int = 10
    """Number of denoising steps to use."""

    save_plot_path: str | None = None
    """Path to save the plot to."""

    input_data_path: str | None = None
    """Path the input data to."""

    save_input_path: str | None = None
    """Path to save the input to."""

    save_output_path: str | None = None
    """Path to save the output to."""

    skip_timing_steps: int = 1
    """Number of initial inference steps to skip when calculating timing statistics (default: 1 to exclude warmup)."""

    get_performance_stats: bool = True
    """Agreegate and summarize timing and accuracy stats across several runs"""

    seed: int = 42
    """Seed to use for reproducibility."""

    perf: bool = False
    """Whether to get performance statistics."""


def main(args: ArgsConfig):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logging.info("\n" + "=" * 80)
    logging.info("=" * 80)
    logging.info(f"Model Path: {args.model_path}")
    logging.info(f"Dataset Path: {args.dataset_path}")
    logging.info(f"Embodiment Tag: {args.embodiment_tag}")
    logging.info(f"Trajectories: {args.traj_ids}")
    logging.info(f"Steps per trajectory: {args.steps}")
    logging.info(f"Action Horizon: {args.action_horizon}")
    logging.info(f"Skip Timing Steps: {args.skip_timing_steps}")
    logging.info(f"Inference Mode: {args.inference_mode}")
    logging.info(f"Get Performance Stats: {args.get_performance_stats}")
    if args.inference_mode == "tensorrt":
        logging.info(f"TensorRT Engine: {args.trt_engine_path}")
    logging.info(f"Seed: {args.seed}")
    set_seed(args.seed)
    logging.info("=" * 80)

    # Download model checkpoint
    local_model_path = args.model_path

    # Extract global_step and checkpoint directory name from checkpoint path
    assert local_model_path is not None, "Provide valid model_path for inference"

    # Model loading
    logging.info("\n" + "=" * 80)
    logging.info("=== Step 1: Loading Policy ===")
    logging.info("=" * 80)
    model_load_start = time.time()

    if local_model_path is not None:
        config = _config.get_config(args.config_name)
#        checkpoint_dir = download.maybe_download(
#           "gs://openpi-assets/checkpoints/pi0_fast_droid")
        checkpoint_dir = local_model_path
        # Create a trained policy.
        policy = policy_config.create_trained_policy(config, checkpoint_dir)

        # Apply inference mode: TensorRT or PyTorch
        if args.inference_mode == "tensorrt":
            from model_optimizer.infer.tensorrt.pi05_executor import Pi05TensorRTExecutor
            print(colored(" TensorRT mode enabled", "yellow"))
            if args.precision == "fp16":
                precision = torch.float16
            elif args.precision == "bf16":
                precision = torch.bfloat16
            else:
                precision = torch.float32
            print(colored(f"Use Precision: {precision}", "green"))
            executor = Pi05TensorRTExecutor(policy, precision)
            config = None

            if args.trt_engine_path:
                config = {
                    "engine_path": args.trt_engine_path,
                }

            if args.expert_engine:
                config["expert_engine"] = args.expert_engine

            if args.vit_engine:
                config["vit_engine"] = args.vit_engine

            if args.llm_engine:
                config["llm_engine"] = args.llm_engine

            if config is not None:
                config = addict.Dict(config)
            # config = None
#            import pdb; pdb.set_trace()
            executor.load_model(config)
            logging.info(" TensorRT mode enabled")
        else:
            from model_optimizer.infer.pytorch.pi05_executor import Pi05PyTorchExecutor
            print(colored(" PyTorch mode enabled", "yellow"))
            executor = Pi05PyTorchExecutor(policy)
            executor.load_model()
            logging.info(" PyTorch mode enabled")

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    else:
        assert 0, "Please provide valid model_path argument for inference"
    model_load_time = time.time() - model_load_start
    logging.info(f"Model loading time: {model_load_time:.4f} seconds")

    # Dataset creation
    logging.info("\n" + "=" * 80)
    logging.info("=== Step 2: Creating Dataset Loader ===")
    logging.info("=" * 80)
    dataset_load_start = time.time()

    # config = dataclasses.replace(config, batch_size=1)
    # data_loader = _data_loader.create_data_loader(
    #    config,
    #    # Skip since we may not have the data available.
    #    skip_norm_stats=True,
    #    num_batches=2,
    #    shuffle=True,
    #    framework='pytorch')
    # dataset_load_time = time.time() - dataset_load_start
    # logging.info(
    #    f"Dataset loader creation time: {dataset_load_time:.4f} seconds")

    # dataset = data_loader._data_loader.torch_loader.dataset
    # dataset_len = len(dataset)
    # print(colored(f"Dataset length: {dataset_len}", "green"))
    # logging.info(f"Dataset length: {dataset_len}")
    # logging.info(f"Running evaluation on trajectories: {args.traj_ids}")

    # Evaluation loop
    logging.info("\n" + "=" * 80)
    logging.info("=== Step 3: Running Evaluation ===")
    logging.info("=" * 80)

    all_mse = []
    all_mae = []
    all_timings = []
    pred_actions = []

    run_single_trajectory(
        policy,
        None,
        0,
        args.embodiment_tag,
        steps=args.steps,
        action_horizon=args.action_horizon,
        skip_timing_steps=args.skip_timing_steps,
        perf=args.perf,
        args=args,
    )

    return pred_actions


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
