import argparse

import torch

from typing import Optional, Any

from model_optimizer.evaluate.compare.utils import compare_predictions
from model_optimizer.utils.data import load_saved_data

def compare_cli(args: Optional[dict[str, Any]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path1', type=str, required=True)
    parser.add_argument('--data_path2', type=str, required=True)
    print(f'[cli] compare_data args {args[1:]}')
    args = parser.parse_args(args[1:])

    data1 = load_saved_data(args.data_path1)
    data2 = load_saved_data(args.data_path2)
    for data_1, data_2 in zip(data1, data2):
        compare_predictions(data_1, data_2, filter_keys=["prompt"])