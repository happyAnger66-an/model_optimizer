import argparse
import numpy as np

from termcolor import colored
from typing import Optional, Any

from model_optimizer.evaluate.compare.utils import compare_predictions, plot_compare_results
from model_optimizer.utils.data import load_saved_data

FILTER_KEYS = ["prompt", "policy_timing"]


def compare_cli(args: Optional[list[str] | None] = None) -> None:
    parser = argparse.ArgumentParser(description="对比两组推理结果")
    parser.add_argument("--data_path1", type=str, required=True, help="第一组数据 npz 路径")
    parser.add_argument("--data_path2", type=str, required=True, help="第二组数据 npz 路径")
    parser.add_argument(
        "--plot_output",
        type=str,
        default=None,
        help="折线图保存路径（指定后生成对比折线图）",
    )
    parser.add_argument(
        "--key1",
        type=str,
        default="data1",
        help="折线图中第一组数据图例名",
    )
    parser.add_argument(
        "--key2",
        type=str,
        default="data2",
        help="折线图中第二组数据图例名",
    )

    parsed = parser.parse_args(args[1:] if args else [])
    print(f"[cli] compare_data args {args[1:] if args else []}")

    data1 = load_saved_data(parsed.data_path1)
    data2 = load_saved_data(parsed.data_path2)

    collected_metrics = []
    all_mean_diff, all_max_diff = [], []
    for data_1, data_2 in zip(data1, data2):
        metrics = compare_predictions(
            data_1,
            data_2,
            filter_keys=FILTER_KEYS,
            key1=parsed.key1,
            key2=parsed.key2,
            return_metrics=bool(parsed.plot_output),
        )
        all_mean_diff.append(metrics["l1_mean"])
        all_max_diff.append(metrics["l1_max"])
        if metrics is not None:
            collected_metrics.append(metrics)

    avg_mean_diff = np.mean(all_mean_diff)
    avg_max_diff = np.mean(all_max_diff)
    print(colored(f"Mean difference: {avg_mean_diff:.4f}", "green"))
    print(colored(f"Max difference: {avg_max_diff:.4f}", "green"))

    if avg_mean_diff < 0.01:
        print(colored("✅ Mean difference is less than 0.01 excellent", "green"))
    elif avg_mean_diff < 0.1:
        print(colored("✅ Mean difference is less than 0.1 good", "green"))
    elif avg_mean_diff < 0.5:
        print(colored("⚠️  accuracy is acceptable (< 0.5)", "yellow"))
    else:
        print(colored("❌ accuracy is poor (> 0.5)", "red"))

    if parsed.plot_output and collected_metrics:
        plot_compare_results(
            collected_metrics,
            parsed.plot_output,
            key1=parsed.key1,
            key2=parsed.key2,
        )