import argparse

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
    for data_1, data_2 in zip(data1, data2):
        metrics = compare_predictions(
            data_1,
            data_2,
            filter_keys=FILTER_KEYS,
            key1=parsed.key1,
            key2=parsed.key2,
            return_metrics=bool(parsed.plot_output),
        )
        if metrics is not None:
            collected_metrics.append(metrics)

    if parsed.plot_output and collected_metrics:
        plot_compare_results(
            collected_metrics,
            parsed.plot_output,
            key1=parsed.key1,
            key2=parsed.key2,
        )