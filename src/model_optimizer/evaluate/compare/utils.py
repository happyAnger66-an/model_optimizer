import numpy as np
import torch

from termcolor import colored

cur_item = 0

def plot_compare_results(
    collected_metrics: list[dict],
    output_path: str,
    key1: str = "data1",
    key2: str = "data2",
) -> None:
    """
    根据 compare_predictions 收集的结果绘制对比折线图。

    横轴为数据 id，纵轴为各 key 的 mean 值，每个 key 一张子图，每张图两条线
    （mean_1 与 mean_2，来自 compare_predictions 的计算结果）。

    Args:
        collected_metrics: compare_predictions 返回的指标列表，每项为
            {key: {mean_1, mean_2, cosine_sim, l1_mean, ...}, ...}
        output_path: 图片保存路径
        key1: 第一条线图例名（对应 mean_1）
        key2: 第二条线图例名（对应 mean_2）
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "绘制折线图需要 matplotlib，请安装: pip install matplotlib"
        ) from e

    if not collected_metrics:
        raise ValueError("无有效对比数据")

    common_keys = list(collected_metrics[0].keys())
    if not common_keys:
        raise ValueError("无有效比较 key")

    n = len(collected_metrics)
    ncols = min(3, len(common_keys))
    nrows = (len(common_keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if len(common_keys) == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    x = np.arange(n)

    for idx, key in enumerate(common_keys):
        ax = axes[idx]
        y1 = [collected_metrics[i][key]["mean_1"] for i in range(n)]
        y2 = [collected_metrics[i][key]["mean_2"] for i in range(n)]
        ax.plot(x, y1, "o-", label=key1, markersize=4)
        ax.plot(x, y2, "s--", label=key2, markersize=4)
        ax.set_xlabel("Data ID")
        ax.set_ylabel("Mean Value")
        ax.set_title(key)
        ax.legend()
        ax.grid(True, alpha=0.3)

    for j in range(len(common_keys), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(colored(f"对比折线图已保存: {output_path}", "green"))


def compare_predictions(
    pred_tensorrt,
    pred_torch,
    key1="PyTorch",
    key2="TensorRT",
    filter_keys=None,
    return_metrics=False,
):
    """
    比较 TensorRT 与 PyTorch 推理结果，可返回各 key 的对比指标。

    Args:
        pred_tensorrt: 第一组预测结果（字典，key -> numpy array）
        pred_torch: 第二组预测结果
        key1: 第一组图例/标签名
        key2: 第二组图例/标签名
        filter_keys: 要跳过的 key
        return_metrics: 若为 True，返回各 key 的指标 dict，否则回 None

    Returns:
        当 return_metrics=True 时返回 {key: {cosine_sim, l1_mean, l1_max, mean_1, mean_2, ...}}
        否则返回 None
    """
    print("\n\n=== Prediction Comparison ===")
    global cur_item
    print(colored(f"Comparing item {cur_item}", "yellow"))
    cur_item += 1
    assert pred_tensorrt.keys() == pred_torch.keys(), "Prediction keys do not match"

    max_label_width = max(
        len(f"Cosine Similarity ({key1}/{key2}):"),
        len(f"L1 Mean/Max Distance ({key1}/{key2}):"),
        len(f"Max Output Values ({key1}/{key2}):"),
        len(f"Mean Output Values ({key1}/{key2}):"),
        len(f"Min Output Values ({key1}/{key2}):"),
    )

    metrics = {}
    for key in pred_tensorrt.keys():
        tensorrt_array = pred_tensorrt[key]
        torch_array = pred_torch[key]

        if filter_keys and key in filter_keys:
            print(
                colored(f"Skipping {key} because it is in the filter keys", "yellow")
            )
            continue

        tensorrt_tensor = torch.from_numpy(
            np.array(tensorrt_array, dtype=np.float32)
        ).to(torch.float32)
        torch_tensor = torch.from_numpy(
            np.array(torch_array, dtype=np.float32)
        ).to(torch.float32)

        assert tensorrt_tensor.shape == torch_tensor.shape, (
            f"{key} shapes do not match: "
            f"{tensorrt_tensor.shape} vs {torch_tensor.shape}"
        )

        flat_tensorrt = tensorrt_tensor.flatten()
        flat_torch = torch_tensor.flatten()

        dot_product = torch.dot(flat_tensorrt, flat_torch)
        norm_tensorrt = torch.norm(flat_tensorrt)
        norm_torch = torch.norm(flat_torch)
        cos_sim = (dot_product / (norm_tensorrt * norm_torch + 1e-8)).item()

        l1_dist = torch.abs(flat_tensorrt - flat_torch)
        l1_mean = l1_dist.mean().item()
        l1_max = l1_dist.max().item()

        mean_1 = torch_tensor.mean().item()
        mean_2 = tensorrt_tensor.mean().item()
        max_1 = torch_tensor.max().item()
        max_2 = tensorrt_tensor.max().item()
        min_1 = torch_tensor.min().item()
        min_2 = tensorrt_tensor.min().item()

        metrics[key] = {
            "cosine_sim": cos_sim,
            "l1_mean": l1_mean,
            "l1_max": l1_max,
            "mean_1": mean_1,
            "mean_2": mean_2,
            "max_1": max_1,
            "max_2": max_2,
            "min_1": min_1,
            "min_2": min_2,
        }

        print(colored(f"\n{key}:", "yellow"))
        print(f"{f'Cosine Similarity ({key1}/{key2}):':<{max_label_width}} {cos_sim}")
        print(
            f"{f'L1 Mean/Max Distance ({key1}/{key2}):':<{max_label_width}} "
            f"{l1_mean:.4f}/{l1_max:.4f}"
        )
        print(
            f"{f'Max Output Values ({key1}/{key2}):':<{max_label_width}} "
            f"{max_1:.4f}/{max_2:.4f}"
        )
        print(
            f"{f'Mean Output Values ({key1}/{key2}):':<{max_label_width}} "
            f"{mean_1:.4f}/{mean_2:.4f}"
        )
        print(
            f"{f'Min Output Values ({key1}/{key2}):':<{max_label_width}} "
            f"{min_1:.4f}/{min_2:.4f}"
        )

    return metrics if return_metrics else None
