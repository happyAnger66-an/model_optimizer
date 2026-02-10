import torch

from termcolor import colored


def compare_predictions(pred_tensorrt, pred_torch, key1="PyTorch",
                        key2="TensorRT", filter_keys=None):
    """
    Compare the similarity between TensorRT and PyTorch predictions

    Args:
        pred_tensorrt: TensorRT prediction results (numpy array)
        pred_torch: PyTorch prediction results (numpy array)
    """
    print("\n\n=== Prediction Comparison ===")

    # Ensure both predictions contain the same keys
    assert pred_tensorrt.keys() == pred_torch.keys(), "Prediction keys do not match"

    # Calculate max label width for alignment
    max_label_width = max(
        len(f"Cosine Similarity ({key1}/{key2}):"),
        len(f"L1 Mean/Max Distance ({key1}/{key2}):"),
        len(f"Max Output Values ({key1}/{key2}):"),
        len(f"Mean Output Values ({key1}/{key2}):"),
        len(f"Min Output Values ({key1}/{key2}):"),
    )

    for key in pred_tensorrt.keys():
        tensorrt_array = pred_tensorrt[key]
        torch_array = pred_torch[key]

        if filter_keys and key in filter_keys:
            print(
                colored(f"Skipping {key} because it is in the filter keys", "yellow"))
            continue

        # Convert to PyTorch tensors
        tensorrt_tensor = torch.from_numpy(tensorrt_array).to(torch.float32)
        torch_tensor = torch.from_numpy(torch_array).to(torch.float32)

        # Ensure tensor shapes are the same
        assert (
            tensorrt_tensor.shape == torch_tensor.shape
        ), f"{key} shapes do not match: {tensorrt_tensor.shape} vs {torch_tensor.shape}"

        # Calculate cosine similarity
        flat_tensorrt = tensorrt_tensor.flatten()
        flat_torch = torch_tensor.flatten()

        # Manually calculate cosine similarity
        dot_product = torch.dot(flat_tensorrt, flat_torch)
        norm_tensorrt = torch.norm(flat_tensorrt)
        norm_torch = torch.norm(flat_torch)
        cos_sim = dot_product / (norm_tensorrt * norm_torch)

        # Calculate L1 distance
        l1_dist = torch.abs(flat_tensorrt - flat_torch)

        print(colored(f"\n{key}:", "yellow"))
        print(
            f'{f"Cosine Similarity ({key1}/{key2}):".ljust(max_label_width)} {cos_sim.item()}')
        print(
            f'{f"L1 Mean/Max Distance ({key1}/{key2}):".ljust(max_label_width)} {l1_dist.mean().item():.4f}/{l1_dist.max().item():.4f}'
        )
        print(
            f'{f"Max Output Values ({key1}/{key2}):".ljust(max_label_width)} {torch_tensor.max().item():.4f}/{tensorrt_tensor.max().item():.4f}'
        )
        print(
            f'{f"Mean Output Values ({key1}/{key2}):".ljust(max_label_width)} {torch_tensor.mean().item():.4f}/{tensorrt_tensor.mean().item():.4f}'
        )
        print(
            f'{f"Min Output Values ({key1}/{key2}):".ljust(max_label_width)} {torch_tensor.min().item():.4f}/{tensorrt_tensor.min().item():.4f}'
        )
