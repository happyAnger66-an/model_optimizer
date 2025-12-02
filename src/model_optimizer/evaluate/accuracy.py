import torch

def compare_model_accuracy(original_model, quantized_model, test_data, device="cuda"):
    """
    Compare accuracy between original and quantized models.

    Args:
        original_model: The original model before quantization
        quantized_model: The quantized model after quantization
        test_data: Test data for comparison
        device: Device to run comparison on

    Returns:
        dict: Comparison results including mean difference, max difference, etc.
    """
    print("\n🔍 COMPARING MODEL ACCURACY BEFORE AND AFTER QUANTIZATION...")

    original_model.eval()
    quantized_model.eval()

    differences = []
    max_diffs = []

    with torch.no_grad():
        for i, data in enumerate(test_data):
            if i >= 5:  # Limit to 5 samples for comparison
                break

            # Move data to device
            data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

            try:
                # Get outputs from both models
                original_output = original_model(**data)
                quantized_output = quantized_model(**data)

                # Calculate difference
                if isinstance(original_output, torch.Tensor) and isinstance(
                    quantized_output, torch.Tensor
                ):
                    diff = torch.abs(original_output - quantized_output)
                    mean_diff = torch.mean(diff).item()
                    max_diff = torch.max(diff).item()
                    # Compute cosine similarity
                    orig_flat = original_output.flatten()
                    quant_flat = quantized_output.flatten()
                    cos_sim = torch.nn.functional.cosine_similarity(
                        orig_flat.unsqueeze(0), quant_flat.unsqueeze(0)
                    ).item()
                    print(f"Cosine similarity: {cos_sim:.6f}")

                    differences.append(mean_diff)
                    max_diffs.append(max_diff)
                    # INSERT_YOUR_CODE
                    print(f"Sample {i+1}: Mean diff: {mean_diff:.6f}, Max diff: {max_diff:.6f}")
                    # Print the position and the original value of the max diff
                    max_diff_pos = (diff == torch.max(diff)).nonzero(as_tuple=True)
                    if len(max_diff_pos[0]) > 0:
                        idx = tuple(pos[0].item() for pos in max_diff_pos)
                        print(f"Position of max diff: {idx}")
                        print(
                            f"Original value at this position: {original_output[idx] if isinstance(original_output, torch.Tensor) else original_output.logits[idx]}"
                        )

                elif hasattr(original_output, "logits") and hasattr(quantized_output, "logits"):
                    # Handle model outputs with logits
                    diff = torch.abs(original_output.logits - quantized_output.logits)
                    mean_diff = torch.mean(diff).item()
                    max_diff = torch.max(diff).item()

                    differences.append(mean_diff)
                    max_diffs.append(max_diff)

                    print(f"Sample {i+1}: Mean diff: {mean_diff:.6f}, Max diff: {max_diff:.6f}")

            except Exception as e:
                print(f"Error comparing sample {i+1}: {e}")
                continue

    if differences:
        avg_mean_diff = sum(differences) / len(differences)
        avg_max_diff = sum(max_diffs) / len(max_diffs)
        overall_max_diff = max(max_diffs)

        print("\n📊 QUANTIZATION ACCURACY SUMMARY:")
        print(f"Average mean difference: {avg_mean_diff:.6f}")
        print(f"Average max difference: {avg_max_diff:.6f}")
        print(f"Overall max difference: {overall_max_diff:.6f}")

        # Determine if quantization is acceptable
        if avg_mean_diff < 0.01:
            print("✅ Quantization accuracy is excellent (< 0.01)")
        elif avg_mean_diff < 0.1:
            print("✅ Quantization accuracy is good (< 0.1)")
        elif avg_mean_diff < 0.5:
            print("⚠️  Quantization accuracy is acceptable (< 0.5)")
        else:
            print("❌ Quantization accuracy is poor (> 0.5)")

        return {
            "avg_mean_diff": avg_mean_diff,
            "avg_max_diff": avg_max_diff,
            "overall_max_diff": overall_max_diff,
            "num_samples": len(differences),
        }
    else:
        print("❌ No valid comparisons could be made")
        return None