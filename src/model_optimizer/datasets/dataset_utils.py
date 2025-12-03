from typing import Any

# Use dict to store the config for each dataset.
# If we want to export more options to user like target languages, we need more standardized approach like dataclass.
SUPPORTED_DATASET_CONFIG: dict[str, Any] = {
    "open_code_reasoning": {
        "config": {"path": "nvidia/OpenCodeReasoning", "name": "split_0", "split": ["split_0"]},
        "preprocess": lambda sample: "\n".join([sample["input"], sample["output"]]),
    },
    "open_math_reasoning": {
        "config": {
            "path": "nvidia/OpenMathReasoning",
            "split": ["cot", "tir", "genselect"],
        },
        "preprocess": lambda sample: "\n".join([sample["problem"], sample["generated_solution"]]),
    },
    "llama-nemotron-post-training-dataset": {
        "config": {
            "path": "nvidia/Llama-Nemotron-Post-Training-Dataset",
            "name": "SFT",
            "split": ["code", "math", "science", "chat", "safety"],
        },
        "preprocess": lambda sample: "\n".join(turn["content"] for turn in sample["input"])
        + "\n"
        + sample["output"],
    },
    "nemotron-post-training-dataset-v2": {
        "config": {
            "path": "nvidia/Nemotron-Post-Training-Dataset-v2",
            "split": ["stem", "chat", "math", "code"],
        },
        "preprocess": lambda sample: "\n".join(turn["content"] for turn in sample["messages"]),
    },
    "nemotron-post-training-dataset-v1": {
        "config": {
            "path": "nvidia/Nemotron-Post-Training-Dataset-v1",
            "split": ["stem", "chat", "math", "code", "tool_calling"],
        },
        "preprocess": lambda sample: "\n".join(turn["content"] for turn in sample["messages"]),
    },
    "magpie": {
        "config": {
            "path": "Magpie-Align/Magpie-Pro-MT-300K-v0.1",
            "split": ["train"],
        },
        "preprocess": lambda sample: "\n".join(turn["value"] for turn in sample["conversations"]),
    },
    "cnn_dailymail": {
        "config": {"path": "cnn_dailymail", "name": "3.0.0", "split": ["train"]},
        "preprocess": lambda sample: sample["article"],
    },
    "pile": {
        "config": {"path": "monology/pile-uncopyrighted", "name": "v1.0", "split": ["train"]},
        "preprocess": lambda sample: sample["text"],
    },
    "pg19": {
        "config": {"path": "pg19", "name": "v1.0", "split": ["train"]},
        "preprocess": lambda sample: sample["text"],
    },
    "wikipedia": {
        "config": {"path": "wikipedia", "name": "20220301.en", "split": ["train"]},
        "preprocess": lambda sample: sample["text"],
    },
    "c4": {
        "config": {"path": "c4", "name": "en", "split": ["train"]},
        "preprocess": lambda sample: sample["text"],
    },
}

def _get_dataset_samples(dataset_name: str, data_dir, num_samples: int) -> list[str]:
    """Load a portion of train dataset with the dataset name and a given size.

    Args:
        dataset_name: Name of the dataset to load.
        num_samples: Number of samples to load from the dataset.

    Returns:
        Samples: The list of samples.
    """
    # Load the dataset
    if dataset_name not in SUPPORTED_DATASET_CONFIG:
        raise NotImplementedError(
            f"dataset {dataset_name} is not supported. Please use one of the following:"
            f" {get_supported_datasets()}."
        )

    from datasets import load_dataset

    dataset_config = SUPPORTED_DATASET_CONFIG[dataset_name]
    # It's unfortunate that the load_dataset function does not support split a list while streaming.
    # So we need to load the dataset for each split.
    config = dataset_config["config"].copy()
    splits = config.pop("split", [None])
    dataset_splits = [
        load_dataset(
            streaming=True,
            **config,
            split=split,
        )
        for split in splits
    ]

    # Split the samples evenly across the splits
    # For streaming datasets, there is no reliable way to get the number of samples in each split
    # other than loading the entire dataset. So, we just use the same number of samples for each split.
    num_samples_splits = [num_samples // len(dataset_splits) for _ in dataset_splits]
    num_samples_splits[-1] += num_samples - sum(num_samples_splits)
    samples = []
    for dataset, num_samples_split in zip(dataset_splits, num_samples_splits):
        for i, sample in enumerate(dataset):
            if i >= num_samples_split:
                break

            # Apply preprocess function to the sample
            samples.append(dataset_config["preprocess"](sample))

    return samples