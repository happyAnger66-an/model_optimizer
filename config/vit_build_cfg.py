build_cfg = {
    "precision": "fp16",
    "workspace_mb": 8192,
    "min_shapes": {
        "pixel_values": (1, 3, 224, 224),
    },
    "opt_shapes": {
        "pixel_values": (1, 3, 224, 224),
    },
    "max_shapes": {
        "pixel_values": (1, 3, 224, 224),
    }
}