build_cfg = {
    "precision": "bf16",
    "workspace_mb": 8192,
    "min_shapes": {
        "inputs_embeds": (1, 968, 2048),
        "attention_mask": (1, 1, 968, 968),
        "position_ids": (1, 968),
    },
    "opt_shapes": {
        "inputs_embeds": (1, 968, 2048),
        "attention_mask": (1, 1, 968, 968),
        "position_ids": (1, 968),
    },
    "max_shapes": {
        "inputs_embeds": (1, 968, 2048),
        "attention_mask": (1, 1, 968, 968),
        "position_ids": (1, 968),
    },
    "plugin_lib_paths": [
        "/srcs/codes/llmOnEdge/build-fmha/libNvInfer_edgellm_plugin.so"
    ]
}