build_cfg = {
    "precision": "fp16",
    "workspace_mb": 8192,
    "min_shapes": {
        "attention_mask": (1, 1, 10, 978),
        "position_ids": (1, 10),
        "inputs_embeds": (1, 10, 1024),
        "adarms_cond": (1, 1024),
    },
    "opt_shapes": {
        "attention_mask": (1, 1, 10, 978),
        "position_ids": (1, 10),
        "inputs_embeds": (1, 10, 1024),
        "adarms_cond": (1, 1024),
    },
    "max_shapes": {
        "attention_mask": (1, 1, 10, 978),
        "position_ids": (1, 10),
        "inputs_embeds": (1, 10, 1024),
        "adarms_cond": (1, 1024),
    }
}