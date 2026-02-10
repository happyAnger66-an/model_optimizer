build_cfg = {
    "precision": "bf16",
    "workspace_mb": 8192,
    "min_shapes": {
        "attention_mask": (1, 1, 10, 978),
        "position_ids": (1, 10),
        "inputs_embeds": (1, 10, 1024),
        "adarms_cond": (1, 1024),
        "past_keys": (18, 1, 968, 256),
        "past_values": (18, 1, 968, 256),
    },
    "opt_shapes": {
        "attention_mask": (1, 1, 10, 978),
        "position_ids": (1, 10),
        "inputs_embeds": (1, 10, 1024),
        "adarms_cond": (1, 1024),
        "past_keys": (18, 1, 968, 256),
        "past_values": (18, 1, 968, 256),
    },
    "max_shapes": {
        "attention_mask": (1, 1, 10, 978),
        "position_ids": (1, 10),
        "inputs_embeds": (1, 10, 1024),
        "adarms_cond": (1, 1024),
        "past_keys": (18, 1, 968, 256),
        "past_values": (18, 1, 968, 256),
    }
}