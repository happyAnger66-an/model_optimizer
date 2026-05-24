# TensorRT build config for π0.5 LLM with model_optimizer CuTe DSL FMHA D=256 plugin.
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PLUGIN_SO = _REPO / "cpp" / "build" / "lib" / "libmodel_opt_plugin.so"

build_cfg = {
    "precision": "fp16",
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
        str(_PLUGIN_SO),
    ],
}
