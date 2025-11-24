import os
import datetime
import signal

from typing import Any, Optional, Union
import json

from psutil import Process
from yaml import safe_dump, safe_load

from .extras import logging

DEFAULT_DATA_DIR = "data"
DEFAULT_CACHE_DIR = "llamaboard_cache"
DEFAULT_CONFIG_DIR = "llamaboard_config"
DEFAULT_SAVE_DIR = "saves"
USER_CONFIG = "user_config.yaml"

logger = logging.get_logger(__name__)

def get_time() -> str:
    r"""Get current date and time."""
    return datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S")

def _get_config_path() -> os.PathLike:
    r"""Get the path to user config."""
    return os.path.join(DEFAULT_CACHE_DIR, USER_CONFIG)

def load_config() -> dict[str, Union[str, dict[str, Any]]]:
    r"""Load user config if exists."""
    try:
        with open(_get_config_path(), encoding="utf-8") as f:
            return safe_load(f)
    except Exception:
        return {"lang": None, "hub_name": None, "last_model": None, "path_dict": {}, "cache_dir": None}

def save_config(
    lang: str, hub_name: Optional[str] = None, model_name: Optional[str] = None, model_path: Optional[str] = None
) -> None:
    r"""Save user config."""
    os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
    user_config = load_config()
    user_config["lang"] = lang or user_config["lang"]
    if hub_name:
        user_config["hub_name"] = hub_name

    if model_name:
        user_config["last_model"] = model_name

    if model_name and model_path:
        user_config["path_dict"][model_name] = model_path

    with open(_get_config_path(), "w", encoding="utf-8") as f:
        safe_dump(user_config, f)

def abort_process(pid: int) -> None:
    r"""Abort the processes recursively in a bottom-up way."""
    try:
        children = Process(pid).children()
        if children:
            for child in children:
                abort_process(child.pid)

        os.kill(pid, signal.SIGABRT)
    except Exception:
        pass

def get_save_dir(*paths: str) -> os.PathLike:
    r"""Get the path to saved model checkpoints."""
    if os.path.sep in paths[-1]:
        logger.warning_rank0("Found complex path, some features may be not available.")
        return paths[-1]

    paths = (path.replace(" ", "").strip() for path in paths)
    return os.path.join(DEFAULT_SAVE_DIR, *paths)

def _clean_cmd(args: dict[str, Any]) -> dict[str, Any]:
    r"""Remove args with NoneType or False or empty string value."""
    no_skip_keys = [
        "packing",
        "enable_thinking",
        "use_reentrant_gc",
        "double_quantization",
        "freeze_vision_tower",
        "freeze_multi_modal_projector",
    ]
    return {k: v for k, v in args.items() if (k in no_skip_keys) or (v is not None and v is not False and v != "")}

def save_cmd(args: dict[str, Any]) -> str:
    r"""Save CLI commands to launch training."""
    output_dir = args["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
#    with open(os.path.join(output_dir, TRAINING_ARGS), "w", encoding="utf-8") as f:
#        safe_dump(_clean_cmd(args), f)

    return os.path.join(output_dir, '')

def load_eval_results(path: os.PathLike) -> str:
    r"""Get scores after evaluation."""
    return ""
#    with open(path, encoding="utf-8") as f:
#        result = json.dumps(json.load(f), indent=4)
#
#    return f"```json\n{result}\n```\n"