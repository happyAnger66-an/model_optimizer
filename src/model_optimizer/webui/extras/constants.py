from collections import OrderedDict, defaultdict
from enum import Enum
from typing import Optional

SUPPORTED_MODELS = OrderedDict()

DEFAULT_TEMPLATE = defaultdict(str)

RUNNING_LOG = "running_log.txt"
PROGRESS_LOG = "progress.jsonl"
QUANTIZE_LOG = "quantize_log.jsonl"
TRAINER_LOG = "trainer_log.jsonl"

class DownloadSource(str, Enum):
    DEFAULT = "hf"
    MODELSCOPE = "ms"
    OPENMIND = "om"

def register_model_group(
    models: dict[str, dict[DownloadSource, str]],
    template: Optional[str] = None,
) -> None:
    for name, path in models.items():
        SUPPORTED_MODELS[name] = path

    DEFAULT_TEMPLATE[name] = template

register_model_group(
    models={
        "yolo_tube.pt": {
            DownloadSource.DEFAULT: "CohereForAI/aya-23-8B",
        }
    },
    template="yolo",
)

register_model_group(
    models={
        "resnet50-v1-12.pt": {
            DownloadSource.DEFAULT: "CohereForAI/aya-23-8B",
        }
    },
    template="resnet",
)

register_model_group(
    models={
        "pi05_libero": {
            DownloadSource.DEFAULT: "lerobot/pi05_base",
        }
    },
    template="pi05",
)

register_model_group(
    models={
        "pi05_libero/vit": {
            DownloadSource.DEFAULT: "lerobot/pi05_base",
        }
    },
    template="pi05",
)