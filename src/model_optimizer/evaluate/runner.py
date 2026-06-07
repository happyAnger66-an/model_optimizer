"""Generic evaluation runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvalRequest:
    model_name: str
    model_path: str
    dataset: str
    output_dir: str
    batch_size: int = 1
    max_data: int = 100
    train_config: str | None = None


@dataclass
class EvalResult:
    metric: Any = None


def run_eval(request: EvalRequest) -> EvalResult:
    from model_optimizer.models.registry import get_model_cls

    model_cls = get_model_cls(request.model_name)
    model = model_cls.construct_from_name_path(
        request.model_name,
        request.model_path,
        request.train_config,
    )
    metric = model.val(
        request.dataset,
        request.batch_size,
        request.max_data,
        request.output_dir,
    )
    return EvalResult(metric=metric)
