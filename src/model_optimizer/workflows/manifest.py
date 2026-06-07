"""Workflow manifest model.

A workflow manifest is user-authored input that describes which optimization
steps to run. It is intentionally separate from artifact manifests, which are
tool-authored output records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WORKFLOW_VERSION = "workflow_v1"
SUPPORTED_ACTIONS = ("quantize", "export", "build")


def _load_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with open(path, encoding="utf-8") as f:
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(
                    "YAML workflow manifests require PyYAML. "
                    "Use JSON or install pyyaml."
                ) from exc
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"workflow manifest root must be object, got {type(data).__name__}")
    return data


@dataclass
class WorkflowAction:
    name: str
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: str | dict[str, Any]) -> "WorkflowAction":
        if isinstance(raw, str):
            return cls(name=raw)
        if isinstance(raw, dict):
            if "name" in raw:
                name = str(raw["name"])
                cfg = {k: v for k, v in raw.items() if k != "name"}
                return cls(name=name, config=cfg)
            if len(raw) == 1:
                name, cfg = next(iter(raw.items()))
                return cls(name=str(name), config=dict(cfg or {}))
        raise ValueError(f"invalid workflow action: {raw!r}")


@dataclass
class WorkflowStage:
    name: str
    actions: list[WorkflowAction]
    model_name: str | None = None
    model_path: str | None = None
    output_dir: str | None = None
    quantize: dict[str, Any] = field(default_factory=dict)
    export: dict[str, Any] = field(default_factory=dict)
    build: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowStage":
        if "name" not in data:
            raise ValueError("workflow stage requires 'name'")
        raw_actions = data.get("actions", [])
        if not isinstance(raw_actions, list) or not raw_actions:
            raise ValueError(f"workflow stage {data['name']!r} requires non-empty actions")
        actions = [WorkflowAction.from_raw(a) for a in raw_actions]
        for action in actions:
            if action.name not in SUPPORTED_ACTIONS:
                raise ValueError(
                    f"unsupported action {action.name!r}; supported={SUPPORTED_ACTIONS}"
                )
        return cls(
            name=str(data["name"]),
            actions=actions,
            model_name=data.get("model_name"),
            model_path=data.get("model_path"),
            output_dir=data.get("output_dir"),
            quantize=dict(data.get("quantize", {}) or {}),
            export=dict(data.get("export", {}) or {}),
            build=dict(data.get("build", {}) or {}),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    def action_config(self, action: WorkflowAction) -> dict[str, Any]:
        merged = dict(getattr(self, action.name, {}) or {})
        merged.update(action.config)
        return merged


@dataclass
class WorkflowManifest:
    version: str
    stages: list[WorkflowStage]
    architecture: str = "pi05"
    model_name: str = ""
    model_path: str = ""
    output_dir: str = ""
    train_config: str | None = None
    feature_config: str | None = None
    dry_run: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowManifest":
        version = str(data.get("version", WORKFLOW_VERSION))
        if version != WORKFLOW_VERSION:
            raise ValueError(f"unsupported workflow version {version!r}")
        raw_stages = data.get("stages", [])
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("workflow manifest requires non-empty 'stages'")
        return cls(
            version=version,
            architecture=str(data.get("architecture", "pi05")),
            model_name=str(data.get("model_name", "")),
            model_path=str(data.get("model_path", data.get("checkpoint", ""))),
            output_dir=str(data.get("output_dir", "")),
            train_config=data.get("train_config"),
            feature_config=data.get("feature_config"),
            dry_run=bool(data.get("dry_run", False)),
            metadata=dict(data.get("metadata", {}) or {}),
            stages=[WorkflowStage.from_dict(s) for s in raw_stages],
        )

    @classmethod
    def load(cls, path: str | Path) -> "WorkflowManifest":
        return cls.from_dict(_load_mapping(Path(path).expanduser().resolve()))

    def stage_output_dir(self, stage: WorkflowStage) -> Path:
        if stage.output_dir:
            return Path(stage.output_dir).expanduser()
        if not self.output_dir:
            raise ValueError("workflow output_dir is required when stage.output_dir is omitted")
        return Path(self.output_dir).expanduser() / stage.name

    def stage_model_name(self, stage: WorkflowStage) -> str:
        if stage.model_name:
            return stage.model_name
        if not self.model_name:
            raise ValueError(f"model_name is required for stage {stage.name!r}")
        if "/" in self.model_name:
            return self.model_name
        return f"{self.model_name}/{stage.name}"

    def stage_model_path(self, stage: WorkflowStage) -> str:
        return str(stage.model_path or self.model_path)
