"""Artifact manifest data model."""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "artifact_manifest.json"
MANIFEST_VERSION = "artifact_manifest_v1"


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class ArtifactManifest:
    """Manifest describing optimization artifacts in a stage/output directory."""

    version: str = MANIFEST_VERSION
    artifact_type: str = ""
    architecture: str = ""
    stage: str = ""
    model_name: str = ""
    precision: str = ""
    quant_format: str = ""
    paths: dict[str, str] = field(default_factory=dict)
    configs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(default_factory=utc_now_iso)
    updated_at_utc: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "artifact_type": self.artifact_type,
            "architecture": self.architecture,
            "stage": self.stage,
            "model_name": self.model_name,
            "precision": self.precision,
            "quant_format": self.quant_format,
            "paths": _jsonable(self.paths),
            "configs": _jsonable(self.configs),
            "metrics": _jsonable(self.metrics),
            "metadata": _jsonable(self.metadata),
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactManifest":
        if not isinstance(data, dict):
            raise TypeError(f"manifest root must be object, got {type(data).__name__}")
        return cls(
            version=str(data.get("version", MANIFEST_VERSION)),
            artifact_type=str(data.get("artifact_type", "")),
            architecture=str(data.get("architecture", "")),
            stage=str(data.get("stage", "")),
            model_name=str(data.get("model_name", "")),
            precision=str(data.get("precision", "")),
            quant_format=str(data.get("quant_format", "")),
            paths=dict(data.get("paths", {}) or {}),
            configs=dict(data.get("configs", {}) or {}),
            metrics=dict(data.get("metrics", {}) or {}),
            metadata=dict(data.get("metadata", {}) or {}),
            created_at_utc=str(data.get("created_at_utc", utc_now_iso())),
            updated_at_utc=str(data.get("updated_at_utc", utc_now_iso())),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ArtifactManifest":
        p = Path(path).expanduser().resolve()
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_dict(raw)

    def save(self, path: str | Path) -> None:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)

    def merge_update(
        self,
        *,
        artifact_type: str | None = None,
        architecture: str | None = None,
        stage: str | None = None,
        model_name: str | None = None,
        precision: str | None = None,
        quant_format: str | None = None,
        paths: dict[str, str] | None = None,
        configs: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ArtifactManifest":
        if artifact_type is not None:
            self.artifact_type = artifact_type
        if architecture is not None:
            self.architecture = architecture
        if stage is not None:
            self.stage = stage
        if model_name is not None:
            self.model_name = model_name
        if precision is not None:
            self.precision = precision
        if quant_format is not None:
            self.quant_format = quant_format
        if paths:
            self.paths.update(paths)
        if configs:
            self.configs.update(configs)
        if metrics:
            self.metrics.update(metrics)
        if metadata:
            self.metadata.update(metadata)
        self.updated_at_utc = utc_now_iso()
        return self
