"""Generic model architecture and stage specifications."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StageSpec:
    """A named execution stage in a model architecture.

    The stage abstraction is intentionally model-agnostic: a stage may be a VLA
    vision encoder, an LLM block, a world-model dynamics module, or any other
    deployable subgraph.
    """

    name: str
    default_backend: str = "pytorch"
    supported_backends: tuple[str, ...] = ("pytorch",)
    quantizable: bool = False
    description: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ArchitectureSpec:
    """A model architecture composed from named stages."""

    name: str
    stages: tuple[StageSpec, ...]
    default_backend: str = "pytorch"
    description: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages)

    @property
    def quantizable_stage_names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages if stage.quantizable)

    def get_stage(self, name: str) -> StageSpec:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(f"Stage {name!r} is not registered for architecture {self.name!r}")

    def validate_backend(self, stage_name: str, backend: str) -> None:
        stage = self.get_stage(stage_name)
        if backend not in stage.supported_backends:
            raise ValueError(
                f"Backend {backend!r} is not supported for stage {stage_name!r} "
                f"in architecture {self.name!r}; allowed={stage.supported_backends}"
            )
