"""Policy adapter interface for model families and external runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class PolicyAdapter(ABC):
    """Bridge between model_optimizer and a model family's policy API."""

    architecture: str

    @abstractmethod
    def load_train_config(self, config_name: str) -> Any:
        """Load the model-family training/runtime config."""

    @abstractmethod
    def create_policy(
        self,
        train_config: Any,
        checkpoint: str,
        *,
        pytorch_device: str | None = None,
        default_prompt: str | None = None,
        unify_action_mode: bool | None = None,
        robot_type: str | None = None,
    ) -> Any:
        """Create a policy object for inference."""

    def unwrap_model(self, policy: Any) -> Any | None:
        """Return the underlying torch model if one is available."""

        return getattr(policy, "_model", None)

    def metadata(self, policy: Any) -> dict[str, Any]:
        """Return policy metadata if available."""

        meta = getattr(policy, "metadata", None) or {}
        return dict(meta)
