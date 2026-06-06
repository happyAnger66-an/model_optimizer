"""OpenPI Pi0.5 policy adapter."""

from __future__ import annotations

from typing import Any

from model_optimizer.architectures.pi05 import PI05_ARCHITECTURE_NAME

from .base import PolicyAdapter
from .registry import register_policy_adapter


class OpenPiPi05PolicyAdapter(PolicyAdapter):
    """Adapter for OpenPI Pi0.5 policies.

    Imports from ``openpi`` are intentionally lazy so registry tests can run in
    environments where the heavy model stack is not installed.
    """

    architecture = PI05_ARCHITECTURE_NAME

    def load_train_config(self, config_name: str) -> Any:
        from openpi.training import config as _config

        return _config.get_config(config_name)

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
        from openpi.policies import policy_config

        kwargs: dict[str, Any] = {}
        if pytorch_device is not None:
            kwargs["pytorch_device"] = pytorch_device
        if default_prompt is not None:
            kwargs["default_prompt"] = default_prompt
        if unify_action_mode is not None:
            kwargs["unify_action_mode"] = unify_action_mode
        if robot_type is not None:
            kwargs["robot_type"] = robot_type
        return policy_config.create_trained_policy(train_config, checkpoint, **kwargs)

    def unwrap_model(self, policy: Any) -> Any | None:
        inner = getattr(policy, "_policy", None)
        if inner is not None:
            model = getattr(inner, "_model", None)
            if model is not None:
                return model
        return getattr(policy, "_model", None)


def register() -> OpenPiPi05PolicyAdapter:
    """Register and return the OpenPI Pi0.5 policy adapter."""

    adapter = OpenPiPi05PolicyAdapter()
    register_policy_adapter(adapter)
    return adapter


register()
