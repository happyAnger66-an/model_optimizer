"""Architecture spec and policy adapter E2E regression tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest


@pytest.mark.e2e
def test_pi05_architecture_spec_registered():
    from model_optimizer.architectures import get_architecture_spec, list_architecture_specs

    spec = get_architecture_spec("pi05")

    assert "pi05" in list_architecture_specs()
    assert spec.stage_names == ("vit", "embed_prefix", "llm", "expert", "denoise")
    assert spec.get_stage("denoise").supported_backends == (
        "pytorch",
        "tensorrt",
        "onnxrt",
        "native",
        "flashrt",
    )


@pytest.mark.e2e
def test_server_config_resolves_non_vla_architecture_stages():
    from model_optimizer.architectures import ArchitectureSpec, StageSpec, register_architecture_spec
    from model_optimizer.infer.server.config import ServerConfig

    register_architecture_spec(
        ArchitectureSpec(
            name="world_model_e2e",
            stages=(
                StageSpec(name="encoder", supported_backends=("pytorch", "tensorrt")),
                StageSpec(name="dynamics", supported_backends=("pytorch",)),
                StageSpec(name="decoder", supported_backends=("pytorch", "onnxrt")),
            ),
            description="Synthetic non-VLA architecture for regression tests.",
        )
    )

    cfg = ServerConfig(
        architecture="world_model_e2e",
        checkpoint="/tmp/fake-checkpoint",
        mode="pytorch",
    )

    assert cfg.resolve_stages() == {
        "encoder": "pytorch",
        "dynamics": "pytorch",
        "decoder": "pytorch",
    }


@pytest.mark.e2e
def test_server_config_rejects_unknown_architecture():
    from model_optimizer.infer.server.config import ServerConfig

    cfg = ServerConfig(architecture="missing_architecture_e2e", checkpoint="/tmp/fake")

    with pytest.raises(ValueError, match="Architecture .* not found"):
        cfg.validate()


@pytest.mark.e2e
def test_pi05_policy_adapter_unwraps_policy_model_without_openpi():
    from model_optimizer.policies import get_policy_adapter

    adapter = get_policy_adapter("pi05")
    model = object()

    class DirectPolicy:
        _model = model

    class InnerPolicy:
        _model = model

    class RecorderPolicy:
        _policy = InnerPolicy()

    assert adapter.unwrap_model(DirectPolicy()) is model
    assert adapter.unwrap_model(RecorderPolicy()) is model


@pytest.mark.e2e
def test_policy_loader_uses_registered_policy_adapter_without_openpi():
    from model_optimizer.architectures import ArchitectureSpec, StageSpec, register_architecture_spec
    from model_optimizer.infer.server.config import ServerConfig
    from model_optimizer.infer.server.policy_loader import _load_pytorch_policy
    from model_optimizer.policies import PolicyAdapter, register_policy_adapter

    @dataclass
    class FakePolicyAdapter(PolicyAdapter):
        architecture: str = "fake_policy_arch_e2e"
        train_config_calls: list[str] = field(default_factory=list)
        create_policy_calls: list[dict[str, Any]] = field(default_factory=list)

        def load_train_config(self, config_name: str) -> dict[str, str]:
            self.train_config_calls.append(config_name)
            return {"name": config_name}

        def create_policy(
            self,
            train_config: Any,
            checkpoint: str,
            *,
            pytorch_device: str | None = None,
            default_prompt: str | None = None,
            unify_action_mode: bool | None = None,
            robot_type: str | None = None,
        ) -> dict[str, Any]:
            call = {
                "train_config": train_config,
                "checkpoint": checkpoint,
                "pytorch_device": pytorch_device,
                "default_prompt": default_prompt,
                "unify_action_mode": unify_action_mode,
                "robot_type": robot_type,
            }
            self.create_policy_calls.append(call)
            return {"policy": call}

    register_architecture_spec(
        ArchitectureSpec(
            name="fake_policy_arch_e2e",
            stages=(StageSpec(name="core", supported_backends=("pytorch",)),),
        )
    )
    adapter = FakePolicyAdapter()
    register_policy_adapter(adapter)

    cfg = ServerConfig(
        architecture="fake_policy_arch_e2e",
        config_name="fake_config",
        checkpoint="/tmp/fake-policy",
        device="cpu",
    )
    progress: list[tuple[str, str]] = []

    policy, train_cfg = _load_pytorch_policy(
        cfg, lambda stage, message: progress.append((stage, message))
    )

    assert train_cfg == {"name": "fake_config"}
    assert policy["policy"]["checkpoint"] == "/tmp/fake-policy"
    assert policy["policy"]["pytorch_device"] == "cpu"
    assert adapter.train_config_calls == ["fake_config"]
    assert len(adapter.create_policy_calls) == 1
    assert [stage for stage, _ in progress] == ["config", "policy_pt", "policy_pt"]
