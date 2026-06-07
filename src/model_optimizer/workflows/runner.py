"""Workflow runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_optimizer.artifacts import (
    ArtifactManifest,
    load_or_create_manifest,
    manifest_path_for,
)

from .manifest import WorkflowAction, WorkflowManifest, WorkflowStage


@dataclass
class WorkflowCommand:
    stage: str
    action: str
    argv: list[str]
    output_dir: str


@dataclass
class WorkflowResult:
    commands: list[WorkflowCommand] = field(default_factory=list)


class WorkflowRunner:
    """Run a workflow manifest by delegating to existing CLI implementations."""

    def __init__(self, manifest: WorkflowManifest):
        self.manifest = manifest

    def plan(self) -> WorkflowResult:
        result = WorkflowResult()
        for stage in self.manifest.stages:
            for action in stage.actions:
                result.commands.append(self._build_command(stage, action))
        return result

    def run(self) -> WorkflowResult:
        result = WorkflowResult()
        for stage in self.manifest.stages:
            stage_output = self.manifest.stage_output_dir(stage)
            stage_output.mkdir(parents=True, exist_ok=True)
            for action in stage.actions:
                cmd = self._build_command(stage, action)
                result.commands.append(cmd)
                if self.manifest.dry_run:
                    continue
                self._run_command(cmd)
                self._record_step(stage, action, cmd)
        return result

    def _build_command(self, stage: WorkflowStage, action: WorkflowAction) -> WorkflowCommand:
        cfg = stage.action_config(action)
        output_dir = Path(str(cfg.get("output_dir") or self.manifest.stage_output_dir(stage))).expanduser()
        model_name = str(cfg.get("model_name") or self.manifest.stage_model_name(stage))
        model_path = str(cfg.get("model_path") or self._default_model_path(stage, action, cfg))
        train_config = cfg.get("train_config", self.manifest.train_config)
        feature_config = cfg.get("feature_config", self.manifest.feature_config)

        if action.name == "quantize":
            argv = [
                "workflow",
                "--model_name",
                model_name,
                "--model_path",
                model_path,
                "--quantize_cfg",
                self._required(cfg, "quantize_cfg", stage, action),
                "--calibrate_data",
                self._required(cfg, "calibrate_data", stage, action),
                "--export_dir",
                str(output_dir),
            ]
            self._append_optional(argv, "--calibrate_method", cfg.get("calibrate_method"))
            self._append_optional(argv, "--verify_data", cfg.get("verify_data"))
            self._append_optional(argv, "--train_config", train_config)
            self._append_optional(argv, "--feature_config", feature_config)
            self._append_optional(argv, "--native-quant-spec-export", cfg.get("native_quant_spec_export"))
            self._append_optional(argv, "--native-calib-component", cfg.get("native_calib_component"))
            self._append_optional(argv, "--native-calib-percentile", cfg.get("native_calib_percentile"))
            self._append_optional(argv, "--native-calib-max-samples", cfg.get("native_calib_max_samples"))
            if bool(cfg.get("verify", False)):
                self._append_optional(argv, "--verify", "True")
            if bool(cfg.get("measure_quant_error", False)):
                argv.append("--measure-quant-error")
        elif action.name == "export":
            argv = [
                "workflow",
                "--model_name",
                model_name,
                "--model_path",
                model_path,
                "--export_dir",
                str(output_dir),
            ]
            self._append_optional(argv, "--export_type", cfg.get("export_type"))
            self._append_optional(argv, "--simplifier", cfg.get("simplifier"))
            self._append_optional(argv, "--verify_data", cfg.get("verify_data"))
            self._append_optional(argv, "--train_config", train_config)
            self._append_optional(argv, "--mode", cfg.get("mode"))
            self._append_optional(argv, "--feature_config", feature_config)
        elif action.name == "build":
            argv = [
                "workflow",
                "--model_path",
                model_path,
                "--build_cfg",
                self._required(cfg, "build_cfg", stage, action),
                "--export_dir",
                str(cfg.get("engine_path") or output_dir / self._default_engine_name(stage)),
            ]
            self._append_optional(argv, "--use_cudagraph", cfg.get("use_cudagraph"))
        else:
            raise ValueError(f"unsupported action {action.name!r}")

        return WorkflowCommand(
            stage=stage.name,
            action=action.name,
            argv=argv,
            output_dir=str(output_dir),
        )

    def _run_command(self, cmd: WorkflowCommand) -> None:
        if cmd.action == "quantize":
            from model_optimizer.quantization.cli import quantize_cli

            quantize_cli(cmd.argv)
        elif cmd.action == "export":
            from model_optimizer.convert.convert_formt import convert_model

            convert_model(cmd.argv)
        elif cmd.action == "build":
            from model_optimizer.trt_build.cli import build_cli

            build_cli(cmd.argv)
        else:
            raise ValueError(f"unsupported action {cmd.action!r}")

    def _record_step(self, stage: WorkflowStage, action: WorkflowAction, cmd: WorkflowCommand) -> None:
        output_path = (
            Path(self._flag_value(cmd.argv, "--export_dir")).expanduser()
            if action.name == "build"
            else Path(cmd.output_dir).expanduser()
        )
        paths: dict[str, str] = {}
        if action.name in ("quantize", "export"):
            onnx_path = self._find_onnx(Path(cmd.output_dir).expanduser(), stage)
            if onnx_path is not None:
                paths["onnx"] = str(onnx_path.resolve())
        manifest_path = manifest_path_for(output_path)
        manifest = load_or_create_manifest(manifest_path)
        workflow = manifest.metadata.get("workflow", {})
        if not isinstance(workflow, dict):
            workflow = {}
        actions = workflow.get("actions", [])
        if not isinstance(actions, list):
            actions = []
        actions.append(
            {
                "name": action.name,
                "argv": list(cmd.argv[1:]),
            }
        )
        workflow.update(
            {
                "version": self.manifest.version,
                "actions": actions,
            }
        )
        manifest.merge_update(
            architecture=self.manifest.architecture,
            stage=stage.name,
            model_name=self.manifest.stage_model_name(stage),
            paths=paths,
            metadata={"workflow": workflow},
        )
        manifest.save(manifest_path)

    def _default_model_path(
        self,
        stage: WorkflowStage,
        action: WorkflowAction,
        cfg: dict[str, Any],
    ) -> str:
        if action.name == "build":
            candidate = self._manifest_onnx_path(stage)
            if candidate:
                return candidate
            stage_output = Path(str(cfg.get("output_dir") or self.manifest.stage_output_dir(stage))).expanduser()
            return str(stage_output / f"{stage.name}.onnx")
        model_path = self.manifest.stage_model_path(stage)
        if not model_path:
            raise ValueError(f"model_path is required for {stage.name}.{action.name}")
        return model_path

    def _manifest_onnx_path(self, stage: WorkflowStage) -> str:
        manifest_path = manifest_path_for(self.manifest.stage_output_dir(stage))
        if not manifest_path.is_file():
            return ""
        manifest = ArtifactManifest.load(manifest_path)
        return str(manifest.paths.get("onnx", ""))

    def _find_onnx(self, output_dir: Path, stage: WorkflowStage) -> Path | None:
        preferred = output_dir / f"{stage.name}.onnx"
        if preferred.is_file():
            return preferred
        matches = sorted(output_dir.glob("*.onnx"))
        if len(matches) == 1:
            return matches[0]
        return None

    @staticmethod
    def _default_engine_name(stage: WorkflowStage) -> str:
        return f"{stage.name}.engine"

    @staticmethod
    def _append_optional(argv: list[str], flag: str, value: Any) -> None:
        if value is None or value == "":
            return
        argv.extend([flag, str(value)])

    @staticmethod
    def _flag_value(argv: list[str], flag: str) -> str:
        try:
            idx = argv.index(flag)
        except ValueError as exc:
            raise ValueError(f"missing required flag {flag!r} in argv: {argv}") from exc
        try:
            return argv[idx + 1]
        except IndexError as exc:
            raise ValueError(f"missing value for flag {flag!r} in argv: {argv}") from exc

    @staticmethod
    def _required(
        cfg: dict[str, Any],
        key: str,
        stage: WorkflowStage,
        action: WorkflowAction,
    ) -> str:
        value = cfg.get(key)
        if value is None or value == "":
            raise ValueError(f"{stage.name}.{action.name} requires {key!r}")
        return str(value)
