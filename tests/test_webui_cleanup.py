"""Tests to verify LlamaFactory remnants have been cleaned from webui modules.

These tests validate:
1. Copyright headers no longer reference LlamaFactory
2. LOCALES/ALERTS contain only used entries, no LlamaFactory leftovers
3. Dead code (engine.resume) has been removed
4. LlamaFactory naming (llamaboard, LLAMAFACTORY_VERBOSITY, etc.) is gone
5. Docstrings/comments no longer reference train.* / infer.*
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
WEBUI = SRC / "model_optimizer" / "webui"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_source(relpath: str) -> str:
    """Read a source file relative to src/model_optimizer/webui/."""
    return (WEBUI / relpath).read_text(encoding="utf-8")


def _all_py_files() -> list[Path]:
    """Return all .py files under src/model_optimizer/webui/."""
    return list(WEBUI.rglob("*.py"))


# ===================================================================
# 1. Copyright headers
# ===================================================================

class TestCopyrightHeaders:
    """All webui .py files must NOT mention LlamaFactory / LlamaOptimizer."""

    FORBIDDEN = ["LlamaFactory", "LlamaOptimizer"]

    @pytest.fixture(params=[
        "engine.py",
        "css.py",
        "manager.py",
        "locales.py",
        "components/eval.py",
        "components/footer.py",
        "components/compile.py",
        "components/profile.py",
        "extras/logging.py",
    ])
    def source_file(self, request):
        return request.param

    def test_no_llamafactory_in_copyright(self, source_file):
        content = _read_source(source_file)
        for word in self.FORBIDDEN:
            assert word not in content, (
                f"{source_file} still contains '{word}'"
            )

    def test_has_model_optimizer_copyright(self, source_file):
        content = _read_source(source_file)
        assert "model_optimizer team" in content, (
            f"{source_file} missing 'model_optimizer team' in copyright"
        )


# ===================================================================
# 2. No LlamaFactory remnants in any webui source
# ===================================================================

class TestNoLlamaFactoryRemnants:
    """No .py file under webui/ should contain LlamaFactory-related strings."""

    FORBIDDEN_STRINGS = [
        "LlamaFactory",
        "LlamaOptimizer",
        "llamafactory",
        "llamaboard",
        "LLAMAFACTORY",
        "hiyouga",
        "TRAINER_LOG",
    ]

    def test_no_forbidden_strings_in_webui(self):
        violations = []
        for pyfile in _all_py_files():
            content = pyfile.read_text(encoding="utf-8", errors="replace")
            for s in self.FORBIDDEN_STRINGS:
                if s in content:
                    violations.append(f"{pyfile.relative_to(SRC)}: contains '{s}'")
        assert not violations, "LlamaFactory remnants found:\n" + "\n".join(violations)


# ===================================================================
# 3. LOCALES & ALERTS validation
# ===================================================================

class TestLocales:
    """Verify LOCALES and ALERTS integrity after cleanup."""

    def _get_locales_alerts(self):
        from model_optimizer.webui.locales import LOCALES, ALERTS
        return LOCALES, ALERTS

    # --- removed keys must NOT exist ---

    REMOVED_LOCALE_KEYS = [
        "template", "rope_scaling", "booster", "training_stage",
        "data_preview_btn", "preview_count", "page_index",
        "prev_btn", "next_btn", "close_btn", "preview_samples",
        "batch_size", "warmup_steps", "extra_args",
        "infer_backend", "infer_dtype", "load_btn", "info_box",
        "query", "submit_btn", "max_length", "clear_btn",
        "export_size", "export_quantization_bit",
        "export_quantization_dataset", "export_device",
        "hub_name", "checkpoint_path",
    ]

    REMOVED_ALERT_KEYS = [
        "err_exists", "err_no_adapter", "err_no_reward_model",
        "err_no_export_dir", "err_gptq_lora", "err_demo",
        "err_tool_name", "err_json_schema", "err_config_not_found",
        "warn_output_dir_exists", "warn_no_instruct",
        "info_config_saved", "info_config_loaded",
        "info_loading", "info_unloading", "info_loaded", "info_unloaded",
        "info_exporting", "info_exported",
    ]

    def test_removed_locale_keys_absent(self):
        locales, _ = self._get_locales_alerts()
        for key in self.REMOVED_LOCALE_KEYS:
            assert key not in locales, f"LOCALES still contains removed key '{key}'"

    def test_removed_alert_keys_absent(self):
        _, alerts = self._get_locales_alerts()
        for key in self.REMOVED_ALERT_KEYS:
            assert key not in alerts, f"ALERTS still contains removed key '{key}'"

    # --- kept keys must still exist ---

    KEPT_LOCALE_KEYS = [
        "title", "subtitle", "lang", "model_name", "model_path",
        "quantization_bit", "simplifier", "export_format",
        "quantization_method", "calibrate_method", "quantize_cfg",
        "calibrate_data", "progress_bar", "do_perf", "build_cfg",
        "export_dir", "e2e_profile", "layer_profile", "e2e_prof",
        "layer_prof", "extra_compile_args", "shapes", "dataset_dir",
        "dataset", "start_btn", "stop_btn", "output_dir", "output_box",
        "predict", "export_btn", "device_memory",
    ]

    KEPT_ALERT_KEYS = [
        "err_conflict", "err_profile_conflict", "err_profile_model_type",
        "err_no_model", "err_no_path", "err_no_dataset",
        "err_no_output_dir", "err_failed", "warn_no_cuda",
        "info_aborting", "info_aborted", "info_finished",
    ]

    def test_kept_locale_keys_present(self):
        locales, _ = self._get_locales_alerts()
        for key in self.KEPT_LOCALE_KEYS:
            assert key in locales, f"LOCALES missing expected key '{key}'"

    def test_kept_alert_keys_present(self):
        _, alerts = self._get_locales_alerts()
        for key in self.KEPT_ALERT_KEYS:
            assert key in alerts, f"ALERTS missing expected key '{key}'"

    # --- each entry must have both en and zh ---

    def test_locales_have_en_zh(self):
        locales, _ = self._get_locales_alerts()
        for key, langs in locales.items():
            assert "en" in langs, f"LOCALES['{key}'] missing 'en'"
            assert "zh" in langs, f"LOCALES['{key}'] missing 'zh'"

    def test_alerts_have_en_zh(self):
        _, alerts = self._get_locales_alerts()
        for key, langs in alerts.items():
            assert "en" in langs, f"ALERTS['{key}'] missing 'en'"
            assert "zh" in langs, f"ALERTS['{key}'] missing 'zh'"

    # --- links should point to real project ---

    def test_subtitle_links_correct(self):
        locales, _ = self._get_locales_alerts()
        for lang in ("en", "zh"):
            value = locales["subtitle"][lang]["value"]
            assert "happyAnger66-an/model_optimizer" in value, (
                f"subtitle[{lang}] has wrong project URL"
            )
            assert "hiyouga" not in value
            assert "llamafactory" not in value.lower()

    def test_title_not_finetuning(self):
        locales, _ = self._get_locales_alerts()
        en_title = locales["title"]["en"]["value"]
        assert "Fine-Tuning" not in en_title
        assert "LLMs" not in en_title

    # --- Chinese text should not reference training ---

    def test_alerts_zh_no_training_reference(self):
        _, alerts = self._get_locales_alerts()
        for key in ("err_conflict", "info_aborting"):
            zh = alerts[key]["zh"]
            assert "训练" not in zh, (
                f"ALERTS['{key}']['zh'] still references training: {zh}"
            )


# ===================================================================
# 4. engine.py dead code removal
# ===================================================================

class TestEngineCleanup:
    """engine.py must not have resume() or references to non-existent components."""

    def test_no_resume_method(self):
        src = _read_source("engine.py")
        assert "def resume" not in src

    def test_no_chatter_reference(self):
        src = _read_source("engine.py")
        assert "self.chatter" not in src

    def test_no_train_references(self):
        src = _read_source("engine.py")
        assert "train." not in src

    def test_no_infer_references(self):
        src = _read_source("engine.py")
        assert "infer." not in src

    def test_no_get_time_import(self):
        src = _read_source("engine.py")
        assert "get_time" not in src


# ===================================================================
# 5. commom.py naming cleanup
# ===================================================================

class TestCommomCleanup:
    """commom.py constants must use model_optimizer naming."""

    def test_cache_dir_renamed(self):
        from model_optimizer.webui.commom import DEFAULT_CACHE_DIR
        assert DEFAULT_CACHE_DIR == "model_optimizer_cache"
        assert "llamaboard" not in DEFAULT_CACHE_DIR

    def test_config_dir_renamed(self):
        from model_optimizer.webui.commom import DEFAULT_CONFIG_DIR
        assert DEFAULT_CONFIG_DIR == "model_optimizer_config"
        assert "llamaboard" not in DEFAULT_CONFIG_DIR

    def test_clean_cmd_no_llamafactory_keys(self):
        from model_optimizer.webui.commom import _clean_cmd
        # LlamaFactory-specific keys should not appear in no_skip_keys
        test_args = {
            "packing": False,
            "enable_thinking": False,
            "freeze_vision_tower": False,
            "model_path": "/some/path",
            "output_dir": "/output",
            "empty_val": "",
            "none_val": None,
        }
        result = _clean_cmd(test_args)
        # False values of LlamaFactory keys should be filtered out now
        assert "packing" not in result
        assert "enable_thinking" not in result
        assert "freeze_vision_tower" not in result
        # Real values should pass through
        assert result["model_path"] == "/some/path"
        assert result["output_dir"] == "/output"
        # Empty/None should be filtered
        assert "empty_val" not in result
        assert "none_val" not in result


# ===================================================================
# 6. logging.py cleanup
# ===================================================================

class TestLoggingCleanup:
    """logging.py must use MODEL_OPTIMIZER_VERBOSITY, not LLAMAFACTORY_VERBOSITY."""

    def test_env_var_renamed(self):
        src = _read_source("extras/logging.py")
        assert "MODEL_OPTIMIZER_VERBOSITY" in src
        assert "LLAMAFACTORY_VERBOSITY" not in src

    def test_docstring_updated(self):
        src = _read_source("extras/logging.py")
        assert "LLaMA Board" not in src
        assert "Model Optimizer" in src

    def test_env_var_respected(self):
        """MODEL_OPTIMIZER_VERBOSITY env var should control logging level."""
        # Reimport with fresh state
        import importlib
        from model_optimizer.webui.extras import logging as molog

        old = os.environ.get("MODEL_OPTIMIZER_VERBOSITY")
        try:
            os.environ["MODEL_OPTIMIZER_VERBOSITY"] = "DEBUG"
            # Force re-evaluation
            import logging
            level = molog._get_default_logging_level()
            assert level == logging.DEBUG
        finally:
            if old is None:
                os.environ.pop("MODEL_OPTIMIZER_VERBOSITY", None)
            else:
                os.environ["MODEL_OPTIMIZER_VERBOSITY"] = old


# ===================================================================
# 7. constants.py cleanup
# ===================================================================

class TestConstantsCleanup:
    """constants.py must not export TRAINER_LOG."""

    def test_no_trainer_log(self):
        from model_optimizer.webui.extras import constants
        assert not hasattr(constants, "TRAINER_LOG")

    def test_required_constants_exist(self):
        from model_optimizer.webui.extras.constants import (
            RUNNING_LOG,
            PROGRESS_LOG,
            QUANTIZE_LOG,
        )
        assert RUNNING_LOG == "running_log.txt"
        assert PROGRESS_LOG == "progress.jsonl"
        assert QUANTIZE_LOG == "quantize_log.jsonl"


# ===================================================================
# 8. Docstring / comment cleanup
# ===================================================================

class TestDocstringCleanup:
    """Comments and docstrings must not reference train.* or have typos."""

    def test_control_docstring_no_train(self):
        src = _read_source("control.py")
        # Should reference quantize.* not train.*
        assert "train.output_path" not in src
        assert "train.output_box" not in src
        assert "train.loss_viewer" not in src
        assert "train.swanlab_link" not in src

    def test_manager_docstring_no_train(self):
        src = _read_source("manager.py")
        assert "train.dataset" not in src

    def test_runner_no_monitorgit_typo(self):
        src = _read_source("runner.py")
        assert "Monitorgit" not in src

    def test_runners_init_no_monitorgit_typo(self):
        src = _read_source("runners/__init__.py")
        assert "Monitorgit" not in src
