"""CLI 入口 E2E 测试。"""

import importlib.util

import pytest


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _can_import(name: str) -> bool:
    """Check if a module can actually be imported (not just found on disk)."""
    try:
        __import__(name)
        return True
    except Exception:
        return False


@pytest.mark.e2e
class TestCliEntrypoint:
    """测试 model-optimizer-cli 入口与基本命令。"""

    def test_version(self, cli_cmd):
        """测试 version 子命令。"""
        result = cli_cmd("version")
        assert result.returncode == 0
        assert "Model optimizer" in result.stdout or "Welcome" in result.stdout

    def test_no_args_shows_usage(self, cli_cmd):
        """无参数时显示使用说明。"""
        result = cli_cmd()
        # 无子命令时可能返回非 0，但应打印 usage
        output = result.stdout + result.stderr
        assert "quantize" in output or "Usage" in output

    @pytest.mark.skipif(not _can_import("modelopt.torch"), reason="quantize 需 modelopt (且依赖可正常导入)")
    def test_quantize_help(self, cli_cmd):
        """quantize 子命令 help。"""
        result = cli_cmd("quantize", "--help")
        assert result.returncode == 0
        assert "model_path" in result.stdout
        assert "quantize_cfg" in result.stdout
        assert "export_dir" in result.stdout

    @pytest.mark.skipif(not _has_module("ultralytics"), reason="export 需 ultralytics")
    def test_export_help(self, cli_cmd):
        """export 子命令 help。"""
        result = cli_cmd("export", "--help")
        assert result.returncode == 0
        assert "model_name" in result.stdout

    def test_build_help(self, cli_cmd):
        """build 子命令 help。"""
        result = cli_cmd("build", "--help")
        assert result.returncode == 0
        assert "model_path" in result.stdout
