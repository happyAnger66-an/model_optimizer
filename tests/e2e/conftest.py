"""pytest E2E 测试公共配置与 Fixtures。"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 将 src 加入 path，以便 import model_optimizer
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture(scope="session")
def project_root():
    """项目根目录。"""
    return PROJECT_ROOT


@pytest.fixture
def tmp_workspace(tmp_path):
    """临时工作目录，用于 E2E 测试的导入/导出。"""
    workspace = tmp_path / "e2e_workspace"
    workspace.mkdir(exist_ok=True)
    return workspace


@pytest.fixture
def cli_cmd():
    """运行 model-optimizer-cli。优先使用已安装的 entry point。"""

    def _run(*args, cwd=None, env=None, timeout=60, check=True):
        import shutil

        cli_bin = shutil.which("model-optimizer-cli") or shutil.which("model-opt")
        if cli_bin:
            cmd = [cli_bin] + list(args)
        else:
            cmd = [sys.executable, "-m", "model_optimizer.cli"] + list(args)
        cwd = cwd or str(PROJECT_ROOT)
        full_env = os.environ.copy()
        # 确保 src 在 PYTHONPATH 中，支持开发模式
        src_path = str(PROJECT_ROOT / "src")
        full_env["PYTHONPATH"] = src_path + os.pathsep + full_env.get("PYTHONPATH", "")
        if env:
            full_env.update(env)
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=full_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if check and result.returncode != 0:
            pytest.fail(
                f"CLI failed: {result.stderr or result.stdout}\nargs: {args}"
            )
        return result

    return _run


@pytest.fixture
def run_cli(cli_cmd):
    """运行 model-optimizer-cli，返回 (returncode, stdout, stderr)。"""

    def _run(*args, check=False, **kwargs):
        kwargs["check"] = check
        result = cli_cmd(*args, **kwargs)
        return result.returncode, result.stdout, result.stderr

    return _run
