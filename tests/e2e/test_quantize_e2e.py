"""量化流程 E2E 测试。需提供真实模型与校准数据时取消 skip。"""

import pytest


@pytest.mark.e2e
@pytest.mark.e2e_gpu
@pytest.mark.skip(reason="需要真实 pi05 模型与校准数据，CI 中跳过")
class TestQuantizeE2E:
    """量化完整流程 E2E 测试。"""

    def test_quantize_pi05_llm_nvfp4(
        self, cli_cmd, tmp_workspace, project_root
    ):
        """pi05 LLM NVFP4 量化 E2E。"""
        model_path = "/path/to/pytorch_pi05_libero"
        calib_data = "/path/to/pi05_llm_calib_datas.pt"
        quant_cfg = str(project_root / "config" / "quant" / "llm_nvfp4_quant_cfg.py")
        export_dir = str(tmp_workspace / "nvfp4_out")

        result = cli_cmd(
            "quantize",
            "--model_name", "pi05_libero/llm",
            "--model_path", model_path,
            "--quantize_cfg", quant_cfg,
            "--calibrate_data", calib_data,
            "--export_dir", export_dir,
        )
        assert result.returncode == 0
        assert (tmp_workspace / "nvfp4_out" / "llm.onnx").exists()
