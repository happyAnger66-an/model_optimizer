"""数据对比模块测试，使用 mock 数据。"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# 确保 src 在 path 中
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# 跳过条件：缺少依赖
pytest.importorskip("termcolor", reason="compare 模块需 termcolor")
pytest.importorskip("matplotlib", reason="plot 测试需 matplotlib")


def _make_mock_pair():
    """生成一对 mock 数据，格式与 compare_predictions 期望一致。"""
    np.random.seed(42)
    data1 = {
        "actions": np.random.randn(1, 10, 7).astype(np.float32) * 0.5,
    }
    data2 = {
        "actions": data1["actions"] + np.random.randn(1, 10, 7).astype(np.float32) * 0.1,
    }
    return data1, data2


def _make_mock_pair_multi_key():
    """多 key 的 mock 数据。"""
    np.random.seed(42)
    data1 = {
        "actions": np.random.randn(1, 10, 7).astype(np.float32) * 0.5,
        "logits": np.random.randn(1, 16).astype(np.float32) * 0.3,
    }
    data2 = {
        "actions": data1["actions"] + np.random.randn(1, 10, 7).astype(np.float32) * 0.05,
        "logits": data1["logits"] + np.random.randn(1, 16).astype(np.float32) * 0.02,
    }
    return data1, data2


def _save_mock_npz(data_list: list[dict], path: str) -> None:
    """按 load_saved_data 期望的格式保存为 npz。"""
    save_dict = {}
    for i, d in enumerate(data_list):
        for k, v in d.items():
            save_dict[f"item_{i}_{k}"] = v
    save_dict["metadata"] = np.array(
        {"num_items": len(data_list), "item_keys": list(data_list[0].keys()) if data_list else []},
        dtype=object,
    )
    np.savez(path, **save_dict)


class TestComparePredictions:
    """测试 compare_predictions。"""

    def test_compare_predictions_returns_none_when_return_metrics_false(self):
        from model_optimizer.evaluate.compare.utils import compare_predictions

        data1, data2 = _make_mock_pair()
        result = compare_predictions(data1, data2, return_metrics=False)
        assert result is None

    def test_compare_predictions_returns_metrics_when_return_metrics_true(self):
        from model_optimizer.evaluate.compare.utils import compare_predictions

        data1, data2 = _make_mock_pair()
        result = compare_predictions(data1, data2, return_metrics=True)

        assert result is not None
        assert "actions" in result
        m = result["actions"]
        assert "cosine_sim" in m
        assert "l1_mean" in m
        assert "mean_1" in m
        assert "mean_2" in m
        assert 0 <= m["cosine_sim"] <= 1 or m["cosine_sim"] <= 1

    def test_compare_predictions_filters_keys(self):
        from model_optimizer.evaluate.compare.utils import compare_predictions

        data1, data2 = _make_mock_pair_multi_key()
        data1["prompt"] = np.zeros(3)
        data2["prompt"] = np.zeros(3)

        result = compare_predictions(
            data1, data2, filter_keys=["prompt"], return_metrics=True
        )
        assert "prompt" not in result
        assert "actions" in result
        assert "logits" in result


class TestPlotCompareResults:
    """测试 plot_compare_results。"""

    def test_plot_compare_results_saves_file(self):
        from model_optimizer.evaluate.compare.utils import (
            compare_predictions,
            plot_compare_results,
        )

        data1, data2 = _make_mock_pair()
        metrics = compare_predictions(data1, data2, return_metrics=True)
        collected = [metrics]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out_path = f.name

        try:
            plot_compare_results(collected, out_path)
            assert Path(out_path).exists()
            assert Path(out_path).stat().st_size > 0
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_plot_compare_results_multi_key(self):
        from model_optimizer.evaluate.compare.utils import (
            compare_predictions,
            plot_compare_results,
        )

        data1, data2 = _make_mock_pair_multi_key()
        metrics = compare_predictions(data1, data2, return_metrics=True)
        collected = [metrics]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out_path = f.name

        try:
            plot_compare_results(collected, out_path, key1="A", key2="B")
            assert Path(out_path).exists()
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_plot_compare_results_empty_raises(self):
        from model_optimizer.evaluate.compare.utils import plot_compare_results

        with pytest.raises(ValueError, match="无有效对比数据"):
            plot_compare_results([], "/tmp/out.png")


class TestCompareCli:
    """测试 compare_cli 完整流程（mock npz）。"""

    def test_compare_cli_with_mock_npz(self):
        from model_optimizer.compare.cli import compare_cli

        data1_list = []
        data2_list = []
        for _ in range(3):
            d1, d2 = _make_mock_pair()
            data1_list.append(d1)
            data2_list.append(d2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = str(Path(tmpdir) / "data1.npz")
            path2 = str(Path(tmpdir) / "data2.npz")
            _save_mock_npz(data1_list, path1)
            _save_mock_npz(data2_list, path2)

            # 不传 plot_output，只做对比
            args = [
                "compare",
                "--data_path1", path1,
                "--data_path2", path2,
            ]
            compare_cli(args)

    def test_compare_cli_with_plot_output(self):
        from model_optimizer.compare.cli import compare_cli

        data1_list = []
        data2_list = []
        for _ in range(3):
            d1, d2 = _make_mock_pair()
            data1_list.append(d1)
            data2_list.append(d2)

        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = str(Path(tmpdir) / "data1.npz")
            path2 = str(Path(tmpdir) / "data2.npz")
            plot_path = str(Path(tmpdir) / "compare_plot.png")
            _save_mock_npz(data1_list, path1)
            _save_mock_npz(data2_list, path2)

            args = [
                "compare",
                "--data_path1", path1,
                "--data_path2", path2,
                "--plot_output", plot_path,
            ]
            compare_cli(args)

            assert Path(plot_path).exists()
            assert Path(plot_path).stat().st_size > 0
