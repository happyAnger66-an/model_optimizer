"""模型注册表与构造 E2E 测试。"""

import importlib.util

import pytest


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# 注册表初始化会导入 Pi05Model (需 jax)、YoloModel (需 ultralytics)
@pytest.mark.e2e
@pytest.mark.skipif(not _has_module("jax"), reason="registry 需 jax (pi05)")
class TestModelRegistry:
    """测试模型注册表与 construct_from_name_path。"""

    def test_get_model_cls_yolo(self):
        """可获取 yolo 模型类。"""
        from model_optimizer.models.registry import get_model_cls

        cls = get_model_cls("yolo")
        assert cls is not None

    def test_get_model_cls_pi05_llm(self):
        """可获取 pi05_libero/llm 模型类。"""
        from model_optimizer.models.registry import get_model_cls

        cls = get_model_cls("pi05_libero/llm")
        assert cls is not None

    def test_invalid_model_raises(self):
        """无效 model_name 抛出 ValueError。"""
        from model_optimizer.models.registry import get_model_cls

        with pytest.raises(ValueError, match="not found"):
            get_model_cls("invalid_model_xyz")
