"""多路 infer backend 应对 obs 做深拷贝，避免第一路 transform 污染第二路输入。"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.deployment.pi05.lerobot_eval_webui.infer_backends import (
    PtPtqCompareBackend,
    PtTrtCompareBackend,
    TrtOrtCompareBackend,
)


def _sample_obs() -> dict[str, object]:
    return {
        "state": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "image": {"cam": np.zeros((2, 2), dtype=np.uint8)},
    }


class _MutatingPolicy:
    """模拟 openpi input transform：就地改写 obs 内嵌数组。"""

    def __init__(self, action_value: float) -> None:
        self._action_value = action_value

    def infer(self, obs: dict[str, object], *, noise: np.ndarray | None = None) -> dict[str, np.ndarray]:
        del noise
        obs["state"][0] = 999.0
        img = obs["image"]
        assert isinstance(img, dict)
        cam = img["cam"]
        assert isinstance(cam, np.ndarray)
        cam[0, 0] = 255
        return {"actions": np.full((10, 7), self._action_value, dtype=np.float32)}


class _ObsProbePolicy:
    def __init__(self) -> None:
        self.seen_state0: float | None = None
        self.seen_cam00: int | None = None

    def infer(self, obs: dict[str, object], *, noise: np.ndarray | None = None) -> dict[str, np.ndarray]:
        del noise
        state = obs["state"]
        assert isinstance(state, np.ndarray)
        self.seen_state0 = float(state[0])
        img = obs["image"]
        assert isinstance(img, dict)
        cam = img["cam"]
        assert isinstance(cam, np.ndarray)
        self.seen_cam00 = int(cam[0, 0])
        return {"actions": np.ones((10, 7), dtype=np.float32)}


@pytest.mark.parametrize(
    "backend_cls",
    [PtTrtCompareBackend, PtPtqCompareBackend, TrtOrtCompareBackend],
)
def test_compare_backends_copy_obs_before_each_infer(backend_cls) -> None:
    obs = _sample_obs()
    probe = _ObsProbePolicy()
    backend = backend_cls()
    pack = backend.predict(
        _MutatingPolicy(0.0),
        probe,
        probe if backend_cls is PtPtqCompareBackend else None,
        obs,
        np.zeros(7, dtype=np.float32),
        action_horizon=7,
    )
    assert pack.pred_h.shape == (7, 7)
    assert probe.seen_state0 == pytest.approx(1.0)
    assert probe.seen_cam00 == 0
    assert float(obs["state"][0]) == pytest.approx(1.0)
    img = obs["image"]
    assert isinstance(img, dict)
    cam = img["cam"]
    assert isinstance(cam, np.ndarray)
    assert int(cam[0, 0]) == 0
