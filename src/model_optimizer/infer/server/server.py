"""InferServer — 纯推理库 Facade。

推理与后处理完全解耦：

- ``infer_chunk(idx)`` 只做 ``backend.predict()``，返回轻量 ``ChunkPayload``。
- 后处理（metrics / 图像编码 / StepResult）在可选的 ``ResultWorker`` 线程中异步完成。
- ``enable_result=False`` 时不启动后处理线程，推理路径零额外开销。

用法::

    from model_optimizer.infer.server import InferServer, load_config

    config = load_config("my_config.json")
    server = InferServer(config)
    server.load()

    # 方式 1：纯推理，拿 ChunkPayload
    payload = server.infer_chunk(0)

    # 方式 2：遍历推理 + 异步后处理回调
    result = server.run_all(on_step=lambda s: print(s.metrics["mse"]))

    # 方式 3：只跑推理不要结果（enable_result=False）
    server.run_all()  # 返回 None

    server.close()
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .backends.base import InferBackend
from .config import ServerConfig
from .result import ChunkPayload, InferResult, StepResult

logger = logging.getLogger(__name__)

StepCallback = Callable[[StepResult], None]
ProgressCallback = Callable[[str, str], None]


class InferServer:
    """Pi0.5 推理服务核心 Facade。

    推理路径 (``infer_chunk``) 只做 ``backend.predict()`` + 轻量数据拷贝，
    不包含 metrics 计算和图像 JPEG 编码。后处理由 ``ResultWorker`` 在独立线程异步完成。
    当 ``config.enable_result=False`` 时后处理完全不启动，零开销。
    """

    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._loaded = False

        # 加载后填充
        self._backend: InferBackend | None = None
        self._dataset: Any = None
        self._repack_fn: Any = None
        self._ep_per_frame: np.ndarray | None = None
        self._action_horizon: int = 0
        self._start_index: int = 0
        self._end: int = 0
        self._n: int = 0
        self._policies: dict[str, Any] = {}
        self._calib_collectors: list[Any] | None = None
        self._meta: dict[str, Any] = {}
        self._data_config: Any = None

    @property
    def config(self) -> ServerConfig:
        return self._config

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def start_index(self) -> int:
        return self._start_index

    @property
    def end(self) -> int:
        return self._end

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    def load(self, on_progress: ProgressCallback | None = None) -> None:
        """加载模型、数据集，构建推理后端。"""
        if self._loaded:
            raise RuntimeError("InferServer is already loaded. Call close() first.")

        if on_progress is None:
            on_progress = lambda stage, msg: None

        from .policy_loader import load_policies

        # 1. 加载策略
        result = load_policies(self._config, on_progress=on_progress)
        self._policies = result
        policy = result["policy"]
        policy_trt = result["policy_trt"]
        policy_ptq = result["policy_ptq"]
        train_cfg = result["train_cfg"]

        # 2. 加载数据集
        on_progress("dataset", "加载 LeRobot 数据集 …")
        data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
        if not data_config.repo_id:
            raise ValueError("当前配置未设置 repo_id，无法加载 LeRobot 数据。")
        self._data_config = data_config

        self._action_horizon = train_cfg.model.action_horizon

        from ._dataset_helpers import (
            build_repack_only,
            global_episode_id_per_frame,
            make_lerobot_dataset,
            unwrap_lerobot_base,
        )

        action_keys = tuple(data_config.action_sequence_keys)
        ds_root = Path(self._config.dataset.root) if self._config.dataset.root else None
        self._dataset = make_lerobot_dataset(
            repo_id=data_config.repo_id,
            action_horizon=self._action_horizon,
            action_sequence_keys=action_keys,
            prompt_from_task=data_config.prompt_from_task,
            dataset_root=ds_root,
        )
        self._repack_fn = build_repack_only(data_config)
        on_progress("dataset", f"数据集就绪（共 {len(self._dataset)} 条）")

        self._n = len(self._dataset)
        self._start_index = self._config.dataset.start_index
        self._end = min(self._start_index + self._config.dataset.num_samples, self._n)
        if self._start_index >= self._n:
            raise ValueError(f"start_index={self._start_index} >= dataset len={self._n}")

        base_ds = unwrap_lerobot_base(self._dataset)
        self._ep_per_frame = global_episode_id_per_frame(base_ds, self._n)

        # 3. 可选：校准收集器
        if self._config.calib.save_path:
            self._start_calib_collectors(policy, on_progress)

        # 4. 构建推理后端
        self._backend = self._create_backend(policy, policy_trt, policy_ptq)

        # 5. 构建 meta
        self._meta = self._build_meta(data_config)

        self._loaded = True
        on_progress("ready", "加载完成")

    def infer_chunk(self, idx: int) -> ChunkPayload | None:
        """纯推理单个 chunk，返回轻量 ChunkPayload。

        只做 ``backend.predict()`` + 提取 prompt 和原始图像 numpy，
        **不计算 metrics、不编码 JPEG**。

        如果 idx 不在有效对齐位置或跨越 episode，返回 ``None``。
        """
        if not self._loaded:
            raise RuntimeError("Call load() before infer_chunk()")

        ah = self._action_horizon
        stride_ok = (idx - self._start_index) % ah == 0
        chunk_fits = idx + ah <= self._n and idx + ah <= self._end
        if not (stride_ok and chunk_fits):
            return None

        ep0 = int(self._ep_per_frame[idx])
        ep_last = int(self._ep_per_frame[idx + ah - 1])
        if ep0 != ep_last:
            return None

        from ._dataset_helpers import tree_to_numpy

        raw = tree_to_numpy(self._dataset[idx])
        packed = self._repack_fn(dict(raw))
        if "actions" not in packed:
            raise KeyError("repack 后缺少 actions，请检查数据配置。")

        gt = np.asarray(packed["actions"])
        obs = {k: v for k, v in packed.items() if k != "actions"}

        pack = self._backend.predict(obs, gt, ah)

        if pack.pred_h.shape[0] < ah or pack.gt_h.shape[0] < ah:
            logger.warning("index %s: pred/gt dim < action_horizon, skip", idx)
            return None

        # 提取 prompt（轻量字符串，不影响推理速度）
        prompt = None
        if "prompt" in packed:
            try:
                prompt = str(packed["prompt"])
            except Exception:
                pass

        # 提取原始图像 numpy（仅引用/轻量拷贝，不做 JPEG 编码）
        raw_images: dict[str, np.ndarray] | None = None
        if "observation/image" in packed:
            raw_images = {"base_rgb": np.asarray(packed["observation/image"])}
            if (
                self._config.websocket.send_wrist
                and "observation/wrist_image" in packed
            ):
                raw_images["wrist_rgb"] = np.asarray(packed["observation/wrist_image"])

        return ChunkPayload(
            idx=idx,
            episode_id=ep0,
            action_horizon=ah,
            pack=pack,
            prompt=prompt,
            raw_images=raw_images,
        )

    def run_all(
        self,
        on_step: StepCallback | None = None,
    ) -> InferResult | None:
        """遍历 [start_index, end) 内所有有效 chunk 进行推理。

        - 推理在当前线程顺序执行（``infer_chunk``）。
        - 当 ``config.enable_result=True`` 时，每个 ``ChunkPayload`` 异步投递到
          ``ResultWorker`` 线程做后处理（metrics + 图像编码 + StepResult 构建），
          ``on_step`` 回调在后处理线程中触发。
        - 当 ``config.enable_result=False`` 时，不启动后处理线程，不计算 metrics，
          不编码图像，返回 ``None``。适用于只关心推理吞吐或校准数据收集。

        Args:
            on_step: 可选回调，每个 StepResult 构建后在后处理线程中调用。
                仅 ``enable_result=True`` 时生效。

        Returns:
            ``InferResult``（enable_result=True）或 ``None``（enable_result=False）。
        """
        if not self._loaded:
            raise RuntimeError("Call load() before run_all()")

        enable_result = self._config.enable_result

        worker = None
        if enable_result:
            from ._result_worker import ResultWorker

            worker = ResultWorker(self._config, on_step=on_step)
            worker.start()

        try:
            for idx in range(self._start_index, self._end):
                payload = self.infer_chunk(idx)
                if payload is not None and worker is not None:
                    worker.submit(payload)
        finally:
            if worker is not None:
                worker.stop()

        if worker is not None:
            return InferResult(
                steps=worker.results,
                meta=dict(self._meta),
                start_index=self._start_index,
                end_index_exclusive=self._end,
            )
        return None

    def close(self) -> None:
        """释放资源（校准收集器等）。"""
        if self._calib_collectors:
            for c in self._calib_collectors:
                try:
                    c.stop_collect()
                except Exception:
                    logger.exception("calib stop_collect failed")
            self._calib_collectors = None
        self._backend = None
        self._dataset = None
        self._policies = {}
        self._loaded = False

    # ── private ──

    def _create_backend(
        self, policy: Any, policy_trt: Any | None, policy_ptq: Any | None
    ) -> InferBackend:
        mode = self._config.mode
        if mode == "pytorch":
            from .backends.pytorch import SinglePyTorchBackend

            return SinglePyTorchBackend(policy)
        elif mode == "tensorrt":
            from .backends.tensorrt import SingleTensorRTBackend

            return SingleTensorRTBackend(policy)
        elif mode == "pt_trt_compare":
            from .backends.pt_trt_compare import PtTrtCompareBackend

            if policy_trt is None:
                raise RuntimeError("pt_trt_compare requires policy_trt")
            return PtTrtCompareBackend(policy, policy_trt)
        elif mode == "pt_ptq_compare":
            from .backends.pt_ptq_compare import PtPtqCompareBackend

            if policy_ptq is None:
                raise RuntimeError("pt_ptq_compare requires policy_ptq")
            return PtPtqCompareBackend(policy, policy_ptq)
        elif mode == "ptq_trt_compare":
            from .backends.ptq_trt_compare import PtqTrtCompareBackend

            if policy_trt is None:
                raise RuntimeError("ptq_trt_compare requires policy_trt")
            return PtqTrtCompareBackend(policy, policy_trt)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    def _start_calib_collectors(self, policy: Any, on_progress: ProgressCallback) -> None:
        try:
            from model_optimizer.calibrate.collector.pi05 import (
                Pi05DenoiseCalibCollector,
                Pi05EmbedPrefixCalibCollector,
                Pi05ExpertCalibCollector,
                Pi05LLMCalibCollector,
                Pi05VitCalibCollector,
            )

            from .policy_loader import _unwrap_pi05_model

            torch_model = _unwrap_pi05_model(policy)
            if torch_model is None:
                logger.warning("Cannot unwrap model for calib; skipping")
                return

            save_dir = str(Path(self._config.calib.save_path).expanduser().resolve())
            max_s = self._config.calib.max_samples
            item = self._config.calib.item

            collectors: list[Any] = []
            if item in ("all", "llm"):
                collectors.append(Pi05LLMCalibCollector(torch_model, save_dir, max_samples=max_s))
            if item in ("all", "expert"):
                collectors.append(Pi05ExpertCalibCollector(torch_model, save_dir, max_samples=max_s))
            if item in ("all", "vit"):
                collectors.append(Pi05VitCalibCollector(torch_model, save_dir, max_samples=max_s))
            if item in ("all", "denoise"):
                collectors.append(Pi05DenoiseCalibCollector(torch_model, save_dir, max_samples=max_s))
            if item in ("all", "embed_prefix"):
                collectors.append(Pi05EmbedPrefixCalibCollector(torch_model, save_dir, max_samples=max_s))

            self._calib_collectors = collectors
            on_progress("calib", f"calib 收集器已启动 → {save_dir}")
        except Exception as exc:
            logger.warning("Failed to start calib collectors: %s", exc, exc_info=True)

    def _build_meta(self, data_config: Any) -> dict[str, Any]:
        cfg = self._config
        mode = cfg.mode

        if mode == "pt_trt_compare":
            backend = "pytorch+tensorrt"
        elif mode == "pt_ptq_compare":
            backend = "pytorch+ptq"
        elif mode == "ptq_trt_compare":
            backend = "pytorch_ptq+tensorrt"
        else:
            backend = mode

        meta: dict[str, Any] = {
            "repo_id": data_config.repo_id,
            "backend": backend,
            "mode": mode,
            "action_horizon": self._action_horizon,
            "start_index": self._start_index,
            "end_index_exclusive": self._end,
            "precision": cfg.precision,
            "rel_eps": cfg.rel_eps,
        }

        if mode == "pt_trt_compare":
            meta["pred1_name"] = "PT"
            meta["pred2_name"] = "TRT"
            meta["pair_name"] = "PT-TRT"
        elif mode == "pt_ptq_compare":
            meta["pred1_name"] = "PT"
            meta["pred2_name"] = "PTQ"
            meta["pair_name"] = "PT-PTQ"
        elif mode == "ptq_trt_compare":
            meta["pred1_name"] = "PTQ"
            meta["pred2_name"] = "TRT"
            meta["pair_name"] = "PTQ-TRT"

        if mode in ("tensorrt", "pt_trt_compare", "ptq_trt_compare"):
            meta["tensorrt"] = {
                "engine_path": cfg.tensorrt.engine_path,
                "vit_engine": cfg.tensorrt.vit_engine,
                "llm_engine": cfg.tensorrt.llm_engine,
                "expert_engine": cfg.tensorrt.expert_engine,
                "denoise_engine": cfg.tensorrt.denoise_engine,
                "embed_prefix_engine": cfg.tensorrt.embed_prefix_engine,
            }

        if mode in ("pt_ptq_compare", "ptq_trt_compare"):
            meta["ptq"] = {
                "quant_cfg": cfg.ptq.quant_cfg,
                "calib_dir": cfg.ptq.calib_dir,
                "parts": list(cfg.ptq.parts),
            }

        return meta
