#!/usr/bin/env python3
"""
LeRobot 离线评估 WebUI（client-server）：

- Server 通过 WebSocket **流式**推送“时间对齐后的 step 事件”：一次推理产生 action_horizon 个 steps，
  每个 step 对齐到数据集 label 的同一行（idx+k）。
- 事件包含：episode_id/global_index/k_in_chunk、gt/pred action、误差指标、prompt、RGB 图像（默认 base，可选 wrist）。
- Client 侧用浏览器订阅展示（见 ./webui_client/）。

协议（v1）：

- `type="meta"`：连接建立后 server 先发 1 条元数据
- `type="step"`：按时间顺序持续推送 step
- `type="log"`：可选日志事件
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import logging
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Deque, Literal

import numpy as np
import tyro

import openpi.transforms as _transforms
from openpi.policies import policy_config
from openpi.training import config as _config
from openpi.training.data_loader import TransformedDataset

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
except ImportError:  # pragma: no cover
    try:
        import lerobot.datasets.lerobot_dataset as lerobot_dataset  # type: ignore[no-redef]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "需要安装 lerobot（openpi 使用 lerobot.common.datasets.lerobot_dataset；"
            "新版包可能为 lerobot.datasets.lerobot_dataset）。"
        ) from e


def _tree_to_numpy(obj):
    """LeRobot 常返回 torch.Tensor；openpi transforms 期望 numpy。"""
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {k: _tree_to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_tree_to_numpy(x) for x in obj)
    return obj


def _build_repack_only(data_config: _config.DataConfig) -> _transforms.DataTransformFn:
    return _transforms.compose([*data_config.repack_transforms.inputs])


def _unwrap_lerobot_base(dataset) -> object:
    d = dataset
    while hasattr(d, "_dataset"):
        d = d._dataset
    return d


def _global_episode_id_per_frame(base_ds, n_frames: int) -> np.ndarray:
    try:
        hf = getattr(base_ds, "hf_dataset", None)
    except Exception as exc:  # pragma: no cover
        logging.warning("读取 hf_dataset 失败: %s；episode 视为全 0。", exc)
        return np.zeros(n_frames, dtype=np.int64)
    if hf is not None:
        cols = getattr(hf, "column_names", []) or []
        for col in ("episode_index", "episode_idx", "ep_index"):
            if col in cols:
                ep = np.asarray(hf[col][:n_frames])
                if ep.shape[0] != n_frames:
                    raise RuntimeError(
                        f"episode 列 {col!r} 长度 {ep.shape[0]} 与 len(dataset)={n_frames} 不一致"
                    )
                return ep.astype(np.int64, copy=False)
    logging.warning(
        "未在 hf_dataset 中找到 episode_index；将把 [start_index, end) 整段当作 episode_id=0。"
    )
    return np.zeros(n_frames, dtype=np.int64)


def _align_action_dim(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred[np.newaxis, :]
    if gt.ndim == 1:
        gt = gt[np.newaxis, :]
    d = min(pred.shape[-1], gt.shape[-1])
    return pred[..., :d], gt[..., :d]


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if np.issubdtype(img.dtype, np.floating):
        img = (255.0 * img).clip(0.0, 255.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8, copy=False)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    return img


def _encode_jpeg_b64(rgb_hwc_uint8: np.ndarray, *, quality: int = 85) -> str:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("缺少 opencv-python-headless，无法编码 JPEG（requirements-base.txt 已包含）。")
    img = np.asarray(rgb_hwc_uint8)
    if img.ndim != 3 or img.shape[-1] != 3 or img.dtype != np.uint8:
        raise ValueError(f"expect uint8(H,W,3), got {img.dtype} {img.shape}")
    bgr = img[..., ::-1]
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _make_lerobot_dataset(
    *,
    repo_id: str,
    action_horizon: int,
    action_sequence_keys: tuple[str, ...],
    prompt_from_task: bool,
    dataset_root: Path | None,
):
    meta_kw: dict = {"repo_id": repo_id}
    if dataset_root is not None:
        meta_kw["root"] = str(dataset_root)
    try:
        meta = lerobot_dataset.LeRobotDatasetMetadata(**meta_kw)
    except TypeError:
        meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    ds_kwargs: dict = {
        "repo_id": repo_id,
        "delta_timestamps": {
            key: [t / meta.fps for t in range(action_horizon)] for key in action_sequence_keys
        },
    }
    if dataset_root is not None:
        ds_kwargs["root"] = str(dataset_root)
    try:
        dataset = lerobot_dataset.LeRobotDataset(**ds_kwargs)
    except TypeError:
        ds_kwargs.pop("root", None)
        logging.warning("当前 lerobot 版本忽略 dataset_root，请用 HF 缓存或升级 lerobot。")
        dataset = lerobot_dataset.LeRobotDataset(**ds_kwargs)
    if prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(meta.tasks)])
    return dataset


def _load_tensorrt_engines(
    policy,
    *,
    engine_path: str,
    precision: Literal["fp16", "bf16", "fp32"],
    vit_engine: str,
    llm_engine: str,
    expert_engine: str,
    denoise_engine: str,
    embed_prefix_engine: str,
) -> None:
    import addict
    import torch

    from model_optimizer.infer.tensorrt.pi05_executor import Pi05TensorRTExecutor

    if precision == "fp16":
        prec = torch.float16
    elif precision == "bf16":
        prec = torch.bfloat16
    else:
        prec = torch.float32

    executor = Pi05TensorRTExecutor(policy, prec)
    cfg: dict = {"engine_path": engine_path}
    if vit_engine:
        cfg["vit_engine"] = vit_engine
    if expert_engine:
        cfg["expert_engine"] = expert_engine
    if llm_engine:
        cfg["llm_engine"] = llm_engine
    if denoise_engine:
        cfg["denoise_engine"] = denoise_engine
    if embed_prefix_engine:
        cfg["embed_prefix_engine"] = embed_prefix_engine
    executor.load_model(addict.Dict(cfg))


@dataclasses.dataclass(frozen=True)
class StepEvent:
    type: Literal["step"]
    run_id: str
    episode_id: int
    global_index: int
    k_in_chunk: int
    is_chunk_start: bool
    action_horizon: int
    prompt: str | None
    gt_action: list[float]
    pred_action: list[float]
    metrics: dict[str, Any]
    images: dict[str, str] | None
    server_timing: dict[str, float] | None


def _event_to_json(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False, separators=(",", ":"))


class WebsocketBroadcaster:
    def __init__(self, *, history_size: int) -> None:
        self._clients: set[Any] = set()
        self._history: Deque[str] = deque(maxlen=max(history_size, 0))

    def add_history(self, msg: str) -> None:
        if self._history.maxlen and self._history.maxlen > 0:
            self._history.append(msg)

    async def register(self, ws) -> None:
        self._clients.add(ws)

    async def unregister(self, ws) -> None:
        self._clients.discard(ws)

    async def send_history(self, ws) -> None:
        for msg in list(self._history):
            await ws.send(msg)

    async def broadcast(self, msg: str) -> None:
        if not self._clients:
            return
        dead: list[Any] = []
        for ws in self._clients:
            try:
                await ws.send(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.unregister(ws)


@dataclasses.dataclass
class Args:
    checkpoint: Path
    config: str = "pi05_libero"
    num_samples: int = 500
    start_index: int = 0
    dataset_root: Path | None = None
    device: str | None = None

    inference_mode: Literal["pytorch", "tensorrt"] = "pytorch"
    precision: Literal["fp16", "bf16", "fp32"] = "bf16"
    engine_path: str = ""
    vit_engine: str = ""
    llm_engine: str = ""
    expert_engine: str = ""
    denoise_engine: str = ""
    embed_prefix_engine: str = ""

    host: str = "0.0.0.0"
    port: int = 8765
    path: str = "/ws"

    send_wrist: bool = False
    jpeg_quality: int = 85

    max_fps: float = 0.0
    """限制推送帧率（step event/s）。0 表示不限制。"""

    history_size: int = 0
    """缓存最近 N 条消息，新 client 连接后先回放（0 表示不缓存）。"""


async def _run_server(args: Args) -> None:
    import websockets.asyncio.server as _server

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_id = uuid.uuid4().hex[:12]

    train_cfg = _config.get_config(args.config)
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if not data_config.repo_id:
        raise ValueError("当前配置未设置 repo_id，无法加载 LeRobot 数据。")
    action_horizon = train_cfg.model.action_horizon
    action_keys = tuple(data_config.action_sequence_keys)

    dataset = _make_lerobot_dataset(
        repo_id=data_config.repo_id,
        action_horizon=action_horizon,
        action_sequence_keys=action_keys,
        prompt_from_task=data_config.prompt_from_task,
        dataset_root=args.dataset_root,
    )
    repack_fn = _build_repack_only(data_config)

    policy = policy_config.create_trained_policy(
        train_cfg,
        args.checkpoint,
        pytorch_device=args.device,
    )
    if args.inference_mode == "tensorrt":
        if not args.engine_path:
            raise ValueError("inference_mode=tensorrt 时必须设置 --engine-path（引擎目录）。")
        _load_tensorrt_engines(
            policy,
            engine_path=args.engine_path,
            precision=args.precision,
            vit_engine=args.vit_engine,
            llm_engine=args.llm_engine,
            expert_engine=args.expert_engine,
            denoise_engine=args.denoise_engine,
            embed_prefix_engine=args.embed_prefix_engine,
        )

    n = len(dataset)
    end = min(args.start_index + args.num_samples, n)
    if args.start_index >= n:
        raise ValueError(f"start_index={args.start_index} >= dataset len={n}")

    base_ds = _unwrap_lerobot_base(dataset)
    ep_per_frame = _global_episode_id_per_frame(base_ds, n)

    broadcaster = WebsocketBroadcaster(history_size=args.history_size)

    meta = {
        "type": "meta",
        "run_id": run_id,
        "repo_id": data_config.repo_id,
        "backend": args.inference_mode,
        "action_horizon": int(action_horizon),
        "start_index": int(args.start_index),
        "end_index_exclusive": int(end),
        "send_wrist": bool(args.send_wrist),
        "jpeg_quality": int(args.jpeg_quality),
    }
    meta_msg = _event_to_json(meta)
    broadcaster.add_history(meta_msg)

    async def handler(ws):  # websockets passes ServerConnection
        if getattr(ws, "path", None) not in (None, args.path):
            await ws.close(code=1008, reason="invalid path")
            return
        await broadcaster.register(ws)
        try:
            await ws.send(meta_msg)
            if args.history_size > 0:
                await broadcaster.send_history(ws)
            async for _ in ws:
                # 当前只做单向推送；忽略 client 消息（后续可扩展控制命令）
                pass
        finally:
            await broadcaster.unregister(ws)

    async with _server.serve(
        handler,
        args.host,
        args.port,
        compression=None,
        max_size=None,
    ) as server:
        logging.info("WebUI WS server: ws://%s:%s%s", args.host, args.port, args.path)
        logging.info("run_id=%s | repo_id=%s | backend=%s", run_id, data_config.repo_id, args.inference_mode)

        min_step_period = (1.0 / args.max_fps) if args.max_fps and args.max_fps > 0 else 0.0
        last_send_t = 0.0

        for idx in range(args.start_index, end):
            stride_ok = (idx - args.start_index) % action_horizon == 0
            chunk_fits = idx + action_horizon <= n and idx + action_horizon <= end
            if not (stride_ok and chunk_fits):
                continue

            ep0 = int(ep_per_frame[idx])
            ep_last = int(ep_per_frame[idx + action_horizon - 1])
            if ep0 != ep_last:
                continue

            raw = _tree_to_numpy(dataset[idx])
            packed = repack_fn(dict(raw))
            if "actions" not in packed:
                raise KeyError("repack 后缺少 actions，请检查数据配置与数据集列名是否一致。")

            gt = np.asarray(packed["actions"])
            obs = {k: v for k, v in packed.items() if k != "actions"}

            t0 = time.monotonic()
            out = policy.infer(obs)
            infer_ms = (time.monotonic() - t0) * 1000.0
            pred = np.asarray(out["actions"])

            pred_a, gt_a = _align_action_dim(pred, gt)
            pred_h = pred_a[:action_horizon]
            gt_h = gt_a[:action_horizon]
            if pred_h.shape[0] < action_horizon or gt_h.shape[0] < action_horizon:
                logging.warning(
                    "index %s: pred/gt 时间维 %s/%s 小于 action_horizon=%s，跳过。",
                    idx,
                    pred_h.shape[0],
                    gt_h.shape[0],
                    action_horizon,
                )
                continue

            prompt: str | None = None
            if "prompt" in packed:
                try:
                    prompt = str(packed["prompt"])
                except Exception:
                    prompt = None

            images: dict[str, str] | None = None
            if "observation/image" in packed:
                try:
                    base_rgb = _to_hwc_uint8(packed["observation/image"])
                    images = {
                        "base_rgb_jpeg_b64": _encode_jpeg_b64(base_rgb, quality=args.jpeg_quality),
                    }
                    if args.send_wrist and "observation/wrist_image" in packed:
                        wrist_rgb = _to_hwc_uint8(packed["observation/wrist_image"])
                        images["wrist_rgb_jpeg_b64"] = _encode_jpeg_b64(
                            wrist_rgb, quality=args.jpeg_quality
                        )
                except Exception as exc:
                    logging.warning("index %s: 图像编码失败（继续只发数值）: %s", idx, exc)
                    images = None

            for k in range(action_horizon):
                g = idx + k
                diff = pred_h[k] - gt_h[k]
                mse = float(np.mean(diff**2))
                mae = float(np.mean(np.abs(diff)))

                step_images = images if k == 0 else None
                step_event = StepEvent(
                    type="step",
                    run_id=run_id,
                    episode_id=ep0,
                    global_index=int(g),
                    k_in_chunk=int(k),
                    is_chunk_start=bool(k == 0),
                    action_horizon=int(action_horizon),
                    prompt=prompt if k == 0 else None,
                    gt_action=[float(x) for x in gt_h[k].astype(np.float64).tolist()],
                    pred_action=[float(x) for x in pred_h[k].astype(np.float64).tolist()],
                    metrics={"mse": mse, "mae": mae},
                    images=step_images,
                    server_timing={"infer_ms": float(infer_ms)} if k == 0 else None,
                )
                msg = _event_to_json(dataclasses.asdict(step_event))
                broadcaster.add_history(msg)
                await broadcaster.broadcast(msg)

                if min_step_period > 0:
                    now = time.monotonic()
                    dt = now - last_send_t
                    if dt < min_step_period:
                        await asyncio.sleep(min_step_period - dt)
                    last_send_t = time.monotonic()

        await server.wait_closed()


def main() -> None:
    args = tyro.cli(Args)
    asyncio.run(_run_server(args))


if __name__ == "__main__":
    main()

