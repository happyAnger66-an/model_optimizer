"""评估会话（Facade）：bundle 加载、chunk 循环、经 Janus 出站（与 WebSocket 解耦）。"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from termcolor import colored

from .bundle import load_infer_bundle
from .calib import stop_pi05_calib_collectors
from .chunk_infer import process_infer_chunk
from .config import Args
from .ports import SyncOutboundPort
from .protocol import event_to_json


def run_infer_worker(
    *,
    args: Args,
    run_id: str,
    meta_ready: dict[str, Any],
    infer_paused: threading.Event,
    bridge: SyncOutboundPort,
) -> None:
    """在专用线程中运行：加载、逐 chunk 推理、投递 JSON；结束时 ``bridge.sync_close()``。"""
    bundle: dict[str, Any] | None = None
    try:
        print(colored("[infer] 线程启动，开始加载 bundle…", "cyan"), flush=True)

        def on_progress(stage: str, message: str) -> None:
            bridge.sync_emit(
                event_to_json(
                    {
                        "type": "server_progress",
                        "run_id": run_id,
                        "stage": stage,
                        "message": message,
                    }
                ),
                add_history=False,
            )

        bundle = load_infer_bundle(args, run_id, on_progress=on_progress)
        meta_msg = bundle["meta_msg"]
        meta_ready["msg"] = meta_msg
        print(colored("[infer] 加载完成，向主循环投递 meta …", "cyan"), flush=True)
        bridge.sync_emit(meta_msg)

        start_index = bundle["start_index"]
        end = bundle["end"]
        min_step_period = (1.0 / args.max_fps) if args.max_fps and args.max_fps > 0 else 0.0
        last_send_t = 0.0

        for idx in range(start_index, end):
            while infer_paused.is_set():
                time.sleep(0.05)
            msgs = process_infer_chunk(bundle, idx)
            for msg in msgs:
                bridge.sync_emit(msg)
                if min_step_period > 0:
                    now = time.monotonic()
                    dt = now - last_send_t
                    if dt < min_step_period:
                        time.sleep(min_step_period - dt)
                    last_send_t = time.monotonic()

        done_msg = event_to_json(
            {
                "type": "done",
                "phase": "finished",
                "run_id": run_id,
                "message": (
                    "推理序列已全部推送完毕；本进程即将关闭 WebSocket 并退出。"
                    "若需再次评估请重新启动本 server。"
                ),
                "start_index": int(start_index),
                "end_index_exclusive": int(end),
            }
        )
        bridge.sync_emit(done_msg)
        print(colored(f"[infer] 已推送 type=done，序列完毕 run_id={run_id}", "green"), flush=True)
        print(colored(f"[infer] 序列推送完毕 run_id={run_id}", "cyan"), flush=True)
    except Exception as exc:  # pragma: no cover
        logging.exception("推理管线失败: %s", exc)
        err_msg = event_to_json({"type": "error", "run_id": run_id, "message": str(exc)})
        try:
            bridge.sync_emit(err_msg)
        except Exception:
            pass
    finally:
        stop_pi05_calib_collectors(bundle.get("calib_collectors") if bundle else None)
        print(colored("[infer] 线程退出", "cyan"), flush=True)
        try:
            bridge.sync_close()
        except Exception:
            pass
