"""评估会话（Facade）：bundle 加载、chunk 循环、经 Janus 出站（与 WebSocket 解耦）。"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from termcolor import colored

from .bundle import load_infer_bundle
from .calib import stop_pi05_calib_collectors
from .chunk_infer import dump_perf_final_summary, process_infer_chunk
from .config import Args
from .ports import SyncOutboundPort
from .protocol import event_to_json


def run_infer_worker(
    *,
    args: Args,
    run_id: str,
    meta_ready: dict[str, Any],
    infer_paused: threading.Event,
    client_connected: threading.Event,
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

        if bool(getattr(args, "wait_for_client", False)):
            print(
                colored(
                    "[infer] wait_for_client：meta 已推送，等待浏览器 WebSocket 连接后再开始推理…",
                    "yellow",
                ),
                flush=True,
            )
            client_connected.wait()
            print(colored("[infer] 已检测到客户端，开始 chunk 推理", "green"), flush=True)

        start_index = bundle["start_index"]
        end = bundle["end"]
        min_step_period = (1.0 / args.max_fps) if args.max_fps and args.max_fps > 0 else 0.0
        last_send_t = 0.0

        print(
            colored(
                f"[infer] 进入 chunk 循环：start={start_index} end={end} "
                f"action_horizon={bundle.get('action_horizon')} "
                f"clients={'有' if client_connected.is_set() else '无'}（开始推理首个 chunk，"
                f"compare/cuda_graph 首段可能较慢）…",
                "cyan",
            ),
            flush=True,
        )
        _emitted_steps = 0
        _first_step_done = False
        for idx in range(start_index, end):
            while infer_paused.is_set():
                time.sleep(0.05)
            _chunk_t0 = time.monotonic()
            msgs = process_infer_chunk(bundle, idx)
            if msgs:
                if not _first_step_done:
                    print(
                        colored(
                            f"[infer] 首个 step 产出 idx={idx} "
                            f"(首段耗时 {(time.monotonic() - _chunk_t0):.1f}s, n={len(msgs)})，开始推送浏览器",
                            "green",
                        ),
                        flush=True,
                    )
                    _first_step_done = True
            for msg in msgs:
                bridge.sync_emit(msg)
                _emitted_steps += 1
                if min_step_period > 0:
                    now = time.monotonic()
                    dt = now - last_send_t
                    if dt < min_step_period:
                        time.sleep(min_step_period - dt)
                    last_send_t = time.monotonic()
            if _emitted_steps and _emitted_steps % 50 == 0:
                print(
                    colored(f"[infer] 已累计推送 {_emitted_steps} 个 step（至 idx={idx}）", "cyan"),
                    flush=True,
                )

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
        try:
            dump_perf_final_summary(bundle)
        except Exception as exc:
            logging.warning("打印最终性能汇总失败: %s", exc)
        print(colored("[infer] 线程退出", "cyan"), flush=True)
        try:
            bridge.sync_close()
        except Exception:
            pass
