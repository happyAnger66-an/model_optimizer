"""服务启动时的终端提示（与 WebSocket / 推理逻辑无关）。"""

from __future__ import annotations

from pathlib import Path

from termcolor import colored

from .config import Args


def print_server_startup_hints(args: Args, hint_url: str) -> None:
    print(
        colored(
            f"WebUI WebSocket 监听: {args.host}:{args.port}{args.path} · "
            f"client 默认地址（已写入 webui_client/server_hint.json）: {hint_url}",
            "green",
        ),
        flush=True,
    )
    client_dir = Path(__file__).resolve().parent.parent / "webui_client"
    print(
        colored(
            f"浏览器请用 HTTP 打开静态页（勿 file://）：cd {client_dir} && "
            f"python -m http.server 8080  →  http://127.0.0.1:8080/",
            "green",
        ),
        flush=True,
    )
    if bool(getattr(args, "wait_for_client", False)):
        print(
            colored(
                "[main] wait_for_client=true：加载完成后将等待 WebSocket 连接再开始推理",
                "yellow",
            ),
            flush=True,
        )
