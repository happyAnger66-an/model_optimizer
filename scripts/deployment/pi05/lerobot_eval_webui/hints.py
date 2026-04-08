"""向静态 webui_client 写入默认 WebSocket URL。"""

from __future__ import annotations

import json
from pathlib import Path

from .config import Args

# 包位于 scripts/deployment/pi05/lerobot_eval_webui/，webui_client 为同级目录
_PI05_DIR = Path(__file__).resolve().parent.parent


def default_client_ws_url(args: Args) -> str:
    if args.client_ws_url and str(args.client_ws_url).strip():
        return str(args.client_ws_url).strip()
    h = args.host.strip()
    if h in ("0.0.0.0", "::", ""):
        h = "127.0.0.1"
    return f"ws://{h}:{args.port}{args.path}"


def write_webui_server_hint(args: Args) -> str:
    """将默认 ws URL 写入 ``webui_client/server_hint.json``，供静态页面 fetch。"""
    url = default_client_ws_url(args)
    client_dir = _PI05_DIR / "webui_client"
    client_dir.mkdir(parents=True, exist_ok=True)
    hint_path = client_dir / "server_hint.json"
    hint_path.write_text(
        json.dumps({"default_ws_url": url}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return url
