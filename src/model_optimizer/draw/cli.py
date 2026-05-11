# Copyright 2025 the model_optimizer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI for ``model_optimizer-cli draw …``."""

from __future__ import annotations

import argparse
import os
import sys

from .trt_profile_viewer import (
    build_profile_html,
    load_trtexec_profile_rows,
    pick_free_port,
    serve_profile_html,
)


def draw_cli(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="model_optimizer-cli draw",
        description="Visual tools (e.g. trtexec layer profile in the browser).",
    )
    sub = parser.add_subparsers(dest="sub", required=True)

    p_prof = sub.add_parser(
        "profile",
        help="Open trtexec --exportProfile JSON in a local web table (sort / filter).",
    )
    p_prof.add_argument(
        "json_path",
        type=str,
        help="Path to trtexec profile JSON (--exportProfile=<file> or equivalent).",
    )
    p_prof.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1).",
    )
    p_prof.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port (0 = pick a free port).",
    )
    p_prof.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser tab automatically.",
    )

    args = parser.parse_args(argv[1:])
    if args.sub != "profile":
        parser.print_help()
        sys.exit(2)

    path = os.path.abspath(os.path.expanduser(args.json_path))
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    rows, iteration_count = load_trtexec_profile_rows(path)
    title = os.path.basename(path)
    html = build_profile_html(rows, title, iteration_count)
    port = pick_free_port(args.host, args.port)
    serve_profile_html(
        html,
        host=args.host,
        port=port,
        open_browser=not args.no_browser,
    )
