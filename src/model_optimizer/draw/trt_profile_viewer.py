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

"""Parse trtexec --exportProfile / profiling JSON and serve an interactive HTML table."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from tempfile import TemporaryDirectory
from typing import Any


def _coerce_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_trtexec_profile_rows(path: str) -> tuple[list[dict[str, Any]], int | None]:
    """Load layer rows from trtexec JSON profile.

    TensorRT trtexec writes an array: first object may be ``{\"count\": N}``;
    subsequent objects are per-layer timing (name, timeMs, averageMs, medianMs, percentage).
    Also accepts a bare list of layer dicts or ``{\"layers\": [...]}`` wrappers.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    iteration_count: int | None = None
    rows: list[dict[str, Any]] = []

    if isinstance(raw, dict):
        if "layers" in raw and isinstance(raw["layers"], list):
            raw = raw["layers"]
        elif "Layers" in raw and isinstance(raw["Layers"], list):
            raw = raw["Layers"]
        else:
            raw = [raw]

    if not isinstance(raw, list):
        raise ValueError(f"Unexpected JSON root type in {path!r}: {type(raw).__name__}")

    for rec in raw:
        if not isinstance(rec, dict):
            continue
        if "count" in rec and "name" not in rec:
            try:
                iteration_count = int(rec["count"])
            except (TypeError, ValueError):
                iteration_count = None
            continue
        name = rec.get("name")
        if name is None:
            continue
        rows.append(
            {
                "name": str(name),
                "timeMs": _coerce_float(rec.get("timeMs")),
                "averageMs": _coerce_float(rec.get("averageMs")),
                "medianMs": _coerce_float(rec.get("medianMs")),
                "percentage": _coerce_float(rec.get("percentage")),
            }
        )

    if not rows:
        raise ValueError(
            f"No layer records found in {path!r}. "
            "Expected trtexec profile JSON (see --exportProfile=<file>)."
        )
    return rows, iteration_count


def build_profile_html(
    rows: list[dict[str, Any]],
    title: str,
    iteration_count: int | None,
) -> str:
    payload = json.dumps(
        {"title": title, "iterationCount": iteration_count, "rows": rows},
        ensure_ascii=False,
    )
    # Single-file viewer: sortable columns, name filter, min % filter.
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>TRT layer profile — {title}</title>
  <style>
    :root {{
      --bg: #0f1419;
      --panel: #1a2332;
      --text: #e6edf3;
      --muted: #8b9cb3;
      --accent: #58a6ff;
      --border: #30363d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font-family: ui-sans-serif, system-ui, sans-serif;
      background: var(--bg); color: var(--text); line-height: 1.45;
    }}
    header {{
      padding: 1rem 1.25rem; background: var(--panel); border-bottom: 1px solid var(--border);
    }}
    h1 {{ font-size: 1.1rem; font-weight: 600; margin: 0 0 0.35rem 0; }}
    .meta {{ color: var(--muted); font-size: 0.85rem; }}
    .toolbar {{
      display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center;
      padding: 0.75rem 1.25rem; border-bottom: 1px solid var(--border);
    }}
    label {{ font-size: 0.8rem; color: var(--muted); }}
    input[type="text"], input[type="number"] {{
      background: var(--panel); color: var(--text); border: 1px solid var(--border);
      border-radius: 6px; padding: 0.4rem 0.55rem; min-width: 12rem;
    }}
    input[type="number"] {{ min-width: 5rem; width: 6rem; }}
    table {{
      width: 100%; border-collapse: collapse; font-size: 0.88rem;
    }}
    th, td {{
      text-align: left; padding: 0.45rem 1rem; border-bottom: 1px solid var(--border);
    }}
    th {{
      cursor: pointer; user-select: none; color: var(--accent); white-space: nowrap;
    }}
    th:hover {{ text-decoration: underline; }}
    td.num {{ font-variant-numeric: tabular-nums; text-align: right; }}
    td.name {{ word-break: break-all; max-width: 48vw; }}
    tr:hover td {{ background: rgba(88, 166, 255, 0.06); }}
    .wrap {{ overflow: auto; max-height: calc(100vh - 12rem); }}
    .hint {{ font-size: 0.75rem; color: var(--muted); margin-left: 0.25rem; }}
    .agg {{
      margin: 0 1.25rem 0.75rem 1.25rem; padding: 0.65rem 0.85rem;
      background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
      font-size: 0.82rem; color: var(--text);
    }}
    .agg-title {{ color: var(--muted); font-size: 0.75rem; margin-bottom: 0.35rem; }}
    .agg-grid {{
      display: grid; grid-template-columns: repeat(auto-fill, minmax(11rem, 1fr));
      gap: 0.35rem 1rem;
    }}
    .agg-grid span {{ color: var(--muted); }}
    .agg-grid strong {{ font-variant-numeric: tabular-nums; color: var(--accent); }}
  </style>
</head>
<body>
  <header>
    <h1 id="hdr">Layer timing</h1>
    <div class="meta" id="sub"></div>
  </header>
  <div class="toolbar">
    <div>
      <label for="q">Filter name</label><br/>
      <input type="text" id="q" placeholder="substring or regex…" autocomplete="off"/>
      <span class="hint">regex if enclosed in /…/</span>
    </div>
    <div>
      <label for="minp">Min time %</label><br/>
      <input type="number" id="minp" value="0" min="0" step="0.1"/>
    </div>
    <div>
      <label for="mint">Min avg (ms)</label><br/>
      <input type="number" id="mint" value="0" min="0" step="0.0001"/>
    </div>
    <div style="margin-top:1.2rem;color:var(--muted);font-size:0.8rem;">
      Click column headers to sort. Rows: <strong id="rc">0</strong> / <span id="tc"></span>
    </div>
  </div>
  <div class="agg">
    <div class="agg-title">Filtered aggregate（当前过滤结果）</div>
    <div class="agg-grid">
      <div><span>Σ timeMs</span> <strong id="agg-sum-time">—</strong></div>
      <div><span>mean(averageMs)</span> <strong id="agg-mean-avg">—</strong></div>
      <div><span>median(medianMs)</span> <strong id="agg-med-med">—</strong></div>
      <div><span>Σ percentage</span> <strong id="agg-sum-pct">—</strong></div>
    </div>
  </div>
  <div class="wrap">
    <table>
      <thead>
        <tr>
          <th data-k="name">Layer</th>
          <th data-k="timeMs" class="num">timeMs</th>
          <th data-k="averageMs" class="num">averageMs</th>
          <th data-k="medianMs" class="num">medianMs</th>
          <th data-k="percentage" class="num">percentage</th>
        </tr>
      </thead>
      <tbody id="tb"></tbody>
    </table>
  </div>
  <script>
  const DATA = {payload};

  let sortKey = "percentage";
  let sortDir = -1;

  function parseFilter(s) {{
    s = (s || "").trim();
    if (s.length >= 2 && s[0] === "/" && s.lastIndexOf("/") > 0) {{
      const last = s.lastIndexOf("/");
      const body = s.slice(1, last);
      const flags = s.slice(last + 1);
      try {{ return new RegExp(body, flags); }} catch (e) {{ return null; }}
    }}
    return null;
  }}

  function nameMatch(row, q) {{
    if (!q) return true;
    const re = parseFilter(q);
    if (re) return re.test(row.name);
    return row.name.toLowerCase().includes(q.toLowerCase());
  }}

  function cmp(a, b, k) {{
    const va = a[k], vb = b[k];
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    if (typeof va === "string") return va.localeCompare(vb);
    return va - vb;
  }}

  function medianNumeric(vals) {{
    const a = vals.filter(v => v != null && !Number.isNaN(v)).slice().sort((x, y) => x - y);
    if (!a.length) return null;
    const m = Math.floor(a.length / 2);
    return (a.length % 2 === 1) ? a[m] : (a[m - 1] + a[m]) / 2;
  }}

  function updateAggregate(rows) {{
    const set = (id, v) => {{ document.getElementById(id).textContent = v; }};
    if (!rows.length) {{
      set("agg-sum-time", "—");
      set("agg-mean-avg", "—");
      set("agg-med-med", "—");
      set("agg-sum-pct", "—");
      return;
    }}
    let sumTime = 0, sumPct = 0, nPct = 0;
    const avgs = [], medms = [];
    for (const r of rows) {{
      if (r.timeMs != null && !Number.isNaN(r.timeMs)) sumTime += r.timeMs;
      if (r.percentage != null && !Number.isNaN(r.percentage)) {{ sumPct += r.percentage; nPct++; }}
      if (r.averageMs != null && !Number.isNaN(r.averageMs)) avgs.push(r.averageMs);
      if (r.medianMs != null && !Number.isNaN(r.medianMs)) medms.push(r.medianMs);
    }}
    const meanAvg = avgs.length ? avgs.reduce((x, y) => x + y, 0) / avgs.length : null;
    const medMed = medianNumeric(medms.length ? medms : avgs);
    set("agg-sum-time", fmt(sumTime));
    set("agg-mean-avg", meanAvg == null ? "—" : fmt(meanAvg));
    set("agg-med-med", medMed == null ? "—" : fmt(medMed));
    set("agg-sum-pct", nPct ? sumPct.toFixed(2) + " %" : "—");
  }}

  function render() {{
    const q = document.getElementById("q").value;
    const minp = parseFloat(document.getElementById("minp").value) || 0;
    const mint = parseFloat(document.getElementById("mint").value) || 0;
    let rows = DATA.rows.filter(r =>
      nameMatch(r, q) &&
      (r.percentage == null || r.percentage >= minp) &&
      (r.averageMs == null || r.averageMs >= mint)
    );
    rows = rows.slice().sort((a, b) => sortDir * cmp(a, b, sortKey));
    const tb = document.getElementById("tb");
    tb.innerHTML = rows.map(r => `
      <tr>
        <td class="name">${{escapeHtml(r.name)}}</td>
        <td class="num">${{fmt(r.timeMs)}}</td>
        <td class="num">${{fmt(r.averageMs)}}</td>
        <td class="num">${{fmt(r.medianMs)}}</td>
        <td class="num">${{fmt(r.percentage)}}</td>
      </tr>`).join("");
    document.getElementById("rc").textContent = rows.length;
    document.getElementById("tc").textContent = DATA.rows.length;
    updateAggregate(rows);
  }}

  function escapeHtml(s) {{
    return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  }}
  function fmt(v) {{
    if (v == null || Number.isNaN(v)) return "—";
    if (Math.abs(v) >= 1000) return v.toFixed(2);
    if (Math.abs(v) >= 1) return v.toFixed(4);
    return v.toPrecision(4);
  }}

  document.querySelectorAll("th[data-k]").forEach(th => {{
    th.addEventListener("click", () => {{
      const k = th.getAttribute("data-k");
      if (k === sortKey) sortDir *= -1;
      else {{ sortKey = k; sortDir = k === "name" ? 1 : -1; }}
      render();
    }});
  }});
  ["q", "minp", "mint"].forEach(id => {{
    document.getElementById(id).addEventListener("input", render);
  }});

  document.getElementById("hdr").textContent = "Layer timing — " + (DATA.title || "profile");
  let sub = "";
  if (DATA.iterationCount != null) sub += "Profiler iterations: " + DATA.iterationCount + ". ";
  sub += "Source: trtexec profile JSON.";
  document.getElementById("sub").textContent = sub;
  render();
  </script>
</body>
</html>
"""


def serve_profile_html(
    html: str,
    host: str,
    port: int,
    open_browser: bool,
) -> None:
    with TemporaryDirectory(prefix="moc_trt_profile_") as tmp:
        index_path = os.path.join(tmp, "index.html")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(html)

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args: Any, **kwargs: Any):
                super().__init__(*args, directory=tmp, **kwargs)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        httpd = ThreadingHTTPServer((host, port), Handler)
        url = f"http://{host}:{port}/"
        print(f"TRT profile viewer: {url}")
        print("Press Ctrl+C to stop.")

        def _open_browser() -> None:
            time.sleep(0.35)
            webbrowser.open(url)

        if open_browser:
            threading.Thread(target=_open_browser, daemon=True).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.shutdown()
            httpd.server_close()


def pick_free_port(host: str, preferred: int) -> int:
    if preferred > 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, preferred))
                return preferred
            except OSError:
                pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]
