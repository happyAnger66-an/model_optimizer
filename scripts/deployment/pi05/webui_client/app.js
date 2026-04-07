/* global Plotly */

let ws = null;
let meta = null;
let connected = false;

const el = (id) => document.getElementById(id);

let chartLayout = null;

const state = {
  windowSize: 200,
  dims: [0, 1, 2],
  x: [],
  gt: new Map(), // dim -> []
  pred: new Map(), // dim -> []
};

function setBadge(ok) {
  const badge = el("connStatus");
  if (ok) {
    badge.textContent = "connected";
    badge.className = "badge badge-green";
  } else {
    badge.textContent = "disconnected";
    badge.className = "badge badge-red";
  }
}

function resetSeries() {
  state.x = [];
  state.gt = new Map();
  state.pred = new Map();
  for (const d of state.dims) {
    state.gt.set(d, []);
    state.pred.set(d, []);
  }
}

function parseDims(text) {
  const parts = text
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  const dims = [];
  for (const p of parts) {
    const v = Number.parseInt(p, 10);
    if (Number.isFinite(v) && v >= 0) dims.push(v);
  }
  return Array.from(new Set(dims)).slice(0, 12);
}

function updateTop(event) {
  el("runId").textContent = event.run_id ?? "-";
  el("repoId").textContent = meta?.repo_id ?? "-";
  el("backend").textContent = meta?.backend ?? "-";
  el("episodeId").textContent = event.episode_id ?? "-";
  el("globalIndex").textContent = event.global_index ?? "-";
  el("kInChunk").textContent = event.k_in_chunk ?? "-";
  if (event.server_timing && typeof event.server_timing.infer_ms === "number") {
    el("inferMs").textContent = event.server_timing.infer_ms.toFixed(2);
  }
  if (event.metrics) {
    if (typeof event.metrics.mae === "number") el("mae").textContent = event.metrics.mae.toFixed(6);
    if (typeof event.metrics.mse === "number") el("mse").textContent = event.metrics.mse.toFixed(6);
  }
}

function setImage(imgEl, jpegB64) {
  if (!jpegB64) return;
  imgEl.src = `data:image/jpeg;base64,${jpegB64}`;
}

function initChart() {
  resetSeries();
  const traces = [];
  for (const d of state.dims) {
    traces.push({
      x: [],
      y: [],
      mode: "lines",
      name: `gt[d${d}]`,
      line: { width: 2 },
    });
    traces.push({
      x: [],
      y: [],
      mode: "lines",
      name: `pred[d${d}]`,
      line: { width: 2, dash: "dot" },
    });
  }
  chartLayout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 50, r: 20, t: 30, b: 40 },
    xaxis: { title: "global_index", color: "#9ca3af" },
    yaxis: { title: "action value", color: "#9ca3af" },
    legend: { font: { color: "#e5e7eb" } },
    font: { color: "#e5e7eb" },
  };
  Plotly.newPlot("chart", traces, chartLayout, { displayModeBar: false, responsive: true });
}

function pushPoint(event) {
  const x = event.global_index;
  state.x.push(x);
  while (state.x.length > state.windowSize) state.x.shift();

  const gtAction = event.gt_action || [];
  const predAction = event.pred_action || [];

  for (const d of state.dims) {
    const gArr = state.gt.get(d);
    const pArr = state.pred.get(d);
    if (!gArr || !pArr) continue;
    gArr.push(gtAction[d] ?? null);
    pArr.push(predAction[d] ?? null);
    while (gArr.length > state.windowSize) gArr.shift();
    while (pArr.length > state.windowSize) pArr.shift();
  }

  Plotly.react("chart", buildTraces(), chartLayout, {
    displayModeBar: false,
    responsive: true,
  });
}

function buildTraces() {
  const traces = [];
  for (const d of state.dims) {
    traces.push({
      x: [...state.x],
      y: [...(state.gt.get(d) || [])],
      mode: "lines",
      name: `gt[d${d}]`,
      line: { width: 2 },
    });
    traces.push({
      x: [...state.x],
      y: [...(state.pred.get(d) || [])],
      mode: "lines",
      name: `pred[d${d}]`,
      line: { width: 2, dash: "dot" },
    });
  }
  return traces;
}

function connect() {
  const url = el("wsUrl").value.trim();
  if (!url) return;

  ws = new WebSocket(url);

  ws.onopen = () => {
    connected = true;
    setBadge(true);
    el("btnConnect").disabled = true;
    el("btnDisconnect").disabled = false;
  };

  ws.onclose = () => {
    connected = false;
    setBadge(false);
    el("btnConnect").disabled = false;
    el("btnDisconnect").disabled = true;
  };

  ws.onerror = () => {
    // onclose will handle UI
  };

  ws.onmessage = (evt) => {
    let msg = null;
    try {
      msg = JSON.parse(evt.data);
    } catch (e) {
      return;
    }
    if (!msg || !msg.type) return;

    if (msg.type === "meta") {
      meta = msg;
      el("runId").textContent = meta.run_id ?? "-";
      el("repoId").textContent = meta.repo_id ?? "-";
      el("backend").textContent = meta.backend ?? "-";
      return;
    }

    if (msg.type === "step") {
      updateTop(msg);
      if (msg.prompt) el("prompt").textContent = msg.prompt;

      if (msg.images) {
        if (msg.images.base_rgb_jpeg_b64) setImage(el("imgBase"), msg.images.base_rgb_jpeg_b64);
        const showWrist = el("toggleWrist").checked;
        if (showWrist && msg.images.wrist_rgb_jpeg_b64) {
          el("imgWrist").style.display = "block";
          setImage(el("imgWrist"), msg.images.wrist_rgb_jpeg_b64);
        }
      }

      pushPoint(msg);
    }
  };
}

function disconnect() {
  if (ws) ws.close();
  ws = null;
}

function applyDims() {
  const newWindow = Number.parseInt(el("windowSize").value, 10);
  if (Number.isFinite(newWindow) && newWindow >= 50) state.windowSize = newWindow;

  const dims = parseDims(el("dims").value);
  if (dims.length > 0) state.dims = dims;
  initChart();
}

function setup() {
  setBadge(false);
  initChart();

  el("btnConnect").addEventListener("click", () => connect());
  el("btnDisconnect").addEventListener("click", () => disconnect());
  el("btnApplyDims").addEventListener("click", () => applyDims());

  el("toggleWrist").addEventListener("change", () => {
    const show = el("toggleWrist").checked;
    el("imgWrist").style.display = show ? "block" : "none";
  });
}

setup();

