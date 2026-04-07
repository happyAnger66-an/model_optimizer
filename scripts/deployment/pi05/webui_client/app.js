/* global Plotly */

let ws = null;
let meta = null;
let connected = false;

/** 非用户主动断开时，每 RECONNECT_MS 自动重连一次 */
const RECONNECT_MS = 3000;
let manualDisconnect = false;
let reconnectTimer = null;

const THEME_STORAGE_KEY = "pi05_webui_theme";
const DEFAULT_WS_FALLBACK = "ws://127.0.0.1:8765/ws";

const PLOTLY_THEME = {
  light: {
    axisTitle: "#4b5563",
    tick: "#6b7280",
    grid: "#e5e7eb",
    zero: "#d1d5db",
    legend: "#374151",
    font: "#374151",
    gt: "#2563eb",
    pred: "#dc2626",
  },
  dark: {
    axisTitle: "#94a3b8",
    tick: "#cbd5e1",
    grid: "#334155",
    zero: "#475569",
    legend: "#e2e8f0",
    font: "#e2e8f0",
    gt: "#60a5fa",
    pred: "#f87171",
  },
};

const el = (id) => document.getElementById(id);

function getPlotlyPalette() {
  const t = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
  return PLOTLY_THEME[t];
}

function buildPlotlyLayout() {
  const p = getPlotlyPalette();
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 48, r: 12, t: 8, b: 32 },
    xaxis: {
      title: "global_index",
      color: p.axisTitle,
      tickfont: { size: 10, color: p.tick },
      gridcolor: p.grid,
      zerolinecolor: p.zero,
    },
    yaxis: {
      title: "value",
      color: p.axisTitle,
      tickfont: { size: 10, color: p.tick },
      gridcolor: p.grid,
      zerolinecolor: p.zero,
    },
    legend: { font: { size: 10, color: p.legend }, orientation: "h", y: 1.12 },
    font: { color: p.font, size: 11 },
    height: 200,
  };
}

function refreshChartsTheme() {
  for (const d of state.dims) {
    const layout = buildPlotlyLayout();
    chartLayouts.set(d, layout);
    const id = chartDivId(d);
    if (document.getElementById(id)) {
      Plotly.react(id, buildTracesForDim(d), layout, {
        displayModeBar: false,
        responsive: true,
      });
    }
  }
}

function applyTheme(mode) {
  const m = mode === "dark" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", m);
  try {
    localStorage.setItem(THEME_STORAGE_KEY, m);
  } catch (e) {
    /* ignore */
  }
  const sel = el("themeSelect");
  if (sel) sel.value = m;
  refreshChartsTheme();
}

function applyThemeFromStorage() {
  let m = null;
  try {
    m = localStorage.getItem(THEME_STORAGE_KEY);
  } catch (e) {
    m = null;
  }
  if (m !== "dark" && m !== "light") m = "light";
  document.documentElement.setAttribute("data-theme", m);
  const sel = el("themeSelect");
  if (sel) sel.value = m;
}

async function loadDefaultWsUrl() {
  const input = el("wsUrl");
  if (!input) return;
  try {
    const r = await fetch("./server_hint.json", { cache: "no-store" });
    if (r.ok) {
      const j = await r.json();
      if (j && typeof j.default_ws_url === "string" && j.default_ws_url.trim()) {
        input.value = j.default_ws_url.trim();
        return;
      }
    }
  } catch (e) {
    /* 本地 file:// 或尚未生成 hint */
  }
  if (!input.value.trim()) input.value = DEFAULT_WS_FALLBACK;
}

/** dim -> Plotly layout object */
const chartLayouts = new Map();

const state = {
  windowSize: 200,
  dims: [0, 1, 2],
  x: [],
  gt: new Map(), // dim -> []
  pred: new Map(), // dim -> []
};

/** 已收到的 ``type=step`` 条数（当前 meta run） */
let stepReceiveCount = 0;

function setProgress(text) {
  const n = el("serverProgress");
  if (n) n.textContent = text;
}

function updateProgressFromStep(msg) {
  const H = meta && typeof meta.action_horizon === "number" ? meta.action_horizon : "?";
  stepReceiveCount += 1;
  const chunkMark = msg.is_chunk_start ? " [chunk 起点]" : "";
  setProgress(
    `推送中${chunkMark} · 累计 step 消息 ${stepReceiveCount} · global_index=${msg.global_index} · ` +
      `k_in_chunk=${msg.k_in_chunk}/${H} · episode_id=${msg.episode_id}`
  );
}

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

function setBadgeRunFinished() {
  const badge = el("connStatus");
  badge.textContent = "connected · 推理已结束";
  badge.className = "badge badge-amber";
}

function clearReconnectTimer() {
  if (reconnectTimer !== null) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
}

function setBadgeReconnecting() {
  const badge = el("connStatus");
  badge.textContent = `reconnecting in ${RECONNECT_MS / 1000}s…`;
  badge.className = "badge badge-amber";
  setProgress(`连接已断开，将在 ${RECONNECT_MS / 1000}s 后自动重连…（或点 Connect）`);
}

function scheduleReconnect() {
  clearReconnectTimer();
  if (manualDisconnect) return;
  const url = el("wsUrl").value.trim();
  if (!url) return;
  setBadgeReconnecting();
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    if (!manualDisconnect) connectInternal();
  }, RECONNECT_MS);
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

function chartDivId(d) {
  return `chart-dim-${d}`;
}

function initChart() {
  resetSeries();
  chartLayouts.clear();
  const container = el("chartsContainer");
  container.innerHTML = "";

  for (const d of state.dims) {
    const wrap = document.createElement("div");
    wrap.className = "chartDimWrap";
    const title = document.createElement("div");
    title.className = "chartDimTitle";
    title.textContent = `Action dim ${d}`;
    const plotDiv = document.createElement("div");
    plotDiv.id = chartDivId(d);
    plotDiv.className = "chart";
    wrap.appendChild(title);
    wrap.appendChild(plotDiv);
    container.appendChild(wrap);

    const p = getPlotlyPalette();
    const traces = [
      {
        x: [],
        y: [],
        mode: "lines",
        name: "gt",
        line: { width: 2, color: p.gt },
      },
      {
        x: [],
        y: [],
        mode: "lines",
        name: "pred",
        line: { width: 2, dash: "dot", color: p.pred },
      },
    ];
    const layout = buildPlotlyLayout();
    chartLayouts.set(d, layout);
    Plotly.newPlot(chartDivId(d), traces, layout, { displayModeBar: false, responsive: true });
  }
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

  for (const d of state.dims) {
    const layout = chartLayouts.get(d);
    if (!layout) continue;
    Plotly.react(chartDivId(d), buildTracesForDim(d), layout, {
      displayModeBar: false,
      responsive: true,
    });
  }
}

function buildTracesForDim(d) {
  const p = getPlotlyPalette();
  return [
    {
      x: [...state.x],
      y: [...(state.gt.get(d) || [])],
      mode: "lines",
      name: "gt",
      line: { width: 2, color: p.gt },
    },
    {
      x: [...state.x],
      y: [...(state.pred.get(d) || [])],
      mode: "lines",
      name: "pred",
      line: { width: 2, dash: "dot", color: p.pred },
    },
  ];
}

function connect() {
  manualDisconnect = false;
  clearReconnectTimer();
  connectInternal();
}

function connectInternal() {
  const url = el("wsUrl").value.trim();
  if (!url) return;
  if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;

  ws = new WebSocket(url);

  ws.onopen = () => {
    clearReconnectTimer();
    connected = true;
    setBadge(true);
    el("btnConnect").disabled = true;
    el("btnDisconnect").disabled = false;
    setProgress("WebSocket 已连接，等待服务端 meta / 数据流…");
  };

  ws.onclose = () => {
    ws = null;
    connected = false;
    setBadge(false);
    el("btnConnect").disabled = false;
    el("btnDisconnect").disabled = true;
    if (manualDisconnect) {
      setProgress("已断开（手动 Disconnect）。");
    } else {
      scheduleReconnect();
    }
  };

  ws.onerror = () => {
    // onclose will handle UI 与重连调度
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
      if (msg.phase === "loading") {
        el("repoId").textContent = "…";
        el("backend").textContent = "…";
        el("prompt").textContent = "—";
        setProgress(msg.message || "服务端：正在加载数据集与策略…");
        return;
      }
      meta = msg;
      stepReceiveCount = 0;
      el("runId").textContent = meta.run_id ?? "-";
      el("repoId").textContent = meta.repo_id ?? "-";
      el("backend").textContent = meta.backend ?? "-";
      {
        const lo = meta.start_index ?? "?";
        const hi = meta.end_index_exclusive ?? "?";
        const H = meta.action_horizon ?? "?";
        setProgress(
          `服务端就绪 · run_id=${meta.run_id ?? "-"} · 评估下标 [${lo}, ${hi}) · action_horizon=${H} · 等待 step 流…`
        );
      }
      return;
    }

    if (msg.type === "done") {
      setProgress(msg.message || `推理已结束 run_id=${msg.run_id ?? "-"}`);
      if (connected) setBadgeRunFinished();
      return;
    }

    if (msg.type === "error") {
      setProgress(`错误：${msg.message || "服务器推理管线异常"}`);
      return;
    }

    if (msg.type === "step") {
      updateTop(msg);
      updateProgressFromStep(msg);
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
  manualDisconnect = true;
  clearReconnectTimer();
  setProgress("正在断开…");
  if (ws) {
    try {
      ws.close();
    } catch (e) {
      /* ignore */
    }
  }
  ws = null;
  connected = false;
  setBadge(false);
  el("btnConnect").disabled = false;
  el("btnDisconnect").disabled = true;
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

  el("themeSelect").addEventListener("change", () => {
    applyTheme(el("themeSelect").value);
  });

  el("toggleWrist").addEventListener("change", () => {
    const show = el("toggleWrist").checked;
    el("imgWrist").style.display = show ? "block" : "none";
  });
}

async function bootstrap() {
  await loadDefaultWsUrl();
  applyThemeFromStorage();
  setup();
}

bootstrap();

