/* global Plotly */

let ws = null;
let meta = null;
let connected = false;

/** 非用户主动断开时，每 RECONNECT_MS 自动重连一次 */
const RECONNECT_MS = 3000;
let manualDisconnect = false;
let reconnectTimer = null;

/** 每次发起新连接自增，用于忽略已被替换或已放弃的旧 WebSocket 的 onopen/onclose */
let wsConnectGeneration = 0;

const THEME_STORAGE_KEY = "pi05_webui_theme";
const CHARTS_COLS_STORAGE_KEY = "pi05_webui_charts_cols";
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
    maeLine: "#059669",
    mseLine: "#d97706",
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
    maeLine: "#34d399",
    mseLine: "#fbbf24",
  },
};

const el = (id) => document.getElementById(id);

function setInferLoadingSub(text) {
  const s = el("inferLoadingSub");
  if (s) s.textContent = text;
}

/** 顶栏下内嵌加载条；true = 等待首次 step（不遮挡页面交互） */
function setInferLoadingBanner(visible) {
  const o = el("inferLoadingBanner");
  if (!o) return;
  const on = Boolean(visible);
  if (on) {
    o.removeAttribute("hidden");
    o.setAttribute("aria-hidden", "false");
  } else {
    o.setAttribute("hidden", "");
    o.setAttribute("aria-hidden", "true");
  }
}

function getPlotlyPalette() {
  const t = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
  return PLOTLY_THEME[t];
}

function buildMetricsLayout(yAxisTitle) {
  const p = getPlotlyPalette();
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 52, r: 12, t: 8, b: 32 },
    xaxis: {
      title: "global_index",
      color: p.axisTitle,
      tickfont: { size: 10, color: p.tick },
      gridcolor: p.grid,
      zerolinecolor: p.zero,
    },
    yaxis: {
      title: yAxisTitle,
      color: p.axisTitle,
      tickfont: { size: 10, color: p.tick },
      gridcolor: p.grid,
      zerolinecolor: p.zero,
    },
    showlegend: false,
    font: { color: p.font, size: 11 },
    height: 200,
  };
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
    const id = chartDivId(d);
    const div = document.getElementById(id);
    if (!div) continue;
    try {
      Plotly.purge(div);
    } catch (e) {
      /* ignore */
    }
    const layout = buildPlotlyLayout();
    chartLayouts.set(d, layout);
    Plotly.newPlot(div, buildTracesForDim(d), layout, { displayModeBar: false, responsive: true });
  }
  refreshMetricsChartsTheme();
}

function refreshMetricsChartsTheme() {
  const p = getPlotlyPalette();
  for (const def of METRIC_CHART_DEFS) {
    const div = document.getElementById(def.id);
    if (!div) continue;
    try {
      Plotly.purge(div);
    } catch (e) {
      /* ignore */
    }
    const layout = buildMetricsLayout(def.yLabel);
    metricsLayouts.set(def.id, layout);
    const ys = state[def.seriesKey];
    const trace = [
      {
        x: [...state.x],
        y: [...ys],
        mode: "lines",
        name: def.traceName,
        line: { width: 2, color: p[def.colorKey] },
      },
    ];
    Plotly.newPlot(div, trace, layout, { displayModeBar: false, responsive: true });
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

function loadChartsColsPreference() {
  try {
    const raw = localStorage.getItem(CHARTS_COLS_STORAGE_KEY);
    const n = raw != null ? Number.parseInt(raw, 10) : NaN;
    const sel = el("chartsCols");
    if (sel && Number.isFinite(n) && n >= 1 && n <= 6) {
      sel.value = String(n);
    }
  } catch (e) {
    /* ignore */
  }
}

function readChartsColsFromUI() {
  const sel = el("chartsCols");
  if (!sel) return 2;
  const v = Number.parseInt(sel.value, 10);
  if (!Number.isFinite(v)) return 2;
  return Math.min(6, Math.max(1, v));
}

/** 设置 #chartsContainer 的 grid 列数（默认 2），并写入 localStorage */
function applyChartsCols() {
  const n = readChartsColsFromUI();
  const c = el("chartsContainer");
  if (c) c.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
  try {
    localStorage.setItem(CHARTS_COLS_STORAGE_KEY, String(n));
  } catch (e) {
    /* ignore */
  }
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
/** chart-mae / chart-mse -> layout */
const metricsLayouts = new Map();

const METRIC_CHART_DEFS = [
  { id: "chart-mae", cardTitle: "MAE", yLabel: "MAE", seriesKey: "mae", traceName: "mae", colorKey: "maeLine" },
  { id: "chart-mse", cardTitle: "MSE", yLabel: "MSE", seriesKey: "mse", traceName: "mse", colorKey: "mseLine" },
];

const state = {
  windowSize: 120,
  dims: [0, 1, 2, 3, 4, 5],
  x: [],
  mae: [],
  mse: [],
  gt: new Map(), // dim -> []
  pred: new Map(), // dim -> []
};

/** 已收到的 ``type=step`` 条数（当前 meta run） */
let stepReceiveCount = 0;

/** 实测 infer 频率：相邻 ``is_chunk_start`` 间隔的 EMA（Hz） */
const INFER_HZ_SMOOTH = 0.82;
/** 实测展开后 step 流频率：最近窗口内 (n-1)/Δt（Hz） */
const STEP_HZ_WINDOW_MS = 1000;
let lastChunkStartPerf = null;
let emaInferHz = null;
const stepReceivePerfRing = [];

function resetHzEstimators() {
  lastChunkStartPerf = null;
  emaInferHz = null;
  stepReceivePerfRing.length = 0;
  const ih = el("inferHz");
  const sh = el("stepHz");
  if (ih) ih.textContent = "-";
  if (sh) sh.textContent = "-";
}

/**
 * 每个 step 更新：infer_hz 由 chunk 起点间隔；step_hz 由最近窗口内 step 到达间隔。
 */
function updateHzEstimatorsFromStep(msg) {
  const now = performance.now();

  if (msg.is_chunk_start) {
    if (lastChunkStartPerf != null) {
      const dtS = (now - lastChunkStartPerf) / 1000;
      if (dtS > 1e-6) {
        const inst = 1 / dtS;
        emaInferHz = emaInferHz == null ? inst : INFER_HZ_SMOOTH * emaInferHz + (1 - INFER_HZ_SMOOTH) * inst;
        const ih = el("inferHz");
        if (ih) ih.textContent = emaInferHz.toFixed(2);
      }
    }
    lastChunkStartPerf = now;
  }

  stepReceivePerfRing.push(now);
  const cut = now - STEP_HZ_WINDOW_MS;
  while (stepReceivePerfRing.length > 0 && stepReceivePerfRing[0] < cut) {
    stepReceivePerfRing.shift();
  }
  const sh = el("stepHz");
  if (!sh) return;
  if (stepReceivePerfRing.length < 2) {
    sh.textContent = "-";
    return;
  }
  const t0 = stepReceivePerfRing[0];
  const spanS = (now - t0) / 1000;
  if (spanS < 0.05) {
    sh.textContent = "-";
    return;
  }
  const stepHz = (stepReceivePerfRing.length - 1) / spanS;
  sh.textContent = stepHz.toFixed(2);
}

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

/** 根据服务端暂停状态更新按钮；未连接时两项均禁用 */
function setInferPauseButtons(serverPaused) {
  const bp = el("btnPauseInfer");
  const br = el("btnResumeInfer");
  if (!bp || !br) return;
  const live = connected && ws && ws.readyState === WebSocket.OPEN;
  if (!live) {
    bp.disabled = true;
    br.disabled = true;
    return;
  }
  bp.disabled = serverPaused;
  br.disabled = !serverPaused;
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
  state.mae = [];
  state.mse = [];
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

/** 与子图初始状态一致的两条空曲线（gt / pred） */
function traceTemplatesEmpty() {
  const p = getPlotlyPalette();
  return [
    { x: [], y: [], mode: "lines", name: "gt", line: { width: 2, color: p.gt } },
    { x: [], y: [], mode: "lines", name: "pred", line: { width: 2, dash: "dot", color: p.pred } },
  ];
}

/**
 * 强制 purge + newPlot，避免 Plotly.react 多次合并后出现多余 trace（例如 3 条线）。
 */
function emptyMetricTrace(def) {
  const p = getPlotlyPalette();
  return [
    {
      x: [],
      y: [],
      mode: "lines",
      name: def.traceName,
      line: { width: 2, color: p[def.colorKey] },
    },
  ];
}

function purgeAndNewPlotMetricsChartsSync() {
  for (const def of METRIC_CHART_DEFS) {
    const div = document.getElementById(def.id);
    if (!div) continue;
    try {
      Plotly.purge(div);
    } catch (e) {
      /* ignore */
    }
    const layout = buildMetricsLayout(def.yLabel);
    metricsLayouts.set(def.id, layout);
    Plotly.newPlot(div, emptyMetricTrace(def), layout, { displayModeBar: false, responsive: true });
  }
}

function purgeAndNewPlotAllChartsSync() {
  for (const d of state.dims) {
    const id = chartDivId(d);
    const div = document.getElementById(id);
    if (!div) continue;
    try {
      Plotly.purge(div);
    } catch (e) {
      /* 尚无图实例 */
    }
    const traces = traceTemplatesEmpty();
    const layout = buildPlotlyLayout();
    chartLayouts.set(d, layout);
    Plotly.newPlot(div, traces, layout, { displayModeBar: false, responsive: true });
  }
  purgeAndNewPlotMetricsChartsSync();
}

/** 清空本地序列数据、图像与 Run 数值区，并重绘空图（保留当前 meta 中的 repo/backend 供参考） */
function clearClientDisplay() {
  stepReceiveCount = 0;
  resetSeries();
  el("prompt").textContent = "—";
  const imgB = el("imgBase");
  const imgW = el("imgWrist");
  if (imgB) imgB.removeAttribute("src");
  if (imgW) imgW.removeAttribute("src");
  el("runId").textContent = "-";
  el("episodeId").textContent = "-";
  el("globalIndex").textContent = "-";
  el("kInChunk").textContent = "-";
  el("inferMs").textContent = "-";
  el("mae").textContent = "-";
  el("mse").textContent = "-";
  resetHzEstimators();
  if (meta) {
    el("repoId").textContent = meta.repo_id ?? "-";
    el("backend").textContent = meta.backend ?? "-";
  }
  setProgress("已清空本页曲线与图像（WebSocket 未断开；可继续接收后续 step）。");
  purgeAndNewPlotAllChartsSync();
}

function raf() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

/** 先铺好 DOM，再分帧 newPlot，避免 6 个子图在同一次任务里卡死主线程导致「白屏很久」 */
async function initMetricsChartsInContainer() {
  const mcont = el("metricsChartsContainer");
  if (!mcont) return;
  mcont.innerHTML = "";
  metricsLayouts.clear();
  const jobs = [];
  for (const def of METRIC_CHART_DEFS) {
    const wrap = document.createElement("div");
    wrap.className = "chartDimWrap";
    const title = document.createElement("div");
    title.className = "chartDimTitle";
    title.textContent = `${def.cardTitle} vs global_index`;
    const plotDiv = document.createElement("div");
    plotDiv.id = def.id;
    plotDiv.className = "chart";
    wrap.appendChild(title);
    wrap.appendChild(plotDiv);
    mcont.appendChild(wrap);
    const layout = buildMetricsLayout(def.yLabel);
    metricsLayouts.set(def.id, layout);
    jobs.push([def.id, emptyMetricTrace(def), layout]);
  }
  await raf();
  for (const [id, traces, layout] of jobs) {
    Plotly.newPlot(id, traces, layout, { displayModeBar: false, responsive: true });
    await raf();
  }
}

async function initChart() {
  resetSeries();
  chartLayouts.clear();
  metricsLayouts.clear();
  await initMetricsChartsInContainer();
  const container = el("chartsContainer");
  container.innerHTML = "";

  const jobs = [];
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

    const traces = traceTemplatesEmpty();
    const layout = buildPlotlyLayout();
    chartLayouts.set(d, layout);
    jobs.push([chartDivId(d), traces, layout]);
  }

  await raf();
  for (const [id, traces, layout] of jobs) {
    Plotly.newPlot(id, traces, layout, { displayModeBar: false, responsive: true });
    await raf();
  }
  applyChartsCols();
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

  const met = event.metrics;
  let maeV = null;
  let mseV = null;
  if (met && typeof met === "object") {
    if (typeof met.mae === "number") maeV = met.mae;
    if (typeof met.mse === "number") mseV = met.mse;
  }
  state.mae.push(maeV);
  state.mse.push(mseV);
  while (state.mae.length > state.windowSize) state.mae.shift();
  while (state.mse.length > state.windowSize) state.mse.shift();

  const xArr = [...state.x];
  for (const d of state.dims) {
    const id = chartDivId(d);
    const gd = document.getElementById(id);
    if (!gd) continue;
    const gArr = state.gt.get(d);
    const pArr = state.pred.get(d);
    if (!gArr || !pArr) continue;
    try {
      if (!gd.data || gd.data.length < 2) {
        const layout = chartLayouts.get(d) || buildPlotlyLayout();
        chartLayouts.set(d, layout);
        Plotly.newPlot(gd, buildTracesForDim(d), layout, { displayModeBar: false, responsive: true });
        continue;
      }
      Plotly.restyle(gd, { x: [xArr, xArr], y: [[...gArr], [...pArr]] }, [0, 1]);
    } catch (e) {
      const layout = chartLayouts.get(d) || buildPlotlyLayout();
      chartLayouts.set(d, layout);
      Plotly.newPlot(gd, buildTracesForDim(d), layout, { displayModeBar: false, responsive: true });
    }
  }

  const maeArr = [...state.mae];
  const mseArr = [...state.mse];
  for (const def of METRIC_CHART_DEFS) {
    const gd = document.getElementById(def.id);
    if (!gd) continue;
    const yArr = def.seriesKey === "mae" ? maeArr : mseArr;
    try {
      if (!gd.data || gd.data.length < 1) {
        const layout = metricsLayouts.get(def.id) || buildMetricsLayout(def.yLabel);
        metricsLayouts.set(def.id, layout);
        const p = getPlotlyPalette();
        Plotly.newPlot(
          gd,
          [
            {
              x: [...state.x],
              y: yArr,
              mode: "lines",
              name: def.traceName,
              line: { width: 2, color: p[def.colorKey] },
            },
          ],
          layout,
          { displayModeBar: false, responsive: true }
        );
        continue;
      }
      Plotly.restyle(gd, { x: [xArr], y: [yArr] }, [0]);
    } catch (e) {
      const layout = metricsLayouts.get(def.id) || buildMetricsLayout(def.yLabel);
      metricsLayouts.set(def.id, layout);
      const p = getPlotlyPalette();
      Plotly.newPlot(
        gd,
        [
          {
            x: [...state.x],
            y: yArr,
            mode: "lines",
            name: def.traceName,
            line: { width: 2, color: p[def.colorKey] },
          },
        ],
        layout,
        { displayModeBar: false, responsive: true }
      );
    }
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

  wsConnectGeneration += 1;
  const myGen = wsConnectGeneration;

  if (ws) {
    try {
      ws.close();
    } catch (e) {
      /* ignore */
    }
    ws = null;
  }

  const socket = new WebSocket(url);
  ws = socket;

  socket.onopen = () => {
    if (myGen !== wsConnectGeneration || ws !== socket) return;
    clearReconnectTimer();
    connected = true;
    setBadge(true);
    el("btnConnect").disabled = true;
    el("btnDisconnect").disabled = false;
    setProgress("WebSocket 已连接，等待服务端 meta / 数据流…");
    setInferLoadingSub("已连接，等待服务端 meta…");
    setInferLoadingBanner(true);
    setInferPauseButtons(false);
  };

  socket.onclose = () => {
    if (myGen !== wsConnectGeneration) return;
    ws = null;
    connected = false;
    setBadge(false);
    el("btnConnect").disabled = false;
    el("btnDisconnect").disabled = true;
    setInferPauseButtons(false);
    setInferLoadingBanner(false);
    if (manualDisconnect) {
      setProgress("已断开（手动 Disconnect）。");
    } else {
      scheduleReconnect();
    }
  };

  socket.onerror = () => {
    // onclose will handle UI 与重连调度
  };

  socket.onmessage = (evt) => {
    if (myGen !== wsConnectGeneration || ws !== socket) return;
    let msg = null;
    try {
      msg = JSON.parse(evt.data);
    } catch (e) {
      return;
    }
    if (!msg || !msg.type) return;

    if (msg.type === "control_ack") {
      if (typeof msg.paused === "boolean") {
        setInferPauseButtons(msg.paused);
      }
      return;
    }

    if (msg.type === "meta") {
      if (msg.phase === "loading") {
        el("repoId").textContent = "…";
        el("backend").textContent = "…";
        el("prompt").textContent = "—";
        setProgress(msg.message || "服务端：正在加载数据集与策略…");
        setInferLoadingSub("服务端正在加载数据集与策略，请稍候…");
        setInferLoadingBanner(true);
        return;
      }
      meta = msg;
      stepReceiveCount = 0;
      resetSeries();
      resetHzEstimators();
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
      setInferLoadingSub("模型与数据已就绪，等待首次推理结果（step）推送…");
      setInferLoadingBanner(true);
      queueMicrotask(() => {
        purgeAndNewPlotAllChartsSync();
      });
      return;
    }

    if (msg.type === "done") {
      setInferLoadingBanner(false);
      setProgress(msg.message || `推理已结束 run_id=${msg.run_id ?? "-"}`);
      if (connected) setBadgeRunFinished();
      el("btnPauseInfer").disabled = true;
      el("btnResumeInfer").disabled = true;
      return;
    }

    if (msg.type === "error") {
      setInferLoadingBanner(false);
      setProgress(`错误：${msg.message || "服务器推理管线异常"}`);
      el("btnPauseInfer").disabled = true;
      el("btnResumeInfer").disabled = true;
      return;
    }

    if (msg.type === "step") {
      setInferLoadingBanner(false);
      updateTop(msg);
      updateProgressFromStep(msg);
      updateHzEstimatorsFromStep(msg);
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
  wsConnectGeneration += 1;
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
  setInferPauseButtons(false);
  setInferLoadingBanner(false);
}

async function applyDims() {
  const newWindow = Number.parseInt(el("windowSize").value, 10);
  if (Number.isFinite(newWindow) && newWindow >= 50) state.windowSize = newWindow;

  const dims = parseDims(el("dims").value);
  if (dims.length > 0) state.dims = dims;
  await initChart();
}

async function setup() {
  setBadge(false);
  loadChartsColsPreference();
  await initChart();

  el("btnConnect").addEventListener("click", () => connect());
  el("btnDisconnect").addEventListener("click", () => disconnect());
  el("btnPauseInfer").addEventListener("click", () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "control", action: "pause" }));
  });
  el("btnResumeInfer").addEventListener("click", () => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "control", action: "resume" }));
  });
  el("btnApplyDims").addEventListener("click", () => void applyDims());
  el("btnResetDisplay").addEventListener("click", () => clearClientDisplay());

  el("chartsCols").addEventListener("change", () => applyChartsCols());

  el("themeSelect").addEventListener("change", () => {
    applyTheme(el("themeSelect").value);
  });

  el("toggleWrist").addEventListener("change", () => {
    const show = el("toggleWrist").checked;
    el("imgWrist").style.display = show ? "block" : "none";
  });
}

async function bootstrap() {
  applyThemeFromStorage();
  // 与 setup 并行：不阻塞首屏；hint 稍后写入 ws 输入框即可
  const hint = loadDefaultWsUrl().catch(() => {});
  await setup();
  await hint;
}

bootstrap();

