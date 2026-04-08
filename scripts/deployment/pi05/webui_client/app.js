/* global Plotly */

let ws = null;
let meta = null;
let connected = false;
/** 由服务端 meta.compare_mode 设置：误差曲线为 PT / TRT / 两路差 三路 */
let compareMode = false;
let latestDimMsePctMean = null; // number[] | null
let latestDimRelP99 = null; // number[] | null
let latestDimMsePctMeanTrt = null; // number[] | null，仅 compare_mode
let latestDimRelP99Trt = null; // number[] | null，仅 compare_mode
let latestDimMsePtTrtDimMean = null; // number[] | null，各维累计 mean((pt-trt)^2)，仅 compare_mode

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
    predTrt: "#9333ea",
    comparePair: "#64748b",
    compareMseTrt: "#a855f7",
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
    predTrt: "#c084fc",
    comparePair: "#94a3b8",
    compareMseTrt: "#e879f9",
  },
};

const el = (id) => document.getElementById(id);

/** 各 dim 子图标题占位（与 updatePerDimMsePctFromStep 展示格式一致） */
function dimChartTitlePlaceholder(d) {
  const di = Number(d);
  if (compareMode) {
    return `Action dim ${di} · PT mse% - · PT p99 - · TRT mse% - · TRT p99 - · PT-TRT mse -`;
  }
  return `Action dim ${di} · mse_pct_mean - · rel_p99 -`;
}

function metaStr(v) {
  if (v === undefined || v === null) return "—";
  const s = String(v).trim();
  return s.length ? s : "—";
}

/** 根据 meta.tensorrt 显示 / 隐藏 TensorRT 引擎路径与各引擎文件名（仅 tensorrt 或 compare 模式有） */
function applyTensorrtMetaFromMsg(m) {
  const block = el("trtEnginesBlock");
  if (!block) return;
  const trt = m && m.tensorrt;
  if (!trt || typeof trt !== "object") {
    block.hidden = true;
    return;
  }
  block.hidden = false;
  const set = (id, v) => {
    const n = el(id);
    if (n) n.textContent = metaStr(v);
  };
  set("trtPrecision", trt.precision);
  set("trtEnginePath", trt.engine_path);
  set("trtVit", trt.vit_engine);
  set("trtLlm", trt.llm_engine);
  set("trtExpert", trt.expert_engine);
  set("trtDenoise", trt.denoise_engine);
  set("trtEmbedPrefix", trt.embed_prefix_engine);
}

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
    showlegend: compareMode,
    legend: compareMode
      ? { font: { size: 9, color: p.legend }, orientation: "h", y: 1.2, x: 0 }
      : undefined,
    font: { color: p.font, size: 11 },
    height: compareMode ? 220 : 200,
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
    const sk = def.seriesKey;
    let traces;
    if (!compareMode) {
      traces = [
        {
          x: [...state.x],
          y: [...state[sk]],
          mode: "lines",
          name: def.traceName,
          line: { width: 2, color: p[def.colorKey] },
        },
      ];
    } else {
      const c2 = sk === "mae" ? p.predTrt : p.compareMseTrt;
      const s2 = sk === "mae" ? "mae_trt" : "mse_trt";
      const s3 = sk === "mae" ? "mae_pt_trt" : "mse_pt_trt";
      traces = [
        {
          x: [...state.x],
          y: [...state[sk]],
          mode: "lines",
          name: `${sk}_pt`,
          line: { width: 2, color: p[def.colorKey] },
        },
        {
          x: [...state.x],
          y: [...state[s2]],
          mode: "lines",
          name: `${sk}_trt`,
          line: { width: 2, color: c2 },
        },
        {
          x: [...state.x],
          y: [...state[s3]],
          mode: "lines",
          name: `${sk}_pt_trt`,
          line: { width: 2, dash: "dot", color: p.comparePair },
        },
      ];
    }
    Plotly.newPlot(div, traces, layout, { displayModeBar: false, responsive: true });
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
  mae_trt: [],
  mse_trt: [],
  mae_pt_trt: [],
  mse_pt_trt: [],
  gt: new Map(), // dim -> []
  pred: new Map(), // dim -> []
  pred_trt: new Map(), // dim -> [] compare_mode
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
  state.mae_trt = [];
  state.mse_trt = [];
  state.mae_pt_trt = [];
  state.mse_pt_trt = [];
  state.gt = new Map();
  state.pred = new Map();
  state.pred_trt = new Map();
  for (const d of state.dims) {
    state.gt.set(d, []);
    state.pred.set(d, []);
    state.pred_trt.set(d, []);
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
  if (event.server_timing) {
    const st = event.server_timing;
    if (typeof st.infer_ms === "number") el("inferMs").textContent = st.infer_ms.toFixed(2);
    if (compareMode) {
      if (typeof st.infer_ms_pt === "number") el("inferMsPt").textContent = st.infer_ms_pt.toFixed(2);
      if (typeof st.infer_ms_trt === "number") el("inferMsTrt").textContent = st.infer_ms_trt.toFixed(2);
    }
  }
  if (event.metrics) {
    const m = event.metrics;
    if (typeof m.mae === "number") el("mae").textContent = m.mae.toFixed(6);
    if (typeof m.mse === "number") el("mse").textContent = m.mse.toFixed(6);
    if (compareMode) {
      if (typeof m.mae_trt === "number") el("maeTrt").textContent = m.mae_trt.toFixed(6);
      if (typeof m.mse_trt === "number") el("mseTrt").textContent = m.mse_trt.toFixed(6);
      if (typeof m.mae_pt_trt === "number") el("maePtTrt").textContent = m.mae_pt_trt.toFixed(6);
      if (typeof m.mse_pt_trt === "number") el("msePtTrt").textContent = m.mse_pt_trt.toFixed(6);
    }
  }
}

function setImage(imgEl, jpegB64) {
  if (!jpegB64) return;
  imgEl.src = `data:image/jpeg;base64,${jpegB64}`;
}

function chartDivId(d) {
  return `chart-dim-${d}`;
}

/** 与子图初始状态一致的空曲线（gt / pred_pt / 可选 pred_trt） */
function traceTemplatesEmpty() {
  const p = getPlotlyPalette();
  const traces = [
    { x: [], y: [], mode: "lines", name: "gt", line: { width: 2, color: p.gt } },
    { x: [], y: [], mode: "lines", name: "pred_pt", line: { width: 2, dash: "dot", color: p.pred } },
  ];
  if (compareMode) {
    traces.push({
      x: [],
      y: [],
      mode: "lines",
      name: "pred_trt",
      line: { width: 2, dash: "dash", color: p.predTrt },
    });
  }
  return traces;
}

/**
 * 强制 purge + newPlot，避免 Plotly.react 多次合并后出现多余 trace（例如 3 条线）。
 */
function emptyMetricTraces(def) {
  const p = getPlotlyPalette();
  const c1 = p[def.colorKey];
  if (!compareMode) {
    return [
      {
        x: [],
        y: [],
        mode: "lines",
        name: def.traceName,
        line: { width: 2, color: c1 },
      },
    ];
  }
  const c2 = def.seriesKey === "mae" ? p.predTrt : p.compareMseTrt;
  return [
    { x: [], y: [], mode: "lines", name: `${def.seriesKey}_pt`, line: { width: 2, color: c1 } },
    { x: [], y: [], mode: "lines", name: `${def.seriesKey}_trt`, line: { width: 2, color: c2 } },
    {
      x: [],
      y: [],
      mode: "lines",
      name: `${def.seriesKey}_pt_trt`,
      line: { width: 2, dash: "dot", color: p.comparePair },
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
    Plotly.newPlot(div, emptyMetricTraces(def), layout, { displayModeBar: false, responsive: true });
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
  latestDimMsePctMean = null;
  latestDimRelP99 = null;
  latestDimMsePctMeanTrt = null;
  latestDimRelP99Trt = null;
  latestDimMsePtTrtDimMean = null;
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
  el("inferMsPt").textContent = "-";
  el("inferMsTrt").textContent = "-";
  el("gpuUtil").textContent = "-";
  el("mae").textContent = "-";
  el("mse").textContent = "-";
  el("maeTrt").textContent = "-";
  el("mseTrt").textContent = "-";
  el("maePtTrt").textContent = "-";
  el("msePtTrt").textContent = "-";
  resetHzEstimators();
  // reset per-dim titles
  for (const d of state.dims) {
    const t = document.getElementById(`dimTitle-${d}`);
    if (t) t.textContent = dimChartTitlePlaceholder(d);
  }
  if (meta) {
    el("repoId").textContent = meta.repo_id ?? "-";
    el("backend").textContent = meta.backend ?? "-";
  }
  applyTensorrtMetaFromMsg(meta);
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
    jobs.push([def.id, emptyMetricTraces(def), layout]);
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
    title.id = `dimTitle-${d}`;
    title.textContent = dimChartTitlePlaceholder(d);
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

function updatePerDimMsePctFromStep(event) {
  const met = event.metrics;
  if (!met || typeof met !== "object") return;
  const arr = met.mse_pct_dim_mean;
  const p99 = met.rel_p99_dim;
  const arrTrt = met.mse_pct_dim_mean_trt;
  const p99Trt = met.rel_p99_dim_trt;
  const pairDm = met.mse_pt_trt_dim_mean;
  if (
    !Array.isArray(arr) &&
    !Array.isArray(p99) &&
    !Array.isArray(arrTrt) &&
    !Array.isArray(p99Trt) &&
    !Array.isArray(pairDm)
  ) {
    return;
  }
  if (Array.isArray(arr)) latestDimMsePctMean = arr;
  if (Array.isArray(p99)) latestDimRelP99 = p99;
  if (compareMode) {
    if (Array.isArray(arrTrt)) latestDimMsePctMeanTrt = arrTrt;
    if (Array.isArray(p99Trt)) latestDimRelP99Trt = p99Trt;
    if (Array.isArray(pairDm)) latestDimMsePtTrtDimMean = pairDm;
  }

  for (const d of state.dims) {
    const t = document.getElementById(`dimTitle-${d}`);
    if (!t) continue;
    const v = Array.isArray(arr) && typeof arr[d] === "number" ? arr[d] : null;
    const sPt = v === null ? "-" : `${v.toFixed(3)}%`;
    const rp = Array.isArray(p99) && typeof p99[d] === "number" ? p99[d] : null;
    const rsPt = rp === null ? "-" : `${(rp * 100.0).toFixed(2)}%`;
    if (compareMode && (Array.isArray(arrTrt) || Array.isArray(p99Trt) || Array.isArray(pairDm))) {
      const vt = Array.isArray(arrTrt) && typeof arrTrt[d] === "number" ? arrTrt[d] : null;
      const sTrt = vt === null ? "-" : `${vt.toFixed(3)}%`;
      const rpt = Array.isArray(p99Trt) && typeof p99Trt[d] === "number" ? p99Trt[d] : null;
      const rsTrt = rpt === null ? "-" : `${(rpt * 100.0).toFixed(2)}%`;
      const pm =
        Array.isArray(pairDm) && typeof pairDm[d] === "number" ? pairDm[d].toFixed(6) : "-";
      t.textContent = `Action dim ${d} · PT mse% ${sPt} · PT p99 ${rsPt} · TRT mse% ${sTrt} · TRT p99 ${rsTrt} · PT-TRT mse ${pm}`;
    } else {
      t.textContent = `Action dim ${d} · mse_pct_mean ${sPt} · rel_p99 ${rsPt}`;
    }
  }
}

function csvEscape(v) {
  const s = String(v ?? "");
  if (/[\",\\n]/.test(s)) return `"${s.replaceAll("\"", "\"\"")}"`;
  return s;
}

function downloadCsv(filename, rows) {
  const csv = rows.map((r) => r.map(csvEscape).join(",")).join("\n") + "\n";
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function downloadDimStatsCsv() {
  if (
    !latestDimMsePctMean &&
    !latestDimRelP99 &&
    !latestDimMsePctMeanTrt &&
    !latestDimRelP99Trt &&
    !latestDimMsePtTrtDimMean
  ) {
    setProgress("暂无可下载数据：请先接收至少一条 step。");
    return;
  }
  let rows;
  if (compareMode) {
    rows = [
      [
        "dim",
        "pt_mse_pct_mean_pct",
        "pt_rel_p99_pct",
        "trt_mse_pct_mean_pct",
        "trt_rel_p99_pct",
        "pt_trt_mse_mean",
      ],
    ];
    for (const d of state.dims) {
      const msePct =
        Array.isArray(latestDimMsePctMean) && typeof latestDimMsePctMean[d] === "number"
          ? latestDimMsePctMean[d]
          : "";
      const relP99Pct =
        Array.isArray(latestDimRelP99) && typeof latestDimRelP99[d] === "number"
          ? latestDimRelP99[d] * 100.0
          : "";
      const msePctTrt =
        Array.isArray(latestDimMsePctMeanTrt) && typeof latestDimMsePctMeanTrt[d] === "number"
          ? latestDimMsePctMeanTrt[d]
          : "";
      const relP99PctTrt =
        Array.isArray(latestDimRelP99Trt) && typeof latestDimRelP99Trt[d] === "number"
          ? latestDimRelP99Trt[d] * 100.0
          : "";
      const pairMse =
        Array.isArray(latestDimMsePtTrtDimMean) && typeof latestDimMsePtTrtDimMean[d] === "number"
          ? latestDimMsePtTrtDimMean[d]
          : "";
      rows.push([d, msePct, relP99Pct, msePctTrt, relP99PctTrt, pairMse]);
    }
  } else {
    rows = [["dim", "mse_pct_mean_pct", "rel_p99_pct"]];
    for (const d of state.dims) {
      const msePct =
        Array.isArray(latestDimMsePctMean) && typeof latestDimMsePctMean[d] === "number"
          ? latestDimMsePctMean[d]
          : "";
      const relP99Pct =
        Array.isArray(latestDimRelP99) && typeof latestDimRelP99[d] === "number"
          ? latestDimRelP99[d] * 100.0
          : "";
      rows.push([d, msePct, relP99Pct]);
    }
  }
  const rid = meta?.run_id ?? "no_run_id";
  const ts = new Date().toISOString().replaceAll(":", "-");
  downloadCsv(`pi05_webui_dim_stats_${rid}_${ts}.csv`, rows);
  setProgress(
    compareMode
      ? "已下载：各 dim 的 PT/TRT mse_pct、rel_p99 及 PT-TRT mse（CSV）。"
      : "已下载：各 action dim mse_pct_mean / rel_p99（CSV）。"
  );
}

function pushPoint(event) {
  const x = event.global_index;
  state.x.push(x);
  while (state.x.length > state.windowSize) state.x.shift();

  const gtAction = event.gt_action || [];
  const predAction = event.pred_action || [];
  const predTrtAction = event.pred_action_trt || [];

  for (const d of state.dims) {
    const gArr = state.gt.get(d);
    const pArr = state.pred.get(d);
    if (!gArr || !pArr) continue;
    gArr.push(gtAction[d] ?? null);
    pArr.push(predAction[d] ?? null);
    while (gArr.length > state.windowSize) gArr.shift();
    while (pArr.length > state.windowSize) pArr.shift();
    if (compareMode) {
      const tArr = state.pred_trt.get(d);
      if (tArr) {
        tArr.push(predTrtAction[d] ?? null);
        while (tArr.length > state.windowSize) tArr.shift();
      }
    }
  }

  const met = event.metrics;
  let maeV = null;
  let mseV = null;
  let maeTrtV = null;
  let mseTrtV = null;
  let maePtTrtV = null;
  let msePtTrtV = null;
  if (met && typeof met === "object") {
    if (typeof met.mae === "number") maeV = met.mae;
    if (typeof met.mse === "number") mseV = met.mse;
    if (compareMode) {
      if (typeof met.mae_trt === "number") maeTrtV = met.mae_trt;
      if (typeof met.mse_trt === "number") mseTrtV = met.mse_trt;
      if (typeof met.mae_pt_trt === "number") maePtTrtV = met.mae_pt_trt;
      if (typeof met.mse_pt_trt === "number") msePtTrtV = met.mse_pt_trt;
    }
  }
  state.mae.push(maeV);
  state.mse.push(mseV);
  while (state.mae.length > state.windowSize) state.mae.shift();
  while (state.mse.length > state.windowSize) state.mse.shift();
  if (compareMode) {
    state.mae_trt.push(maeTrtV);
    state.mse_trt.push(mseTrtV);
    state.mae_pt_trt.push(maePtTrtV);
    state.mse_pt_trt.push(msePtTrtV);
    while (state.mae_trt.length > state.windowSize) state.mae_trt.shift();
    while (state.mse_trt.length > state.windowSize) state.mse_trt.shift();
    while (state.mae_pt_trt.length > state.windowSize) state.mae_pt_trt.shift();
    while (state.mse_pt_trt.length > state.windowSize) state.mse_pt_trt.shift();
  }

  const xArr = [...state.x];
  const needTracesDim = compareMode ? 3 : 2;
  for (const d of state.dims) {
    const id = chartDivId(d);
    const gd = document.getElementById(id);
    if (!gd) continue;
    const gArr = state.gt.get(d);
    const pArr = state.pred.get(d);
    if (!gArr || !pArr) continue;
    try {
      if (!gd.data || gd.data.length < needTracesDim) {
        const layout = chartLayouts.get(d) || buildPlotlyLayout();
        chartLayouts.set(d, layout);
        Plotly.newPlot(gd, buildTracesForDim(d), layout, { displayModeBar: false, responsive: true });
        continue;
      }
      if (compareMode) {
        const tArr = state.pred_trt.get(d) || [];
        Plotly.restyle(gd, { x: [xArr, xArr, xArr], y: [[...gArr], [...pArr], [...tArr]] }, [0, 1, 2]);
      } else {
        Plotly.restyle(gd, { x: [xArr, xArr], y: [[...gArr], [...pArr]] }, [0, 1]);
      }
    } catch (e) {
      const layout = chartLayouts.get(d) || buildPlotlyLayout();
      chartLayouts.set(d, layout);
      Plotly.newPlot(gd, buildTracesForDim(d), layout, { displayModeBar: false, responsive: true });
    }
  }

  const maeArr = [...state.mae];
  const mseArr = [...state.mse];
  const maeTrtArr = [...state.mae_trt];
  const mseTrtArr = [...state.mse_trt];
  const maePtTrtArr = [...state.mae_pt_trt];
  const msePtTrtArr = [...state.mse_pt_trt];

  for (const def of METRIC_CHART_DEFS) {
    const gd = document.getElementById(def.id);
    if (!gd) continue;
    const sk = def.seriesKey;
    const yArr = sk === "mae" ? maeArr : mseArr;
    const yTrt = sk === "mae" ? maeTrtArr : mseTrtArr;
    const yPair = sk === "mae" ? maePtTrtArr : msePtTrtArr;
    const needTracesMet = compareMode ? 3 : 1;
    try {
      if (!gd.data || gd.data.length < needTracesMet) {
        const layout = metricsLayouts.get(def.id) || buildMetricsLayout(def.yLabel);
        metricsLayouts.set(def.id, layout);
        const p = getPlotlyPalette();
        let traces;
        if (!compareMode) {
          traces = [
            {
              x: xArr,
              y: yArr,
              mode: "lines",
              name: def.traceName,
              line: { width: 2, color: p[def.colorKey] },
            },
          ];
        } else {
          const c2 = sk === "mae" ? p.predTrt : p.compareMseTrt;
          traces = [
            { x: xArr, y: yArr, mode: "lines", name: `${sk}_pt`, line: { width: 2, color: p[def.colorKey] } },
            { x: xArr, y: yTrt, mode: "lines", name: `${sk}_trt`, line: { width: 2, color: c2 } },
            {
              x: xArr,
              y: yPair,
              mode: "lines",
              name: `${sk}_pt_trt`,
              line: { width: 2, dash: "dot", color: p.comparePair },
            },
          ];
        }
        Plotly.newPlot(gd, traces, layout, { displayModeBar: false, responsive: true });
        continue;
      }
      if (compareMode) {
        Plotly.restyle(gd, { x: [xArr, xArr, xArr], y: [yArr, yTrt, yPair] }, [0, 1, 2]);
      } else {
        Plotly.restyle(gd, { x: [xArr], y: [yArr] }, [0]);
      }
    } catch (e) {
      const layout = metricsLayouts.get(def.id) || buildMetricsLayout(def.yLabel);
      metricsLayouts.set(def.id, layout);
      const p = getPlotlyPalette();
      const traces = !compareMode
        ? [
            {
              x: xArr,
              y: yArr,
              mode: "lines",
              name: def.traceName,
              line: { width: 2, color: p[def.colorKey] },
            },
          ]
        : [
            {
              x: xArr,
              y: yArr,
              mode: "lines",
              name: `${sk}_pt`,
              line: { width: 2, color: p[def.colorKey] },
            },
            {
              x: xArr,
              y: yTrt,
              mode: "lines",
              name: `${sk}_trt`,
              line: { width: 2, color: sk === "mae" ? p.predTrt : p.compareMseTrt },
            },
            {
              x: xArr,
              y: yPair,
              mode: "lines",
              name: `${sk}_pt_trt`,
              line: { width: 2, dash: "dot", color: p.comparePair },
            },
          ];
      Plotly.newPlot(gd, traces, layout, { displayModeBar: false, responsive: true });
    }
  }
}

function buildTracesForDim(d) {
  const p = getPlotlyPalette();
  const traces = [
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
      name: "pred_pt",
      line: { width: 2, dash: "dot", color: p.pred },
    },
  ];
  if (compareMode) {
    traces.push({
      x: [...state.x],
      y: [...(state.pred_trt.get(d) || [])],
      mode: "lines",
      name: "pred_trt",
      line: { width: 2, dash: "dash", color: p.predTrt },
    });
  }
  return traces;
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

    if (msg.type === "gpu_stats") {
      const gu =
        typeof msg.gpu_util_pct === "number" ? msg.gpu_util_pct.toFixed(0) : "?";
      const mu =
        typeof msg.mem_util_pct === "number" ? msg.mem_util_pct.toFixed(0) : "?";
      const di =
        typeof msg.device_index === "number" ? `[${msg.device_index}] ` : "";
      el("gpuUtil").textContent = `${di}${gu}% · mem ${mu}%`;
      return;
    }

    if (msg.type === "meta") {
      if (msg.phase === "loading") {
        el("repoId").textContent = "…";
        el("backend").textContent = "…";
        const te = el("trtEnginesBlock");
        if (te) te.hidden = true;
        el("gpuUtil").textContent = "-";
        el("prompt").textContent = "—";
        setProgress(msg.message || "服务端：正在加载数据集与策略…");
        setInferLoadingSub("服务端正在加载数据集与策略，请稍候…");
        setInferLoadingBanner(true);
        return;
      }
      meta = msg;
      compareMode = Boolean(meta.compare_mode);
      const cmb = el("compareMetricsBlock");
      if (cmb) cmb.hidden = !compareMode;
      stepReceiveCount = 0;
      resetSeries();
      resetHzEstimators();
      latestDimMsePctMean = null;
      latestDimRelP99 = null;
      latestDimMsePctMeanTrt = null;
      latestDimRelP99Trt = null;
      latestDimMsePtTrtDimMean = null;
      el("runId").textContent = meta.run_id ?? "-";
      el("repoId").textContent = meta.repo_id ?? "-";
      el("backend").textContent = meta.backend ?? "-";
      applyTensorrtMetaFromMsg(meta);
      el("gpuUtil").textContent =
        typeof meta.gpu_stats_interval_sec === "number" && meta.gpu_stats_interval_sec > 0
          ? "…"
          : "—";
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
        for (const d of state.dims) {
          const t = document.getElementById(`dimTitle-${d}`);
          if (t) t.textContent = dimChartTitlePlaceholder(d);
        }
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
      updatePerDimMsePctFromStep(msg);
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
  el("btnDownloadStats").addEventListener("click", () => downloadDimStatsCsv());

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

