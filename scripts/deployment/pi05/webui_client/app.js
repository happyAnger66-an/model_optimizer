/* global Plotly */

let ws = null;
let meta = null;
let connected = false;
/** 由服务端 meta.compare_mode 设置：双路之一为 TensorRT */
let compareMode = false;
/** 由服务端 meta.ptq_compare 设置：双路之二为 PyTorch PTQ（与 compare_mode 互斥） */
let ptqCompareMode = false;

/** 服务端 load_infer_bundle 阶段：{ stage, message, done }[]，用于 Server 进度分步展示 */
let serverLoadStepList = [];
function dualPredMode() {
  return compareMode || ptqCompareMode;
}
let latestDimMsePctMean = null; // number[] | null
let latestDimRelP99 = null; // number[] | null
let latestDimMsePctMeanTrt = null; // number[] | null，仅 compare_mode
let latestDimRelP99Trt = null; // number[] | null，仅 compare_mode
let latestDimMsePtTrtDimMean = null; // number[] | null，各维累计 mean((pt-trt)^2)，仅 compare_mode

/** PTQ 分层报告：完整 layers，供模块名子串过滤 */
let ptqLayerReportLayersAll = null;

/** 非用户主动断开时，每 RECONNECT_MS 自动重连一次 */
const RECONNECT_MS = 3000;
let manualDisconnect = false;
let reconnectTimer = null;

/** 每次发起新连接自增，用于忽略已被替换或已放弃的旧 WebSocket 的 onopen/onclose */
let wsConnectGeneration = 0;

const THEME_STORAGE_KEY = "pi05_webui_theme";
const CHARTS_COLS_STORAGE_KEY = "pi05_webui_charts_cols";
/** 折叠误差曲线 / 各 dim 子图（1=折叠），减少 Plotly 每步重绘 */
const FOLD_METRICS_STORAGE_KEY = "pi05_webui_fold_metrics";
const FOLD_DIM_CHARTS_STORAGE_KEY = "pi05_webui_fold_dim_charts";
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

/** 误差曲线各 dim 折线颜色（与动作子图 dim 顺序一致） */
const METRIC_DIM_COLORS = [
  "#2563eb",
  "#dc2626",
  "#059669",
  "#d97706",
  "#7c3aed",
  "#db2777",
  "#0ea5e9",
  "#ca8a04",
  "#16a34a",
  "#4f46e5",
  "#ea580c",
  "#0d9488",
];

const el = (id) => document.getElementById(id);

function metricDimLineColor(i) {
  return METRIC_DIM_COLORS[i % METRIC_DIM_COLORS.length];
}

/** 各 dim 子图标题（compare 详细指标见「按维累计对比」表） */
function dimChartTitlePlaceholder(d) {
  const di = Number(d);
  if (dualPredMode()) {
    return `Action dim ${di}`;
  }
  return `Action dim ${di} · mse_pct_mean - · rel_p99 -`;
}

function resetCompareScalarTable() {
  for (const id of [
    "cmpMaePt",
    "cmpMsePt",
    "cmpMaeTrt",
    "cmpMseTrt",
    "cmpMaePair",
    "cmpMsePair",
    "cmpInferPt",
    "cmpInferTrt",
  ]) {
    const n = el(id);
    if (n) n.textContent = "-";
  }
}

/** compare 模式：Run 区隐藏单行 mae/mse，改用表；显示按维累计表 */
function setCompareLayoutVisible(on) {
  const ptBlock = el("runScalarPtBlock");
  const cmpBlock = el("compareMetricsBlock");
  const dimCard = el("compareDimTableCard");
  if (ptBlock) ptBlock.style.display = on ? "none" : "";
  if (cmpBlock) cmpBlock.hidden = !on;
  if (dimCard) {
    dimCard.hidden = !on;
    if (!on) {
      const tb = el("compareDimTableBody");
      if (tb) tb.innerHTML = "";
    }
  }
  if (!on) resetCompareScalarTable();
  else refreshCompareDimTable();
  updateMetricChartWrapTitles();
  setDimTraceToggleUi();
}

/** 按维累计表：直接显示服务端 JSON 中的原始数值（不做 %/科学计数转换） */
function fmtCompareDimRawCell(v) {
  if (typeof v !== "number" || !Number.isFinite(v)) return "—";
  return String(v);
}

/** 按当前 state.dims 与 latest* 缓存刷新「按维累计对比」tbody */
function refreshCompareDimTable() {
  const tbody = el("compareDimTableBody");
  const card = el("compareDimTableCard");
  if (!tbody || !card || !dualPredMode()) return;

  const arr = latestDimMsePctMean;
  const p99 = latestDimRelP99;
  const arrTrt = latestDimMsePctMeanTrt;
  const p99Trt = latestDimRelP99Trt;
  const pairDm = latestDimMsePtTrtDimMean;

  tbody.innerHTML = "";
  const dims = [...state.dims].sort((a, b) => a - b);
  for (const d of dims) {
    const tr = document.createElement("tr");
    const tdDim = document.createElement("td");
    tdDim.textContent = String(d);
    tr.appendChild(tdDim);

    const vPt = Array.isArray(arr) && typeof arr[d] === "number" ? arr[d] : null;
    const pPt = Array.isArray(p99) && typeof p99[d] === "number" ? p99[d] : null;
    const vTrt = Array.isArray(arrTrt) && typeof arrTrt[d] === "number" ? arrTrt[d] : null;
    const pTrt = Array.isArray(p99Trt) && typeof p99Trt[d] === "number" ? p99Trt[d] : null;
    const pM = Array.isArray(pairDm) && typeof pairDm[d] === "number" ? pairDm[d] : null;

    for (const text of [
      fmtCompareDimRawCell(vPt),
      fmtCompareDimRawCell(pPt),
      fmtCompareDimRawCell(vTrt),
      fmtCompareDimRawCell(pTrt),
      fmtCompareDimRawCell(pM),
    ]) {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

function fmtPtqLayerMetric(x) {
  if (typeof x !== "number" || !Number.isFinite(x)) return "—";
  const ax = Math.abs(x);
  if (ax === 0) return "0";
  if (ax < 1e-5 || ax >= 1e5) return x.toExponential(3);
  if (ax < 1e-2) return x.toExponential(3);
  return x.toFixed(6);
}

function resetPtqLayerReportDom() {
  ptqLayerReportLayersAll = null;
  const sumEl = el("ptqLayerReportSummary");
  const noteEl = el("ptqLayerReportNote");
  const body = el("ptqLayerReportBody");
  const heatRow = el("ptqLayerReportHeatRow");
  const filterInp = el("ptqLayerReportFilter");
  const filterRow = el("ptqLayerReportFilterRow");
  const filterHint = el("ptqLayerReportFilterHint");
  if (sumEl) sumEl.textContent = "—";
  if (noteEl) {
    noteEl.textContent = "";
    noteEl.hidden = true;
  }
  if (filterInp) filterInp.value = "";
  if (filterRow) filterRow.hidden = true;
  if (filterHint) filterHint.textContent = "";
  if (body) body.innerHTML = "";
  if (heatRow) {
    heatRow.innerHTML = "";
    heatRow.hidden = true;
    heatRow.setAttribute("aria-hidden", "true");
  }
}

/** 模块路径包含输入子串即保留（区分大小写，与 Excel「包含」一致） */
function filterPtqLayersByModuleSubstring(layers, rawQ) {
  const q = String(rawQ ?? "").trim();
  if (!q) return layers;
  return layers.filter((r) => {
    const mod = typeof r.module === "string" ? r.module : "";
    return mod.includes(q);
  });
}

function updatePtqLayerReportFilterHint(shown, total) {
  const hint = el("ptqLayerReportFilterHint");
  if (!hint) return;
  if (typeof total !== "number" || total <= 0) {
    hint.textContent = "";
    return;
  }
  const s = typeof shown === "number" ? shown : 0;
  hint.textContent = s === total ? `共 ${total} 层` : `显示 ${s} / ${total} 层`;
}

/** 根据当前过滤结果重绘条形区与表格（不改变摘要/说明） */
function renderPtqLayerReportHeatAndTable(layers) {
  const body = el("ptqLayerReportBody");
  const heatRow = el("ptqLayerReportHeatRow");
  if (!body) return;
  body.innerHTML = "";
  if (heatRow) {
    heatRow.innerHTML = "";
    heatRow.hidden = true;
    heatRow.setAttribute("aria-hidden", "true");
  }

  const totalAll =
    Array.isArray(ptqLayerReportLayersAll) && ptqLayerReportLayersAll.length
      ? ptqLayerReportLayersAll.length
      : layers.length;
  updatePtqLayerReportFilterHint(layers.length, totalAll);

  const topN = 6;
  const top = layers.slice(0, topN);
  const maxMse =
    top.length > 0
      ? Math.max(
          ...top.map((r) => (typeof r.mse_mean === "number" ? r.mse_mean : 0)),
          1e-30
        )
      : 0;

  if (heatRow && top.length > 0 && maxMse > 0) {
    heatRow.hidden = false;
    heatRow.setAttribute("aria-hidden", "false");
    const cap = document.createElement("div");
    cap.className = "sub muted";
    cap.style.marginBottom = "0.15rem";
    cap.textContent = `MSE 较高层（Top ${top.length}，条长 ∝ MSE）`;
    heatRow.appendChild(cap);
    for (const r of top) {
      const row = document.createElement("div");
      row.className = "ptqLayerReportBarRow";
      const lab = document.createElement("span");
      lab.className = "ptqLayerReportBarRowLabel";
      const fullMod = typeof r.module === "string" ? r.module : "";
      lab.title = fullMod;
      lab.textContent = fullMod ? fullMod.split(".").pop() || fullMod : "—";
      const track = document.createElement("div");
      track.className = "ptqLayerReportBarTrack";
      const fill = document.createElement("div");
      fill.className = "ptqLayerReportBarFill";
      const mse = typeof r.mse_mean === "number" ? r.mse_mean : 0;
      fill.style.width = `${Math.min(100, (mse / maxMse) * 100)}%`;
      track.appendChild(fill);
      const val = document.createElement("span");
      val.className = "ptqLayerReportBarVal";
      val.textContent = fmtPtqLayerMetric(mse);
      row.appendChild(lab);
      row.appendChild(track);
      row.appendChild(val);
      heatRow.appendChild(row);
    }
  }

  for (const r of layers) {
    const tr = document.createElement("tr");
    const mod = typeof r.module === "string" ? r.module : "—";
    const tdM = document.createElement("td");
    tdM.textContent = mod.length > 52 ? `${mod.slice(0, 50)}…` : mod;
    tdM.title = mod;
    tr.appendChild(tdM);
    for (const key of ["mse_mean", "mae_mean", "max_abs_mean"]) {
      const td = document.createElement("td");
      td.textContent = fmtPtqLayerMetric(r[key]);
      tr.appendChild(td);
    }
    const tdS = document.createElement("td");
    const sn = r.samples;
    tdS.textContent =
      typeof sn === "number" && Number.isFinite(sn) ? String(Math.round(sn)) : "—";
    tr.appendChild(tdS);
    body.appendChild(tr);
  }
}

function refreshPtqLayerReportFilteredTable() {
  if (!Array.isArray(ptqLayerReportLayersAll)) return;
  const filt = el("ptqLayerReportFilter");
  const q = filt ? filt.value : "";
  const filtered = filterPtqLayersByModuleSubstring(ptqLayerReportLayersAll, q);
  renderPtqLayerReportHeatAndTable(filtered);
}

/** ptq_compare：展示 meta.ptq_layer_report（条形摘要 + 全表） */
function renderPtqLayerReportFromMeta(m) {
  resetPtqLayerReportDom();
  const sumEl = el("ptqLayerReportSummary");
  const noteEl = el("ptqLayerReportNote");
  const body = el("ptqLayerReportBody");
  if (!sumEl || !body) return;

  const rep = m && m.ptq_layer_report;
  if (!rep || typeof rep !== "object") {
    ptqLayerReportLayersAll = null;
    sumEl.textContent = m.ptq_layer_report_path
      ? "已配置报告路径，但本次 meta 未附带 JSON（请确认服务端已更新）。"
      : "未生成分层报告：启动时添加 --ptq-layer-report-path；JSON 会随首次 meta 下发。";
    return;
  }
  if (rep.error) {
    ptqLayerReportLayersAll = null;
    sumEl.textContent = `读取报告失败：${rep.error}`;
    if (noteEl && rep.path) {
      noteEl.textContent = String(rep.path);
      noteEl.hidden = false;
    }
    return;
  }

  const parts = Array.isArray(rep.parts) ? rep.parts.join(", ") : "—";
  const isStart = rep.indices_start != null ? rep.indices_start : "—";
  const iu = rep.indices_used != null ? rep.indices_used : "—";
  const lc = rep.layer_count != null ? rep.layer_count : "—";
  sumEl.textContent = `parts: ${parts} · 起始下标 ${isStart} · 使用帧数 ${iu} · 统计层数 ${lc}`;

  if (rep.note && noteEl) {
    noteEl.textContent = String(rep.note);
    noteEl.hidden = false;
  }

  const layers = Array.isArray(rep.layers) ? rep.layers : [];
  ptqLayerReportLayersAll = layers;
  const filterInp = el("ptqLayerReportFilter");
  const filterRow = el("ptqLayerReportFilterRow");
  if (filterInp) filterInp.value = "";
  if (filterRow) filterRow.hidden = layers.length === 0;

  renderPtqLayerReportHeatAndTable(layers);
}

function syncPtqLayerReportCard() {
  const card = el("ptqLayerReportCard");
  if (!card) return;
  if (!ptqCompareMode) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  renderPtqLayerReportFromMeta(meta || {});
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

function buildMetricsLayout(yAxisTitle, options) {
  const showLegend = options?.showLegend === true;
  const p = getPlotlyPalette();
  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: showLegend ? { l: 52, r: 88, t: 8, b: 32 } : { l: 52, r: 12, t: 8, b: 32 },
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
    showlegend: showLegend,
    font: { color: p.font, size: 11 },
    height: showLegend ? 260 : 200,
  };
  if (showLegend) {
    layout.legend = {
      font: { size: 9, color: p.legend },
      orientation: "v",
      x: 1.02,
      y: 1,
      xref: "paper",
      yref: "paper",
      xanchor: "left",
      yanchor: "top",
    };
  }
  return layout;
}

function metricsChartLayout(def) {
  return buildMetricsLayout(metricsChartYAxisTitle(def), { showLegend: state.dims.length > 1 });
}

/** 误差曲线图 y 轴标题：双路时画 PT−第二路（TRT 或 PTQ） */
function metricsChartYAxisTitle(def) {
  if (!dualPredMode()) return def.yLabel;
  if (ptqCompareMode) return `PT−PTQ ${def.yLabel}`;
  return `PT−TRT ${def.yLabel}`;
}

function updateMetricChartWrapTitles() {
  for (const def of METRIC_CHART_DEFS) {
    const t = el(`metricTitle-${def.id}`);
    if (!t) continue;
    const pairLabel = ptqCompareMode ? "PT−PTQ" : "PT−TRT";
    const base = dualPredMode()
      ? `${pairLabel} ${def.cardTitle} vs global_index`
      : `${def.cardTitle} vs global_index`;
    t.textContent = state.dims.length > 1 ? `${base}（各 dim 一条线）` : base;
  }
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

function refreshDimChartsTheme() {
  if (isDimChartsFoldCollapsed()) {
    return;
  }
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
}

function refreshMetricsChartsTheme() {
  if (isMetricsFoldCollapsed()) {
    return;
  }
  for (const def of METRIC_CHART_DEFS) {
    const div = document.getElementById(def.id);
    if (!div) continue;
    try {
      Plotly.purge(div);
    } catch (e) {
      /* ignore */
    }
    const layout = metricsChartLayout(def);
    metricsLayouts.set(def.id, layout);
    const traces = buildMetricTracesFromState(def);
    Plotly.newPlot(div, traces, layout, { displayModeBar: false, responsive: true });
  }
}

function refreshChartsTheme() {
  refreshDimChartsTheme();
  refreshMetricsChartsTheme();
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
  /** 误差曲线：与 state.dims 对齐，每 dim 一条序列 */
  maePerDim: new Map(),
  msePerDim: new Map(),
  maePtTrtPerDim: new Map(),
  msePtTrtPerDim: new Map(),
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

function resetServerLoadSteps() {
  serverLoadStepList = [];
  const ul = el("serverProgressSteps");
  if (ul) {
    ul.innerHTML = "";
    ul.hidden = true;
  }
}

function renderServerLoadSteps() {
  const ul = el("serverProgressSteps");
  if (!ul) return;
  if (serverLoadStepList.length === 0) {
    ul.hidden = true;
    ul.innerHTML = "";
    return;
  }
  ul.hidden = false;
  ul.innerHTML = "";
  for (const s of serverLoadStepList) {
    const li = document.createElement("li");
    li.className = s.done ? "progressStepItem done" : "progressStepItem active";
    li.textContent = (s.done ? "✓ " : "→ ") + s.message;
    ul.appendChild(li);
  }
}

/** 处理 WebSocket type=server_progress（与 bundle.load_infer_bundle 中 on_progress 对应） */
function applyServerProgressMsg(msg) {
  const stage = msg.stage || "_";
  const message = msg.message || "";
  const active = serverLoadStepList.find((x) => !x.done);
  if (active && active.stage !== stage) {
    active.done = true;
  }
  let entry = serverLoadStepList.find((x) => x.stage === stage && !x.done);
  if (!entry) {
    entry = { stage, message, done: false };
    serverLoadStepList.push(entry);
  } else {
    entry.message = message;
  }
  renderServerLoadSteps();
  const run = serverLoadStepList.find((x) => !x.done);
  const line = el("serverProgress");
  if (line) {
    line.textContent = run ? run.message : message;
  }
}

function finalizeServerLoadSteps() {
  for (const s of serverLoadStepList) s.done = true;
  renderServerLoadSteps();
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
  state.maePerDim = new Map();
  state.msePerDim = new Map();
  state.maePtTrtPerDim = new Map();
  state.msePtTrtPerDim = new Map();
  state.gt = new Map();
  state.pred = new Map();
  state.pred_trt = new Map();
  for (const d of state.dims) {
    state.gt.set(d, []);
    state.pred.set(d, []);
    state.pred_trt.set(d, []);
    state.maePerDim.set(d, []);
    state.msePerDim.set(d, []);
    state.maePtTrtPerDim.set(d, []);
    state.msePtTrtPerDim.set(d, []);
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
    if (dualPredMode()) {
      const ipt = el("cmpInferPt");
      const itr = el("cmpInferTrt");
      if (ipt && typeof st.infer_ms_pt === "number") ipt.textContent = st.infer_ms_pt.toFixed(2);
      if (itr && typeof st.infer_ms_trt === "number") itr.textContent = st.infer_ms_trt.toFixed(2);
    }
  }
  if (event.metrics) {
    const m = event.metrics;
    const setNum = (id, v) => {
      const n = el(id);
      if (n && typeof v === "number") n.textContent = v.toFixed(6);
    };
    if (dualPredMode()) {
      setNum("cmpMaePt", m.mae);
      setNum("cmpMsePt", m.mse);
      if (compareMode) {
        setNum("cmpMaeTrt", m.mae_trt);
        setNum("cmpMseTrt", m.mse_trt);
        setNum("cmpMaePair", m.mae_pt_trt);
        setNum("cmpMsePair", m.mse_pt_trt);
      } else {
        setNum("cmpMaeTrt", m.mae_ptq);
        setNum("cmpMseTrt", m.mse_ptq);
        setNum("cmpMaePair", m.mae_pt_ptq);
        setNum("cmpMsePair", m.mse_pt_ptq);
      }
    } else {
      if (typeof m.mae === "number") el("mae").textContent = m.mae.toFixed(6);
      if (typeof m.mse === "number") el("mse").textContent = m.mse.toFixed(6);
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

function isMetricsFoldCollapsed() {
  const b = el("metricsFoldBody");
  return !!(b && b.hidden);
}

function isDimChartsFoldCollapsed() {
  const b = el("dimChartsFoldBody");
  return !!(b && b.hidden);
}

function setFoldUi(section, collapsed) {
  const isMetrics = section === "metrics";
  const btn = el(isMetrics ? "btnFoldMetrics" : "btnFoldDimCharts");
  const body = el(isMetrics ? "metricsFoldBody" : "dimChartsFoldBody");
  if (!btn || !body) return;
  body.hidden = !!collapsed;
  btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
  btn.textContent = collapsed ? "►" : "▼";
  try {
    localStorage.setItem(isMetrics ? FOLD_METRICS_STORAGE_KEY : FOLD_DIM_CHARTS_STORAGE_KEY, collapsed ? "1" : "0");
  } catch (e) {
    /* ignore */
  }
}

function loadFoldPreference(key) {
  try {
    const v = localStorage.getItem(key);
    return v === "1";
  } catch (e) {
    return false;
  }
}

function applyFoldPreferencesFromStorage() {
  setFoldUi("metrics", loadFoldPreference(FOLD_METRICS_STORAGE_KEY));
  setFoldUi("dim", loadFoldPreference(FOLD_DIM_CHARTS_STORAGE_KEY));
}

function purgeMetricsPlotDivsOnly() {
  for (const def of METRIC_CHART_DEFS) {
    const div = document.getElementById(def.id);
    if (!div) continue;
    try {
      Plotly.purge(div);
    } catch (e) {
      /* ignore */
    }
  }
}

function purgeDimPlotDivsOnly() {
  for (const d of state.dims) {
    const div = document.getElementById(chartDivId(d));
    if (!div) continue;
    try {
      Plotly.purge(div);
    } catch (e) {
      /* ignore */
    }
  }
}

/** 与各 dim 子图 trace 顺序一致：gt、pred_pt、[pred_trt] */
function getDimTraceVisibility() {
  const gt = el("dimShowGt")?.checked !== false;
  const predPt = el("dimShowPredPt")?.checked !== false;
  if (!dualPredMode()) return [gt, predPt];
  const predTrt = el("dimShowPredTrt")?.checked !== false;
  return [gt, predPt, predTrt];
}

function applyVisibilityToDimTraces(traces) {
  const vis = getDimTraceVisibility();
  for (let i = 0; i < traces.length; i++) {
    traces[i].visible = i < vis.length ? vis[i] !== false : true;
  }
}

/** 非 compare 时隐藏 pred_trt 勾选（该 trace 不存在） */
function setDimTraceToggleUi() {
  const wrap = el("dimShowPredTrtWrap");
  if (wrap) wrap.style.display = dualPredMode() ? "" : "none";
}

function restyleAllDimChartsVisibility() {
  const vis = getDimTraceVisibility();
  for (const d of state.dims) {
    const gd = document.getElementById(chartDivId(d));
    if (!gd || !gd.data || gd.data.length !== vis.length) continue;
    try {
      Plotly.restyle(gd, { visible: vis });
    } catch (e) {
      /* ignore */
    }
  }
}

/** dim -> 误差曲线是否显示（dims 变化时保留仍存在的 dim 的勾选） */
const metricDimCheckboxPreference = new Map();

function pruneMetricDimPreferences() {
  const ds = new Set(state.dims);
  for (const k of [...metricDimCheckboxPreference.keys()]) {
    if (!ds.has(k)) metricDimCheckboxPreference.delete(k);
  }
}

function getMetricDimVisibility() {
  return state.dims.map((d) => {
    const inp = el(`metricDimShow-${d}`);
    if (inp) return inp.checked === true;
    return metricDimCheckboxPreference.get(d) !== false;
  });
}

function applyVisibilityToMetricTraces(traces) {
  const vis = getMetricDimVisibility();
  for (let i = 0; i < traces.length; i++) {
    traces[i].visible = vis[i] !== false;
  }
}

function restyleAllMetricChartsVisibility() {
  const vis = getMetricDimVisibility();
  for (const def of METRIC_CHART_DEFS) {
    const gd = document.getElementById(def.id);
    if (!gd || !gd.data || gd.data.length !== vis.length) continue;
    try {
      Plotly.restyle(gd, { visible: vis });
    } catch (e) {
      /* ignore */
    }
  }
}

function renderMetricDimToggles() {
  const box = el("metricDimToggles");
  if (!box) return;
  pruneMetricDimPreferences();
  box.innerHTML = "";
  for (const d of state.dims) {
    if (!metricDimCheckboxPreference.has(d)) {
      metricDimCheckboxPreference.set(d, true);
    }
    const label = document.createElement("label");
    label.className = "checkbox metricDimToggleItem";
    const inp = document.createElement("input");
    inp.type = "checkbox";
    inp.id = `metricDimShow-${d}`;
    inp.checked = metricDimCheckboxPreference.get(d) !== false;
    const dimVal = d;
    inp.addEventListener("change", () => {
      metricDimCheckboxPreference.set(dimVal, inp.checked);
      restyleAllMetricChartsVisibility();
    });
    label.appendChild(inp);
    label.appendChild(document.createTextNode(` dim ${d}`));
    box.appendChild(label);
  }
}

/** 与子图初始状态一致的空曲线（gt / pred_pt / 可选 pred_trt） */
function traceTemplatesEmpty() {
  const p = getPlotlyPalette();
  const traces = [
    { x: [], y: [], mode: "lines", name: "gt", line: { width: 2, color: p.gt } },
    { x: [], y: [], mode: "lines", name: "pred_pt", line: { width: 2, dash: "dot", color: p.pred } },
  ];
  if (dualPredMode()) {
    traces.push({
      x: [],
      y: [],
      mode: "lines",
      name: "pred_trt",
      line: { width: 2, dash: "dash", color: p.predTrt },
    });
  }
  applyVisibilityToDimTraces(traces);
  return traces;
}

function buildMetricTracesFromState(def) {
  const xArr = [...state.x];
  const sk = def.seriesKey;
  const traces = [];
  for (let i = 0; i < state.dims.length; i++) {
    const d = state.dims[i];
    let yArr;
    if (!dualPredMode()) {
      yArr = sk === "mae" ? [...(state.maePerDim.get(d) || [])] : [...(state.msePerDim.get(d) || [])];
    } else {
      yArr = sk === "mae" ? [...(state.maePtTrtPerDim.get(d) || [])] : [...(state.msePtTrtPerDim.get(d) || [])];
    }
    traces.push({
      x: xArr,
      y: yArr,
      mode: "lines",
      name: `dim ${d}`,
      line: { width: 1.5, color: metricDimLineColor(i) },
    });
  }
  applyVisibilityToMetricTraces(traces);
  return traces;
}

/**
 * 强制 purge + newPlot，避免 Plotly.react 多次合并后出现多余 trace（例如 3 条线）。
 */
function emptyMetricTraces(def) {
  const traces = [];
  for (let i = 0; i < state.dims.length; i++) {
    const d = state.dims[i];
    traces.push({
      x: [],
      y: [],
      mode: "lines",
      name: `dim ${d}`,
      line: { width: 1.5, color: metricDimLineColor(i) },
    });
  }
  applyVisibilityToMetricTraces(traces);
  return traces;
}

function purgeAndNewPlotMetricsChartsSync() {
  if (isMetricsFoldCollapsed()) {
    purgeMetricsPlotDivsOnly();
    return;
  }
  for (const def of METRIC_CHART_DEFS) {
    const div = document.getElementById(def.id);
    if (!div) continue;
    try {
      Plotly.purge(div);
    } catch (e) {
      /* ignore */
    }
    const layout = metricsChartLayout(def);
    metricsLayouts.set(def.id, layout);
    Plotly.newPlot(div, emptyMetricTraces(def), layout, { displayModeBar: false, responsive: true });
  }
}

function purgeAndNewPlotDimChartsSync() {
  if (isDimChartsFoldCollapsed()) {
    purgeDimPlotDivsOnly();
    return;
  }
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
}

function purgeAndNewPlotAllChartsSync() {
  purgeAndNewPlotDimChartsSync();
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
  el("gpuUtil").textContent = "-";
  el("mae").textContent = "-";
  el("mse").textContent = "-";
  resetCompareScalarTable();
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
  refreshCompareDimTable();
  syncPtqLayerReportCard();
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
    title.id = `metricTitle-${def.id}`;
    title.textContent = `${def.cardTitle} vs global_index`;
    const plotDiv = document.createElement("div");
    plotDiv.id = def.id;
    plotDiv.className = "chart";
    wrap.appendChild(title);
    wrap.appendChild(plotDiv);
    mcont.appendChild(wrap);
    const layout = metricsChartLayout(def);
    metricsLayouts.set(def.id, layout);
    jobs.push([def.id, emptyMetricTraces(def), layout]);
  }
  if (!isMetricsFoldCollapsed()) {
    await raf();
    for (const [id, traces, layout] of jobs) {
      Plotly.newPlot(id, traces, layout, { displayModeBar: false, responsive: true });
      await raf();
    }
  } else {
    purgeMetricsPlotDivsOnly();
  }
  updateMetricChartWrapTitles();
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

  if (!isDimChartsFoldCollapsed()) {
    await raf();
    for (const [id, traces, layout] of jobs) {
      Plotly.newPlot(id, traces, layout, { displayModeBar: false, responsive: true });
      await raf();
    }
  } else {
    purgeDimPlotDivsOnly();
  }
  setDimTraceToggleUi();
  renderMetricDimToggles();
  applyChartsCols();
  refreshCompareDimTable();
}

function updatePerDimMsePctFromStep(event) {
  const met = event.metrics;
  if (!met || typeof met !== "object") return;
  const arr = met.mse_pct_dim_mean;
  const p99 = met.rel_p99_dim;
  const arrTrt = met.mse_pct_dim_mean_trt;
  const p99Trt = met.rel_p99_dim_trt;
  const pairDm = met.mse_pt_trt_dim_mean;
  const arrPtq = met.mse_pct_dim_mean_ptq;
  const p99Ptq = met.rel_p99_dim_ptq;
  const pairPpq = met.mse_pt_ptq_dim_mean;
  if (
    !Array.isArray(arr) &&
    !Array.isArray(p99) &&
    !Array.isArray(arrTrt) &&
    !Array.isArray(p99Trt) &&
    !Array.isArray(pairDm) &&
    !Array.isArray(arrPtq) &&
    !Array.isArray(p99Ptq) &&
    !Array.isArray(pairPpq)
  ) {
    return;
  }
  if (Array.isArray(arr)) latestDimMsePctMean = arr;
  if (Array.isArray(p99)) latestDimRelP99 = p99;
  if (compareMode) {
    if (Array.isArray(arrTrt)) latestDimMsePctMeanTrt = arrTrt;
    if (Array.isArray(p99Trt)) latestDimRelP99Trt = p99Trt;
    if (Array.isArray(pairDm)) latestDimMsePtTrtDimMean = pairDm;
  } else if (ptqCompareMode) {
    if (Array.isArray(arrPtq)) latestDimMsePctMeanTrt = arrPtq;
    if (Array.isArray(p99Ptq)) latestDimRelP99Trt = p99Ptq;
    if (Array.isArray(pairPpq)) latestDimMsePtTrtDimMean = pairPpq;
  }

  for (const d of state.dims) {
    const t = document.getElementById(`dimTitle-${d}`);
    if (!t) continue;
    if (dualPredMode()) {
      t.textContent = `Action dim ${d}`;
    } else {
      const v = Array.isArray(arr) && typeof arr[d] === "number" ? arr[d] : null;
      const sPt = v === null ? "-" : `${v.toFixed(3)}%`;
      const rp = Array.isArray(p99) && typeof p99[d] === "number" ? p99[d] : null;
      const rsPt = rp === null ? "-" : `${(rp * 100.0).toFixed(2)}%`;
      t.textContent = `Action dim ${d} · mse_pct_mean ${sPt} · rel_p99 ${rsPt}`;
    }
  }
  if (dualPredMode()) refreshCompareDimTable();
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
  if (compareMode || ptqCompareMode) {
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
    compareMode || ptqCompareMode
      ? "已下载：各 dim 的双路 mse_pct、rel_p99 及两路预测 mse（CSV）。"
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
  const predPtqAction = event.pred_action_ptq || [];
  const predSecond = ptqCompareMode ? predPtqAction : predTrtAction;

  for (const d of state.dims) {
    const gArr = state.gt.get(d);
    const pArr = state.pred.get(d);
    if (!gArr || !pArr) continue;
    gArr.push(gtAction[d] ?? null);
    pArr.push(predAction[d] ?? null);
    while (gArr.length > state.windowSize) gArr.shift();
    while (pArr.length > state.windowSize) pArr.shift();
    if (dualPredMode()) {
      const tArr = state.pred_trt.get(d);
      if (tArr) {
        tArr.push(predSecond[d] ?? null);
        while (tArr.length > state.windowSize) tArr.shift();
      }
    }
  }

  const met = event.metrics;
  if (met && typeof met === "object") {
    if (!dualPredMode()) {
      const ma = met.mae_per_dim;
      const ms = met.mse_per_dim;
      const fallbackMae = typeof met.mae === "number" ? met.mae : null;
      const fallbackMse = typeof met.mse === "number" ? met.mse : null;
      for (const d of state.dims) {
        const maeP = state.maePerDim.get(d);
        const mseP = state.msePerDim.get(d);
        if (!maeP || !mseP) continue;
        const maeVal = Array.isArray(ma) && typeof ma[d] === "number" ? ma[d] : fallbackMae;
        const mseVal = Array.isArray(ms) && typeof ms[d] === "number" ? ms[d] : fallbackMse;
        maeP.push(maeVal);
        mseP.push(mseVal);
        while (maeP.length > state.windowSize) maeP.shift();
        while (mseP.length > state.windowSize) mseP.shift();
      }
    } else if (compareMode) {
      const ma = met.mae_pt_trt_per_dim;
      const ms = met.mse_pt_trt_per_dim;
      const fallbackMae = typeof met.mae_pt_trt === "number" ? met.mae_pt_trt : null;
      const fallbackMse = typeof met.mse_pt_trt === "number" ? met.mse_pt_trt : null;
      for (const d of state.dims) {
        const maeP = state.maePtTrtPerDim.get(d);
        const mseP = state.msePtTrtPerDim.get(d);
        if (!maeP || !mseP) continue;
        const maeVal = Array.isArray(ma) && typeof ma[d] === "number" ? ma[d] : fallbackMae;
        const mseVal = Array.isArray(ms) && typeof ms[d] === "number" ? ms[d] : fallbackMse;
        maeP.push(maeVal);
        mseP.push(mseVal);
        while (maeP.length > state.windowSize) maeP.shift();
        while (mseP.length > state.windowSize) mseP.shift();
      }
    } else {
      const ma = met.mae_pt_ptq_per_dim;
      const ms = met.mse_pt_ptq_per_dim;
      const fallbackMae = typeof met.mae_pt_ptq === "number" ? met.mae_pt_ptq : null;
      const fallbackMse = typeof met.mse_pt_ptq === "number" ? met.mse_pt_ptq : null;
      for (const d of state.dims) {
        const maeP = state.maePtTrtPerDim.get(d);
        const mseP = state.msePtTrtPerDim.get(d);
        if (!maeP || !mseP) continue;
        const maeVal = Array.isArray(ma) && typeof ma[d] === "number" ? ma[d] : fallbackMae;
        const mseVal = Array.isArray(ms) && typeof ms[d] === "number" ? ms[d] : fallbackMse;
        maeP.push(maeVal);
        mseP.push(mseVal);
        while (maeP.length > state.windowSize) maeP.shift();
        while (mseP.length > state.windowSize) mseP.shift();
      }
    }
  }

  const xArr = [...state.x];

  if (!isDimChartsFoldCollapsed()) {
    const needTracesDim = dualPredMode() ? 3 : 2;
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
        const visDim = getDimTraceVisibility();
        if (dualPredMode()) {
          const tArr = state.pred_trt.get(d) || [];
          Plotly.restyle(gd, { x: [xArr, xArr, xArr], y: [[...gArr], [...pArr], [...tArr]] }, [0, 1, 2]);
          Plotly.restyle(gd, { visible: visDim });
        } else {
          Plotly.restyle(gd, { x: [xArr, xArr], y: [[...gArr], [...pArr]] }, [0, 1]);
          Plotly.restyle(gd, { visible: visDim });
        }
      } catch (e) {
        const layout = chartLayouts.get(d) || buildPlotlyLayout();
        chartLayouts.set(d, layout);
        Plotly.newPlot(gd, buildTracesForDim(d), layout, { displayModeBar: false, responsive: true });
      }
    }
  }

  if (!isMetricsFoldCollapsed()) {
    const needTracesMet = state.dims.length;
    for (const def of METRIC_CHART_DEFS) {
      const gd = document.getElementById(def.id);
      if (!gd) continue;
      const sk = def.seriesKey;
      const yPayload = state.dims.map((d) => {
        if (!dualPredMode()) {
          const arr = sk === "mae" ? state.maePerDim.get(d) : state.msePerDim.get(d);
          return [...(arr || [])];
        }
        const arr = sk === "mae" ? state.maePtTrtPerDim.get(d) : state.msePtTrtPerDim.get(d);
        return [...(arr || [])];
      });
      const layoutFallback = () => metricsLayouts.get(def.id) || metricsChartLayout(def);
      try {
        if (!gd.data || gd.data.length !== needTracesMet) {
          const layout = layoutFallback();
          metricsLayouts.set(def.id, layout);
          Plotly.newPlot(gd, buildMetricTracesFromState(def), layout, { displayModeBar: false, responsive: true });
          continue;
        }
        const xPay = state.dims.map(() => xArr);
        Plotly.restyle(gd, { x: xPay, y: yPayload }, state.dims.map((_, i) => i));
        Plotly.restyle(gd, { visible: getMetricDimVisibility() });
      } catch (e) {
        const layout = layoutFallback();
        metricsLayouts.set(def.id, layout);
        Plotly.newPlot(gd, buildMetricTracesFromState(def), layout, { displayModeBar: false, responsive: true });
      }
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
  if (dualPredMode()) {
    traces.push({
      x: [...state.x],
      y: [...(state.pred_trt.get(d) || [])],
      mode: "lines",
      name: "pred_trt",
      line: { width: 2, dash: "dash", color: p.predTrt },
    });
  }
  applyVisibilityToDimTraces(traces);
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
    resetServerLoadSteps();
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

    if (msg.type === "server_progress") {
      applyServerProgressMsg(msg);
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
        resetServerLoadSteps();
        el("repoId").textContent = "…";
        el("backend").textContent = "…";
        compareMode = false;
        ptqCompareMode = false;
        setCompareLayoutVisible(false);
        const prc = el("ptqLayerReportCard");
        if (prc) prc.hidden = true;
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
      ptqCompareMode = Boolean(meta.ptq_compare);
      setCompareLayoutVisible(dualPredMode());
      syncPtqLayerReportCard();
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
      finalizeServerLoadSteps();
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
        updateMetricChartWrapTitles();
        setDimTraceToggleUi();
        renderMetricDimToggles();
        refreshCompareDimTable();
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
  applyFoldPreferencesFromStorage();
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
  const ptqFilt = el("ptqLayerReportFilter");
  if (ptqFilt) {
    ptqFilt.addEventListener("input", () => refreshPtqLayerReportFilteredTable());
  }
  el("btnDownloadStats").addEventListener("click", () => downloadDimStatsCsv());

  el("chartsCols").addEventListener("change", () => applyChartsCols());

  for (const id of ["dimShowGt", "dimShowPredPt", "dimShowPredTrt"]) {
    const inp = el(id);
    if (inp) inp.addEventListener("change", () => restyleAllDimChartsVisibility());
  }

  {
    const bm = el("btnFoldMetrics");
    if (bm) {
      bm.addEventListener("click", () => {
        const collapsed = !isMetricsFoldCollapsed();
        setFoldUi("metrics", collapsed);
        if (collapsed) {
          purgeMetricsPlotDivsOnly();
        } else {
          purgeAndNewPlotMetricsChartsSync();
          renderMetricDimToggles();
          updateMetricChartWrapTitles();
        }
      });
    }
    const bd = el("btnFoldDimCharts");
    if (bd) {
      bd.addEventListener("click", () => {
        const collapsed = !isDimChartsFoldCollapsed();
        setFoldUi("dim", collapsed);
        if (collapsed) {
          purgeDimPlotDivsOnly();
        } else {
          purgeAndNewPlotDimChartsSync();
          setDimTraceToggleUi();
        }
      });
    }
  }

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

