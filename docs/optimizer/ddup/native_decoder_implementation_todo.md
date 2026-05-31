# Pi0.5 Native Decoder 实施跟踪（不依赖 FlashRT）

> 目标：在 `model_optimizer` 内实现 `expert + denoise` 原生高性能后端（不依赖 FlashRT 代码），并支持基于真实数据的量化校准闭环。  
> 范围约束：**不通过减少 token / 减少 denoise 步数** 获得收益。

---

## 1. 背景与当前性能基线

最近实测（已开启 ViT multi-view batching）：

- `e2e/chunk`: `143.79 ms`
- `predict_ms`: `109.53 ms`
- `load_ms`: `28.91 ms`
- `post_ms`: `5.28 ms`
- `python_overhead`: `34.26 ms (23.8%)`
- `llm.execute`: `34.73 ms`
- `denoise.execute`: `4.24 ms / step`（10 步约 42ms 级别）
- `embed_prefix`: `13.86 -> 8.72 ms`（已优化完成）

结论（仅看模型链路）：

1. 下一阶段最大优化空间在 `expert + denoise`（launch-bound）。
2. 继续优化 ViT 的边际收益已下降，应该切换到 decoder 路径。
3. 需要将“调度/图执行”与“量化校准”一起设计，避免只做算子微优化。

---

## 2. 目标架构（MVP 到完整体）

### 2.1 核心方向

- 新增 `native`（或 `mo_native`）后端，接入现有 stages 路由。
- 首版阶段矩阵建议：
  - `vit=tensorrt`
  - `llm=tensorrt`
  - `expert=native`
  - `denoise=native`
- `native` 路径先聚焦：
  1) 静态 buffer 池  
  2) denoise 全循环 CUDA Graph capture/replay  
  3) 最小化 Python/launch 调度开销

### 2.2 量化闭环

- 新增 native decoder 校准链路：
  - 分层采样（episode × frame_position × denoise_step）
  - percentile 统计（默认 99.9）
  - 导出 `NativeQuantSpec`（json sidecar）
  - 支持真实数据 recalibration（可配置）

---

## 3. 分阶段实施分析

### Phase A（MVP，先拿性能）

目标：先跑通 `expert/denoise -> native`，拿到可见 `predict_ms` 收益。

- 交付重点：
  - native executor + runner
  - graph capture + static buffers
  - webui/standalone 接线
  - 统一 perf summary 输出

风险：

- shape 漂移导致 graph 失效（需 fallback 机制）
- 与现有 TRT 阶段混合时的 stream 同步处理

验收门槛：

- `predict_ms` 相比当前基线下降（以同数据集、同配置 A/B）
- 功能正确（动作输出无 NaN，流程稳定跑完）

### Phase B（量化校准闭环）

目标：在 native 路径上支持数据驱动量化与可复现规格输出。

- 交付重点：
  - native calibrator
  - `NativeQuantSpec` 导出/加载
  - recalibration 开关

风险：

- step 分布不均导致后段误差放大
- 校准样本构成与线上分布偏移

验收门槛：

- 精度指标不低于当前 TRT 路径约定门限
- 规格文件可复现加载，重复推理差异可控

### Phase C（热点算子深化）

目标：逐步将最热点子路径替换为 native kernel/更优实现。

- 交付重点：
  - kernel 接口层稳定
  - 分模块灰度开关
  - profile 持续闭环

风险：

- 早期过度 kernel 化导致维护成本飙升
- 与校准规格不一致引发精度问题

验收门槛：

- 每个热点替换项均有独立 A/B 数据与回滚开关

---

## 4. 文件实施清单（长期跟踪）

> 状态建议：`TODO / IN_PROGRESS / DONE / BLOCKED`

### 4.1 配置与路由

| 文件 | 变更点 | 状态 |
|---|---|---|
| `src/model_optimizer/infer/server/config.py` | 新增 `NativeConfig`，扩展 mode/stage backend 枚举 | TODO |
| `src/model_optimizer/infer/server/policy_loader.py` | 新增 `_mount_native`，支持 stages 路由到 native | TODO |
| `src/model_optimizer/infer/server/backends/native.py` | 新增 `SingleNativeBackend` | TODO |
| `src/model_optimizer/infer/server/backends/__init__.py` | 导出 native backend | TODO |

### 4.2 Native 执行器与运行时

| 文件 | 变更点 | 状态 |
|---|---|---|
| `src/model_optimizer/infer/native/pi05_executor.py` | 新增 `Pi05NativeExecutor`，替换 expert/denoise 调用 | TODO |
| `src/model_optimizer/infer/native/decoder_runner.py` | 新增 `NativeDenoiseLoopRunner`（run/warmup） | TODO |
| `src/model_optimizer/infer/native/graph_capture.py` | 新增 capture/replay 与 signature cache | TODO |
| `src/model_optimizer/infer/native/buffers.py` | 新增静态 buffer 池管理 | TODO |
| `src/model_optimizer/infer/native/profile.py` | 新增 native 分层 profile 统计 | TODO |

### 4.3 WebUI / 脚本接线

| 文件 | 变更点 | 状态 |
|---|---|---|
| `scripts/deployment/pi05/lerobot_eval_webui/config.py` | 新增 native 配置项 | TODO |
| `scripts/deployment/pi05/lerobot_eval_webui/bundle.py` | 新增 `inference_mode=native` 分支 | TODO |
| `scripts/deployment/pi05/lerobot_eval_webui/chunk_infer.py` | 汇总中加入 native 路径分项 | TODO |
| `scripts/deployment/pi05/standalone_inference_script.py` | 新增 native 模式参数与加载逻辑 | TODO |

### 4.4 量化与校准

| 文件 | 变更点 | 状态 |
|---|---|---|
| `src/model_optimizer/quantization/native_decoder/spec.py` | 新增 `NativeQuantSpec` schema | TODO |
| `src/model_optimizer/quantization/native_decoder/calibrator.py` | 新增 native 校准入口 | TODO |
| `src/model_optimizer/quantization/native_decoder/collectors.py` | 新增 step/layer 统计聚合 | TODO |
| `src/model_optimizer/quantization/native_decoder/export.py` | 导出 quant spec + 报告 | TODO |
| `src/model_optimizer/calibrate/pi05_calib_load.py` | 新增 decoder 分层采样模式 | TODO |
| `src/model_optimizer/quantization/cli.py` | 新增 native quant spec 相关 CLI 参数 | TODO |

### 4.5 测试与文档

| 文件 | 变更点 | 状态 |
|---|---|---|
| `tests/infer/native/test_native_executor.py` | 新增执行器基本链路测试 | TODO |
| `tests/infer/native/test_graph_capture.py` | 新增 graph capture/replay 测试 | TODO |
| `tests/quantization/native_decoder/test_calibrator.py` | 新增校准/规格导出测试 | TODO |
| `tests/deployment/test_webui_native_mode.py` | 新增 webui native 模式集成测试 | TODO |
| `docs/optimizer/roadmap.md` | 同步 native 路线进展与里程碑 | TODO |
| `config/webui_configs/native_full.yaml` | 新增 native 全链路示例配置 | TODO |
| `config/webui_configs/native_hybrid.yaml` | 新增 native+trt 混合示例配置 | TODO |

---

## 5. TODO（可执行任务单）

## T0（架构准备）
- [ ] `T0-1`：确定 `native` 命名（`native` or `mo_native`），冻结配置字段。
- [ ] `T0-2`：确定 MVP shape 约束（batch、horizon、denoise steps 固定/动态策略）。

## T1（MVP 跑通）
- [ ] `T1-1`：落地 `Pi05NativeExecutor` + `NativeDenoiseLoopRunner`。
- [ ] `T1-2`：落地 denoise 全循环 CUDA Graph capture/replay。
- [ ] `T1-3`：接线 webui/standalone，支持 native 模式运行。
- [ ] `T1-4`：补齐最终汇总中的 native 分解耗时。

**T1 验收**：
- [ ] 同数据同配置 A/B，`predict_ms` 显著下降。
- [ ] 推理流程稳定，无 NaN/崩溃。

## T2（量化校准闭环）
- [ ] `T2-1`：实现分层采样校准（episode/frame/step）。
- [ ] `T2-2`：导出并加载 `NativeQuantSpec`。
- [ ] `T2-3`：实现可选真实数据 recalibration。
- [ ] `T2-4`：补齐 CLI 参数与文档示例。

**T2 验收**：
- [ ] 精度不低于当前门限。
- [ ] 规格可复现、可追踪（版本号 + 配置摘要）。

## T3（热点深化）
- [ ] `T3-1`：建立 kernel 接口层 + fallback 路径。
- [ ] `T3-2`：按 profile 排名前 1/2 热点逐项替换，逐项 A/B。
- [ ] `T3-3`：完善回滚开关，确保线上可控。

**T3 验收**：
- [ ] 每一项替换都有独立收益报告与回归结论。

---

## 6. 里程碑与记录

| 里程碑 | 目标日期 | 实际日期 | 结果 | 备注 |
|---|---:|---:|---|---|
| M1 |  |  |  | native MVP 跑通 |
| M2 |  |  |  | native quant spec 闭环 |
| M3 |  |  |  | 第一批热点替换完成 |

### 变更日志

| 日期 | 变更 | 人员 |
|---|---|---|
| 2026-05-31 | 初版：实施分析 + 文件清单 + TODO |  |

