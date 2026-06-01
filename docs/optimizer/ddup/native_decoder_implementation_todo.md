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
| 2026-06-01 | 新增 §7：DenoiseBackend 接口抽象 + 与 TRT engine 对齐 + flash_rt 降级可选 |  |

---

## 7. 架构抽象：DenoiseBackend（与 TRT denoise engine 接口对齐）

### 7.1 设计约定（再次确认）

- **不依赖 `flash_rt` 库**：把 FlashRT 的 denoise *运行时设计*（静态 buffer、AdaRMS modulation
  预计算、整步 Graph replay、静态 FP8 scale）迁移进 `model_optimizer` 自实现，计算图复用本仓库
  `Pi05DenoiseStep` / expert 前向。
- `flash_rt` 直连前端（`infer/flash/policy_adapter.py`、`backends/flashrt.py`、
  `backends/pt_flashrt_compare.py`）**降级为可选实验路径**，仅在显式 `mode=flashrt` /
  `mode=pt_flashrt_compare` 时加载；**不再**由 stages 矩阵自动接管整条流水线。

### 7.2 阶段后端矩阵（其它阶段可继续用 TensorRT）

pi05 的 5 个阶段（`vit / embed_prefix / llm / expert / denoise`）可各自独立选后端。
典型组合（即本次目标）：

| 阶段 | 后端 | 入口 |
|---|---|---|
| vit / embed_prefix / llm | tensorrt | `_mount_tensorrt_engines`（webui: `load_tensorrt_engines`） |
| expert / denoise | native | `_mount_native`（webui: `native_overlay_on_tensorrt`） |

- server 路径：`mode=tensorrt` + `stages.denoise=native` → 先挂 TRT 引擎，再用
  `_mount_native` 覆盖 `denoise_step`（native 后挂，胜出）。
- webui 路径：`tensorrt_native_denoise.yaml`（`inference_mode=tensorrt` +
  `native_overlay_on_tensorrt=true` + `native_enable_denoise=true`）。

### 7.3 统一接口（核心）

`DenoiseBackend` 规范调用接口**与 TRT denoise engine 完全一致**，使 TRT/native 在 `denoise_step`
hook 处可互换：

```
backend(prefix_pad_masks, past_keys, past_values, x_t, timestep) -> v_t
```

- `past_keys/past_values` 为堆叠张量 `[num_layers, batch, prefix_len, head_dim]`，与
  `Pi05TensorRTExecutor._stack_past_key_value_tensors` 一致。
- 实现：`infer/native/denoise_backend.py`
  - `DenoiseBackend`（ABC）：`__call__` + `calibrate` + `dump_summary`
  - `NativeDenoiseBackend`：包装 `NativeDenoiseLoopRunner`（eager + 可选 Graph）+
    `NativeQuantRuntime`（静态 scale + 可选在线重标定）；内部把堆叠 KV 还原为 `DynamicCache`
    复用原生 `denoise_step` 计算。
- `Pi05NativeExecutor.denoise_step_native` hook 把模型给的 `DynamicCache` 堆叠成 TRT 风格张量后
  调用 backend，从而 native 与 TRT 在该 hook 处真正等价、可由配置切换。

### 7.4 降 launch 开销的机制（分层、可插拔）

> 约定：不依赖 flash_rt、不强制 torch cuda graph。机制做成可插拔层，按收益逐步替换：

1. L0（已就位）：eager 复用 `Pi05DenoiseStep`，接口对齐 TRT。
2. L1：`NativeDenoiseLoopRunner` 整步/整循环 Graph replay（按 signature 缓存、可全局降级 eager）。
3. L2：热点算子用自研融合 kernel（`kernelSrc` Triton/CuTe）替换，逐项 A/B（T3）。

### 7.5 量化校准（数据驱动，仓内闭环）

- 离线：`quantization/native_decoder/` 用真实数据采样统计 → `NativeDecoderQuantSpec`（JSON）。
- 在线：`NativeQuantRuntime` 加载 spec 给输入做静态 fake-quant；可选用真实样本做轻量重标定
  （EMA 更新 active amax），影响后续推理 scale。

---

## 8. FlashRT decoder 直接移植（不 import flash_rt 包）

> 决策：denoise 不再走 graph-replay 试验路线，**直接把 FlashRT 的 Pi0.5 Thor decoder 实现
> 搬进 `model_optimizer`**，并支持离线量化。源码读取自 `third_party/FlashRT` /
> `~/codes/FlashRT`。

### 8.1 已落地（vendored 叶子，纯指针编排，编译通过）

`src/model_optimizer/infer/native/flashrt_decoder/`：

| 文件 | 对应 FlashRT 源 | 说明 |
|---|---|---|
| `pipeline.py` | `models/pi05/pipeline_thor.py` | `decoder_forward` / `_decoder_forward_fp16` / `decoder_forward_calibrate` 逐字移植，`fvk` 注入 |
| `cuda_helpers.py` | `hardware/thor/shared_primitives.py`（GPU 指针工具）| ctypes + libcudart，无 flash_rt 依赖 |
| `kernels.py` | `flash_rt.flash_rt_kernels`（.so）| 按 `.so` 路径 `importlib` 加载，**不 import flash_rt 包** |
| `driver.py` | `frontends/torch/pi05_thor.py`（AE 驱动）| `build_decoder_dims` + `DecoderBuffers` + `DecoderWeights` + `fill_prefix_kv_from_trt`（方案B KV 适配器）+ `Pi05ThorDecoderLoop`（整循环） |
| `weights.py` | `_pi05_thor_spec.build_spec` + `executors/torch_weights.py` + `core/thor_frontend_utils.py` | **权重 repack（已落地）**：`interleave_qk`/`quant_fp8` + 每层 `dec_{qkv,o,gu,d}_flat` FP8 + `_ae_w_scales` + 单例 `ain/aow/aob`（烘 -1/steps）|
| `precompute.py` | `pi05_thor.py` set_prompt 段 | **AdaRMS 预计算（已落地）**：`build_dec_rope`（suffix RoPE 表）+ `precompute_adarms_styles`（sa/sf/fs）|
| `backend.py` | （装配层，新增）| `FlashRtDecoderBackend`：repack+预计算+driver+离线量化 scale 导出/加载，`setup_prompt/run/calibrate/save_act_scales/load_act_scales` |

### 8.2 两个硬事实（决定整体工作量）

1. **kernel 不可仓内重写**：FlashRT 算子（`fp8_gemm_descale_fp16` / `fused_adarms_fp8_static_fp16`
   / `attention_qkv_fp16` …）由其 `csrc`（≈27MB / 700+ 文件，含 CUTLASS FMHA、FP8/FP4 GEMM、
   megakernel）在 **Thor(SM110)** 上 cmake 构建为 `flash_rt_kernels*.so`（+ `libfmha_fp16_strided.so`）。
   本仓库**复用其编译产物**（按路径加载），不重写 CUDA，也不 import flash_rt python 包。
   - 构建：`cmake --build build -j --target flash_rt_kernels`
   - 装载：`MO_FLASHRT_BUILD_DIR` / `MO_FLASHRT_KERNELS_SO` / `MO_FLASHRT_FMHA_SO`

2. **KV-cache 契约不一致**：FlashRT decoder 读取的 prefix KV（`Kc/Vc`）是其 **encoder 用
   `qkv_split_rope_kvcache_fp16` 写入的 FlashRT 私有 FP8/RoPE/offset 布局**。
   TRT 的 llm 引擎产出的是 bf16 stacked KV，布局/精度/RoPE 约定都不同。
   → 「vit/llm=TRT + denoise=FlashRT」并非干净 drop-in，必须二选一：
   - **(A) 同时 vendored FlashRT encoder**：prefix（siglip+llm）也走 FlashRT，KV 自洽；
     此时 vit/llm **不再是 TRT**（与 §7.2 矩阵冲突）。
   - **(B) 写 KV 适配器**：把 TRT bf16 KV 转成 FlashRT decoder 期望的布局/scale 后喂入；
     保留 vit/llm=TRT，但需要在 Thor 上验证数值一致性（RoPE 是否已应用、KV 量化点等）。

### 8.3 移植闭环进展

- [x] **权重 repack** → `weights.py`：每层 `dec_{qkv,o,gu,d}_flat` FP8 + `_ae_w_scales`（q→o→gu→d/层）+
  单例 `ain/aow/aob`。与 `pipeline.decoder_forward` 的 GEMM K/N 步进逐一核对一致。
- [x] **AdaRMS 预计算 + RoPE 表** → `precompute.py`：`sa/sf/fs`（`[steps,layers,S,3D]` 展平）+ `dec_rope`。
- [x] **驱动 / dims / buffers** → `driver.py`：`build_decoder_dims` + `DecoderBuffers` + `DecoderWeights`。
- [x] **装配 + 离线量化** → `backend.py`：`FlashRtDecoderBackend`（`calibrate` 导 act scales，启动 `load_act_scales`）。
- [x] **接入 `sample_actions`** → `pi05_executor.py::_install_flashrt_loop_runtime`：prefix（vit/llm=TRT）→ prefix KV
  →（方案B 适配）→ `Pi05ThorDecoderLoop.run` 整 10 步替代 denoise loop；失败安全回退原 `sample_actions`。
- [x] 配置接入：`native_flashrt_*`（config.py/native_backend.py/bundle.py）+ 示例 `config/webui_configs/tensorrt_flashrt_denoise.yaml`。
- [ ] **Thor 上端到端验证**：kernel `.so` 构建后跑通；重点验证 §8.2 的 RoPE 约定 / KV 数值一致性 + 精度对齐。
- [ ] 方案 (A) 可选：vendored encoder（若放弃 vit/llm=TRT、追求 KV 完全自洽）。

> 用法：`native_flashrt_decoder: true` + `native_flashrt_build_dir` 指向 Thor 构建目录。
> 首跑离线量化：`native_flashrt_calibrate: true` + `native_flashrt_act_scales_path`，导出后改回 `false` 自动加载。

### 8.4 构建/装载约定（Thor）

```bash
# 1) 在 FlashRT 源码树构建 kernel 扩展（Thor sm_110a）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target flash_rt_kernels
# 2) 指给 model_optimizer
export MO_FLASHRT_BUILD_DIR=/home/zhangxa/codes/FlashRT/build
export MO_FLASHRT_FMHA_SO=$MO_FLASHRT_BUILD_DIR/libfmha_fp16_strided.so
```

