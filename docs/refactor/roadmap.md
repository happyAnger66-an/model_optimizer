# model_optimizer 重构 Roadmap

## 1. 背景

`model_optimizer` 当前已经具备完整的模型优化工具链：

- 量化：ModelOpt PTQ、ONNX 量化、native decoder quant spec。
- 导出：按模型注册表导出 ONNX，`pi05` 已拆分为 `vit / embed_prefix / llm / expert / denoise`。
- 编译：ONNX 到 TensorRT engine，支持 build config、插件库、精度预检。
- 推理：PyTorch、TensorRT、ONNX Runtime、Native、FlashRT 多后端。
- 精度对齐：NPZ 对比、PT/TRT/PTQ/ORT 多路比较、Pi05Metric。
- 性能分析：stage wall-time、TRT profile、FlashRT benchmark、CUDA Graph。
- 自定义算子：FMHA D256 plugin、CuTe DSL kernel、fused MLP、FlashRT decoder。

但当前核心路径仍强绑定 `pi05/openpi`，尤其是 `policy_loader.py`、`Pi05*Executor`、`calibrate/collector/pi05.py`、`scripts/deployment/pi05/`、`ServerConfig.PI05_STAGES`。后续要支持其它 VLA 模型，应从“pi05 专用优化工具链”演进为“VLA 优化平台 + pi05 architecture plugin”。

## 2. 重构目标

### 2.1 架构目标

- 将 `pi05` 的五阶段思想抽象为可注册的 `VLAArchitectureSpec`。
- 将 openpi 依赖隔离到 `PolicyAdapter`，避免散落在 executor 和 server loader 中。
- 将 TensorRT / ONNX Runtime / Native / FlashRT 等实现抽象为可组合的 `BackendRunner`。
- 将量化、导出、编译、精度对齐、性能分析统一挂到 `StageArtifact` 和 manifest。
- 将自定义算子、结构改写、runtime 开关统一纳入 feature/plugin registry。
- 保持现有 pi05 功能稳定，优先做“外包一层”的渐进式重构。

### 2.2 非目标

- 不在第一阶段重写 `pi05` 的推理逻辑。
- 不立即替换 FlashRT / TensorRT / Native 的现有实现。
- 不一次性统一所有配置格式，先建立 resolved manifest，再逐步收敛。
- 不为了抽象牺牲当前 pi05/Thor 性能优化路径。

## 3. 当前架构问题

### 3.1 pi05 强耦合

- `ServerConfig` 中写死 `PI05_STAGES = ("vit", "embed_prefix", "llm", "expert", "denoise")`。
- `policy_loader.py` 直接调用 openpi 的 `policy_config.create_trained_policy()`。
- `_mount_tensorrt_engines()`、`_mount_native()`、`_apply_selective_ptq()` 都直接引用 pi05 子模块。
- `Pi05NativeExecutor` 直接 monkey-patch `sample_actions` / `denoise_step`，并依赖 openpi `Policy._sample_actions` 缓存行为。
- FlashRT decoder 路径硬编码 pi05 权重、层数、head、action horizon、even seq padding 等细节。

### 3.2 配置和产物不统一

- 量化配置使用 Python `QUANT_CFG`。
- build 配置使用 Python `build_cfg`。
- feature 配置使用 JSON。
- WebUI/server 使用 JSON/YAML。
- 产物路径由用户指定，没有统一 manifest 描述 ONNX、engine、quant spec、profile、accuracy report 之间的关系。

### 3.3 扩展其它 VLA 的阻力

- 缺少通用 VLA stage 定义。
- 缺少通用 Policy/Model adapter。
- 缺少通用 calibration collector manifest。
- 缺少 backend runner registry。
- 部署和评测脚本集中在 `scripts/deployment/pi05/`，可复用逻辑没有沉淀到库内。

## 4. 目标架构

### 4.1 VLAArchitectureSpec

每个 VLA 架构定义自己的 stage、默认后端、量化/导出/编译配置和校准需求。

示例：

```text
architecture: pi05
stages:
  - vit
  - embed_prefix
  - llm
  - expert
  - denoise
backends:
  vit: [pytorch, tensorrt, flashrt]
  embed_prefix: [pytorch, tensorrt, onnxrt]
  llm: [pytorch, tensorrt, onnxrt]
  expert: [pytorch, tensorrt, native]
  denoise: [pytorch, tensorrt, native, flashrt]
```

未来其它 VLA 可以定义不同阶段，例如：

```text
architecture: generic_vla
stages:
  - vision_encoder
  - language_model
  - action_head
```

或：

```text
architecture: diffusion_vla
stages:
  - perception
  - planner
  - diffusion_action_decoder
```

### 4.2 PolicyAdapter

将不同模型生态的加载、输入输出、底层模型访问统一封装。

建议接口：

```text
PolicyAdapter
  load_policy(checkpoint, config) -> PolicyHandle
  infer(obs, noise=None) -> ActionOutput
  preprocess(raw_obs) -> ModelInput
  postprocess(model_out) -> ActionOutput
  get_underlying_model(policy) -> torch.nn.Module
  metadata(policy) -> dict
```

首个实现：

```text
OpenPiPi05Adapter
```

后续可新增：

```text
StarVLAAdapter
OpenVLAAdapter
ShVLAAdapter
```

### 4.3 BackendRunner

将当前 `Pi05TensorRTExecutor`、`Pi05OnnxRTExecutor`、`Pi05NativeExecutor` 的能力拆成按 stage 执行的 backend runner。

建议接口：

```text
BackendRunner
  supports(architecture, stage) -> bool
  prepare(policy_handle, stage_spec, config) -> RuntimeHandle
  run(inputs) -> outputs
  close()
```

候选实现：

- `PytorchRunner`
- `TensorRTRunner`
- `OnnxRTRunner`
- `NativeRunner`
- `FlashRTRunner`

`PipelineAssembler` 根据 `VLAArchitectureSpec` 和用户配置组合各阶段 runner。

### 4.4 StageArtifact 和 Manifest

每个 stage 的产物目录应包含统一 manifest，描述量化、导出、编译、精度、性能产物。

推荐布局：

```text
artifacts/pi05/llm/
  model.onnx
  model.engine
  quant_spec.json
  build_config.resolved.json
  io_schema.json
  accuracy_report.json
  perf_report.json
  artifact_manifest.json
```

manifest 示例字段：

```text
architecture
stage
model_name
checkpoint
precision
quant_format
onnx_path
engine_path
build_config
feature_config
calibration_data
accuracy_report
perf_report
created_at
git_commit
```

### 4.5 Feature Plugin Registry

现有 `models/features.py` 和 `FeatureConfig` 方向正确，应扩展成多类型 feature：

- `ModelFeature`：结构改写，例如 fused MLP。
- `ExportFeature`：ONNX custom op、sidecar、plugin attribute。
- `RuntimeFeature`：CUDA Graph、KV cache、FlashRT。
- `QuantFeature`：FP8、NVFP4、W4A8、per-channel axis。
- `ProfilingFeature`：NVTX、stage perf、trtexec profile。

每个 feature 声明：

```text
supported_architectures
supported_stages
required_artifacts
conflicts
validation_check
```

## 5. 分阶段实施计划

### Phase 0：冻结现状和补基础文档

目标：明确当前 pi05 行为，避免重构期间功能漂移。

任务：

- 梳理现有 pi05 端到端命令：calibrate、quantize、export、build、infer、compare、profile。
- 为当前主路径建立 baseline 文档和示例配置。
- 记录当前 pi05 支持的后端组合矩阵。
- 为现有产物增加最小 manifest 草案，不改变执行逻辑。
- 建立 smoke test 清单，覆盖 registry、quantize dry-run、compare、build config load。

验收：

- `docs/refactor/roadmap.md` 可作为后续拆任务入口。
- 现有 pi05 命令和配置被完整记录。
- 不改变现有行为。

### Phase 1：抽出 ArchitectureSpec

目标：把 `PI05_STAGES` 从通用 server config 中移出。

任务：

- 新增 `src/model_optimizer/architectures/`。
- 定义 `ArchitectureSpec` / `StageSpec` / `ArchitectureRegistry`。
- 实现 `Pi05ArchitectureSpec`，包含 `vit / embed_prefix / llm / expert / denoise`。
- `ServerConfig` 增加 `architecture` 字段，默认 `pi05`。
- `resolve_stages()` 改为从 architecture spec 获取 stage 列表。
- 保持旧配置兼容。

验收：

- 现有 pi05 server config 无需修改即可运行。
- 新增测试验证 `pi05` stage 解析结果与旧逻辑一致。

### Phase 2：抽出 PolicyAdapter

目标：隔离 openpi 加载逻辑。

任务：

- 新增 `src/model_optimizer/policies/`。
- 定义 `PolicyAdapter` 接口。
- 实现 `OpenPiPi05Adapter`，封装：
  - `openpi.training.config.get_config`
  - `policy_config.create_trained_policy`
  - `_unwrap_pi05_model`
  - `metadata`
- 修改 `policy_loader.py`，通过 adapter registry 加载 policy。
- 保持 `load_policies()` 和 `load_policy_for_serve()` 对外接口不变。

验收：

- openpi import 集中到 `OpenPiPi05Adapter`。
- `policy_loader.py` 不再直接知道 pi05 底层模型解包细节。
- 现有 `pytorch/tensorrt/native/onnxrt/flashrt` 模式行为不变。

### Phase 3：后端挂载逻辑注册化

目标：将 `_mount_tensorrt_engines()`、`_mount_native()`、`_mount_onnxrt_engines()` 从 pi05 专用函数演进为 backend registry。

任务：

- 定义 `BackendInstaller` 或轻量 `BackendRunner` 接口。
- 先以兼容方式包装现有：
  - `Pi05TensorRTInstaller`
  - `Pi05NativeInstaller`
  - `Pi05OnnxRTInstaller`
  - `Pi05FlashRTInstaller`
- `policy_loader.py` 根据 architecture + stage backend 矩阵调用 installer。
- 将 `_apply_selective_ptq()` 改为 architecture-specific PTQ installer。

验收：

- `policy_loader.py` 中 pi05 条件分支显著减少。
- 新增一个 mock architecture 能完成 stage resolution 和 backend installer lookup。
- pi05 现有功能不变。

### Phase 4：产物 Manifest 化

目标：统一量化、导出、编译、精度、性能产物描述。

任务：

- 定义 `ArtifactManifest` dataclass。
- `quantize_cli` 量化完成后写入 manifest。
- `convert/export` 导出后写入 ONNX manifest。
- `trt_build/build.py` 编译后补充 engine 信息。
- `compare/profile` 输出报告路径写入 manifest。
- WebUI 优先读取 manifest 展示状态。

验收：

- 任意 stage 产物目录有 `artifact_manifest.json`。
- 可以通过 manifest 找到 ONNX、engine、quant config、build config、校准数据、报告。
- 不破坏旧目录结构。

### Phase 5：校准和精度对齐通用化

目标：让其它 VLA 复用 pi05 的校准数据组织和对比工具。

任务：

- 定义通用 `CalibCollector` 接口。
- 统一 shard + manifest 格式。
- 将 `calibrate/collector/pi05.py` 包装成 `Pi05CalibCollector` plugin。
- 定义通用 `ActionMetric`，保留 `Pi05Metric` 为特化实现。
- 将 `compare_predictions`、PT/TRT/PTQ/ORT 多路对比沉淀为通用 report。

验收：

- pi05 校准数据仍可被现有量化流程读取。
- 新增 mock collector 测试。
- 对比报告可以被 manifest 引用。

### Phase 6：Feature Plugin 扩展

目标：把结构改写、自定义算子、runtime 开关纳入统一 feature 生命周期。

任务：

- 扩展 `models/features.py` 的 feature 类型。
- 给现有 feature 补 metadata：
  - fused MLP
  - FMHA D256 plugin
  - Edge-LLM attention plugin
  - CUDA Graph
  - native quant spec
  - FlashRT decoder
- 增加 feature conflict / validation 机制。
- 导出和 runtime 阶段都记录 applied features。

验收：

- feature_config 拼写错误、模型不支持、冲突组合能给出明确错误或警告。
- manifest 记录实际启用的 features。

### Phase 7：提炼通用 VLA 评测/部署入口

目标：减少 `scripts/deployment/pi05/` 中的重复和专用逻辑。

任务：

- 抽出通用 eval session、GPU stats、stage perf、result worker。
- 将 `lerobot_eval_webui` 可复用部分迁移到库内。
- pi05 deployment script 改为薄 wrapper。
- 为第二个 VLA 准备模板目录。

验收：

- pi05 旧脚本仍可运行。
- 新 VLA 只需提供 adapter/spec/metric，即可复用通用 eval runner。

### Phase 8：接入第二个 VLA 作为验收

目标：验证架构是否真正泛化。

候选：

- `starVLA`
- `sh_vla`
- `openvla`
- `qwen_vl` 草稿路径

任务：

- 新增 architecture spec。
- 新增 policy adapter。
- 注册至少一个可量化/导出的 stage。
- 跑通最小链路：

```text
load -> calibrate/sample -> quantize or export -> build optional -> infer/compare
```

验收：

- 不修改 pi05 executor 主体即可接入新 VLA。
- 新 VLA 能复用 CLI、manifest、compare、profile 基础能力。

## 6. 优先级建议

近期优先做：

1. `ArchitectureSpec`：收益最大，风险低。
2. `PolicyAdapter`：隔离 openpi，给其它 VLA 留入口。
3. `ArtifactManifest`：提升工程可维护性，支撑 WebUI/CI/回归。
4. Backend installer registry：降低 `policy_loader.py` 复杂度。

暂缓：

- 大规模重写 executor。
- 统一所有配置格式。
- 把 FlashRT 完全泛化。
- 重构所有 deployment 脚本。

## 7. 关键文件映射

现有核心文件：

```text
src/model_optimizer/launcher.py
src/model_optimizer/models/registry.py
src/model_optimizer/models/model.py
src/model_optimizer/infer/server/config.py
src/model_optimizer/infer/server/policy_loader.py
src/model_optimizer/infer/native/pi05_executor.py
src/model_optimizer/infer/tensorrt/pi05_executor.py
src/model_optimizer/infer/onnxrt/pi05_executor.py
src/model_optimizer/quantization/cli.py
src/model_optimizer/convert/convert_formt.py
src/model_optimizer/trt_build/build.py
src/model_optimizer/models/features.py
src/model_optimizer/config/feature_config.py
```

建议新增：

```text
src/model_optimizer/architectures/
  __init__.py
  base.py
  registry.py
  pi05.py

src/model_optimizer/policies/
  __init__.py
  base.py
  registry.py
  openpi_pi05.py

src/model_optimizer/backends/
  __init__.py
  base.py
  registry.py
  pi05_tensorrt.py
  pi05_native.py
  pi05_onnxrt.py
  pi05_flashrt.py

src/model_optimizer/artifacts/
  __init__.py
  manifest.py
  writer.py

src/model_optimizer/calibrate/collectors/
  base.py
  registry.py
```

## 8. 风险与约束

- pi05 性能路径依赖 monkey-patch、CUDA Graph、FlashRT，不应在早期强行重写。
- openpi `Policy` 缓存 `_sample_actions`，替换 `sample_actions` 后必须同步引用。
- FlashRT 路径包含大量 pi05 结构假设，短期只注册为 pi05-only backend。
- 量化和导出依赖 ModelOpt、TensorRT、CUDA 版本，manifest 应记录环境信息。
- WebUI 当前依赖日志文件和子进程，重构时要保持进度文件兼容。

## 9. 最终验收标准

重构完成后应满足：

- pi05 现有量化、导出、编译、推理、对比、性能分析能力保持可用。
- 新增一个 VLA 不需要修改 `policy_loader.py` 主流程。
- 新增一个 stage/backend 不需要修改 `ServerConfig` 的硬编码 stage 列表。
- 每个 stage 产物都有 manifest，可追溯配置、模型、精度、性能报告。
- feature/plugin 能声明适用模型、适用 stage、依赖产物和冲突关系。
- WebUI/CLI/CI 可以基于同一套 manifest 和 registry 工作。
