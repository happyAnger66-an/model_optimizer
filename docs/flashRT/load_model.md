# FlashRT `load_model` 流程说明

本文以 **`flash_rt/api.py`** 中的 **`load_model`**（约第 171 行）为入口，梳理从用户调用到得到 **`VLAModel`** 实例的完整路径，并说明与各 **Pipeline / Frontend** 构造的关系。代码路径以 **FlashRT 仓库根目录** 为基准（`flash_rt/` 包）。

---

## 1. 入口与对外契约

### 1.1 函数签名（摘要）

```python
def load_model(
    checkpoint,
    framework="torch",
    num_views=2,
    autotune=3,
    recalibrate=False,
    weight_cache=True,
    config="pi05",
    device=None,
    decode_cuda_graph=False,
    decode_graph_steps=80,
    max_decode_steps=256,
    hardware="auto",
    embodiment_tag=None,
    action_horizon=None,
    use_fp4=False,
    fp4_layers=None,
    use_awq=None,
    awq_alpha=0.5,
    use_p1_split_gu=None,
):
```

### 1.2 返回值

- 返回 **`VLAModel`**（同文件内定义）：内部持有具体 **`pipe`**（如 `Pi05TorchFrontendThor`、`Pi05TorchFrontendRtx` 等）与 **`framework`** 字符串。
- 用户侧典型用法：`model.predict(images=..., prompt=...)` → 内部调用 **`pipe.set_prompt` / `pipe.infer`**（见下文第 7 节）。

---

## 2. 执行顺序总览（高层）

以下顺序与 `api.py` 中 `load_model` 的**实际代码顺序**一致：

| 步骤 | 内容 | 主要涉及模块 |
|------|------|----------------|
| A | 校验 `config` ∈ `{pi05, groot, pi0, pi0fast}`、`framework` ∈ `{torch, jax}` | `api.py` |
| B | **`use_fp4` 默认派生**：开启 FP4 时补全 `fp4_layers` / `use_awq` / `use_p1_split_gu`；关闭时用另一套默认 | `api.py` |
| C | **`hardware`**：`"auto"` → **`detect_arch()`**，否则直接使用传入的 arch 字符串 | `flash_rt/hardware/__init__.py` |
| D | **`recalibrate=True`**：清量化校准缓存；JAX 路径额外尝试清 **权重 FP8 磁盘缓存** | `flash_rt/core/quant/calibrator.py`、`core/weights/weight_cache.py` |
| E | **`framework=="jax"`**：设置默认 `XLA_FLAGS`、`XLA_PYTHON_CLIENT_PREALLOCATE` | `api.py` |
| F | **`resolve_pipeline_class(config, framework, arch)`** → 得到 **pipeline 类** | `hardware/__init__.py` |
| G | **FP4 再路由**（仅 `pi05` + `torch|jax` 且扩展可用）：可能将 `pipe_cls` 换为 `Pi05TorchFrontendThorFP4` / `Pi05JaxFrontendThorFP4` | `api.py`、`frontends/*/pi05_thor_fp4.py` |
| H | 按 **`config` + `inspect.signature(pipe_cls)`** 组装 **`kwargs`**（避免向不接受参数的类传参） | `api.py` |
| I | **`pipe = pipe_cls(checkpoint, **kwargs)`** —— **真正构造推理管线** | 各 `frontends/*/*.py` |
| J | **`return VLAModel(pipe, framework)`** | `api.py` |

---

## 3. 硬件探测与 Pipeline 解析

### 3.1 `detect_arch()`（`hardware == "auto"`）

- 文件：`flash_rt/hardware/__init__.py`
- 依赖 **`torch.cuda.is_available()`** 与 **`torch.cuda.get_device_capability()`**。
- 返回值与 GPU 对应关系（节选，与源码一致）：
  - SM **11.0** → **`"thor"`**（Jetson AGX Thor）
  - SM **12.0** → **`"rtx_sm120"`**
  - SM **8.9** → **`"rtx_sm89"`**
- 不支持的 SM 会 **`RuntimeError`**（刻意严格，避免静默选错后端）。

### 3.2 `resolve_pipeline_class(config, framework, arch)`

- 查表 **`_PIPELINE_MAP[(config, framework, arch)]`** → `(module_path, class_name)`。
- 使用 **`__import__(module_path, fromlist=[class_name])`** 懒加载模块，再 **`getattr`** 取类。
- 若三元组不存在：抛出带 **hint** 的 `RuntimeError`（列出同 config+framework 下支持的 arch）。

### 3.3 映射表要点（`pi05` 示例）

| arch | torch | jax |
|------|--------|-----|
| `thor` | `flash_rt.frontends.torch.pi05_thor.Pi05TorchFrontendThor` | `flash_rt.frontends.jax.pi05_thor.Pi05JaxFrontendThor` |
| `rtx_sm120` / `rtx_sm89` | `pi05_rtx.Pi05TorchFrontendRtx` | `pi05_rtx.Pi05JaxFrontendRtx` |

`pi0`、`groot`、`pi0fast` 等条目见同文件 `_PIPELINE_MAP` 全文。

---

## 4. `kwargs` 装配逻辑（与具体类对齐）

`load_model` 用 **`inspect.signature(pipe_cls)`** 判断目标类是否接受某参数，再写入 `kwargs`，核心意图：

- **`pi0fast`**：固定传入 `autotune`、`decode_cuda_graph`、`decode_graph_steps`、`max_decode_steps`。
- **`groot`**：若签名含 `autotune` / `embodiment_tag` / `action_horizon` 且调用方传入非默认，则传入（Thor / RTX 侧类签名可能不同，故做 feature-detect）。
- **`pi05` / `pi0`**：通常传 `num_views`；若签名含 `autotune` 则传入；JAX 侧若含 **`weight_cache`** 则传入（仅影响 Orbax 路径的 FP8 权重缓存等）。
- **`use_fp4` 为真且类支持**：可能增加 `use_fp4_encoder_ffn`、`fp4_layers`、`use_awq`、`awq_alpha`、`use_p1_split_gu` 等（由 FP4 子类消费）。

---

## 5. 典型 Pipeline 构造时做什么（深入摘要）

以下仅概括 **构造阶段**（`pipe_cls(checkpoint, **kwargs)`）的主要工作；**推理阶段**见第 7 节。

### 5.1 Pi0.5 + Torch + Thor — `Pi05TorchFrontendThor`

- 文件：`flash_rt/frontends/torch/pi05_thor.py`，类 **`Pi05TorchFrontendThor`**。
- **checkpoint**：目录路径；权重来自 **`checkpoint/model.safetensors`**（非 Orbax）。
- 初始化要点（与源码一致）：
  - **`FvkContext`、`GemmRunner`**（`flash_rt_kernels` / `flash_rt.flash_rt_kernels`）。
  - 按固定搜索顺序尝试加载 **`libfmha_fp16_strided.so`**（FMHA）；失败则日志警告并走 fallback。
  - **`_load_norm_stats`**：多候选路径加载 `norm_stats.json`（含 LeRobot / openpi 资产路径）。
  - **`_load_weights`**：从 safetensors 读入并布置到 GPU/内核侧缓冲区（后续在 `_load_weights` 方法体内展开）。
- 与 **`autotune`**：构造函数接收，用于后续 **CUDA Graph** 在 `set_prompt` 等阶段的试验次数（见该 frontend 文档字符串）。

### 5.2 Pi0.5 + Torch + RTX — `Pi05TorchFrontendRtx`

- 文件：`flash_rt/frontends/torch/pi05_rtx.py`，类 **`Pi05TorchFrontendRtx`**。
- **checkpoint**：同样期望目录下有 **`model.safetensors`**（HF 风格 PyTorch 导出）。
- 初始化要点：
  - **`convert_pi05_safetensors`**：将 HF 权重键布局转换为 FlashRT **`Pi05Pipeline`** 所需的张量字典（含 vision/encoder/decoder 重排、RMSNorm fold、时间嵌入、action head 预缩放等）。
  - 张量迁到 **CUDA bf16**；大 GEMM 做 **FP8 预量化**（`_quantize_all_fp8`）。
  - **`_precompute_decoder_styles`**、**`RtxFlashAttnBackend`**、**`GemmRunner`**、预分配 **`_img_buf` / `_noise_buf`** 等。
- 与 Thor 路径差异：RTX 使用 **`flash_rt.models.pi05.pipeline_rtx`** 与 RTX 注意力后端；且公开 API 与 Thor 对齐（`set_prompt`、`infer`、`calibrate_with_real_data` 等，见该文件模块 docstring）。

### 5.3 Pi0.5 + JAX + Thor — `Pi05JaxFrontendThor`

- 文件：`flash_rt/frontends/jax/pi05_thor.py`。
- 模块 import 前已可能设置 **XLA 环境变量**（与 `load_model` 中再次设置互补）。
- 初始化要点：
  - **`load_norm_stats`**（`strict=False` 等差异以源码为准）。
  - **`FvkContext` / FMHA 路径搜索** 与 Torch Thor 类似。
  - **Orbax → FP8 量化 → `CudaBuffer`** 的权重加载链路（含 **`weight_cache`** 命中则加速二次启动）。
- 细节分支（`engine_path` 等）见该类 **`__init__`** 全文。

### 5.4 FP4 变体（`use_fp4=True` 且校验通过）

- Torch：`flash_rt.frontends.torch.pi05_thor_fp4.Pi05TorchFrontendThorFP4`
- JAX：`flash_rt.frontends.jax.pi05_thor_fp4.Pi05JaxFrontendThorFP4`
- 若 config/framework 不匹配或 **`flash_rt_fp4`** 不可用 / `has_nvfp4()` 为假，则 **回退 FP8** 并打 warning。

### 5.5 其他 `config`（`pi0`、`groot`、`pi0fast`）

- 均由 **`_PIPELINE_MAP`** 指向对应 **`frontends/torch/*` 或 `frontends/jax/*`** 中的类；构造参数通过第 4 节的 **`inspect.signature`** 分支注入。
- **`groot`**：文档中强调 **`embodiment_tag`** 与 checkpoint 训练槽位匹配，否则行为异常。
- **`pi0fast`**：额外 CUDA Graph decode 相关参数由 `kwargs` 固定传入。

---

## 6. 缓存与 `recalibrate`

| 行为 | 作用 |
|------|------|
| `recalibrate=True` | 调用 **`clear_calibration(checkpoint)`**；JAX 时再 **`clear_weight_cache(checkpoint)`**（忽略文件不存在） |
| `weight_cache`（JAX） | 在 `kwargs` 中传给支持该参数的 frontend，控制 FP8 权重是否落盘以加速二次加载 |

---

## 7. 加载完成后的运行时路径（`VLAModel`）

`load_model` 返回 **`VLAModel(pipe, framework)`**。

### 7.1 `VLAModel.predict`

- 文件：`flash_rt/api.py`。
- 流程摘要：
  1. **prompt 变化**时：若 `pipe` 有 **`set_prompt`**，按签名决定是否传入 **`state`**。
  2. 将 **`images`** 规范为 **`obs` 字典**（支持 list 或 dict；多相机键名兼容 Thor / rtx 前端）。
  3. 若 `pipe` 有 **`calibrate_with_real_data`**（如部分 RTX Pi0.5），在首次 `predict` **lazy** 调用一次 **`calibrate_with_real_data([obs])`**。
  4. **`result = pipe.infer(obs)`**，返回 **`result['actions']`** 为 **`np.ndarray`**。

### 7.2 与 `load_model` 的关系

- **`load_model` 只负责** 选型、清缓存、**构造 `pipe`** 并包进 **`VLAModel`**。
- **首次真实推理**中的校准 / CUDA Graph 捕获等多发生在 **`set_prompt` / `infer` / `calibrate_with_real_data`** 内，由各 **Frontend** 实现，而非在 `load_model` 内完成。

---

## 8. 相关源码索引（便于跳转）

| 主题 | 路径 |
|------|------|
| 公共 API | `flash_rt/api.py`（`load_model`、`VLAModel`） |
| 硬件探测与映射 | `flash_rt/hardware/__init__.py`（`detect_arch`、`resolve_pipeline_class`、`_PIPELINE_MAP`） |
| Pi0.5 Thor Torch | `flash_rt/frontends/torch/pi05_thor.py` |
| Pi0.5 RTX Torch | `flash_rt/frontends/torch/pi05_rtx.py`（`convert_pi05_safetensors`、`Pi05TorchFrontendRtx`） |
| Pi0.5 Thor JAX | `flash_rt/frontends/jax/pi05_thor.py` |
| Pi0.5 FP4 | `flash_rt/frontends/torch/pi05_thor_fp4.py`、`flash_rt/frontends/jax/pi05_thor_fp4.py` |
| 量化校准缓存 | `flash_rt/core/quant/calibrator.py`（`clear_calibration` 等） |
| JAX 权重缓存 | `flash_rt/core/weights/weight_cache.py` |

---

## 9. 流程图（Mermaid）

```mermaid
flowchart TD
  A[load_model 入口] --> B{config / framework 合法?}
  B -->|否| X[ValueError]
  B -->|是| C[use_fp4 默认与校验]
  C --> D{hardware}
  D -->|auto| D1[detect_arch]
  D -->|显式| D2[arch = hardware]
  D1 --> E
  D2 --> E{recalibrate?}
  E -->|是| E1[clear_calibration + JAX 时 clear_weight_cache]
  E -->|否| F
  E1 --> F{framework jax?}
  F -->|是| F1[默认 XLA 环境变量]
  F -->|否| G
  F1 --> G[resolve_pipeline_class]
  G --> H{use_fp4 且 pi05?}
  H -->|是且扩展 OK| H1[pipe_cls = FP4 Frontend]
  H -->|否| I[保持解析类]
  H1 --> J[按 signature 组装 kwargs]
  I --> J
  J --> K["pipe = pipe_cls(checkpoint, **kwargs)"]
  K --> L[VLAModel(pipe, framework)]
  L --> M[predict -> set_prompt / infer / 可选 lazy 校准]
```

---

## 10. 小结

- **`load_model`** 是 FlashRT 的 **唯一推荐高层入口**：完成 **GPU 架构解析 → Pipeline 类解析 → （可选）FP4 路由 → 构造参数裁剪 → 实例化 `pipe` → 包装为 `VLAModel`**。
- **重计算与 I/O**（读 safetensors / Orbax、FP8 量化、FMHA `.so`、norm_stats、CUDA Graph 等）均在各 **`Pi05TorchFrontend*` / `Pi05JaxFrontend*`** 的 **`__init__` 及后续 `set_prompt`/`infer`** 中完成；理解 **`load_model`** 后应结合具体 **`config` + `arch`** 打开对应 frontend 源文件继续下钻。

本文档随 FlashRT 版本演进，若与源码不一致，以 **`flash_rt/api.py` 与 `flash_rt/hardware/__init__.py`** 为准。
