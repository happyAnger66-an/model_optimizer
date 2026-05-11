# SigLIP FFN FP8（FlashRT 桥接）TensorRT 插件

本文档描述 `SiglipFfFp8FlashrtPlugin` 的 **I/O 规格**、**符号约定**、**Workspace** 与 **计算顺序**，并与 FlashRT 侧实现对齐说明。

## 概述

- **插件类型名**（`getPluginType` / Creator `getPluginName`）：`SiglipFfFp8FlashrtPlugin`
- **版本**：`1`
- **实现位置**：`csrc/trt_plugins/siglip_ffn_fp8/siglip_ffn_fp8_flashrt_plugin.cpp`
- **核心计算**：通过 FlashRT 静态桥 `third_party/FlashRT/csrc/trt_bridge/flash_rt_trt_api.{h,cu}` 调用 `GemmRunner::fp8_nn_gelu_bias`、`quantize_fp8_static_fp16`、`GemmRunner::fp8_nn_bias_res`，与 Thor 路径上 `flash_rt/hardware/thor/shared_primitives.py` 中的融合顺序一致。

## 输入与输出规格

共 **7 路输入 + 1 路输出**。TensorRT `supportsFormatCombination` 中 `pos` 为「按顺序的输入张量，再接输出张量」的下标；所有张量均为 **`kLINEAR`** 排布。

| pos | 名称（语义） | TensorRT dtype | 形状约定 |
|-----|----------------|------------------|----------|
| 0 | `x`：激活 FP8 | `kFP8`（e4m3） | `[…, D]`：除最后一维外各维乘积记为 **S**，最后一维为 **D** |
| 1 | `residual`：残差 FP16 | `kHALF` | 与输出 `y` **同 rank、同尺寸** |
| 2 | `up_w`：Up 权重 FP8 | `kFP8` | **`[D, H]`**（与 `configurePlugin` 中 `m_d`、`m_h` 来源一致：`d[0]=D,d[1]=H`） |
| 3 | `down_w`：Down 权重 FP8 | `kFP8` | **`[H, D]`** |
| 4 | `up_b`：Up bias FP16 | `kHALF` | **`[H]`** |
| 5 | `down_b`：Down bias FP16 | `kHALF` | **`[D]`** |
| 6 | `unit_scale`：静态量化标量 | `kFLOAT` | **`[1]`**（设备上 1 个 `float`） |
| 7 | `y`：输出 FP16 | `kHALF` | 与 **输入 1** 相同 rank/尺寸（`getOutputDimensions` 跟随 `inputs[1]`） |

### 序列化插件字段

引擎序列化 / 反序列化时写入 **两个 `float`**：

- `alpha_up`
- `alpha_down`

与 ONNX 节点属性 `alpha_up` / `alpha_down`（若通过 Creator 的 `PluginFieldCollection` 传入）对应。

### 符号 S、D、H

与 `enqueue` 中逻辑一致：

- **S**：输入 `x`（pos 0）除 **最后一维** 外各维大小的 **乘积**（例如 `[B, T, D]` → `S = B * T`）。
- **D**：运行时以 **`inputDesc[0]` 的最后一维** 为准（与 `x` 的 in_features 一致）。
- **H**：来自 **`up_w`（pos 2）** 的第二维 `inputDesc[2].dims.d[1]`（hidden 宽度）。

## Workspace 规格

`frt_siglip_ffn_fp8_workspace_bytes(S_max, D, H)` 在桥接层实现，为两段 **256 字节对齐** 的缓冲：

1. **FP16 `hidden`**：`S_max × H × sizeof(fp16)`
2. **FP8 `hid_fp8`**：`S_max × H × sizeof(fp8_e4m3)`

用于存放 Up+GELU 后的 FP16 中间结果，以及再量化后的 FP8 中间结果。  
**注意**：cuBLASLt 在 `GemmRunner` 内部使用的大块 workspace 由 `GemmRunner` 构造时单独分配，**不包含**在上述插件 `getWorkspaceSize` 返回值中。

## 计算过程（顺序）

对应 `third_party/FlashRT/csrc/trt_bridge/flash_rt_trt_api.cu` 中的 `frt_siglip_ffn_fp8_enqueue`：

1. **残差拷入输出（必要时）**  
   若 `y` 与 `residual` **不是**同一块设备内存，则执行  
   `y ← residual`（`cudaMemcpyAsync`，大小 `S × D × sizeof(fp16)`）。  
   若指针相同则跳过。

2. **Up：FP8 GEMM + 融合 GELU + bias → FP16 `hidden`**  
   `GemmRunner::fp8_nn_gelu_bias(x_fp8, up_w, hidden, up_b, M=S, N=H, K=D, alpha_up, stream)`  
   得到形状 **`[S, H]`** 的 FP16 `hidden`。

3. **FP16 → FP8 静态量化**  
   `quantize_fp8_static_fp16(hidden, hid_fp8, unit_scale, S*H, stream)`  
   使用设备上的 **`unit_scale`**（标量），与 Thor 上静态 FP8 量化语义一致。

4. **Down：FP8 GEMM + bias + 残差累加到 `y`**  
   `GemmRunner::fp8_nn_bias_res(hid_fp8, down_w, y_fp16, down_b, M=S, N=D, K=H, alpha_down, stream)`  
   FlashRT 实现中该路径对输出使用 **`beta = 1`**：在步骤 1 之后 **`y` 已含 residual**，本步在 `y` 上累加 down 分支结果与 bias（语义与 `fp8_nn_bias_res` 一致；矩阵布局细节以 `third_party/FlashRT/csrc/gemm/gemm_runner.cu` 为准）。

**运行时对象**：插件在 `initialize` 中创建 `GemmRunner`（`frt_gemm_runner_create`），在 `terminate` 中销毁（`frt_gemm_runner_destroy`）；每次 `enqueue` 在给定 `stream` 上按 1→4 顺序执行。

## ONNX / 命名说明

- 图中自定义算子 **`op_type`** 需为 **`SiglipFfFp8FlashrtPlugin`**，`domain` 一般为 **`trt`**，版本 **`1`**，方能与当前 Creator 注册名一致。  
- 若历史 ONNX 仍使用旧名 **`SiglipFfFp8Plugin`**，需改节点类型或改 C++ 中 `getPluginType` / Creator 名与之一致（避免多插件同名注册冲突）。

## 构建与依赖（摘要）

1. 先在 **`third_party/FlashRT`** 构建，生成 **`flash_rt/libflash_rt_trt_bridge.a`**（见该目录下 `CMakeLists.txt` 中 `flash_rt_trt_bridge` 目标）。  
2. 再在 **`model_optimizer/csrc`** 打开 **`MODEL_OPTIMIZER_BUILD_TRT_PLUGINS`**，配置 **`TensorRT_ROOT`**；若找到上述 `.a`，会生成 **`libtrt_siglip_ffn_fp8_flashrt_plugin.so`**（见 `csrc/trt_plugins/siglip_ffn_fp8/CMakeLists.txt`）。  
3. Python 侧可用 **`model_optimizer.ops.siglip_ffn_fp8.discover_siglip_ffn_fp8_flashrt_plugin_so`** 或环境变量 **`MODEL_OPTIMIZER_TRT_SIGLIP_FF8_FLASHRT_PLUGIN`** 指向该 `.so`。

## 实施落地步骤

本节说明 **Pi05 `Vit` 量化 → ONNX 导出 → TensorRT** 与 FlashRT 桥接插件的衔接方式；与代码中 `src/model_optimizer/models/pi05/vit.py` 的 `export` / `quantize` 及 `src/model_optimizer/ops/siglip_ffn_fp8*.py` 一致。

### 实现过程（代码侧摘要）

1. **ONNX 自定义算子与 symbolic**（`src/model_optimizer/ops/siglip_ffn_fp8_flashrt_export.py`）  
   - 注册 `trt::siglip_ffn_fp8_flashrt_plugin`（`torch.library.custom_op`），eager 委托 `siglip_ffn_fp8_eager`。  
   - `register_siglip_ffn_fp8_flashrt_plugin_onnx_symbolic_functions()`：注册 ONNX schema `trt::SiglipFfFp8FlashrtPlugin`，并将上述 custom op 映射到该节点（含 `alpha_up` / `alpha_down` 属性）。

2. **视觉塔 MLP 包装**（`src/model_optimizer/ops/siglip_ffn_fp8.py`）  
   - `SiglipFfFp8FlashrtMlpWrapper`：在 **`MODEL_OPTIMIZER_SIGLIP_FFN_FP8_FLASHRT_TRT_EXPORT=1`** 时，将各层 `SiglipMLP` 前向改为调用 `siglip_ffn_fp8_flashrt_plugin`（权重需为 FP8 e4m3；布局与上表 **up_w `[D,H]`、down_w `[H,D]`** 对齐，包装内对 HF `fc1.weight`/`fc2.weight` 做转置）。  
   - `patch_vision_siglip_ffn_fp8_flashrt_custom_op`：在 `vision_tower` 各 encoder 层上替换 `mlp`。

3. **`Vit` 集成**（`src/model_optimizer/models/pi05/vit.py`）  
   - 构造参数 **`siglip_ffn_fp8_flashrt_custom_op`**，或环境变量 **`MODEL_OPTIMIZER_SIGLIP_FFN_FP8_FLASHRT_CUSTOM_OP=1`**。  
   - 与 **`siglip_mlp_custom_op`（`trt::SiglipMlpPlugin` 路径）互斥**：若两者同时开启，优先 FlashRT FFN 包装并打日志告警。  
   - **`export` / `export_onnx`**：若 **`MODEL_OPTIMIZER_SIGLIP_FFN_FP8_FLASHRT_TRT_EXPORT=1`**，则在 `torch.onnx.export` 前调用 `register_siglip_ffn_fp8_flashrt_plugin_onnx_symbolic_functions()`，并将 **`opset_version` 提升为 20**（否则保持 19）。  
   - **`quantize`**：在 `quantize_model` 之后，若检测到 **NVFP4 量化配置** 且上述 TRT 导出环境变量已开启，则 `set_dynamic_quant(self, "fp16")`；否则仍为 **`bf16`**（与 `utils.set_dynamic_quant` 仅作用于 NVFP4 线性层的行为一致）。

### 推荐操作顺序（可执行清单）

1. **编译产物**  
   按上文「构建与依赖」生成 **`libflash_rt_trt_bridge.a`** 与 **`libtrt_siglip_ffn_fp8_flashrt_plugin.so`**；运行 TensorRT 建引擎时确保能加载该 `.so`（或通过 `MODEL_OPTIMIZER_TRT_SIGLIP_FF8_FLASHRT_PLUGIN` 指定路径）。

2. **构造带 FlashRT 包装的 `Vit`**  
   - 代码：`Vit.construct_model(..., siglip_ffn_fp8_flashrt_custom_op=True)`，或  
   - 环境：`export MODEL_OPTIMIZER_SIGLIP_FFN_FP8_FLASHRT_CUSTOM_OP=1`  
   不要依赖与 **`MODEL_OPTIMIZER_SIGLIP_MLP_CUSTOM_OP` / `siglip_mlp_custom_op`** 同时生效；应只选一条融合路径。

3. **量化（`Vit.quantize`）**  
   - 准备校准数据与量化配置（例如 `config/quant/vit_quant_fp8_cfg.py` 或项目内等价 FP8 配置），对 SigLIP **`fc1`/`fc2`（及链路）** 完成 **`quantize_model`**。  
   - **检查点**：导出 TRT 插件前，MLP 权重应为 **`torch.float8_e4m3fn`**；否则 `SiglipFfFp8FlashrtMlpWrapper` 会退回原始 `inner` 并记录 warning。  
   - **`unit_scale`**：与静态量化语义一致时，应在模块上提供与 FlashRT 标定一致的标量（包装优先读取 `inner.unit_scale`，否则默认 `[1.0]`，生产环境需自行对齐）。

4. **导出 ONNX（含 `quantize` 末尾自动 `export`**）  
   - `export MODEL_OPTIMIZER_SIGLIP_FFN_FP8_FLASHRT_TRT_EXPORT=1`  
   - 效果：`export` 内注册 symbolic、**opset 20**、图中出现 **`trt::SiglipFfFp8FlashrtPlugin`** 节点（经 `trt::siglip_ffn_fp8_flashrt_plugin` 映射）。  
   - **激活 dtype**：插件期望 **FP8 激活**；若当前图中隐状态仍为 BF16/FP16，包装内存在 **cast 到 FP8 的开发回退**（仅便于联调，**不等价**于真实量化激活，生产环境应接入真实 FP8 激活路径）。

5. **TensorRT 建引擎与部署**  
   - 使用与 ONNX 域/类型名一致的插件注册；核对 **I/O dtype**（残差/偏置/输出为 **FP16**，权重/激活为 **FP8 e4m3**，`unit_scale` 为 **FP32 `[1]`**）。  
   - 与 FlashRT 顺序、Workspace 以本文「计算过程」「Workspace 规格」为准做算子级对齐验收。

### 环境变量一览（与本路径相关）

| 变量 | 作用 |
|------|------|
| `MODEL_OPTIMIZER_SIGLIP_FFN_FP8_FLASHRT_CUSTOM_OP` | 为 `1`/`true`/`yes` 时，等价于构造 `Vit` 时开启 FlashRT MLP 包装（未显式传参时）。 |
| `MODEL_OPTIMIZER_SIGLIP_FFN_FP8_FLASHRT_TRT_EXPORT` | 为 `1`/`true`/`yes` 时：ONNX 导出走 custom op + symbolic，且 `Vit.export` 使用 **opset 20**；与 NVFP4 组合时影响 `quantize` 内 `set_dynamic_quant` 的 dtype 选择。 |
| `MODEL_OPTIMIZER_TRT_SIGLIP_FF8_FLASHRT_PLUGIN` | 显式指向 **`libtrt_siglip_ffn_fp8_flashrt_plugin.so`** 路径。 |

## 相关源文件

| 组件 | 路径 |
|------|------|
| TRT 插件 | `csrc/trt_plugins/siglip_ffn_fp8/siglip_ffn_fp8_flashrt_plugin.cpp` |
| C 桥 API 声明 | `third_party/FlashRT/csrc/trt_bridge/flash_rt_trt_api.h` |
| C 桥实现（调用顺序） | `third_party/FlashRT/csrc/trt_bridge/flash_rt_trt_api.cu` |
| GemmRunner FP8 路径 | `third_party/FlashRT/csrc/gemm/gemm_runner.cu` |
| 静态 FP8 量化（FP16 输入） | `third_party/FlashRT/csrc/kernels/quantize.cu` |
| Pi05 Vit 导出 / 量化入口 | `src/model_optimizer/models/pi05/vit.py` |
| FlashRT MLP 包装与插件发现 | `src/model_optimizer/ops/siglip_ffn_fp8.py` |
| ONNX custom op 与 symbolic | `src/model_optimizer/ops/siglip_ffn_fp8_flashrt_export.py` |
| Vit FP8 量化配置示例 | `config/quant/vit_quant_fp8_cfg.py` |
