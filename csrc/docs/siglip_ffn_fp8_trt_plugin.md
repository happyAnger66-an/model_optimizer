# SiglipFfFp8Plugin（TensorRT 自定义插件）

本文说明 **model_optimizer** 仓库内 **`csrc/trt_plugins/siglip_ffn_fp8`** 实现的 SigLIP 编码器 **FFN FP8 融合路径** TensorRT 插件：接口约定、与张量布局、与 FlashRT 的对齐关系、编译与运行时加载方式。

实现目标：在 TensorRT 引擎中 **单节点** 完成与 FlashRT `shared_primitives` 中相同的链路：

`fp8_nn_gelu_bias`（Up GEMM + GELU + bias）→ **`quantize_fp8_static_fp16`**（按设备标量 `unit_scale` 静态量化）→ **`fp8_nn_bias_res`**（Down GEMM + bias + 残差累加）。

参考源码：

- FlashRT：`third_party/FlashRT/flash_rt/hardware/thor/shared_primitives.py`（FFN 段）、`third_party/FlashRT/csrc/gemm/gemm_runner.cu`（`fp8_nn_gelu_bias` / `fp8_nn_bias_res`）、`third_party/FlashRT/csrc/kernels/quantize.cu`（`quantize_fp8_static_fp16`）。
- 本仓库插件：`csrc/trt_plugins/siglip_ffn_fp8/siglip_ffn_fp8_plugin.cpp`、`siglip_ffn_fp8_gelu_bias_res.cu`、`siglip_ffn_fp8_cuda_api.h`。

---

## 1. 插件标识与接口类型

| 项目 | 值 |
|------|-----|
| **插件类型名**（`getPluginType` / Creator `getPluginName`） | `SiglipFfFp8Plugin` |
| **版本字符串** | `"1"` |
| **TensorRT 接口** | `nvinfer1::IPluginV2DynamicExt` |
| **注册方式** | 动态库内 `REGISTER_TENSORRT_PLUGIN(SiglipFfFp8PluginCreator)`；加载该 `.so` 后由 `IPluginRegistry` 解析图中对应自定义层。 |
| **Creator 类**（C++ 命名空间） | `mopt_trt::SiglipFfFp8PluginCreator`（Registrar 使用全局别名注册，图中按 **插件名** 查找即可）。 |

**与 `SiglipMlpPlugin` 的区别**：`src/model_optimizer/ops/siglip_mlp_plugin.py` 导出的是 **FP16 融合 MLP**（`trt::SiglipMlpPlugin`）。本文档描述的是 **FP8 FFN + 静态量化 + 残差** 路径，ONNX 节点类型名应为 **`SiglipFfFp8Plugin`**（若自行扩展 Python symbolic，域一般为 `trt`）。

---

## 2. 输入 / 输出约定

全部为 **`TensorFormat::kLINEAR`**（线性内存，语义上按 **行主序** 二维张量理解）。

| 索引 | 语义 | `DataType` | 形状（逻辑） | 说明 |
|------|------|-------------|--------------|------|
| 0 | 归一化后激活 **`x_fp8`** | `kFP8`（E4M3） | **`[S, D]`** | `S` 可为动态维（如 token 数）；`D` 为隐藏维。 |
| 1 | **残差**（FFN 分支前的短接） | `kHALF` | **`[S, D]`** | 与输出同形；若与输出 **同一设备指针**，插件跳过拷贝，直接在输出缓冲上做 `beta=1` 累加。 |
| 2 | **Up 权重** `up_w` | `kFP8` | **`[D, H]`** | 与 FlashRT / `GemmRunner::fp8_nn_gelu_bias` 的 `B`（`K×N`）行主序约定一致：`K=D`，`N=H`。 |
| 3 | **Down 权重** `down_w` | `kFP8` | **`[H, D]`** | Down GEMM 的 `K×N`：`K=H`，`N=D`。 |
| 4 | **Up bias** `up_b` | `kHALF` | **`[H]`** | |
| 5 | **Down bias** `down_b` | `kHALF` | **`[D]`** | |
| 6 | **`unit_scale`** | `kFLOAT` | **标量**（长度 1 的一维张量即可） | 设备端 `float`，含义与 FlashRT `quantize_fp8_static_fp16(..., d_scale, ...)` 的 `descale` 一致：`inv_scale = 1/max(unit_scale, 1e-12)`。 |
| **输出 0** | **`y`** | `kHALF` | **`[S, D]`** | **`y = residual + alpha_down * ( GELU(alpha_up * x @ up_w + up_b) 量化为 FP8 后与 down_w 相乘 ) + down_b`**（与实现中 cuBLASLt epilogue 顺序一致）。 |

**输出形状推导**：与 **输入 1（残差）** 的维度表达式一致（`getOutputDimensions` 复制 `inputs[1]`）。

**静态 `H`、`D`**：`configurePlugin` 从 **输入 2** 的 `desc.dims` 读取 `D = d[0]`、`H = d[1]`。运行时 **`enqueue`** 中 **`D` 亦可从输入 0 的第二维** 再取一次（与权重维一致即可）。

**动态 `S` 上界**：用 **输入 0** 的 **`max.d[0]`**（当前 profile 下序列维最大值）估算 **workspace**；请保证构建 profile 的 `S_max` 不小于实际推理可能出现的 `S`。

---

## 3. 数学与 FlashRT 对齐说明

记 **`hidden = GELU( alpha_up * (x_fp8 @ up_w) + up_b )`**（FP16），再 **`hid_fp8 = quantize_fp8_static_fp16(hidden, unit_scale)`**，最后：

**`y = residual + alpha_down * (hid_fp8 @ down_w) + down_b`**

其中 GEMM 与 cuBLASLt 描述符、缓存 key 策略（如 `fp8_nn_gelu_bias` 使用 key type `102` 与 `N+3000000` 等）与 FlashRT `gemm_runner.cu` 中实现 **对齐**，便于与 Thor / pi05 侧已校准权重共用。

**注意**：静态 FP8 量化 kernel 与 FlashRT `quantize.cu` 中 **`quantize_fp8_kernel` / `quantize_fp8_static_fp16`** 同结构；**`S*H` 元素个数建议为 4 的倍数**（与 kernel 每线程处理 4 元素一致）。若 `H` 或 `S` 导致总元素非 4 对齐，需自行在图外围 pad 或改用扩展 kernel（当前插件未做尾部处理）。

---

## 4. 序列化与 Creator 字段

### 4.1 引擎序列化（`serialize` / `deserializePlugin`）

| 字段 | 类型 | 含义 |
|------|------|------|
| `alpha_up` | `float` | Up 路径标量，对应 `cublasLtMatmul` 的 `alpha`（默认 **1.0**）。 |
| `alpha_down` | `float` | Down 路径标量（默认 **1.0**）。 |

总长度 **`2 * sizeof(float)`**。

### 4.2 `createPlugin` 的 `PluginFieldCollection`（可选）

`getFieldNames` 当前返回 **空集合**；`createPlugin` 仍支持在 **`fc`** 中传入下列 **可选** 字段（名称需完全匹配）：

| `name` | `PluginFieldType` | `length` | 含义 |
|--------|-------------------|----------|------|
| `alpha_up` | `kFLOAT32` | ≥ 1 | 覆盖默认 `alpha_up`。 |
| `alpha_down` | `kFLOAT32` | ≥ 1 | 覆盖默认 `alpha_down`。 |

未传入时两个标量均为 **1.0f**。

---

## 5. Workspace 与 `enqueue` 返回值

- **`getWorkspaceSize`**：返回在 **`configurePlugin`** 阶段按 **`S_max`、`D`、`H`** 计算的上界，包括：中间 **FP16 hidden**、**FP8 hidden**、以及 cuBLASLt 启发式所需的 **临时工作区**（与内部 `kLtPrefWorkspaceBytes` 偏好一致）。
- **`enqueue` 非 0 返回值**（来自 `mopt_siglip_ffn_enqueue`）：表示参数非法、workspace 不足或 CUDA/cuBLAS 异常等；引擎会按 TensorRT 插件契约终止执行。排障时优先检查 **profile 的 `S_max`**、**`D/H` 与权重形状**、以及 **LT 段 workspace 是否小于启发式需求**。

---

## 6. 编译与产物

### 6.1 CMake 选项

在 **`csrc/CMakeLists.txt`** 中：

- **`MODEL_OPTIMIZER_BUILD_TRT_PLUGINS`**（默认 `OFF`）：置为 **`ON`** 时构建本插件。

子目录 **`csrc/trt_plugins/siglip_ffn_fp8/CMakeLists.txt`** 要求：

- **`TensorRT_ROOT`**（或环境变量 **`TENSORRT_ROOT`**）：含 `include/NvInfer.h` 与可用的 **`libnvinfer`**（CMake 会在 `TensorRT_ROOT/lib`、`lib/x64` 及常见系统路径中查找）。
- **CUDA Toolkit**：需存在 **`cuda_fp8.h`**（文档上对应 **CUDA 12.8+** 一类工具链；若默认 toolkit 较旧，请设置 **`CUDAToolkit_ROOT`** 与 **`CMAKE_CUDA_COMPILER`** 指向新 nvcc）。
- 链接：**`CUDA::cudart`**、**`CUDA::cublasLt`**、**`nvinfer`**。

### 6.2 示例命令

```bash
cmake -S /path/to/model_optimizer/csrc -B build_csrc \
  -DMODEL_OPTIMIZER_BUILD_TRT_PLUGINS=ON \
  -DTensorRT_ROOT=/path/to/TensorRT \
  -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build_csrc -j
```

典型产物路径：

`build_csrc/trt_plugins/siglip_ffn_fp8/libtrt_siglip_ffn_fp8_plugin.so`

---

## 7. 运行时使用方法

1. **加载插件动态库**  
   在构建引擎或 **`deserializeCudaEngine`** 之前，确保进程已加载 **`libtrt_siglip_ffn_fp8_plugin.so`**（例如 `dlopen`、设置 **`LD_LIBRARY_PATH`**，或 TensorRT 提供的插件加载 API / 应用自有加载逻辑）。

2. **网络中的节点**  
   图中自定义层类型名需为 **`SiglipFfFp8Plugin`**，版本 **`1`**，I/O 与上表一致。Python 侧可用 **`model_optimizer.ops.siglip_ffn_fp8_onnx.build_siglip_ffn_fp8_onnx_from_module`** 直接生成仅含该节点的 ONNX（不经 TorchScript FP8 导出）；schema 由 **`register_siglip_ffn_fp8_onnx_schema`** 注册。

3. **FP8 与硬件**  
   FP8 推理依赖 **支持 FP8 的 GPU 架构** 及匹配的 **TensorRT / CUDA** 组合；请在目标环境上单独验证计划文件（plan）与精度。

---

## 8. 仅使用 CUDA 路径（不经 TensorRT）

若要在其它宿主（自研 runtime、测试 harness）中直接调用融合核逻辑，可链接同一目标并包含 **`siglip_ffn_fp8_cuda_api.h`**：

- **`mopt_siglip_ffn_rt_create` / `mopt_siglip_ffn_rt_destroy`**：cuBLASLt 上下文与描述符缓存生命周期。
- **`mopt_siglip_ffn_query_workspace_bytes(S_max, D, H)`**：申请 device workspace 字节数上界。
- **`mopt_siglip_ffn_enqueue(...)`**：在给定 `cudaStream_t` 上执行完整融合；参数顺序与含义见头文件注释。

---

## 9. 相关文档

- FlashRT SigLIP FP8 GEMM 细节：`docs/flashRT/siglip/fp8_nn_bias.md` 等。
- 仓库内 FP16 MLP 插件（ONNX/Python 侧）：`src/model_optimizer/ops/siglip_mlp_plugin.py` 与 `docs/optimizer/vit_attention_plugin.md`（结构可参考，插件名不同）。

---

## 10. 端到端测试（PyTorch → ONNX → TRT）

仓库内 pytest：**`tests/test_siglip_ffn_fp8_trt_e2e.py`**。

| 步骤 | 说明 |
|------|------|
| PyTorch 最小模型 | **`TinySiglipFfFp8Mlp`**（`src/model_optimizer/ops/siglip_ffn_fp8.py`），前向调用 **`siglip_ffn_fp8_eager`**。 |
| ONNX | **`build_siglip_ffn_fp8_onnx_from_module`**（`siglip_ffn_fp8_onnx.py`）生成 **`trt::SiglipFfFp8Plugin`** 单节点图；权重为 FP8 initializer。 |
| 建引擎 | **`load_siglip_ffn_fp8_plugin`** + **`build_engine_from_onnx`**（`siglip_ffn_fp8_trt.py`），需已编译 **`libtrt_siglip_ffn_fp8_plugin.so`**。 |
| 推理对比 | **`run_engine_fp8_ffn`** 与 PyTorch 参考在 **SM ≥ 8.9** 上对比（`gpu_supports_fp8_trt()`）；Ampere 上建引擎通常会失败，测试会 **skip**。 |

**环境变量**：若 CMake 产物不在默认搜索路径，设置 **`MODEL_OPTIMIZER_TRT_SIGLIP_FF8_PLUGIN`** 指向 `.so` 绝对路径。

运行示例（仓库根目录、``PYTHONPATH=src``）：

```bash
MODEL_OPTIMIZER_TRT_SIGLIP_FF8_PLUGIN=/path/to/libtrt_siglip_ffn_fp8_plugin.so \
  PYTHONPATH=src pytest tests/test_siglip_ffn_fp8_trt_e2e.py -q
```
