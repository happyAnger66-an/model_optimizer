# Gemma 融合 MLP（TRT 插件 / CUDA）优化方向与手段

本文档总结 **PaliGemma / Pi05 文本侧 GeGLU FFN**（`GemmaFusedGatedMlp`）在 TensorRT 插件与 CUDA 实现上的 **优化思路与可落地手段**，并区分「已做」「可做」「收益有限」。思路参考业界常见做法（含 FlashRT / CUTLASS 类栈中的 **分桶 GEMM、向量化逐点、workspace 与算法选择** 等），**不照搬**某一条具体 FP8 业务管线。

---

## 1. 算子与数据流（优化靶点）

典型形状（Pi05 文本 LLM 一层、token 维展平）：

| 张量 | 形状（逻辑） | 说明 |
|------|----------------|------|
| 激活 `x` | `[M, H]`，`M=B×T` | 例：`B=1,T=968 → M=968`，`H=2048` |
| `gate_up` 权重 | `[2I, H]` | 两路 Linear 在 K 维拼接；例 `I=16384 → [32768,2048]` |
| `down` 权重 | `[H, I]` | 例 `[2048,16384]` |
| 中间 | `z=[M,2I]`，`h=[M,I]` | GeGLU：`h = gelu_tanh(gate) * up` |

计算链：**GEMM1**（`x @ gate_up^T`）→ **GeGLU 逐点** → **GEMM2**（`h @ down^T`）。瓶颈通常在前两段的 **大 GEMM** 与 **带宽型逐点**。

---

## 2. 可借鉴的优化手段（对照 FlashRT / CUTLASS 思路）

### 2.1 GEMM：算法与实现选型

| 手段 | 目的 | 说明 |
|------|------|------|
| **cuBLASLt + 更大 workspace + 多启发式候选** | 让库在宽 `N=2I` 的 mat1 上选到更合适的 Tensor Core 算法 | 实现成本低，与 TRT 插件 ABI 兼容；需控制 **显存占用**（Jetson） |
| **CUTLASS 3.x / CuTe 分桶**（`t1` / `wide` / `plain` 等） | 针对不同 `(M,N,K)` 用不同 tile / 流水线，提高 SM 利用率 | 需引入 CUTLASS、单独 CMake 与 **数值对齐测试** |
| **GEMM epilogue 融合**（如 GELU_tanh × up） | 减少一次全局写/读 `z` | CUTLASS epilogue 或 cuBLASLt 若支持需逐版本验证 |
| **FP8 权重 + 高精度累加** | 吞吐常显著高于纯 BF16 GEMM | 需改权重布局、标定与 ONNX/建链，与当前 bf16 插件 **不同产品路径** |

### 2.2 逐点 GeGLU：带宽与指令

| 手段 | 目的 | 说明 |
|------|------|------|
| **向量化（如每线程 4 列）** | 降低 load/store 与指令条数 | 要求 `I % 4 == 0`（Pi05 `I=16384` 满足） |
| **融合读写模式** | 提高合并访问比例 | 可按 `row` 分块，后续可试 **half2 / bfloat162** 等更激进打包 |

### 2.3 调度与重叠

| 手段 | 预期收益 | 说明 |
|------|----------|------|
| **多 CUDA stream 重叠** | 理论上重叠独立算子 | 本链 **严格顺序依赖**，重叠空间很小，**优先级低** |

---

## 3. 仓库中已落地的实现（`gemma_fused_gated_mlp_cuda.cu`）

以下为当前代码路径中 **已实现** 的优化，便于与 benchmark 对照：

1. **cuBLASLt**  
   - **gate_up** / **down** 分档 **`MAX_WORKSPACE`**（不显存敏感时默认 **512MiB / 256MiB**），便于宽 `N=2I` 与 down 侧选到更激进实现。  
   - `MatmulAlgoGetHeuristic` 一次最多拉 **32** 路候选；在预算内 **优先 workspace 更大** 的候选；否则退回首个 `SUCCESS`。

2. **GeGLU 向量化**  
   - `inter % 8 == 0`：每线程 **8** 列（`*_x8`），对 `z` 使用 **`__ldg`** 只读提示。  
   - `inter % 4 == 0` 且不满足 8：每线程 **4** 列（`*_x4`，含 `__ldg`）。  
   - 否则标量核。统一经 `launch_gelu_dispatch`（cuBLASLt 主路径与 naive 回退共用）。

3. **插件 workspace**  
   - `gemma_fused_gated_mlp_workspace_bytes` 尾部与 **gate_up 分档上限** 对齐（当前与 **512MiB** 一致），避免 scratch 不足回退 naive。

4. **基准**  
   - C++：`csrc/trt_plugins/gemma_fused_gated_mlp/bench_gemma_fused_gated_mlp.cu`（`bench_gemma_fused_gated_mlp` 目标）。  
   - PyTorch 参考：`scripts/bench_gemma_fused_mlp_torch.py`（同形状、便于对比）。

---

## 4. 建议的后续步骤（CUTLASS / 更强融合）

1. **数值基线**：固定 `(M,H,I)` 与 dtype，建立 **cuBLASLt 路径 vs CUTLASS 路径** 的 `max_abs_diff` / 相对误差门槛。  
2. **CUTLASS 分桶**：对 `GEMM1(M,2I,H)`、`GEMM2(M,H,I)` 分别选 tile（可参考 FlashRT `cutlass_sm100.cu` 中 **按形状分 kernel** 的思想，实现放在本仓库独立 target，**不强制依赖** `third_party/FlashRT` 编译链）。  
3. **Epilogue 融合**：在误差允许范围内尝试 **GEMM1 + GeGLU 部分融合**（或两段 GEMM 间减少中间 `z` 驻留），需与 TRT 插件 **enqueue 内存契约** 一致。  
4. **FP8（可选）**：单独文档与导出路径，涵盖 scale、权重布局与 TRT 插件 I/O dtype。

---

## 5. 风险与调参提示

- **显存**：当前默认按 **不显存敏感** 配置（大 workspace + 大插件尾部）。嵌入式或共享显存场景请 **自行调低** `kLtPrefWorkspaceGateUp` / `Down` 并重新建引擎。  
- **启发式与驱动版本**：不同 JetPack / cuBLASLt 返回的候选集合可能变化，重要版本点建议固化 golden shape 的 profile。  
- **向量化条件**：依赖 `I % 8` 或 `I % 4`；若改 `I` 需重新验证。

---

## 6. TRT 插件 I/O 布局与 `__myl_Move_*`（权重两条边）

### 6.1 插件在 API 层能看到的契约

`IPluginV2DynamicExt` 的 `PluginTensorDesc` / `DynamicPluginTensorDesc` 只提供 **dims、type、format（如 `kLINEAR`）**，**不提供 stride**。本插件在 `supportsFormatCombination` 中要求 **全部输入与输出** 为 **`TensorFormat::kLINEAR`**，且 **dtype 与激活 `x` 一致**（`kHALF` 或 `kBF16`），与 CUDA 侧按 **行主序、连续 `ld = 末维长度`** 使用 `inputs[]` 指针对齐。

| 端口 | 逻辑形状 | 与 PyTorch `nn.Linear` 权重 |
|------|-----------|------------------------------|
| `in[0]` 激活 `x` | `[…, H]`（任意前导维，末维为 `H`） | — |
| `in[1]` `gate_up` | `[2I, H]` | 等价于将 gate/up 两个 `Linear` 的 `weight` 在 **out_features（行）** 维拼接 |
| `in[2]` `down` | `[H, I]` | 与 `down_proj.weight` 同形 |

`enqueue` 中 **leading 维 `M`** 为 **除最后一维外各维的乘积**（与表 1 中 `M=B×T` 一致）。

### 6.2 为何图里常出现「两个 Move ↔ 两个权重」

在 Myelin 视图中，**激活** 往往已在前面算子的 **device 执行缓冲** 上，**可直接喂入插件**，故不一定出现前置 `Move`。**权重** 在引擎里常落在 **权重池 / 常量布局** 与 **插件执行期期望的缓冲** 之间；对 **两个独立张量边**（`[2I,H]` 与 `[H,I]`），规划器可能各插入一次 **`__myl_Move_*`**（本质是 **placement / D2D / 对齐到插件可读缓冲**），与插件是否在 `enqueue` 里再 `memcpy` **无必然关系**。因此 **单靠放宽 `supportsFormatCombination` 往往消不掉这两条 Move**（API 层也看不到 stride 以证明「非连续转连续」）。

### 6.3 排查与缓解

- **排查**：建引擎或首次 `configurePlugin` 时设置环境变量 **`MODEL_OPTIMIZER_GEMMA_TRT_PLUGIN_VERBOSE=1`**，插件会向 **stderr** 打印各输入的 **dims / type / format** 及 **min/opt/max**（动态形状），并校验 **`gate_up` 与 `down` 是否与 `[2I,H]`、`[H,I]` 及 `H` 一致**（不一致时打印告警）。
- **缓解 A（推荐，消除两条权重 Move）**：导出三输入 ONNX 后，调用 **`model_optimizer.ops.gemma_fused_gated_mlp_onnx_embed.embed_gemma_fused_gated_mlp_trt_static_weights`**，将权重折叠为 **张量属性**、节点仅保留 **`x` 输入**；使用 TensorRT 插件 **`GemmaFusedGatedMlp` 版本 `"2"`**（`initialize` 一次 H2D，推理期不再经图输入搬运权重）。Python 推理见 **`run_gemma_fused_gated_mlp_engine(..., gate_up_weight=None, down_weight=None)`**。
- **缓解 B（保守）**：关闭融合导出，回到原生 MatMul/Linear 链（无自定义插件，通常也无插件边界 Move，但失去融合算子收益）。

---

## 7. 相关路径速查

| 内容 | 路径 |
|------|------|
| CUDA 主实现 | `csrc/trt_plugins/gemma_fused_gated_mlp/gemma_fused_gated_mlp_cuda.cu` |
| 对外 C API 头文件 | `csrc/trt_plugins/gemma_fused_gated_mlp/gemma_fused_gated_mlp_cuda.h` |
| TRT 插件（IPlugin） | `csrc/trt_plugins/gemma_fused_gated_mlp/gemma_fused_gated_mlp_plugin.cpp`（verbose：`MODEL_OPTIMIZER_GEMMA_TRT_PLUGIN_VERBOSE=1`） |
| ONNX 权重折叠（消 Move） | `src/model_optimizer/ops/gemma_fused_gated_mlp_onnx_embed.py` |
| CMake（插件 + bench） | `csrc/trt_plugins/gemma_fused_gated_mlp/CMakeLists.txt` |

---

*文档与实现保持同步：若 CUTLASS 或 FP8 路径落地，请在第 4 节补充「版本、形状表、误差门槛与构建开关」。*
