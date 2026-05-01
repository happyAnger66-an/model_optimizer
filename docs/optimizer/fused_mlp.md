# TensorRT-LLM 中的 Fused MLP（`fuse_gate_mlp`）

本文说明 **TensorRT-LLM** 在构建网络时对 **门控 FFN（Gated MLP / SwiGLU 类结构）** 做的融合优化：入口函数为 `fuse_gate_mlp`，定义于上游仓库的 `tensorrt_llm/models/modeling_utils.py`（约第 1141 行）。`model_optimizer` 不实现该逻辑；文档便于对照 Edge-LLM / 量化管线时理解行为。

## 1. 作用概览

- **目标**：把每一层里的 **`GatedMLP`**（或 **`Fp8RowwiseGatedMLP`**）换成 **`FusedGatedMLP`**（或 **`Fp8RowwiseFusedGatedMLP`**），在图里用**单次列线性** `fused_fc: hidden → 2 × ffn_hidden` 替代原来的 **`fc` + `gate` 两次线性**，再配合 **`swiglu` / `geglu`** 激活，数学上与「先算两路再相乘」的写法等价。
- **动机**：
  - 减少 GEMM 次数与中间张量，利于延迟与显存；
  - 在开启 **`gemm_swiglu_plugin` / `low_latency_gemm_swiglu_plugin`**（当前为 FP8）时，可走**融合 Kernel**，进一步把矩阵乘与 SwiGLU 绑在一起。
- **调用链**：`optimize_model(..., use_fused_mlp=True, ...)` 在权重加载等前置步骤之后调用 `fuse_gate_mlp(model, gemm_swiglu_plugin_dtype, low_latency_gemm_swiglu_plugin_dtype)`。

## 2. 融合前后的计算对应关系

### 2.1 `GatedMLP`（未融合）

实现见 `tensorrt_llm/layers/mlp.py` 中 `GatedMLP.forward`：

1. `inter = fc(x)`
2. `inter = activation(inter)`（由 `hidden_act` 决定，融合路径仅支持 **`silu`**、**`gelu`**）
3. `gate = gate_proj(x)`
4. `out = inter * gate`，再可选 `inner_layernorm`，最后 `proj` 压回 `hidden_size`。

即典型的 **SiLU(fc)·gate** 或配合 GeGLU 的变体。

### 2.2 `FusedGatedMLP`（融合后）

实现见同文件 `FusedGatedMLP`：

- 使用 **`ColumnLinear(hidden, 2 * ffn_hidden)`** 一次得到拼接后的两路 logits，再经 **`ACT2FN['swiglu']` / `['geglu']`** 在通道维上完成与「两路分别线性再乘」等价的拆分与组合。
- `forward` 根据 `default_net().plugin_config` 选择：
  - **插件路径**：`fc_gate_plugin` → `gemm_swiglu` / `low_latency_gemm_swiglu`（要求 FP8 权重、`proj` 为 FP8 Row 等，见源码断言）；
  - **非插件路径**：`fc_gate` → 普通 `fused_fc` + `swiglu`/`geglu`。

`fuse_gate_mlp` 负责把**权重与量化参数**从「双线性」迁到「单线性 + 正确 layout」，子模块属性 **`proj` / `inner_layernorm`** 仍指向原模块，保证下游结构不变。

## 3. `fuse_gate_mlp` 流程（按代码顺序）

函数遍历 `model.named_modules_with_parent()`，对每个子模块分两类处理。

### 3.1 `GatedMLP` 分支

1. **`get_init_params(mlp)`** 取出构建参数；**`hidden_act` 必须为 `silu` 或 `gelu`**，否则打日志并 **skip**。
2. 设置 **`inner_layernorm`** 是否与原版一致，**`FusedGatedMLP(**init_params)`**。
3. **量化策略**（`model.config._get_quant_cfg(name + '.fc')`）：
   - 若 **`quant_algo` 非 FP8 且非 `None`**：**直接 `continue`**，不融合该层（避免与未支持的量化格式冲突）。
   - 若 **`fc` 在 `quantization.exclude_modules` 中**：按无该层量化处理（`layer_quant_algo = None`）。
4. **`QuantAlgo.FP8`**：
   - 对 `fused_layer` 做 **`fp8_quantize`**；
   - 将 `gate` / `fc` 权重搬到同一 dtype，必要时按 **`weights_scaling_factor`** 做 **反量化再拼接**；
   - **`torch.cat([gate_weight, fc_weight], dim=0)`** 得到融合权重（与 `ColumnLinear` 输出维 `2 * ffn` 一致）；
   - **`weights_scaling_factor`** 取 gate 与 fc 的 **max**；若仍为 FP8 权重量化，再按新 scale **压回 e4m3**；
   - **`activation_scaling_factor`** 取两者 **max**；
   - 若 **`gemm_swiglu_plugin_dtype` / `low_latency_gemm_swiglu_plugin_dtype` 为 `'fp8'`**：按插件要求的 **(k, n)** 布局写入 `Parameter`；否则直接赋值 `value`；
   - **bias** 存在则 **`np.concatenate` gate/fc 的 bias**。
5. **无量化（`layer_quant_algo is None`）**：对 **`weight` / `bias`** 做 **`np.concatenate(..., axis=0)`**。
6. 其他 `quant_algo`：**抛 `ValueError`**。
7. 挂回 **`proj` / `inner_layernorm`**，用 **`setattr(layer, mlp_name, fused_layer)`** 替换父模块中的 MLP 子模块。

### 3.2 `Fp8RowwiseGatedMLP` 分支

- 同样要求 **`hidden_act` ∈ {`silu`, `gelu`}**。
- 若有 **`clamp_val`**，写入 **`Fp8RowwiseFusedGatedMLP`**。
- **`fused_fc.weight` / `per_channel_scale`** 分别对 gate 与 fc 做 **`np.concatenate(..., axis=0)`**；bias 同理。
- 替换父模块中的对应 MLP。

### 3.3 返回值

- 原地修改 **`PretrainedModel`** 树并 **返回同一 `model` 引用**。

## 4. 限制与注意事项

| 项目 | 说明 |
|------|------|
| 激活函数 | 仅 **`silu`**、**`gelu`**；其他激活会 **跳过融合** 并打 warning。 |
| 量化 | **`GatedMLP`** 路径下，**仅 FP8 与无额外 quant_algo** 会融合；**非 FP8 的 quant_algo** 整层 **不融合**。 |
| 插件 | **`gemm_swiglu` / `low_latency_gemm_swiglu`** 路径在 **`FusedGatedMLP.fc_gate_plugin`** 中限制较多（如 FP8、无 bias、无拆分 LoRA 等），需与构建配置一致。 |
| LoRA | 融合后 LoRA 使用 **`fused_gate_up_lora`** 等字段；与 **拆分 gate/fc 的 LoRA** 在插件路径下 **未实现**（源码 `NotImplementedError`）。 |

## 5. 与 `model_optimizer` 的关系

- 本文描述的是 **TensorRT-LLM 源码**中的图优化行为；若在 `model_optimizer` 或 Edge-LLM 文档中引用「Fused MLP / SwiGLU 融合」，通常对应上游 **`use_fused_mlp` + `fuse_gate_mlp`** 及 **`FusedGatedMLP`** 的实现。
- 具体行号与 API 以所用 **TensorRT-LLM 版本** 为准；集成时请以当前 checkout 的 `modeling_utils.py` / `layers/mlp.py` 为准。

## 6. 参考路径（TensorRT-LLM）

- `tensorrt_llm/models/modeling_utils.py`：`fuse_gate_mlp`，`optimize_model` 中的 `use_fused_mlp`。
- `tensorrt_llm/layers/mlp.py`：`GatedMLP`，`FusedGatedMLP`（含 `fc_gate` / `fc_gate_plugin` / `forward`）。
- `tensorrt_llm/quantization/layers.py`：`Fp8RowwiseGatedMLP`，`Fp8RowwiseFusedGatedMLP`。
