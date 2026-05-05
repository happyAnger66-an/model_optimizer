# awq_lite 原理与工作流程

本文说明 NVIDIA **ModelOpt**（`modelopt.torch.quantization`）中 **`awq_lite`**（AWQ Lite）的数学直觉与代码级工作流程。实现以本仓库 vendored 的 Model-Optimizer 为准，路径：`third_party/Model-Optimizer/modelopt/torch/quantization/model_calib.py` 中的 `awq_lite()`。

与 **INT4 AWQ 在本仓库中的端到端用法**（Pi05 LLM、`forward_loop`、`float32` 适配等）见同目录下的 [int4_awq.md](./int4_awq.md)。

---

## 1. 在解决什么问题

对线性层 \(y = x W^\top\)（\(x\) 为激活，\(W\) 为权重），低比特权重量化会引入误差。AWQ（Activation-aware Weight Quantization，[论文](https://arxiv.org/pdf/2306.00978)）的核心是：对**输入通道**引入可吸收的逐通道缩放 \(s\)（向量，长度等于 `in_features`），在数学上等价重写为

\[
y = (x \odot s^{-1}) \,(W \odot s)^\top
\]

其中 \(\odot\) 表示按最后一维广播的逐元素乘。这样 **\(x\) 与 \(W\) 同时被重标定**，但 **FP 前向的 \(y\) 不变**；量化器作用在「已平滑」的激活与权重上，通常能减小量化误差。

**`awq_lite`** 是 ModelOpt 中的「轻量」实现：用校准数据估计激活/权重的**通道级显著性**，在一个由超参 \(\alpha\) 参数化的尺度族上搜索，使 **带 fake-quant 的线性输出** 与 **无量化参考输出** 的 **MSE** 尽可能小；最后把最优 \(s\) **折叠进权重**（并配置输入侧的 `pre_quant_scale`），再按常规 `max` 校准权重量化器。

**`awq_full`** 在 `awq_lite` 之后还会跑 **`awq_clip`**（对块 amax 等做更重的搜索）；本文只讨论 `awq_lite`。

---

## 2. 通道统计量：`act_scale` 与 `weight_scale`

实现里对每个参与 AWQ 的量化 `Linear`（要求 `weight_quantizer` 已启用）构造 `AWQLiteHelper`，预先计算 **权重侧** 尺度，在校准前向中累积 **激活侧** 尺度。

### 2.1 激活尺度 `act_scale`（cache 阶段）

对某层输入张量经 `input_quantizer` 后的结果（记为与量化配置一致的处理后激活），按 **最后一维（输入通道）** 做：

- 将张量 reshape 为 `(-1, in_features)`；
- 对 **绝对值** 在 batch/token 维上取 **均值**，得到长度为 `in_features` 的 FP32 向量；
- 在 `cache_mode` 下对多步校准前向 **累加**，阶段结束后 **除以 `num_cache_steps`** 得到平均统计。

若该层启用了输入量化，cache 阶段还会用 `max_calibrate` 收集输入 quantizer 的 amax（供后续 postprocess 恢复输入量化形态）。若 `input_quantizer` 的 `axis` 不是 `None` 或 `-1`，该层会被 **禁用 AWQ**（`is_enabled = False`）。

### 2.2 权重尺度 `weight_scale`

对权重 `W`（形状一般为 `[out_features, in_features]`），按 **权重量化器的块大小** `block_size`（由 `_get_awq_quantizer_block_size` 从权重与 quantizer 推断）：

- 沿最后一维必要时 **padding** 到块长整数倍，view 成 `[-1, block_size]`；
- 对每个块取 `abs().amax(dim=1)` 得到块内最大值；
- 计算 `abs(weight) / (block_amax + tiny)`，再 view 回原形状；
- 在 **输出通道维（dim 0）** 上 **取均值**，得到与 **输入通道维** 对齐的尺度（与 `get_act_scale` 向量逐通道对应）。

直觉：该统计刻画 **权重各输入通道在块内的相对幅值模式**，与激活的通道均值幅度一起进入下面的 \(\alpha\)-插值公式，用于生成候选缩放 \(s\)。

---

## 3. 候选尺度 `get_scale` 与 \(\alpha\) 搜索

对离散网格 \(\alpha \in \{0,\ \alpha_{\text{step}},\ 2\alpha_{\text{step}},\ \ldots,\ 1\}\)（默认 `alpha_step=0.1`，见 `AWQLiteCalibConfig`），定义（与测试里 `awq_lite_manual` 及 [llm-awq auto_scale](https://github.com/mit-han-lab/llm-awq) 思路一致）：

\[
s_i = \mathrm{clamp}\left(
\frac{\texttt{act\_scale}_i^{\alpha}}{\texttt{weight\_scale}_i^{1-\alpha} + \varepsilon},\ 10^{-4},\ 10^{4}
\right)
\]

再对整向量做一次 **几何风格归一化**（避免尺度整体漂移）：

\[
s \leftarrow \frac{s}{\sqrt{\max(s)\cdot\min(s)}}
\]

- \(\alpha \to 0\)：更偏向由 **权重尺度** 主导 \(s\)。
- \(\alpha \to 1\)：更偏向由 **激活尺度** 主导 \(s\)。

**张量并行（列并行线性）**：`get_scale` 可对 `scales` 在 tensor parallel group 上做 `all_reduce(SUM)` 再除以 `world_size`，保证各 rank 使用一致的 \(s\)。

---

## 4. 搜索目标：逐 \(\alpha\) 的 MSE（search 阶段）

`SequentialQuantizer.convert_to_single_quantizer` 包裹后，对每个目标模块 **patch `forward`**（保留原始为 `_forward_no_awq`）。

**Search 阶段**（`cache_mode = False`）每次 `forward` 大致做：

1. **关闭** `weight_quantizer`，清除输入/权重 quantizer 上可能残留的 `pre_quant_scale`，用 `_forward_no_awq` 得到 **参考输出** `out_actual`（无量化路径）。
2. **打开** `weight_quantizer`。
3. 对每个网格点 \(\alpha\)：  
   - 用当前累积并已平均的 `act_scale`、`weight_scale` 调用 `get_scale` 得到 `awq_scale`（即候选 \(s\)）；  
   - 设 **`input_quantizer.pre_quant_scale = 1 / awq_scale`**，`**weight_quantizer.pre_quant_scale = awq_scale**`（dtype 对齐到权重）；  
   - 再跑 `_forward_no_awq`，将输出与 `out_actual` 算 **MSE**，累加到 `loss[alpha]`。
4. `num_search_steps` 自增，返回 **仍为** `out_actual`（不把搜索用的量化前向作为最终输出）。

因此：**第二遍校准前向**中，每层在每个 batch 上会 **对每个 \(\alpha\) 多跑一次线性内部路径**，计算量约为 \(\times (1 + N_\alpha)\) 量级（与实现细节有关）。

搜索结束后 **`update_best_params`**：取 **`loss` 最小的 \(\alpha\)** 作为 `best_alpha`，并用该 \(\alpha\) 再算一遍 `get_scale` 得到 **`best_scale`**。

---

## 5. 收尾：写入权重与 `pre_quant_scale`（postprocess）

对每个模块：

- 若 `act_scale` / `weight_scale` 出现 **NaN**，或 cache/search **步数为 0**，则 **禁用 AWQ**，对该层退化为仅对权重做 **`max_calibrate`**。
- 否则调用 **`apply_pre_quant_scale_and_smooth(module, 1.0 / module.awq_lite.best_scale)`**：
  - 输入 quantizer 侧设置 **`pre_quant_scale = 1 / best_scale`**（在 fp32 下做乘法更安全）；
  - 权重侧通过 **`_apply_weight_pre_quant_scale`** 将 **`weight *= best_scale`**（或等价地挂上权重 quantizer 的 pre_quant，取决于全局 `_ENABLE_FOLDING_PQS_TO_WEIGHTS`），再 **`reset_amax` + `max_calibrate`** 权重量化器；
  - 若存在平滑阶段保存的 `_amax_for_smoothing`，会按 `pre_quant_scale` 调整后续 **激活 amax** 形态，并与列/行并行下的 **amax 同步** 逻辑配合。

最后 **`cleanup`**：卸载 patch 的 `forward`，默认 **`delattr(module, "awq_lite")`**（`debug=True` 时保留 `awq_lite` 对象便于检查 `best_scale` / `loss`）。

---

## 6. 配置与入口

| 方式 | 说明 |
|------|------|
| 字符串 | `"algorithm": "awq_lite"` 使用默认 `alpha_step=0.1`（见 `model_quant.calibrate` 对算法配置的解析） |
| 字典 | `"algorithm": {"method": "awq_lite", "alpha_step": 0.1, "debug": false}` |

配置类 **`AWQLiteCalibConfig`**（`modelopt/torch/quantization/config.py`）字段含义：

- **`alpha_step`**：\(\alpha\) 从 0 到 1 的步长，需满足 \(0 < \texttt{alpha\_step} \le 1\)。
- **`debug`**：为 `True` 时在模块上保留 `awq_lite` 属性以便调试。

本仓库 **`NVFP4_AWQ_LITE_CFG`** 将 NVFP4 的权重量化/输入量化与 **`"algorithm": "awq_lite"`** 组合，用于在 FP4 量化前做同样的 AWQ Lite 尺度搜索（配置定义见 `config.py` 中 `NVFP4_AWQ_LITE_CFG`）。

---

## 7. 整体时序小结

```text
quantize(..., algorithm 含 awq_lite)
    → calibrate(..., awq_lite, forward_loop)
        → 为各层安装 AWQLiteHelper + patch forward
        → [Pass 1] cache_mode=True：forward_loop(model)
              累积 act_scale；max_calibrate 消化统计
        → 归一化 act_scale；DP 上可选 all_reduce(AVG)
        → [Pass 2] cache_mode=False：forward_loop(model)
              对每个 α：设 pre_quant_scale，累加 MSE loss
        → postprocess：best_alpha / best_scale
              apply_pre_quant_scale_and_smooth；cleanup
```

**要点**：必须提供可用的 **`forward_loop`**，且数据会 **至少两次** 完整流过模型；若第二层前向某层步数为 0，会触发警告并可能对该层关闭 AWQ。

---

## 8. 参考

- AWQ 论文：<https://arxiv.org/pdf/2306.00978>  
- ModelOpt 配置说明：`AWQLiteCalibConfig`（`third_party/Model-Optimizer/modelopt/torch/quantization/config.py`）  
- 实现：`awq_lite()`（`third_party/Model-Optimizer/modelopt/torch/quantization/model_calib.py`）  
- 与手写参考一致的单元测试：`tests/unit/torch/quantization/test_calib.py` 中 `awq_lite_manual` / `test_awq_lite`
