# Fused MLP 实现总结（π0.5 / LLMWithCuteDsl 路径）

> 对应代码：
> - `src/model_optimizer/models/pi05/fused_mlp.py` — 核心替换层
> - `src/model_optimizer/models/pi05/llm_with_cutedsl.py` — 接入点（特性注册 + ONNX 导出）
>
> 当前方案为 **方案 A：纯 PyTorch 模块替换 + ONNX 原生算子图重写，零 plugin**。
> 深度融合（L2/L3）见 `kernelSrc/docs/fused_mlp.md`。

---

## 一、整体实现架构

方案 A 分布在两个文件、四个环节。

### 1. 核心替换层 `fused_mlp.py`

| 组件 | 职责 |
|---|---|
| `FusedGemmaMLP(nn.Module)` | `GemmaMLP` 的等价替身：构造时把 `gate_proj.weight` 与 `up_proj.weight` 在 dim=0 concat 成单个 `gate_up_proj`，复用原 `down_proj` 和 `act_fn` |
| `install_fused_mlp(gemma)` | in-place 遍历 `gemma.layers[*].mlp` 做替换；幂等、可跳过非匹配层、返回替换数 |
| `_LinearMetaShim` + `@property gate_proj/up_proj` | 后向兼容垫片（见 §一.3） |
| `fused_mlp_state_dict_remap` | 离线 ckpt 加载备用（把旧的分离权重 key 合并） |

### 2. 接入点 `llm_with_cutedsl.py` —— 特性注册表驱动

fused MLP 现在是一个**注册特性**（相比早期 `if fuse_mlp: install_fused_mlp` 的结构变化）：

```python
register_feature(
    "fmha_d256_attention",
    default_enabled=True,
    apply_fn=_apply_fmha_d256_attention,
    description="将各层 self_attn 包成 CuTe DSL FMHA D=256 TRT 插件路径。",
    supported_models=(MODEL_NAME,),
)
register_feature(
    "fused_mlp",
    default_enabled=True,
    apply_fn=_apply_fused_mlp,
    description="合并各层 GemmaMLP 的 gate/up 为单次 GEMM（纯 ONNX 图重写）。",
    supported_models=(MODEL_NAME,),
)
```

启停优先级：**JSON `--feature_config` > 环境变量 > 默认值**。

- 走 CLI 时由 `apply_features()` 根据 `FeatureConfig` 决定（`Pi05CuteDslLanguageModel.__init__` 的 `feature_config` 分支）。
- 直接构造（非 CLI）时回退到布尔参数 `fuse_mlp=True` + 环境变量 `MODEL_OPT_PI05_FUSED_MLP` / 别名 `MODEL_OPT_FEATURE_FUSED_MLP`。

`fused_mlp` 与 `fmha_d256_attention` 两个特性**完全正交**：fused MLP 改的是 `layer.mlp`，attention 包装改的是 `layer.self_attn`，互不干扰。

### 3. 量化阶段（关键设计点）

`LLMWithCuteDsl.quantize` → `quantize_model` → ModelOpt `mtq.quantize` 通过
`named_children()` 按 **`type(child) in registry`** 发现 `nn.Linear`。`FusedGemmaMLP`
用了两层伪装，让 ModelOpt **只看到 `gate_up_proj` + `down_proj` 两个真 Linear**：

- `gate_proj` / `up_proj` 是 `@property`，根本不在 `_modules` 里 → `named_children()` 遍历不到；
- 它们返回的 `_LinearMetaShim` 不是 `nn.Module` → 即便被访问也不会被识别为可量化层。

效果：每层量化器从原来 `gate+up+down = 3 套` 变成 `gate_up+down = 2 套`，且下游
`mlp.gate_proj.weight.dtype` 这类探测代码（如 `gemma_pytorch.py` 检查 dtype）仍可用。

### 4. ONNX 导出

`GemmaModelCuteDslOnnxExport.forward` 直接调用 `layer.mlp(hidden_states)`——因为替换是
in-place 的，导出时自然走 `FusedGemmaMLP.forward`，无需任何特殊处理：

```python
            residual = hidden_states
            hidden_states, gate = layer.post_attention_layernorm(hidden_states, None)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = _gated_residual(residual, hidden_states, gate)
```

导出后 ONNX 图里的 MLP 子图变为：
`MatMul(gate_up_proj) → Split → Gelu(tanh) → Mul → MatMul(down_proj)`，
建引擎时由 TensorRT Myelin 自行做 epilogue 融合。

---

## 二、MLP 计算过程与优化原理

### 原始 GemmaMLP（pi0.5 用 `gelu_pytorch_tanh`，即 GeGLU）

```python
h = down_proj( act_fn(gate_proj(x)) * up_proj(x) )
```

记 `H=2048`（hidden），`I=16384`（intermediate，Gemma 2B）。三个权重矩阵：

- `W_gate ∈ R^{I×H}`
- `W_up   ∈ R^{I×H}`
- `W_down ∈ R^{H×I}`

原始计算是**两次独立的 FC1 GEMM**（gate 和 up），各自从 HBM 读一次 `x`、读一次权重、
写一次中间结果。

### Fused 后

把 gate / up 在输出维度拼接：

```
W_gate_up = [ W_gate ;        ∈ R^{2I × H}
              W_up   ]
```

一次 GEMM 得到 `[g; u] = W_gate_up · x`，再 `chunk(2, dim=-1)` 逻辑切分：

```
y = W_down · ( GeLU(g) ⊙ u )
```

`FusedGemmaMLP.forward` 即：

```python
def forward(self, x):
    gu = self.gate_up_proj(x)          # [..., 2I]
    gate, up = gu.chunk(2, dim=-1)     # 逻辑 split
    return self.down_proj(self.act_fn(gate) * up)
```

### 优化原理（三点）

1. **省一次 input 读取与一次 kernel launch**：原来 gate、up 两次 GEMM 各读一遍 `x`，
   合并后只读一次、只 launch 一次。
2. **提升算术强度（arithmetic intensity）**：两个 `[I,H]` 的 GEMM 合成一个 `[2I,H]`
   的大 GEMM，M/N 维更大，更容易打满 Tensor Core，减少 wave quantization 损失。
3. **数学严格等价**：仅是权重行拼接 + 输出切分，**零精度损失**
   （smoke test 验证 `max|diff|=0`）。

> 注意：方案 A 本身**不消除中间结果落盘**（`Split/Gelu/Mul` 仍是独立 ONNX 节点），
> 这部分（L2/L3 融合）留给 TensorRT Myelin 在建引擎阶段自动做 epilogue 融合。
> 这也是它"零 plugin、零风险"的代价——把深度融合交给下游编译器。

---

## 三、前后性能对比

| 维度 | 原始 GemmaMLP | Fused（方案 A） | 改进 |
|---|---|---|---|
| FC1 GEMM 次数 | 2（gate + up） | 1（gate_up） | kernel launch ↓50% |
| FC1 input 读取 | 2× 读 `x` | 1× 读 `x` | HBM 读带宽 ↓ |
| FC1 权重 layout | 两个分散矩阵 | 一个连续大矩阵 | 访存连续性↑、算术强度↑ |
| 量化器数量/层 | 3 套（gate/up/down） | 2 套（gate_up/down） | scale 元数据↓、量化 kernel↓ |
| 中间结果落盘 | 有 | 仍有（交给 Myelin 融合） | 需引擎侧 epilogue 融合 |
| 数值精度 | — | 严格等价 `max|diff|=0` | 无损 |
| 参数总量 | `2IH + IH` | 不变 | 不变（仅重排） |

### 收益定位

- **直接收益**：FC1 阶段的 kernel 数与 input 访存减半。对于 Gemma 2B（`I=16384` 很大）
  的 MLP 这是热点；在 prefill（968 token 大 M）和 action expert denoise（多步重复）
  都会反复受益。
- **间接收益**：合并成大 GEMM 后，下游 TensorRT 更容易把
  `gate_up GEMM + Split + Gelu + Mul` 融成一个 epilogue，进一步省掉中间 HBM 往返——
  这部分是"为下游优化铺路"。
- **风险**：极低。唯一需关注的是 gate 与 up 量纲差异较大时 per-tensor weight scale
  可能被拉伸——**已由 roadmap #9 解决**（给 `gate_up_proj.weight_quantizer` 设 `axis=0`
  走 per-channel，原理详见 §四）。

---

## 四、per-channel 权重量化（roadmap #9 · fused MLP 的精度收尾）

> 落地位置：`config/quant/llm_quant_nvfp4_fp8_mix_cutedsl_cfg.py`，把 fused
> `mlp.gate_up_proj` 的 **FP8 `weight_quantizer`** 由 per-tensor(`axis=None`) 改为
> per-channel(`axis=0`)；`input_quantizer`（激活）保持 per-tensor。
>
> **这是精度优化，不是速度优化**——per-tensor / per-channel 在 TRT FP8 GEMM 上是同一条
> kernel 路径，区别只在 scale 粒度。收益是量化输出更接近 BF16 baseline。

### 4.1 量化的本质：一个 scale 把浮点压进 FP8 的小动态范围

FP8 E4M3 只能表示约 `[-448, 448]` 且尾数很短。量化先求缩放因子 `s`：

```
s = amax(W) / 448,   W_q = round(W / s),   Ŵ = s · W_q
```

`amax(W)` 是该 scale 覆盖范围内**绝对值最大的元素**。关键：**范围内所有元素共享同一个
scale，而 scale 由最大值决定**。量化误差正比于 `s`——`s` 越大，量化台阶越粗，小数值损失越重。

### 4.2 per-tensor vs per-channel：scale 覆盖范围不同

fused 后权重 `W_gate_up ∈ R^{2I × H}`（行=输出通道，列=输入维）：

| 模式 | scale 数量 | 公式 |
|---|---|---|
| per-tensor (`axis=None`) | 1（整矩阵共用） | `s = max_{全矩阵}|W| / 448` |
| per-channel (`axis=0`) | `2I`（每输出行一个） | `s_k = max_j |W_{k,j}| / 448` |

per-tensor 用**全矩阵最大值**定 scale，只要有一个离群大值（outlier），整矩阵 scale 被拉大，
幅度小的通道被"压扁"、有效量化位数骤降。

> 直观例子：A 行最大值 100，B 行最大值 1。
> - per-tensor：`s=100/448≈0.223`，B 行在 1 附近只有 ~4 个台阶 → 精度极差。
> - per-channel：B 行单独 `s=1/448≈0.0022`，1 附近有 ~448 个台阶 → 精度高约 100×。

per-channel 让每行用自己的局部最大值定 scale，**离群行不再污染其他行**。

### 4.3 为什么这对 "fused gate_up_proj" 尤其关键

这正是 #9 的动机。fusion 把两支在输出维拼接：

```
W_gate_up = [ W_gate ;   ∈ R^{2I × H}
              W_up   ]
```

- `gate` 后接激活（GeGLU）、`up` 是线性直通——两支权重量纲通常不同（常一支明显更大）。
- **per-tensor 会逼 gate 与 up 共用一个 scale**：大幅度分支决定 amax，小幅度分支精度被牺牲。
  这是 **fusion 引入的新隐患**——融合前它俩是两个独立 Linear、各有 per-tensor scale，互不干扰；
  一拼接，per-tensor 把它们绑死了。
- 改 **per-channel(`axis=0`)** 后，`2I` 个输出通道各自独立 scale，gate 行与 up 行互不影响，
  等价于"融合前各自独立量化"的精度，同时保留 fusion 的速度收益。

一句话：**per-channel 把 fused MLP 从"省速度、丢精度"变成"既省速度、又不丢精度"。**

### 4.4 边界说明

- **不影响速度**：per-channel weight scale 在 FP8 GEMM epilogue 阶段按列乘一个向量 scale，
  开销可忽略，kernel 不变。
- **激活仍 per-tensor**：激活 scale 沿收缩维（contraction dim），per-channel 对 GEMM 数学上
  不成立，故 `input_quantizer` 保持 `axis=None`。per-channel 只用在 weight。
- **NVFP4 层无需此改动**：NVFP4 weight 本就沿**输入维**按 16 元素 block 做 block-scale（比
  per-channel 还细），而拼接是沿**输出维**、不混 block，天然安全。只有被显式指定成
  **FP8 per-tensor** 的那几层（mix cfg 第 3–17 层部分）有隐患，#9 正是对症这些层。

### 4.5 代价

仅多存 `2I` 个 scale 标量，可忽略。

---

## 五、踩坑记录

### 5.1 NVFP4 量化后 TensorRT 编译报 `TRT_FP4QDQ ... Plugin not found`

**现象**：cutedsl 路径（`LLMWithCuteDsl`）NVFP4 量化导出的 `llm.onnx`，在 TensorRT
编译时报错，节点常落在 `mlp/gate_up_proj` / `mlp/down_proj`：

```
operator: TRT_FP4QDQ (checkFallbackPluginImporter): INVALID_NODE:
creator && "Plugin not found, are the plugin name, version, and namespace correct?"
```

**根因**：ModelOpt 对 NVFP4 权重导出的是**自定义算子 `TRT_FP4QDQ`**，它不是 TensorRT
原生 plugin，必须经 `modelopt.onnx.quantization.qdq_utils.fp4qdq_to_2dq()` 转换成
**两级标准 `DequantizeLinear`**（block scale + global scale）后 TensorRT 才能识别。

报错节点指向 `gate_up_proj` / `down_proj` 只是日志先撞上 MLP 层；实际上**所有** NVFP4
量化的 `nn.Linear`（含 attention q/k/v/o）都带 `TRT_FP4QDQ`，**与 fused MLP 本身无关**。

**关键约束：三条 LLM 导出路径都必须在 `export()` 后调用 `_nvfp4_post_processing`。**

| 路径 | `quantize()` 是否调 `_nvfp4_post_processing` |
|---|---|
| `llm.py`（`LLM`） | ✅ L320-322 |
| `llm_with_trtedgellm.py`（`LLMWithTrtEdgeLLM`） | ✅ L753-755 |
| `llm_with_cutedsl.py`（`LLMWithCuteDsl`） | ✅ 已补（早期版本缺失，即本坑） |

**注意**：不能直接复用基类 `Model._nvfp4_post_processing`——它首行
`self.model.save_pretrained(...)` 假设 `self.model` 是 HF `PreTrainedModel`；而
cutedsl 路径的 `self.model` 是 `Pi05CuteDslLanguageModel`（普通 `nn.Module`，真正的
HF 解码器藏在 `_hf_gemma`），没有 `save_pretrained`，会抛 `AttributeError`。故
`LLMWithCuteDsl` 用了适配版：从 `_hf_gemma` 取 config 容错保存，核心
`fp4qdq_to_2dq` 转换逻辑与其它路径一致。

**自查**：导出后确认 ONNX 里不再有 `TRT_FP4QDQ` 节点：

```python
import onnx
m = onnx.load("llm.onnx")
assert not any(n.op_type == "TRT_FP4QDQ" for n in m.graph.node), "缺 nvfp4 后处理"
```

---

## 六、后续（方案 B / C）

若需进一步的 L2/L3 深度融合（消除中间落盘、CuTe DSL epilogue fusion），属于方案 B/C
范畴，需要写 plugin；当前方案 A 是无损、零依赖的第一步。详见 `kernelSrc/docs/fused_mlp.md`。
