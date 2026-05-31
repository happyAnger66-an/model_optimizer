# AdaRMS Dense 预计算（Pi0.5 action expert / DiT denoise）

> roadmap #22。FlashRT 实证：**编译器路径 -5.5ms**（v1.4 → v1.6-L2），是 FlashRT 那批优化里
> **最该移植到 model_optimizer（编译器/Myelin 路径）的一项**——因为它靠的是"图简化让编译器选到更好 tactic"，
> 而非"抛弃编译器手写 kernel"。
>
> 关联：`docs/optimizer/roadmap.md` §8 / #22；`docs/flashRT/pipeline.md`；FlashRT `docs/optimization-details.md` §1.4。

---

## 1. 背景：AdaRMS 在 Pi0.5 expert 里是什么

Pi0.5 的 action expert（denoise/DiT）用 **自适应 RMSNorm（Adaptive RMSNorm, AdaRMS）** 把"扩散时间步 t"
作为条件注入每一层归一化。代码见
`third_party/openpi/.../gemma/modeling_gemma.py::GemmaRMSNorm.forward`：

```python
# cond = adarms_cond，形状 [batch, cond_dim]，与 token 无关
modulation = self.dense(cond)              # Linear: cond_dim → dim*3   ← 关键 GEMM
modulation = modulation.unsqueeze(1)       # [batch, 1, dim*3] 广播到 [batch, seq, dim*3]
scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
normed = self._norm(x) * (1 + scale) + shift     # 仿射调制
return normed, gate                              # gate 用于 _gated_residual
```

条件 `adarms_cond` 的来源（`dit.py::_embed_suffix_pi05`）：

```
timestep(标量 t)
   └─ create_sinusoidal_pos_embedding(t)      # sin/cos 位置编码，仅依赖 t
        └─ time_mlp_in → SiLU → time_mlp_out → SiLU
             └─ adarms_cond   [batch, cond_dim]
```

每个 decoder layer 有 **2 个 AdaRMS**（`input_layernorm`、`post_attention_layernorm`），
外加 1 个 **final `norm`**，每个 AdaRMS 内含一个 `dense`（`cond_dim → dim*3` 的 Linear）。

**计数**：`(2 × num_layers + 1) × num_denoise_steps = (2×18 + 1) × 10 = 37 × 10 = 370` 个 Dense GEMM。
这就是 FlashRT 文档里"370 Dense GEMMs computed inside the engine every inference"的来源。

---

## 2. 核心洞察：这 370 个 Dense 全是"静态常量"

关键事实链：

1. **`adarms_cond` 只依赖 `timestep`**，与 `x_t`（噪声动作）、prefix KV、观测输入**完全无关**。
   （链路里没有任何一项来自 image/state/action token。）
2. **denoise 的时间步调度是固定的**：flow-matching 采样用确定的 `num_steps` 与固定 `dt`，
   N 个时间步 `t_0, t_1, …, t_{N-1}` 是**编译/部署期就确定的常量**，不随推理输入变化。
3. **`dense` 是冻结权重**（推理期不变）。

三者叠加 ⇒ 每个 step、每个 norm 的 `modulation = dense(cond(t_k))` 及其拆出的
`(scale, shift, gate)` 都是**与具体推理输入无关的常量**，可以**离线一次性算好**：

```
style[k, i] = dense_i( SiLU(time_mlp_out(SiLU(time_mlp_in(sinusoid(t_k))))) )
            = (scale, shift, gate)            # k=step, i=norm 索引；共 N×37 个常量
```

所以这 370 个 GEMM **根本不需要在每次推理的 GPU 图里跑**——把结果当常量注入即可。
FlashRT 就是在 `set_prompt()` 时用 Python/CPU 预算所有 `style`，再注入。

---

## 3. 为什么能省 5.5ms（不是省 GEMM 算力，是省"图复杂度"）

这 370 个 Dense 每个都极小（`[1, cond_dim] × [cond_dim, dim*3]`，batch 通常 = 1）。
denoise 段 **S=10（action horizon）是 launch-bound**（host 启动开销主导，不是算力主导）。因此：

- **直接收益**：370 个 tiny GEMM 从 GPU 图里消失 → 少 370 次 kernel 调度 / 中间张量。
- **更大的间接收益（关键）**：去掉这些零碎小算子后，**主 transformer 图被显著简化**，
  Myelin 能把剩下的 norm 仿射、gated residual、attention、FFN **融合得更彻底、选到更好的 tactic**。
  roadmap/FlashRT 都标注 “graph simplification → better Myelin tactic”。

> 一句话：**这是"编译器友好"型优化**——交给编译器的图更干净，编译器自己就跑得更快。
> 与 FlashRT 那 26ms 大头（手写 fused kernel、抛弃编译器）性质不同，**这一项在 TRT/Myelin 路径上照搬即可受益**。

---

## 4. 在 model_optimizer 上的实现思路

当前 `dit.py` 的导出形态：**单步** `denoise.onnx`，`timestep` 是**运行时输入**，host 循环喂 N 个不同的 t；
`dense(cond)` 嵌在 `gemma_expert` 的每个 RMSNorm 里，被一起导进 ONNX。目标是把 `dense` 从图里拿掉。

### 方案 A（推荐，改动小）：modulation 作为图输入 + AdaRMS 退化为纯仿射

把"条件 → modulation"的整条链从导出图里剥离，`timestep` 不再进图，改成把**预算好的 modulation**喂进去。

1. **新增导出开关（feature）**：如 `adarms_precompute`，开启时改写 `GemmaRMSNorm.forward`：
   不再 `self.dense(cond)`，而是消费外部传入的 `(scale, shift, gate)`，AdaRMS 变成纯 elementwise：
   `normed*(1+scale)+shift`、返回 `gate`。`dense` 不出现在图里。
2. **改 `Pi05DenoiseStep.forward` / `export` 的签名**：
   - 移除 `timestep` 入图（或保留但不参与计算）；
   - 新增入参 `adarms_mod`（把 37 个 norm 的 modulation 打包，布局如
     `[num_norms, batch, dim*3]` 或按 `(layer, sub-block)` 展平），由 host 按当前 step 选切片喂入。
3. **host 侧预计算（一次性）**：在引擎构建 / prompt-set 期，用原始 PyTorch 子图算出
   `mod[k, i]`（k∈N 步，i∈37 norm），缓存为常量张量。denoise 循环第 k 步直接取 `mod[k]`。
4. **engine 仍是单份**，N 份常量驻留 host，按 step 索引。

> 这样图里彻底没有 `time_mlp_*` / `sinusoid` / `dense`，只剩仿射调制 + 主干。

### 方案 B（最激进，最贴近 FlashRT）：每步常量折叠

把 step-specific 的 modulation 直接 bake 成 ONNX **initializer（常量）**：

- 要么导出 **N 份特化图**（每步把 37 个 modulation 写死成常量），
- 要么导出 **1 份图 + step 索引**，在图里用常量张量按 step gather。

`do_constant_folding=True` 会把这些常量彻底折进相邻算子，Myelin 看到的图最干净。
代价：N 份图 / 较大常量、与固定调度强耦合。

### 改动落点清单（✅ = 已实现，全部封装在 model_optimizer，未改原始 `modeling_gemma.py`）

| 文件 | 改动 | 状态 |
|---|---|---|
| `models/pi05/dit.py` 模块级 `_adarms_injected_forward` | `GemmaRMSNorm.forward` 的"注入版"（读 `_injected_mod`，不调 `dense`），数学等价 | ✅ |
| `dit.py::Pi05DenoiseStep.enable_adarms_precompute` | 运行时把 expert 内 37 个 AdaRMS 的 `forward` 替换为注入版，可还原 | ✅ |
| `dit.py::precompute_adarms_modulation` | 用真实 `dense` 离线预算打包 modulation `[num_norms,batch,dim*3]` | ✅ |
| `dit.py::Pi05DenoiseStep.{forward,export}` | 第 5 入参 `time_or_mod`：默认 `timestep`，预计算模式 `adarms_mod`；导出移除 sinusoid/time_mlp/dense | ✅ |
| `dit.py::quantize` | 校准走 timestep 路径（calib 数据喂 timestep），仅导出时切到预计算 | ✅ |
| `dit.py` 特性注册 `adarms_dense_precompute` + `__init__(feature_config=...)` | 与 `llm.py` 同构，可在 JSON 配置启停 | ✅ |
| `infer/pi05_adarms.py::AdaRmsModulator` | host 侧按时间步值记忆化预算 modulation（dense 只在 torch 全精度跑有限次） | ✅ |
| `infer/{tensorrt,onnxrt}/pi05_executor.py` denoise 段 | 启用时喂 `adarms_mod` 取代 `timestep` | ✅ |
| `infer/server/{config,policy_loader}.py` | 新增 `tensorrt/onnxrt.denoise_adarms_precompute`（serve JSON 可配） | ✅ |

> 注：本实现采用**方案 A**（modulation 作为图输入 + host 记忆化预算），未改原始 `modeling_gemma.py`
> （改用运行时 monkey-patch）。方案 B（ONNX 常量折叠）保留为后续可选项。

### 启用方式（三选一，优先级 高→低）

1. **导出 / 量化（feature_config）**：JSON 写 `{"features": {"adarms_dense_precompute": true}}`，
   传给 `Pi05DenoiseStep.construct_model(..., feature_config=...)` → 导出 `adarms_mod` 引擎。
2. **serve 配置**：服务 JSON 里 `tensorrt.denoise_adarms_precompute=true`（或 `onnxrt.*`）→ host 自动预算并喂入。
3. **环境变量**：`PI05_ADARMS_PRECOMPUTE=1`（导出端与 host 端均识别），便于快速 A/B。

---

## 5. 风险与注意事项

- **调度一致性**：预算依赖固定的 `num_steps / dt / sinusoid(min_period=4e-3, max_period=4.0)`。
  若运行时调度可变，必须在调度变更时（如 prompt-set/build 期）**重算 modulation**，否则结果错。
- **dtype 对齐**：`adarms_cond` 链路在 fp32 计算，`(scale, shift, gate)` 注入时与 expert 权重 dtype（bf16）对齐，
  仿射在 fp32 做（与现实现 `normed*(1+scale.float())+shift.float()` 一致）。
- **batch 维度**：cond 是 `[batch, cond_dim]`，pi05 采样中同一 step 的 `expanded_time` 跨 batch 相同 → modulation 可 batch 广播；
  若未来支持每样本不同 t，预算需按样本展开。
- **量化交互**：dense 是极小 GEMM，移出图后也无需对其做 FP8 量化，量化目标更聚焦在主干 GEMM。
- **数值等价校验**：开关前后必须对 `v_t` 做逐步 cos 比对（建议门禁 cos ≥ 0.999），确保仿射常量注入与原 `dense(cond)` 一致。

---

## 6. 与 FlashRT 的对照

| 维度 | FlashRT（手写） | model_optimizer（本方案，编译器路径） |
|---|---|---|
| 预算时机 | `set_prompt()` Python/CPU 预算 `style` | 引擎构建 / prompt-set 期预算 `mod[k,i]` |
| 注入方式 | 常量指针，`(s*layers+l)*S*D3` 偏移索引 | 方案 A：图输入切片；方案 B：ONNX 常量折叠 |
| 收益来源 | 去 370 GEMM + 全程同套 kernel 复用 | 去 370 GEMM + **图简化让 Myelin 选更优 tactic** |
| 实测 | -5.5ms（v1.4→v1.6-L2） | 预期同量级（编译器友好，可直接移植） |

---

## 7. 使用指南（export → quantize → build → serve 全流程）

> 组件注册名 `pi05_libero/denoise`（`models/registry.py`），类 `Pi05DenoiseStep`（`models/pi05/dit.py`）。
> 统一 CLI：`model-opt`（= `model-optimizer-cli`，`src/model_optimizer/cli.py`）。
> **贯穿全程的开关一致性**：export 一旦开启预计算，ONNX 第 5 输入变成 `adarms_mod`，
> 那么 **build_cfg 必须改名/改 shape**、**serve 必须置 `denoise_adarms_precompute=true`**，三处缺一不可。

### 准备：feature_config JSON

```jsonc
// config/feature_configs/adarms_precompute.json
{
  "version": 1,
  "features": { "adarms_dense_precompute": true },
  "export":   { "dynamo": false, "prefix_len": 968 }
}
```

> `export.prefix_len` 覆盖导出假输入的 prefix 长度（默认 968，仅影响 tracing，prefix 维是动态轴）。
> 把它设成与 `build_cfg` 的 `_PREFIX_LEN` 一致，可消除 export(968) 与 build(818) 的历史不一致。

也可不写 JSON，改用环境变量 `PI05_ADARMS_PRECOMPUTE=1`（export/quantize/serve 均识别）。

### 步骤 1｜Export（bf16，无量化）

```bash
model-opt export \
  --model_name pi05_libero/denoise \
  --model_path /path/to/pytorch_pi05_checkpoint \
  --export_dir /tmp/export/pi05 \
  --feature_config adarms_precompute.json
```

- CLI 经 `convert/convert_formt.py` 把 `FeatureConfig` 透传到 `construct_from_name_path` → `Pi05DenoiseStep`。
- `__init__` 里 `apply_features` 命中 `adarms_dense_precompute` → `enable_adarms_precompute(True)`。
- `export()` 走预计算分支：第 5 输入为 `adarms_mod`，图内**无 sinusoid / time_mlp / dense**。
- **关注日志**：`[adarms] precompute export: N norms × dim*3=D, dense GEMM 已移出图`
  —— 记下 `N`（=2×层数+1，18 层即 **37**）与 `D`（=expert hidden_size×3），build_cfg 要用。

产物：`/tmp/export/pi05/denoise.onnx`（输入 `prefix_pad_masks/past_keys/past_values/x_t/adarms_mod`）。

### 步骤 2｜Quantize（可选，FP8/NVFP4）

```bash
model-opt quantize \
  --model_name pi05_libero/denoise \
  --model_path /path/to/pytorch_pi05_checkpoint \
  --quantize_cfg config/quant/denoise_quant_fp8_cfg.py \
  --calibrate_data /path/to/denoise_calib.pt \
  --export_dir /tmp/quantize/pi05 \
  --feature_config adarms_precompute.json
```

- **关键**：校准数据喂的是 `timestep`，故 `quantize()` 会**自动临时关闭**预计算走时间链路校准，
  校准完再恢复、以预计算模式重新导出（`dit.py::quantize`）。无需手动干预。
- dense 是极小算子，预计算后不出现在导出图，因此其量化器在导出时自然丢弃；量化聚焦主干 GEMM。
- 产物同样是 `denoise.onnx`（含 QDQ；NVFP4 时再过 `_nvfp4_post_processing`）。

### 步骤 3｜Build（ONNX → TensorRT engine）

预计算后输入名/shape 变了，**必须改 build_cfg**。基于 `config/build_configs/denoise_step_build_cfg.py` 复制一份：

```python
# denoise_step_build_cfg_adarms.py
_NUM_LAYERS = 18
_PREFIX_LEN = 818          # 按实际 prefix 长度对齐
_ACTION_HORIZON = 10
_ACTION_DIM = 32
_HEAD_DIM = 256
_NUM_NORMS = 2 * _NUM_LAYERS + 1     # = 37，取 export 日志的 N
_DIM3 = 3072                          # = export 日志的 D（expert hidden_size×3），按实际填

def _shapes():
    return {
        "prefix_pad_masks": (1, _PREFIX_LEN),
        "past_keys":  (_NUM_LAYERS, 1, _PREFIX_LEN, _HEAD_DIM),
        "past_values": (_NUM_LAYERS, 1, _PREFIX_LEN, _HEAD_DIM),
        "x_t": (1, _ACTION_HORIZON, _ACTION_DIM),
        "adarms_mod": (_NUM_NORMS, 1, _DIM3),   # ← 取代 timestep
    }

build_cfg = {
    "precision": "bf16",
    "strongly_typed_network": True,
    "workspace_mb": 8192,
    "min_shapes": _shapes(),
    "opt_shapes": _shapes(),
    "max_shapes": _shapes(),
}
```

```bash
model-opt build \
  --model_path /tmp/export/pi05/denoise.onnx \
  --build_cfg config/build_configs/denoise_step_build_cfg_adarms.py \
  --export_dir /tmp/build/pi05/denoise.engine
```

> `--export_dir` 在 build 子命令里是 **engine 输出文件完整路径**（非目录）。

### 步骤 4｜Serve / Infer（host 侧按步预算并喂入）

serve JSON 置 `denoise_adarms_precompute=true`：

```jsonc
{
  "checkpoint": "/path/to/pytorch_pi05_libero",
  "config_name": "pi05_libero",
  "mode": "tensorrt",
  "precision": "bf16",
  "tensorrt": {
    "engine_path": "/tmp/build/pi05",
    "vit_engine": "vit.engine",
    "llm_engine": "llm.engine",
    "expert_engine": "expert.engine",
    "denoise_engine": "denoise.engine",
    "embed_prefix_engine": "embed_prefix.engine",
    "denoise_adarms_precompute": true
  }
}
```

启动后日志出现 `[adarms] denoise host 侧预计算已启用（喂 adarms_mod）`：
- `policy_loader._mount_tensorrt_engines` 把开关透传给 executor；
- `Pi05TensorRTExecutor` 用 `infer/pi05_adarms.py::AdaRmsModulator` 在每个 denoise step
  按 `timestep` 值**记忆化**预算 `adarms_mod`（dense 只在 torch 全精度跑有限次，首步后命中缓存），
  以 `adarms_mod=` 取代 `timestep=` 喂引擎。

> ONNX Runtime 后端同理：`onnxrt.denoise_adarms_precompute=true`。

### 一致性自检表

| 项 | 默认模式 | 预计算模式 |
|---|---|---|
| ONNX 第 5 输入 | `timestep (B,)` | `adarms_mod (N, B, D)` |
| build_cfg shapes | 含 `timestep` | 改为 `adarms_mod` |
| serve 开关 | 关 | `denoise_adarms_precompute=true` |
| 图内是否含 sinusoid/time_mlp/dense | 是 | **否** |

> 三处任一不匹配的典型报错：build 阶段 profile 输入名找不到、或 serve 阶段引擎绑定输入名不符。

---

## 8. 当前 Pi0.5 整体推理流程总结

### 5 阶段串联（`PI05_STAGES = (vit, embed_prefix, llm, expert, denoise)`）

```
observation
  → [vit]          SigLIP 多视角图像 → 视觉 token        (get_image_features)
  → [embed_prefix] 视觉 token + 语言嵌入 拼成 prefix      (embed_prefix)
  → [llm]          paligemma LLM 对 prefix 做 prefill → past_key_values(KV cache)
  → sample_actions  Euler 去噪循环（PyTorch 宿主，未进引擎）：
        x_t ← 随机噪声;  dt = -1/num_steps;  time = 1.0
        repeat num_steps(默认 10) 次:
          v_t = denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)
                 └─ [expert] action expert 读同一份 prefix KV + AdaRMS(time) → v_t
          x_t ← x_t + dt * v_t      # Euler 更新（图外宿主）
  → actions (= x_t)  → 后处理(LiberoOutputs 截断维度) → 返回动作 chunk
```

- **宿主循环**：`third_party/openpi/.../pi0_pytorch.py::sample_actions`（prefill + Euler 循环）。
  TRT 路径加载后 `_restore_eager_sample_actions`，**循环仍在 PyTorch**，仅**单步 `denoise_step` 进引擎**。
- **denoise 单步**：`Pi05DenoiseStep`（`dit.py`）= `embed_suffix`(action 投影 + 时间条件) + expert forward + `action_out_proj`。

### 各阶段后端替换（`infer/tensorrt/pi05_executor.py` 运行时挂载）

| 阶段 | 替换对象 | 触发条件 |
|---|---|---|
| vit | `paligemma.model.get_image_features` | `tensorrt.vit_engine` |
| embed_prefix | `pi05_model.embed_prefix` | `embed_prefix_engine`，或 `use_flashrt_siglip_embed_prefix`/`stages.vit=flashrt`（FlashRT SigLIP） |
| llm | `language_model.forward` | `tensorrt.llm_engine` |
| expert | `gemma_expert.model.forward` | `tensorrt.expert_engine` |
| **denoise** | `pi05_model.denoise_step` | `tensorrt.denoise_engine`（含 AdaRMS 预计算分支） |

> 每阶段可独立选 PyTorch / TensorRT / ONNX / FlashRT，详见 `flashrt_backend_design.md`（分阶段后端矩阵）。

### AdaRMS 预计算在该流程中的位置

- **离线（export/build 期）**：把 expert 内 37 个 AdaRMS 的 `dense(cond)` 移出 denoise 引擎图。
- **在线（每个 denoise step）**：host 的 `AdaRmsModulator` 按 `timestep` 预算 `adarms_mod` 喂入引擎；
  因调度固定，仅前几步真正算 dense，其后全部命中缓存 → 引擎图更干净、launch 更少。

---

## 9. 叠加 fused MLP（gate/up 合并，roadmap #9）

AdaRMS 预计算把 370 个 `dense` GEMM 移出图后，denoise expert 仍处于 **launch-bound** 区间（suffix S=10）。
此时可叠加 **fused MLP**：把 expert 每层 `GemmaMLP` 的 `gate_proj`/`up_proj` 在输出维 concat 成单个
`gate_up_proj`（一次 FC1 GEMM），与原计算 **数学等价、无精度损失**。

```text
原：g = gate_proj(x); u = up_proj(x); h = down_proj(act(g) * u)   # 2 次 FC1 GEMM/层
新：gu = gate_up_proj(x); g,u = gu.chunk(2,-1); h = down_proj(act(g)*u)  # 1 次 FC1 GEMM/层
```

### 收益（与 AdaRMS 同向，可叠加）

- 每层省 1 个 GEMM launch：18 层 → 每 denoise step 少 18 次；`num_steps=10` → 整段动作生成少约 **180 次** GEMM launch。
- FC1 权重一次连续 HBM load（`[2I, H]`）、算术强度更高；后续 `chunk/act/mul/down` 仍为 ONNX 原生 op，由 Myelin 融合。
- 纯 ONNX 图重写（concat 权重→更大 MatMul），**编译器友好**，与手写融合 kernel（roadmap #13 实测会 regress）性质不同；该实现已在 LLM/cutedsl 路径上线验证。

### 实现（复用 `fused_mlp.py`，仍不改原始 `modeling_gemma.py`）

- `dit.py` 注册 `fused_mlp` 特性（`supported_models=None`、`default_enabled=True`，与 `llm.py` 一致）；
  `apply_features` 的 target 统一传 `gemma_expert`，`install_fused_mlp` 就地替换每层 `.mlp` 为 `FusedGemmaMLP`。
- adarms 经 `ctx.extra["denoise_step"]` 回到 `Pi05DenoiseStep`，与 fused_mlp 共用一次 `apply_features`，互不干扰
  （fused_mlp 改 `.mlp`，adarms 改 layernorm.forward）。
- 直接构造路径新增 `fuse_mlp: bool=False` 开关；`construct_model(..., fuse_mlp=...)` 透传。
- 时序：`fused_mlp` 在 `__init__` 完成，早于 `quantize()`，ModelOpt 直接量化合并后的 `gate_up_proj`；
  `_LinearMetaShim` 保证不会给已合并的 gate/up 重复插量化器。

### 量化注意（必看）⚠️

合并后模块名变为 `*layers.{i}.mlp.gate_up_proj`，**针对 `gate_proj`/`up_proj` 的旧量化规则不再命中**。

**FP8 必须 per-tensor**：标准 ONNX 的 FP8(E4M3) 导出只支持 per-tensor（ModelOpt
`tensor_quantizer._check_onnx_readiness` 要求 `amax` 为标量）。对 `gate_up_proj` 设 per-channel（axis=0）
会在 `torch.onnx.export` 阶段断言失败：

```text
AssertionError: E4M3 supports ONNX export only for per-tensor quantization.
Received non-scalar amax of shape: torch.Size([8192, 1])
```

> per-channel FP8 仅在 **TRT 插件 QDQ 导出路径**（如 cutedsl LLM，会产出 `TRT_FP8QDQ`/`TRT_FP4QDQ` 节点）可用；
> denoise 走标准 `torch.onnx.export`，故 `gate_up_proj` 用 **per-tensor FP8**（`FP8_DEFAULT_CFG` 默认即此）。
> 若担心 gate/up 量纲混叠影响精度，改用 **NVFP4**：沿输入维 block 量化，按输出维 concat 不混 block，天然无此问题。

### 配套配置（新增）

| 用途 | 文件 |
|---|---|
| feature_config（同时开 adarms + fused_mlp） | `config/feature_configs/denoise_fused_adarms.json` |
| 量化（`gate_up_proj` per-tensor FP8） | `config/quant/denoise_fused_adrams_st_quant_fp8_cfg.py` |
| 编译（I/O 与纯 adarms 版相同，单列仅为命名清晰） | `config/build_configs/denoise_fused_adrams_build_cfg.py` |

> 注：`config/feature_configs/adarms_precompute.json`（纯 adarms）已显式写入 `"fused_mlp": {"enabled": false}`，
> 避免 fused_mlp 全局默认 True 被误开、破坏旧量化配置的规则匹配。

### 命令

```bash
# 量化导出（feature_config 同时开 fused_mlp + adarms；用新量化配置）
model-opt quantize \
  --model_name pi05_libero/denoise \
  --model_path /srcs/openpi/pytorch_pi05_libero/ \
  --quantize_cfg config/quant/denoise_fused_adrams_st_quant_fp8_cfg.py \
  --calibrate_data /data/pi05/denoise/ \
  --export_dir /tmp/quantize/pi05/denoise_fused_st \
  --feature_config config/feature_configs/denoise_fused_adarms.json

# 编译
model-opt build \
  --model_path /tmp/quantize/pi05/denoise_fused_st/denoise.onnx \
  --export_dir /tmp/pi05/build/quant/denoise_fused_st_fp8.engine \
  --build_cfg config/build_configs/denoise_fused_adrams_build_cfg.py
```

### 验证

- 数值：开/关 fused_mlp 对 `v_t` 逐元素 cos 应 ≥ 0.999（数学等价）。
- 图：每层 MLP 由两个 MatMul 变一个；denoise engine kernel 数下降。
- 量化：`mtq.print_quant_summary` 出现 `gate_up_proj` 量化器（per-tensor FP8），且无 `gate_proj/up_proj` 重复量化。

---

## 10. 一句话总结

> Pi0.5 denoise 的 AdaRMS 时间调制 **只依赖固定的扩散时间步调度**，与推理输入无关；
> 因此 370 个 `dense(cond)` GEMM 全是**离线可算的常量**。把它们从导出图里剥离（modulation 作为输入或常量折叠），
> 既省掉 370 次 launch，又让 Myelin 在更干净的图上选到更好 tactic —— FlashRT 实测 **-5.5ms**，
> 是最值得优先移植的编译器友好优化。

---

## 11. 修订记录

| 日期 | 内容 |
|---|---|
| 2026-05-31 | 初版：AdaRMS Dense 预计算技术细节与 model_optimizer 实现思路（roadmap #22） |
| 2026-05-31 | 落地方案 A：`dit.py` 运行时 monkey-patch + `adarms_mod` 导出；host 侧 `AdaRmsModulator` 记忆化预算；feature_config / serve JSON / env 三种可配开关。未改原始 `modeling_gemma.py`。 |
| 2026-05-31 | 新增 §7 使用指南（export/quantize/build/serve 全流程 + 一致性自检表）与 §8 整体推理流程总结。 |
| 2026-05-31 | 新增 §9 叠加 fused MLP（gate/up 合并，roadmap #9）：实现接线、per-channel 量化注意、配套配置与命令。 |
