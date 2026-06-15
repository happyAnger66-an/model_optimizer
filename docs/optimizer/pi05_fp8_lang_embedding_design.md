# π0.5 FP8 Language Embedding 设计方案（model_optimizer）

> 目标：在 model_optimizer 中为 π0.5 **语言 prompt token embedding** 接入 Edge-LLM 同构的 FP8 sidecar 路径，并补齐可 A/B 对比的耗时与显存统计，量化「静态表体积 ↓ ~50%」与「单次 lookup 带宽/latency」收益。

---

## 1. 背景与范围

### 1.1 优化对象

| 组件 | 是否纳入 | 说明 |
|------|----------|------|
| PaliGemma `embed_tokens`（语言 prompt） | **是** | `embed_prefix` 中 `lang_tokens → lang_emb`，V=257152，H=2048 |
| SigLIP 视觉 patch embedding | **否** | 独立 ViT 权重；已有 NVFP4 / FlashRT 路径 |
| Action expert `action_in_proj` | **否** | 连续动作，非 token lookup |
| LLM / Expert TRT 主图 | **否** | 只吃 `inputs_embeds`，不含 embedding 表 |

### 1.2 与现有方案的区别

| 路径 | 现状 | 本方案 |
|------|------|--------|
| `embed_prefix_quant_fp8_cfg.py` | ModelOpt **整图** FP8 PTQ（ViT + lang 一起校准） | **仅语言表** sidecar，per-block max-abs，**无需 calibration** |
| Edge-LLM `llm_loader --fp8-embedding` | LLM runtime + `embedding.safetensors` | 算法复用；挂载点改为 π0.5 hybrid `embed_language_tokens` |
| 整图 `embed_prefix.engine` | 1GB+ 词表 baked 进 TRT | Phase 1 **不改造**；优先 hybrid 路径 |

### 1.3 π0.5 三条 embed_prefix 推理路径

```text
vit_batch_views     TRT ViT + PyTorch embed_language_tokens     ← Phase 1 主目标
flashrt_siglip      FlashRT ViT + PyTorch embed_language_tokens ← Phase 1 主目标
whole_graph         embed_prefix.engine（图内含 embed_tokens）  ← Phase 2 可选拆分
```

---

## 2. 总体架构

```text
Export / 离线
─────────────────────────────────────────────────────────────
paligemma.language_model.embed_tokens.weight  [V, H] FP16/BF16
        │
        ▼ quantize_embedding_to_fp8 (block=128, E4M3)
lang_embedding.safetensors
  ├─ embedding:       float8_e4m3fn  [V, H]
  └─ embedding_scale: float32        [V, H/128]
lang_embedding.meta.json
  └─ vocab_size, hidden_size, block_size, format=fp8_e4m3

Runtime / 推理
─────────────────────────────────────────────────────────────
lang_tokens ──► Fp8LangEmbeddingLookup ──► FP16/BF16 [B, S, H]
                      │                              │
                      │ 读 FP8 sidecar                × sqrt(H)
                      ▼                              ▼
                 concat with vision embs ──► prefix_embs → LLM prefill
```

**设计原则**

1. **Sidecar 与 ONNX/TRT 解耦**：大表不进 `embed_prefix.onnx`，与 Edge-LLM 一致。
2. **对外接口不变**：仍替换 `PaliGemmaWithExpertModel.embed_language_tokens`，`sample_actions` 无感。
3. **可回退**：无 sidecar 或 `use_fp8_lang_embedding=false` 时走原 `nn.Embedding`。
4. **可度量**：静态显存 + 子阶段 wall time + 估算带宽，支持 FP16 baseline A/B。

---

## 3. 模块设计

### 3.1 量化与 sidecar 写入

**新文件**：`src/model_optimizer/quantization/fp8_lang_embedding.py`

```python
FP8_EMBEDDING_BLOCK_SIZE = 128
FP8_E4M3_MAX = 448.0

def quantize_embedding_to_fp8(weight: Tensor) -> tuple[Tensor, Tensor]: ...

def save_lang_embedding_sidecar(
    weight: Tensor,
    export_dir: Path,
    *,
    filename: str = "lang_embedding.safetensors",
) -> LangEmbeddingMeta: ...

def load_lang_embedding_sidecar(engine_dir: Path) -> Fp8LangEmbeddingTables: ...
```

- 算法与 Edge-LLM `tensorrt_edgellm/quantization/embedding_quantization.py` **逐行对齐**（避免 C++ kernel 对接时 scale 不一致）。
- 校验：`hidden_size % 128 == 0`（PaliGemma H=2048 满足）。
- 产物与 engine 同目录：`engines/lang_embedding.safetensors` + `lang_embedding.meta.json`。

**接入 export**

| 入口 | 改动 |
|------|------|
| `Pi05EmbedPrefix.export()` | 新增参数 `fp8_lang_embedding: bool = False`；export 末尾写 sidecar |
| `quantize_sub_model("embed_prefix")` CLI | `--fp8-lang-embedding` 或 feature flag |
| 独立工具 | `scripts/tools/export_fp8_lang_embedding.py --checkpoint ... --out-dir ...` |

**注意**：开启 FP8 sidecar 时，**整图 export 仍可用 FP16 `embed_tokens` 占位**（whole_graph 路径 Phase 2 再拆）；hybrid 路径 runtime **不读** PyTorch `nn.Embedding` 权重，只读 sidecar。

### 3.2 Runtime Lookup 模块

**新文件**：`src/model_optimizer/infer/kernels/fp8_lang_embedding.py`

```python
class Fp8LangEmbeddingLookup(nn.Module):
    """token_ids [B,S] int64 → hidden [B,S,H] fp16/bf16"""

    def __init__(self, tables: Fp8LangEmbeddingTables, *, dtype=torch.bfloat16): ...

    def forward(self, token_ids: Tensor) -> Tensor: ...
    def memory_stats(self) -> LangEmbeddingMemoryStats: ...
```

**Phase 1 实现**：PyTorch 向量化 lookup（`embedding_fp8[token_ids]` + block dequant），足够做 A/B 与正确性验证。

**Phase 2 实现**：绑定 Edge-LLM `embeddingLookup` CUDA kernel（`cpp/kernels/embeddingKernels/`），接口不变。

**挂载**（`pi05_trt_engine_setup.py`）：

```python
def _install_fp8_lang_embedding(model, engine_root, opts):
    tables = load_lang_embedding_sidecar(engine_root)
    lookup = Fp8LangEmbeddingLookup(tables, dtype=model.dtype)
    def embed_language_tokens(token_ids):
        with stage_perf.timed("embed_prefix.lang_embedding"):
            return lookup(token_ids)
    model.paligemma_with_expert.embed_language_tokens = embed_language_tokens
```

在 `install_embed_prefix_batched()` / `install_embed_prefix_flashrt()` 内，lang 分支改为：

```python
with stage_perf.timed("embed_prefix.lang_embedding"):
    lang_emb = embed_language_tokens(lang_tokens)
with stage_perf.timed("embed_prefix.lang_scale_concat"):
    lang_emb = lang_emb * math.sqrt(...)
    # concat ...
```

### 3.3 配置项

`ServerConfig` / build config 新增：

```yaml
embed_prefix:
  use_fp8_lang_embedding: true          # 默认 false，A/B 时显式开启
  lang_embedding_sidecar: auto          # auto = engine_dir/lang_embedding.safetensors
```

环境变量（与现有 perf 体系一致）：

| 变量 | 作用 |
|------|------|
| `MO_PI0_STAGE_PROFILE=1` | 启用 `StagePerfCollector` |
| `MO_FP8_LANG_EMB=1` | 强制 FP8 lang embedding（sidecar 存在时） |
| `MO_FP8_LANG_EMB_BASELINE=1` | 微基准：仅跑 lookup N 次，不跑全 pipeline |

---

## 4. 耗时与收益统计设计

### 4.1 现有基础设施

| 组件 | 文件 | 现状 |
|------|------|------|
| 分阶段 wall time | `infer/perf/stage_perf.py` | 已有 `embed_prefix`、`prefix_llm`、`denoise.*` |
| TRT hook | `infer/tensorrt/trt_hook_timer.py` | `trt.embed_prefix_vit_batched` 等 |
| Native FlashRT | `infer/native/pi05_executor.py` | 已有 `timed("embed_prefix")` |
| WebUI 汇总 | `scripts/deployment/pi05/lerobot_eval_webui/chunk_stage_perf.py` | 聚合 engine timing |

**缺口**：无 `embed_prefix.lang_embedding` 子项；无静态显存/带宽元数据；TRT hybrid 路径 `embed_prefix` 未拆 vision/lang。

### 4.2 新增 StagePerf Key

在 `stage_perf.py` 的 `_DEFAULT_SUMMARY_ORDER` 中 **`embed_prefix` 之后**插入：

```text
embed_prefix.vision              # SigLIP / TRT ViT / FlashRT vision 段
embed_prefix.lang_embedding      # FP8/FP16 token lookup + dequant
embed_prefix.lang_scale          # × sqrt(H) 缩放
embed_prefix.concat              # cat + mask 构造（可选，开销极小）
```

**语义**

- `embed_prefix`：整段 wall time（保持兼容，= 子项之和 + 少量 overhead）。
- `embed_prefix.lang_embedding`：**仅** `Fp8LangEmbeddingLookup.forward` 或 `nn.Embedding` forward（含 CUDA sync 可选，见下）。
- `embed_prefix.vision`：batched ViT / FlashRT SigLIP 段。

### 4.3 静态资源统计（非 wall time）

**新 dataclass**：`LangEmbeddingMemoryStats`（挂在 policy 或 executor 上，推理启动时打印一次）

```python
@dataclass
class LangEmbeddingMemoryStats:
    format: str              # "fp16" | "fp8_e4m3"
    vocab_size: int
    hidden_size: int
    table_bytes: int         # GPU 常驻权重
    scales_bytes: int        # FP8 时 scale 表
    total_bytes: int
```

**启动日志示例**：

```text
[fp8-lang-emb] format=fp8_e4m3 vocab=257152 hidden=2048
               table=1054.0 MiB scales=7.9 MiB total=1061.9 MiB
               (fp16 baseline would be 2004.0 MiB, saved 942.1 MiB, 47.0%)
```

### 4.4 每 step 动态带宽估算

在 `embed_prefix.lang_embedding` 计时块内记录 **元样本**（扩展 `StagePerfCollector` 或并行 dict `_meta`）：

```python
valid_lang_tokens = int(lang_masks.sum().item())   # 或 batch 维 sum
bytes_read = valid_lang_tokens * hidden_size * (1 if fp8 else 2)
collector.record_meta("embed_prefix.lang_embedding.valid_tokens", valid_lang_tokens)
collector.record_meta("embed_prefix.lang_embedding.bytes_read", bytes_read)
```

汇总脚本输出：

```text
[summary:emb] lang_embedding.valid_tokens  n=100 mean=48.2
[summary:emb] lang_embedding.bytes_read    n=100 mean=98.4 KiB  (fp8)  vs 196.8 KiB (fp16)
```

### 4.5 A/B 对比流程

**脚本**：`scripts/bench/bench_fp8_lang_embedding.py`

```bash
# 1) 微基准：隔离 lookup（排除 ViT / LLM / denoise）
python scripts/bench/bench_fp8_lang_embedding.py \
  --engine-dir engines/pi05 \
  --mode fp16|fp8 \
  --lang-seq-len 48 \
  --iters 1000

# 2) 端到端（需 MO_PI0_STAGE_PROFILE=1）
MO_PI0_STAGE_PROFILE=1 python scripts/deployment/pi05/run_infer.py \
  --engine-dir engines/pi05_fp16_baseline ...

MO_PI0_STAGE_PROFILE=1 python scripts/deployment/pi05/run_infer.py \
  --engine-dir engines/pi05_fp8_lang_emb ...
```

**输出表格**（脚本自动生成 Markdown / CSV）：

| 指标 | FP16 baseline | FP8 lang emb | 预期 |
|------|---------------|--------------|------|
| `lang_embedding.table_bytes` | ~2.0 GB | ~1.0 GB | **~50%↓** |
| `embed_prefix.lang_embedding` mean ms | X | X−Δ | Δ 小（~48 token 量级） |
| `embed_prefix` mean ms | Y | Y−Δ | Δ ≪ ViT+LLM |
| `sample_actions` mean ms | Z | Z−Δ | **Δ 通常 <1%**（lang 只查一次） |
| `policy.infer` mean ms | W | W−Δ | 同上 |

**收益解读模板**（写入脚本 docstring）：

```text
静态显存收益 ≈ (V×H×2) − (V×H×1 + V×H/128×4)  ≈ 47% @ V=257152,H=2048
单次 action latency 收益 ≈ f(valid_lang_tokens, H, HBM BW)
  valid=50,H=2048: 带宽差 ~100KB/action → 通常 <0.1ms，远小于 prefix_llm (~30ms)
主要价值在 Thor 显存预算；latency 仅在 lang_seq 很长或 batch>1 时可见
```

### 4.6 WebUI / 汇总集成

`chunk_stage_perf.py` 增加行：

```python
"embed_prefix.lang_embedding": stage_perf_ms(policy, "embed_prefix.lang_embedding"),
"embed_prefix.vision": stage_perf_ms(policy, "embed_prefix.vision"),
```

`format_summary_lines()` 扩展：若存在 `_meta`，追加 `[summary:emb]` 段。

### 4.7 CUDA Sync 策略

与现有 `KEY_POLICY_SAMPLE_ACTIONS_CUDA_SYNC` 一致：

- **默认**：`embed_prefix.lang_embedding` 用 `perf_counter` wall time（含 async launch）。
- **可选** `MO_STAGE_PERF_CUDA_SYNC=1`：lookup 后 `torch.cuda.synchronize()`，得到 **真实 GPU 时间**（微基准与 nsys 对齐）。

---

## 5. 分阶段落地

### Phase 1 — 可验证 MVP（≈1 周） ✅ 已实现

| 任务 | 文件 | 状态 |
|------|------|------|
| FP8 量化 + sidecar IO | `quantization/fp8_lang_embedding.py` | ✅ |
| PyTorch lookup 模块 | `infer/kernels/fp8_lang_embedding.py` | ✅ |
| Export 写 sidecar | `models/pi05/embed_prefix.py`（`PI05_FP8_LANG_EMBEDDING=1`） | ✅ |
| Hybrid runtime 挂载 | `infer/tensorrt/pi05_trt_engine_setup.py` | ✅ |
| 子阶段计时 key | `infer/perf/stage_perf.py` + batched/flashrt hooks | ✅ |
| 微基准脚本 | `scripts/bench/bench_fp8_lang_embedding.py` | ✅ |
| 独立 export 工具 | `scripts/tools/export_fp8_lang_embedding.py` | ✅ |
| 单测 | `tests/unit/test_fp8_lang_embedding.py` | ✅ |

**验收**：同一 prompt 下 FP8 vs FP16 `lang_emb` cosine > 0.999；`table_bytes` 降 ~47%；`embed_prefix.lang_embedding` 有样本。

### Phase 2 — CUDA kernel + whole_graph 策略（≈2 周）

| 任务 | 说明 |
|------|------|
| 绑定 Edge-LLM `embeddingLookup` | 替换 PyTorch dequant |
| whole_graph 路径 | export 时 **剥离** `embed_tokens` 出 ONNX，runtime 强制 sidecar |
| C++ `Pi05Runtime` | `edge-llm/src/pi05/` 读 sidecar（若走 C++ 管线） |

### Phase 3 — 与 Feature 体系集成

| 任务 | 说明 |
|------|------|
| `FeatureConfig` | `fp8_lang_embedding: true` |
| `docs/optimizer/roadmap.md` #12 | 状态改为「已支持」并链到本文 |
| CI | bench 回归：sidecar 体积 + cos similarity 阈值 |

---

## 6. 文件清单

```text
src/model_optimizer/quantization/fp8_lang_embedding.py          # NEW
src/model_optimizer/infer/kernels/fp8_lang_embedding.py         # NEW
src/model_optimizer/infer/kernels/__init__.py                   # export
src/model_optimizer/models/pi05/embed_prefix.py                 # export sidecar
src/model_optimizer/infer/tensorrt/pi05_trt_engine_setup.py     # mount + timed hooks
src/model_optimizer/infer/native/pi05_executor.py             # split embed_prefix sub-keys
src/model_optimizer/infer/perf/stage_perf.py                  # new keys + record_meta
scripts/bench/bench_fp8_lang_embedding.py                     # NEW
tests/unit/test_fp8_lang_embedding.py                           # NEW
config/quant/embed_prefix_fp8_lang_sidecar.yaml               # NEW optional preset
docs/optimizer/pi05_fp8_lang_embedding_design.md                # 本文
```

---

## 7. 风险与约束

| 风险 | 缓解 |
|------|------|
| whole_graph 路径仍 bake FP16 大表 | Phase 1 仅 hybrid；文档标明 whole_graph 无 FP8 收益 |
| PyTorch lookup 慢于 CUDA kernel | Phase 1 以显存 A/B 为主；Phase 2 换 kernel |
| 与 ModelOpt 整图 FP8 冲突 | 互斥 flag：`fp8_lang_embedding_sidecar` vs `embed_prefix_quant_fp8_cfg` |
| 精度 | per-block max-abs；用 LIBERO prompt 集验 action error |
| SM < 89 | FP8 storage 可用；kernel 路径需 fallback FP16 dequant on GPU |

---

## 8. 与 flow.md / roadmap 的关系

- **flow.md §9.9.3**：Edge-LLM 通用 FP8 embedding 原理；本文是 **π0.5 model_optimizer 落地**。
- **roadmap.md #12**：实现完成后更新为 DONE，并引用 `bench_fp8_lang_embedding.py` 实测表格。
