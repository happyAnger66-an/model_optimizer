# pi05 / Jetson Thor 性能优化 Roadmap

> 面向 **π0.5 (PaliGemma + Gemma 2B + Gemma action expert, flow-matching DiT)** 在
> **Jetson Thor (SM110, Blackwell)** 上的部署。覆盖 `model_optimizer`（PTQ + ONNX
> export）能直接落地的优化项，以及需要联动 `TensorRT-Edge-LLM` 才能生效的项。
>
> 来源：交叉对比 `TensorRT-LLM`（数据中心 LLM 推理）和 `TensorRT-Edge-LLM`
> （Jetson / DRIVE C++ runtime）两个仓库的优化技术清单。

---

## 0. 背景与约束

| 维度 | 现状 |
|---|---|
| 模型组成 | PaliGemma SigLIP ViT (head_dim=72) + Gemma 2B LLM prefix (head_dim=256) + Gemma action expert DiT (flow matching) |
| 典型时序 | 长 prefix（968 多模态 tokens, 单次 prefill）+ 多次 denoise step（10-50 步）+ 每步 action expert decode |
| `model_optimizer` 角色 | **PTQ 校准 + ONNX export 工具链**——不参与 runtime；通过 plugin attribute + sidecar 文件影响下游引擎行为 |
| 部署目标 | ONNX → `TensorRT-Edge-LLM cpp/builder/llmBuilder` → `LLMInferenceRuntime` (C++) |
| Thor 平台特性 | SM110（CMake 中映射为 SM101 cubin）；Blackwell 算力；DriveOS hugepage；JetPack TRT 10.13.3.9（NVFP4 推荐版本）；统一内存架构 |

### 已落地（baseline）

- ModelOpt PTQ：`FP16 / FP8 / NVFP4 / INT4 AWQ / INT4 AWQ ModelOpt / INT4 GPTQ / INT8 SQ / mixed_precision`
- KV cache FP8：通过 `trt::attention_plugin` 的 `enable_fp8_kv_cache=1` + `qkv_scales`
  （`src/model_optimizer/models/pi05/llm_with_trtedgellm.py`）
- 三条 ONNX 导出路径并存：
  - `llm.py` — native eager attention
  - `llm_with_cutedsl.py` — `trt::FmhaD256AttentionPlugin`（自家 CuTe DSL D=256）
  - `llm_with_trtedgellm.py` — `trt::attention_plugin`（Edge-LLM 内置）
- **Fused MLP 方案 A**（gate/up 权重合并，`src/model_optimizer/models/pi05/fused_mlp.py`，
  默认开启；环境变量 `MODEL_OPT_PI05_FUSED_MLP=0` 可关）—— 详见
  `kernelSrc/docs/fused_mlp.md`
- TRT runtime 已带 CUDA Graph（`src/model_optimizer/infer/tensorrt/trt_torch.py`，
  环境变量 `MODEL_OPT_TRT_CUDA_GRAPH=1` 启用）

---

## 1. 优化项总表（按优先级）

| # | 优化项 | TRT-LLM 来源 | Edge-LLM 来源 | `model_optimizer` 改动点 | Thor 预期收益 | 优先级 | 状态 |
|---|---|---|---|---|---|---|---|
| 1 | NVFP4 backbone + per-channel scales | `_torch/cute_dsl_kernels/blackwell/dense_blockscaled_gemm_*` | `experimental/quantization/`, `kernelSrcs/build_cutedsl.py` | NVFP4 设为 Thor 默认 quant_format；扩校准集到 1k+ 样本 | 30-40% latency↓，35% 显存↓ | ★★★★★ | 部分已支持 |
| 2 | FP8 KV cache 全链路默认 ON | `KvCacheConfig.dtype='fp8'` | `enable_fp8_kv_cache` + `qkv_scales`（`attention_plugin.cpp`） | LLM 已支持；**action expert (`expert.py`) 需补 FP8 KV** | 50% KV 显存↓，长 prefix 1.3x | ★★★★★ | LLM ✓ / Expert ✗ |
| 3 | CuTe DSL FMHA 而非 fmha_v2 cubin | `_torch/cute_dsl_kernels/blackwell/` | `cmake/CuteDslFMHA.cmake`, `cuteDslFMHARunner.*` | export 走 `trt::attention_plugin`（**不是** `FmhaD256AttentionPlugin`）；Edge-LLM 侧预生成 sm_110 artifact | 长 prefix prefill 1.5-2x | ★★★★★ | 路径已存在，需默认化 |
| 4 | System prompt KV cache | `enable_block_reuse` radix tree | `SystemPromptKVCache` + JSON `save_system_prompt_kv_cache` | 校准集需覆盖 system prompt；export sidecar 记录 prompt hash | denoise 多步 30-50% per-action↓ | ★★★★★ | 未启用 |
| 5 | Fused gate+up MLP (方案 A) | `_torch/modules/gated_mlp.py` | TRT Myelin epilogue fusion | `fused_mlp.py` (default ON) | FC1 HBM 带宽 ½ | ✅ 完成 | DONE |
| 6 | NVFP4 weight + FP8 activation (W4A8) | `cutlass_kernels/fpA_intB_gemm/...sm100.h`, `QuantAlgo.W4A8_*` | `mixed_precision` cfg + weight repack | `quantization/cfg.py` 加 W4A8 preset；ModelOpt 已支持 | 再 25% 显存↓；算力换内存的最佳点 | ★★★★☆ | 未启用 |
| 7 | Vision tower 切 `vitAttentionPlugin` (FP8 可选) | `_torch/visual_gen/jit_kernels/flash_attention/cute/flash_fwd_sm100.py` | `vit_fmha_d{64,72,80,128}` + `vitAttentionPlugin` | `vit.py` 当前 SDPA-math → 改 export `trt::vit_attention_plugin`（head_dim=72） | SigLIP encode 1.4-1.6x | ★★★★☆ | 未启用 |
| 8 | Vocab reduction（机器人词表裁剪） | — | `experimental/llm_loader/vocab_reduction/` | 统计校准集 token 频次；调 `tensorrt-edgellm-reduce-vocab`；export `--reduced-vocab-dir` | LM head 显存 ↓ 50%+ (256k→~10k) | ★★★★☆ | 未启用 |
| 9 | Per-channel weight quantizer for `gate_up_proj` | — | — | cfg override：`*gate_up_proj*weight_quantizer.axis=(0,)` | 防止 gate/up 合并后 per-tensor scale 拉伸 | ★★★★☆ | 紧跟 #5 |
| 10 | CuTe DSL sm_110 artifact 预生成 | — | `kernelSrcs/build_cutedsl.py --gpu_arch sm_110 --arch aarch64` | 写入 `convert_format` README + CI | 防止 fallback 到 SM101 cubin (~20% latency 损失) | ★★★★☆ | 文档/CI 缺失 |
| 11 | Action expert (DiT) CUDA Graph (denoise inner loop) | `cuda_graph_runner.py`, piecewise graph | runtime `mCudaGraphs` + pinned H2D | `trt_torch.py` 已有；扩展到 denoise 全循环；`pi05_executor.py` 强制默认开 | 单 action 50 步 ↓ 100-200ms host overhead | ★★★★☆ | 部分支持 |
| 12 | FP8 embedding | — | `--fp8-embedding` sidecar + `embeddingKernels` | `llm.py` export 写 `embedding.safetensors` + per-row scale (block 128) | embed + LM head 显存 ½ | ★★★☆☆ | 未启用 |
| 13 | Fused RMSNorm + Quant (FP8/NVFP4) | `cpp/.../fusedGatedRMSNormQuant.cu` | TRT native fusion | export 时确保 RMSNorm + 紧邻 Q/DQ 在 ONNX 相邻；参考 `denoise_onnx_post_export.py: fp4qdq_to_2dq` | LayerNorm host stall ↓ 5-8% iter time | ★★★☆☆ | 已有 helper，未泛化 |
| 14 | NVTX 标注 + layer-wise profiler | `perf-nsight-systems` skill | `cpp/profiling/{nvtx_wrapper,layerProfiler}.h` | builder 默认带 `-DENABLE_NVTX_PROFILING=ON`（开发版）；`pi05_executor.py` 加 NVTX range | 不直接提速；定位下一轮 | ★★★☆☆ | 未启用 |
| 15 | Chunked prefill | `enable_chunked_prefill` | `LLMInferenceRuntime` | model_optimizer 不动；Edge-LLM 侧配置 | prefix 968 时显存波动 ↓ | ★★☆☆☆ | 收益小 |
| 16 | LoRA hot-swap | — | `experimental/llm_loader/lora/` | export 加 LoRA 通路；目前单任务不需要 | 多任务/多机器人时启用 | ☆ 备选 | — |
| 17 | EAGLE3 speculative decoding | `_torch/speculative/eagle3.py` | `LLMInferenceSpecDecodeRuntime` | — | DiT 非 token-level autoregressive，不适用 | ✗ | N/A |
| 18 | DSA / sparse attention | `attention_backend/sparse/dsa.py` | — | — | prefix < 4K，sparse 不划算 | ✗ | N/A |
| 19 | MoE optimization | trtllm-gen MoE | `cpp/kernels/moe/` | — | Gemma 2B 是 dense | ✗ | N/A |
| 20 | Mamba/SSM kernel | — | `cpp/kernels/mamba/` | — | pi05 是纯 transformer | ✗ | N/A |
| 21 | AutoDeploy piecewise graph | `auto_deploy/compile/piecewise_runner.py` | — | — | 走 C++ runtime，不走 PyExecutor | ✗ | N/A |

---

## 2. 分阶段落地路线

### Phase 1 — "把已有但没默认开的开关全部打开"（≈1-2 周）

**特征：不需要新代码或仅配置改动；零算法风险。**

| 序 | 任务 | 文件 / 改动 |
|---|---|---|
| 1.1 | NVFP4 设为 Thor 默认 quant_format | `src/model_optimizer/quantization/cfg.py`：加 `THOR_DEFAULT = "nvfp4"`；`convert_format` CLI 在 `platform=thor` 时优先选用 |
| 1.2 | Action expert (`expert.py`) 补 FP8 KV cache | 复用 `llm_with_trtedgellm.py:~135` 的 `enable_fp8_kv_cache=1` + `qkv_scales` 写法到 `expert.py` 的 export 路径 |
| 1.3 | CuTe DSL FMHA 路径默认化 | `convert_format` 在 thor 平台默认 `LLMWithTrtEdgeLLM`，不要走 `LLMWithCuteDsl`（D=256 自家 plugin 没 sm_110 artifact） |
| 1.4 | CUDA Graph 默认开 | 在 `pi05_executor.py` 默认设 `MODEL_OPT_TRT_CUDA_GRAPH=1`；action expert denoise step 强制开 |
| 1.5 | NVTX 编译默认 ON | Edge-LLM build 命令在开发/Profile 配置加 `-DENABLE_NVTX_PROFILING=ON`；在 `docs/trt-edge-llm/` 文档化 |
| 1.6 | `gate_up_proj` per-channel weight quantizer | quant_cfg 全局加 `*gate_up_proj*weight_quantizer.axis=(0,)` |

**叠加预期：BF16 baseline → 1.5-2× end-to-end action latency。**

### Phase 2 — "写一两个真正的图改写"（≈2-4 周）

| 序 | 任务 | 文件 / 改动 |
|---|---|---|
| 2.1 | Vision tower 切 `trt::vit_attention_plugin` | 新增 `src/model_optimizer/ops/vit_attention_plugin.py`（类似 `fmha_d256_attention_plugin.py`）；`vit.py` 改用 wrapper 替换 SigLIP attention；head_dim=72 写死 plugin attribute |
| 2.2 | Vocab reduction | 在 `calibrate/pi05_calib_load.py` 统计 token 频次；新增 `tools/reduce_vocab.py` 调 `tensorrt-edgellm-reduce-vocab` 写 mapping；export 时载入 mapping |
| 2.3 | System prompt KV cache 接线 | 在 export sidecar 写入固定 system prompt 文本 + sha256；Edge-LLM runtime 自动触发 `SystemPromptKVCache` |
| 2.4 | RMSNorm + quant 节点紧邻保证 | 把 `denoise_onnx_post_export.py: fp4qdq_to_2dq` 的思路抽出来通用化，所有量化导出后跑一遍 |

### Phase 3 — "上需要新代码的"（≈4-8 周）

| 序 | 任务 | 文件 / 改动 |
|---|---|---|
| 3.1 | W4A8 (NVFP4 weight + FP8 activation) preset | `quantization/cfg.py` 加 W4A8 cfg；weight repack 参考 Edge-LLM `experimental/llm_loader/checkpoint/repacking.py` |
| 3.2 | FP8 embedding | `llm.py` export 增加 `--fp8-embedding`；写 per-row scale (block 128) safetensors sidecar |
| 3.3 | Action expert 独立 NVFP4 + FP8 KV 完整 pipeline | `expert.py` 增加 `LLMWithTrtEdgeLLM` 风格的 wrapper；与 LLM 解耦量化 |
| 3.4 | DiT denoise inner loop CUDA Graph 化（10-50 步固定 shape） | `infer/tensorrt/pi05_executor.py` 把 denoise 子循环改成单个 capture/replay |
| 3.5 | NVTX scope 体系化（Phase 1.5 的延伸） | 把 NVTX 标注覆盖到 vision / prefill / 每个 denoise step；与 Edge-LLM `layerProfiler` 联动出报告 |

---

## 3. 不适用 / 不做（带原因）

| 项 | 为何不做 |
|---|---|
| EAGLE3 / Medusa / MTP | DiT denoise 不是 token-level autoregressive，accept/reject 语义不通 |
| DSA / sparse attention | prefix 968 < 4K，sparse 收益 < 重写成本 |
| MoE 相关（trtllm-gen MoE / fused_moe_*） | Gemma 2B 是 dense |
| Mamba / SSM / GDN kernels | pi05 纯 transformer |
| AutoDeploy piecewise graph / torch.compile | 走 Edge-LLM C++ runtime，非 PyExecutor |
| trtllm-build Legacy 路径优化（weight_sparsity / refit） | Edge-LLM builder 是独立实现，TRT-LLM legacy 的 builder flag 不能直接用 |

---

## 4. pi05 单 action 耗时分解（BF16 baseline 估算）

```text
[Vision SigLIP]            │██│                       ~5 ms      → #1 NVFP4, #7 vit_attention_plugin
[Embed prefix]             │█│                        ~2 ms
[LLM prefix prefill]       │██████████│               ~30 ms     → #3 CuTe FMHA, #2 FP8 KV, #1 NVFP4
[Denoise step ×10]                                                → 主要优化目标
  ├─ Action FC (gate/up/down)  ×10 ≈ ~30 ms                       → #5 fused MLP ✓, #6 W4A8
  ├─ Action attn (cross-attn with prefix KV)  ×10 ≈ ~25 ms        → #3, #2, #4 system prompt KV
  └─ host overhead  ×10 ≈ ~5-10 ms                                → #11 CUDA Graph
─────────────────────────────────────────────────
total ≈ 100-110 ms / action  (BF16)
```

> Phase 1 全开后估计能压到 **≈55-65 ms / action**；Phase 2 之后 **≈40-50 ms**；
> Phase 3 + denoise inner loop graph 化后 **≈30-40 ms**。需 nsys 验证实际数。

---

## 5. `model_optimizer` 在 pipeline 中的"开关位"

```text
HuggingFace checkpoint
   │
   ▼
calibration set ──────►  PTQ (ModelOpt / mtq.quantize)
                          │
                          │ ★ NVFP4 / FP8 / W4A8 / FP8 KV / per-channel weight axis
                          ▼
                        quantized PyTorch model
                          │
                          │ ★ install_fused_mlp ✓  |  Fused RMSNorm-Quant  |  Vocab reduction
                          ▼
                        torch.onnx.export
                          │
                          │ ★ plugin 选择：
                          │     - trt::attention_plugin     (Edge-LLM 默认, 含 FP8 KV)
                          │     - trt::FmhaD256AttentionPlugin (自家 D=256, 需 sm_110 artifact)
                          │     - trt::vit_attention_plugin  (vision tower)
                          │ ★ sidecar：
                          │     - embedding.safetensors (FP8 embedding)
                          │     - reduced_vocab_mapping.npz
                          │     - system_prompt.json (sha256 + text)
                          │ ★ plugin attribute：
                          │     - enable_fp8_kv_cache, qkv_scales
                          │     - head_dim (256 for LLM, 72 for SigLIP)
                          │     - max_seq_len
                          ▼
                        llm.onnx + visual.onnx + action.onnx + sidecars
                          │
                          │ ────── 以下属于 TensorRT-Edge-LLM 的责任 ──────
                          ▼
                        cpp/builder/llmBuilder  ──►  TRT engine
                          │
                          ▼
                        LLMInferenceRuntime  (CUDA graph, prefix KV reuse, FP8 KV runtime)
```

任何 runtime 优化想生效，相关的 plugin attribute / sidecar 必须从 `model_optimizer`
**这一阶段就写对**——这就是 Phase 1 全是"打开开关"而非"写算法"的原因。

---

## 6. 立刻可做的最小集（≤1 周）

> 三件本周即可推进，零阻塞、零依赖外部交付。

1. **`expert.py` 加 FP8 KV cache**
   - 复制 `llm_with_trtedgellm.py:~135` 的 `enable_fp8_kv_cache=1` + `qkv_scales`
     模板到 action expert export
   - 验证：导出 ONNX 后用 `onnx.shape_inference` 确认 plugin attribute 存在
2. **`fused_mlp.py` 配合 per-channel weight quantizer**
   - 全局 cfg 加 `*gate_up_proj*weight_quantizer = {axis: (0,), ...}`
   - 防止 gate/up 合并后 per-tensor scale 被拉伸（gate 与 up 量纲可能差几倍）
3. **`pi05_executor.py` 加 NVTX scope**
   - vision / prefill / 每个 denoise step / sampling 都包 NVTX range
   - 不直接提速；Phase 2/3 全部决策依赖这个 nsys baseline

---

## 7. 验证矩阵（Phase 完成时的 ship gate）

| Phase | 必须项 | 验证手段 |
|---|---|---|
| Phase 1 | NVFP4 PTQ + FP8 KV 全开后 acc 不掉 | `Pi05Metric` action error；BF16 baseline 对比 |
| Phase 1 | sm_110 artifact 已部署 | Edge-LLM verbose log 确认走 CuTe FMHA 而非 fmha_v2 cubin |
| Phase 1 | CUDA Graph 真在跑 | nsys 看到 "graph launched" event |
| Phase 2 | Vision plugin 切完后图像 forward bit-exact within ±1e-3 (FP8) | 单元 test 对比 |
| Phase 2 | Vocab reduction 后 LM head shape 减小、概率分布不变 | output logits top-K 对比 |
| Phase 3 | W4A8 acc 不掉超 1% | 端到端 Pi05Metric |
| Phase 3 | DiT denoise 全循环 graph 化 | nsys 看到单个 graph 内含 N×denoise step |

---

## 8. 相关源码 / 文档索引

### model_optimizer

- `src/model_optimizer/models/pi05/fused_mlp.py` — 方案 A 实现（已落地）
- `src/model_optimizer/models/pi05/llm_with_trtedgellm.py` — `trt::attention_plugin` 路径（FP8 KV 模板）
- `src/model_optimizer/models/pi05/llm_with_cutedsl.py` — `trt::FmhaD256AttentionPlugin` 路径
- `src/model_optimizer/models/pi05/expert.py` — action expert export（待补 FP8 KV）
- `src/model_optimizer/models/pi05/vit.py` — SigLIP export（待切 `vit_attention_plugin`）
- `src/model_optimizer/models/pi05/denoise_onnx_post_export.py` — `fp4qdq_to_2dq` 等图改写
- `src/model_optimizer/infer/tensorrt/{trt_torch,pi05_executor}.py` — runtime CUDA Graph
- `src/model_optimizer/quantization/{cfg,quantization_utils}.py` — PTQ cfg & loop
- `kernelSrc/docs/fused_mlp.md` — fused MLP 三层融合详解（L1/L2/L3）
- `kernelSrc/fmha_d256_cutedsl/README.md` — 自家 CuTe DSL FMHA D=256
- `docs/trt-edge-llm/fmha_v2.md` — fmha_v2 cubin 在 Edge-LLM 的集成

### TensorRT-LLM（数据中心 LLM 推理）

- `tensorrt_llm/_torch/modules/gated_mlp.py` — fused MLP 三层融合参考
- `tensorrt_llm/_torch/modules/swiglu.py` — L2 fused activation kernel 入口
- `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/` — Blackwell CuTe DSL kernel 集
  - `dense_blockscaled_gemm_swiglu_fusion.py` — L3 FC1+SwiGLU epilogue 融合
  - `dense_blockscaled_gemm_persistent.py` — NVFP4 / FP8 dense GEMM
- `tensorrt_llm/_torch/attention_backend/trtllm_gen.py` — trtllm-gen FMHA dispatch
- `tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py` — generation CUDA Graph
- `tensorrt_llm/_torch/compilation/piecewise_optimizer.py` — piecewise CUDA Graph
- `cpp/tensorrt_llm/kernels/fusedGatedRMSNormQuant/` — Norm+Quant 融合 kernel
- `docs/source/features/quantization.md` — FP8/NVFP4/W4A8 文档
- `docs/source/features/kvcache.md` — KV cache 量化与 reuse

### TensorRT-Edge-LLM（Jetson / DRIVE C++ runtime）

- `cpp/plugins/attentionPlugin/attentionPlugin.cpp` — `trt::attention_plugin` 入口
- `cpp/plugins/vitAttentionPlugin/` — vision tower FMHA plugin
- `cpp/kernels/contextAttentionKernels/{contextFMHARunner,cuteDslFMHARunner}.*` — context FMHA
- `cpp/kernels/decodeAttentionKernels/` — XQA + FP8 KV decode cubins
- `cpp/runtime/llmInferenceSpecDecodeRuntime.*` — 统一 runtime (含 EAGLE)
- `cpp/runtime/kvCacheManager.*` + `hybridCacheManager.*` — KV cache + system prompt cache
- `experimental/quantization/` — 新 PTQ pipeline
- `experimental/llm_loader/` — checkpoint-based ONNX export 前端
- `experimental/llm_loader/checkpoint/repacking.py` — AWQ/GPTQ/NVFP4 权重重打包
- `experimental/llm_loader/vocab_reduction/` — 词表裁剪工具
- `experimental/llm_loader/lora/` — LoRA 工具
- `kernelSrcs/build_cutedsl.py` — CuTe DSL artifact 预生成（Thor 必跑）
- `cmake/CuteDslFMHA.cmake` + `cmake/CuteDsl.cmake` — CuTe DSL build 集成
- `docs/source/user_guide/features/{FP8KV,fp8-embedding,system-prompt-cache,reduce-vocab,lora,quantization}.md` — 各 feature 用户文档
- `docs/source/user_guide/getting_started/limitations.md` — Thor / NVFP4 注意事项
- `docs/source/user_guide/performance/performance-benchmarks.md` — Thor 官方 perf

---

## 9. 修订记录

| 日期 | 内容 |
|---|---|
| 2026-05-27 | 初版，基于 `TensorRT-LLM` / `TensorRT-Edge-LLM` 优化技术交叉清单 |
