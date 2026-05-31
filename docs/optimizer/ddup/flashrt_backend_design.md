# FlashRT 推理后端集成设计（分阶段后端矩阵）

> 目标：把 FlashRT（`/home/zhangxa/codes/FlashRT`，手写 C++/CUDA pipeline，pi05/Thor 实测 70.2→44ms）
> 作为 `model_optimizer` 的一种 **推理后端**引入，同时保持**配置化**与**每阶段可独立切换**的灵活性。
>
> 关联文档：`docs/optimizer/roadmap.md` §8（FlashRT 交叉分析）。

---

## 0. 背景：抽象已存在，混合已做了一半

`model_optimizer` 的 `infer/` 已具备本设计所需的全部骨架，**无需从零搭建**：

| 现有件 | 作用 |
|---|---|
| `infer/executor.py::Executor` | 基类：`__getattr__` 透传到 `policy`，`load_model()` 钩子 |
| `infer/tensorrt/pi05_executor.py::Pi05TensorRTExecutor` | 把 openpi policy 的各阶段 forward 替换为 TRT 引擎 |
| `infer/onnxrt/pi05_executor.py::Pi05OnnxRTExecutor` | 同上，ONNX Runtime |
| `infer/flash/siglip.py::FlashRtSiglipVision` | **已存在**：用 FlashRT kernel 跑多视角 SigLIP（+CUDA Graph），`flash_rt` 为可选依赖 |
| `Pi05TensorRTExecutor.load_model` 里的 `use_flashrt_siglip_embed_prefix` | **已存在**：把 SigLIP 阶段切到 FlashRT 的 ad-hoc 开关 = 混合的雏形 |
| `infer/server/backends/{base,tensorrt,onnxrt,...}.py` | `InferBackend` 策略接口 + 各后端薄包装 |
| `infer/server/config.py::ServerConfig` | JSON→dataclass；`mode` 选后端，`tensorrt/onnxrt` 各列 `*_engine` 路径 |
| `infer/server/policy_loader.py` | 按 `mode` 加载 policy 并 mount 引擎 |

**结论**：引入 FlashRT 后端 = 把上面那个 `use_flashrt_siglip_embed_prefix` 的黑魔法，**升级成一等公民的"分阶段后端矩阵"**。

---

## 1. 核心设计：Per-Stage Backend Matrix

pi05 推理是 5 个可独立替换的阶段：

```
vit(SigLIP) → embed_prefix → llm(prefix prefill) → expert → denoise(DiT × N 步)
```

**让每个阶段独立选择后端**，而非整体二选一。依据 roadmap §8 的实测结论——不同阶段最优后端不同：

| 阶段 | §8 结论 | 推荐默认后端 |
|---|---|---|
| `vit` (SigLIP, S=512) | Myelin 全局融合完胜手写（1.0ms vs 6.3ms） | `tensorrt` |
| `embed_prefix` | 轻量、host 侧 | `tensorrt` / `pytorch` |
| `llm` (prefix, Se~776) | FlashRT cuBLASLt 略优，trt 亦可 | `tensorrt`（或 `flashrt`） |
| `expert` / `denoise` (S=10, **launch-bound**) | **FlashRT 大胜（-20ms，kernel 数 0.14×）** | **`flashrt`** |

> 这个矩阵本身就是最有价值的"灵活性"：SigLIP 留编译器、expert/denoise 走 FlashRT，
> 既拿到 FlashRT 在 launch-bound 段的大头收益，又保留每阶段独立切换/降级的能力。

后端取值集合：`pytorch | tensorrt | onnxrt | flashrt`。

---

## 2. 配置 schema（`infer/server/config.py`）

两层粒度，**细粒度 `stages` 优先级高于粗粒度 `mode`**：

```jsonc
{
  "mode": "flashrt",            // 粗粒度语法糖：等价 stages 全填 flashrt（支持的阶段）
  "stages": {                   // 细粒度：逐阶段覆盖
    "vit":          "tensorrt",
    "embed_prefix": "tensorrt",
    "llm":          "flashrt",
    "expert":       "flashrt",
    "denoise":      "flashrt"
  },
  "flashrt": {
    "checkpoint_dir": "/path/to/ckpt",     // safetensors 权重源
    "num_views": 2,
    "use_cuda_graph": true,
    "lib_dir": "/path/to/flash_rt/build",  // libfmha_*.so 搜索目录
    "calib": { "n": 64, "percentile": 99.9, "recalibrate_with_real_data": true }
  }
}
```

新增 dataclass：

- `StagesConfig`：字段 `vit/embed_prefix/llm/expert/denoise`，各取 `pytorch|tensorrt|onnxrt|flashrt`；默认 `None`（表示"跟随 `mode`"）。
- `FlashRtConfig`：`checkpoint_dir / num_views / use_cuda_graph / lib_dir / calib(FlashRtCalibConfig)`。
- `InferMode` 增加 `"flashrt"`、`"pt_flashrt_compare"`。

**解析规则**（`stages` 归一化）：对每个阶段，最终后端 = `stages.<stage>` 若非空，否则由 `mode` 推导（`mode=flashrt` → 全 flashrt；`mode=tensorrt` → 全 tensorrt；以此类推）。`mode` 为 compare 模式时按各自语义。

---

## 3. 统一的阶段协议（新增 `infer/stages.py`）

抽出细粒度接口，让所有后端按阶段对齐（每后端给一份实现）：

```python
class StageRunner(Protocol):
    def vision(self, images) -> Tensor: ...          # SigLIP → enc dim
    def embed_prefix(self, ...) -> Tensor: ...
    def llm_prefix(self, embeds, mask, pos) -> ...: ... # 返回 KV / hidden
    def expert_denoise(self, ...) -> actions: ...    # 含 N 步去噪循环
```

实现：
- `TorchStages` —— 直接调 openpi policy（baseline / 降级目标）。
- `TrtStages` —— 包装现有 `Pi05TensorRTExecutor` 的分阶段调用。
- `FlashRtStages` —— 包装 `flash_rt.frontends.torch.pi05_thor.Pi05TorchFrontendThor` + 已有 `FlashRtSiglipVision`。

> P1 不实现完整协议；P1 只做 **配置 + 路由骨架**（见 §6）。协议在 P2 落地。

---

## 4. 混合执行器（`infer/flash/pi05_executor.py` + dispatcher）

- `Pi05FlashRtExecutor(Executor)`：`load_model(cfg)` 构建 FlashRT 各阶段并 mount 到 policy（与 `Pi05TensorRTExecutor` 平行）。
- `Pi05HybridExecutor`：持有 `{stage: StageRunner}` 映射，按归一化后的 `stages` 把每阶段路由到对应后端。
  现在的 `use_flashrt_siglip_embed_prefix` 是它的特例（`stages.vit=flashrt`）。

---

## 5. 接线（`policy_loader.py` + `backends/`）

- `_mount_flashrt(policy, config)`：仿 `_mount_tensorrt_engines`。
- `backends/flashrt.py::SingleFlashRtBackend(InferBackend)`：和 `SingleTensorRTBackend` 一样，薄包装 `policy.infer(obs)`（执行细节都在 executor）。
- `backends/__init__.py` 导出。
- `load_policies` / `load_policy_for_serve` 增加 `flashrt`、`pt_flashrt_compare` 分支。

---

## 6. 工程要点

### 6.1 可选依赖 + 优雅降级（关键）

`flash_rt` 已是 try/except 可选依赖（见 `infer/flash/siglip.py`）。沿用：
- 某阶段请求 `flashrt` 但 `flash_rt` 不可用 / SM 不匹配 → **明确报错**，或按策略**回退 `pytorch`** 并告警。
- 能力探测用 `fvk.get_sm_version()` / `has_nvfp4()` / `has_cutlass_fmha()`。
- 没有 Thor / 没装 flash_rt 的开发机上，`stages` 里非 flashrt 的项照常工作。

### 6.2 精度 / 校准桥接

FlashRT 自带 W8A8 静态 FP8 校准。两条路（建议都支持）：
- **(a) FlashRT 自校准**（默认、最简）：`flashrt.calib` 喂给 FlashRT `calibrate(obs, percentile, N)`；顺便把 roadmap §8 的 #23（多帧分层）/ #24（实数据 recalib）落在这条路上。
- **(b) model_optimizer 产 spec → FlashRT 消费**（统一精度故事）：导出 `ModelPrecisionSpec.to_json()`，FlashRT `from_json()` 加载。PTQ 校准一次、两后端共享。

### 6.3 权重源

FlashRT 阶段从 `checkpoint_dir/model.safetensors` 经 `flash_rt` 的 `WeightLoader/SafetensorsSource` 独立加载（`FlashRtSiglipVision` 已如此），**与 torch policy 权重解耦**——混合时两边各管各的权重。

### 6.4 风险边界

- FlashRT 自带 CUDA Graph + 预分配 buffer；与 torch 阶段共用 stream 需小心（现有 SigLIP 混合已处理）。
- `FlashRtSiglipVision` 当前 **B=1** 主路径。
- Thor (SM110) 专用 kernel；他卡需回退。
- 精度：FlashRT 静态 FP8 vs torch bf16 须验证 cos（用 `pt_flashrt_compare` 对比，门禁 cos≥0.998）。

---

## 7. 分阶段落地

| Phase | 内容 | 产出 |
|---|---|---|
| **P1** | `StagesConfig`/`FlashRtConfig` + `stages` 归一化路由；把现有 `use_flashrt_siglip_embed_prefix` 迁移成 `stages.vit=flashrt`（保留旧字段兼容） | 配置矩阵可用，**零行为变化** |
| **P2** | `StageRunner` 协议 + `FlashRtStages`（encoder/decoder，复用 `pi05_thor`）+ `Pi05HybridExecutor` | expert/denoise 可走 flashrt |
| **P3** | `mode=flashrt`、`pt_flashrt_compare` 对比 + 精度门禁 | 一键 A/B trt vs flashrt |
| **P4** | 精度桥接 (b)：PrecisionSpec 互通 | 统一校准 |

---

## 8. 一句话总结

> **不要把 FlashRT 当"另一个整体引擎"，而是当"分阶段后端矩阵里的一列"。** 复用现有
> `Executor` + 各阶段 mount 抽象，把已有的 SigLIP 混合升级成可配置的 per-stage 路由：
> SigLIP 留 Myelin、expert/denoise 走 FlashRT，既拿到 launch-bound 段的大头收益，
> 又保持每阶段独立切换 / 降级的灵活性。

---

## 9. 修订记录

| 日期 | 内容 |
|---|---|
| 2026-05-31 | 初版：分阶段后端矩阵设计；P1 = 配置 + 路由骨架 |
