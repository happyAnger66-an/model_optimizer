# Pi0.5 denoise FP8 量化方案

> 结论先行：**denoise 的 FP8 量化能力已经在仓内 FlashRT native decoder 里实现**
> （权重 + 激活全链路 FP8）。问题不是"从零实现"，而是"怎么启用 + 标定质量怎么保证"。
> 相关代码：`src/model_optimizer/infer/native/flashrt_decoder/`，接入见
> `infer/native/pi05_executor.py::_install_flashrt_loop_runtime`，设计/进展见
> [`native_decoder_implementation_todo.md`](native_decoder_implementation_todo.md) §8。

---

## 1. 两条 denoise FP8 路径

| 路径 | FP8 方式 | 现状 |
|---|---|---|
| **A. FlashRT native decoder**（本轮移植） | 权重 per-tensor FP8 + 激活 static FP8（每层 4 个量化点） | **机制已实现**，只差 Thor 上构建 kernel + 标定 |
| **B. TRT denoise engine** | modelopt 量化 → ONNX 导出 → trtllm build | 受 `E4M3 仅支持 per-tensor` 限制 |

- **追求 denoise 低时延 + FP8** → 走 **路径 A**：绕开 per-channel ONNX 限制，W 已离线 FP8、激活有标定闭环。
- **只想留在 TRT 生态** → 走 **路径 B**：denoise 子图用 per-tensor FP8 配置。

---

## 2. 路径 A：FlashRT native decoder 的 FP8（推荐）

### 2.1 量化点（已写死在 `pipeline.decoder_forward`）

18 层 × 每层 4 个 FP8 GEMM，激活/权重各一组 scale
（`act_scales[l*4+k]` / `w_scales[l*4+k]`，k=0..3）：

| k | 量化点 | 触发位置 |
|---|---|---|
| 0 | `qkv` | C1 AdaRMSNorm → FP8 输入 |
| 1 | `o_proj` | C4 attention 输出 → FP8 |
| 2 | `gate_up` | C5 AdaRMSNorm → FP8 |
| 3 | `down` | C6 GeGLU → FP8 |

- **权重 scale（`w_scales`）**：`weights.py::quant_fp8` 离线算好
  （per-tensor E4M3，`scale = max(|w|.max()/448, 1e-12)`），**不需要 GPU，已完成**。
- **激活 scale（`act_scales`）**：必须用数据**标定**。`decoder_forward_calibrate` 两遍法：
  FP16 跑一遍测 amax → 用该 scale 跑 FP8。

### 2.2 启用三步

```yaml
# 0) Thor 上先 build kernel（FP8 GEMM/AdaRMS 算子）
#    bash scripts/deployment/pi05/build_flashrt_kernels.sh
native_flashrt_decoder: true
native_flashrt_use_fp8: true                 # 关键开关：走 FP8 路径（false=fp16 baseline，精度对照用）
native_flashrt_build_dir: "/workspace/flash_rt/build"

# 1) 首跑标定激活 scale（导出 JSON）
native_flashrt_calibrate: true
native_flashrt_act_scales_path: "/tmp/quantize/pi05/flashrt_decoder_act_scales.json"

# 2) 之后正式推理：改回 false，启动时自动 load_act_scales 加载
# native_flashrt_calibrate: false
```

### 2.3 标定质量改进（已落地）

> **归属澄清**：FlashRT 原版标定（`pi05_thor.py::_calibrate` + `decoder_forward_calibrate`）本身
> 就是 **单次前向 + 跨步覆盖（step-9 amax 胜出）**，且每 prompt 用单份随机 noise（按 `Se` 缓存）。
> 这不是移植引入的退化；本节是在其之上的**增强**，FlashRT 与原移植版都没有这两项。

FP8 激活量化对 amax 很敏感，scale 偏小会饱和/掉点。已实现两层 max-merge：

1. **跨步 max（单样本内）**：`pipeline.decoder_forward_calibrate` 新增 `per_step_scales_ptr`
   （`steps*layers*4` 设备 buffer），每个扩散步落盘当步 `calib_buf`；`driver.calibrate` 在 torch
   侧对 step 轴 `amax`，得到"整 10 步最大激活" scale。**不改 in-loop FP8 forward**（仍用当步
   scale 前向，与 FlashRT 一致），只改最终输出。
2. **跨样本 max（多 observation）**：`backend.accumulate_calibration` 对每个样本（KV/noise 各异）
   的跨步 max 结果再与 `self._act_scales` 逐元素取 max；`reset_act_scales` 清零起点。

接入：`pi05_executor::_install_flashrt_loop_runtime` 在标定阶段累计前 `N` 个 observation
（`N = native_flashrt_calib_samples`，默认 8），达到 N 后 `save_act_scales` 固化并冻结。

```yaml
native_flashrt_calibrate: true
native_flashrt_calib_samples: 8     # 跨 8 个 observation + 每样本跨 10 步取 max
native_flashrt_act_scales_path: "/tmp/quantize/pi05/flashrt_decoder_act_scales.json"
```

涉及文件：`pipeline.py`（`per_step_scales_ptr`）、`driver.py`（`calibrate` 返回跨步 max）、
`backend.py`（`reset_act_scales` / `accumulate_calibration`）、`pi05_executor.py`（N 样本累计）。

---

## 3. 路径 B：TRT denoise engine 的 FP8

走现有 modelopt 流程，但 **denoise 子图只能 per-tensor E4M3**：

- 量化配置用 **FP8 per-tensor**（不要 per-channel；per-channel 会触发
  `AssertionError: E4M3 supports ONNX export only for per-tensor quantization`，
  因为给了非标量 amax，如 `[8192, 1]`）。
- 流程：`quantize → onnx export → trtllm build`，把 `denoise_*.engine` 填回 `denoise_engine`。
- 对比：LLM 能用 per-channel FP8 是因为走 **TRT 插件 QDQ**（产出 `TRT_FP8QDQ` 节点）；
  denoise 走标准导出，不适用该插件路径。

---

## 4. 行动建议

1. **Thor 上构建 kernel**（路径 A 的前置硬条件）。
2. ~~改造标定：多样本 + 跨步 max-merge~~ **已落地**（见 §2.3）。
3. 用 `flashrt_decoder_smoke.py --mode compare` 对比 FP8 vs PyTorch raw `sample_actions`，
   确认 `max_abs_diff / cosine` 达标；如掉点，增大 `native_flashrt_calib_samples`。
4. 若需要保留 TRT denoise，则按路径 B 用 per-tensor FP8 配置导出。

---

## 5. 相关文件

| 文件 | 角色 |
|---|---|
| `infer/native/flashrt_decoder/weights.py` | 权重 per-tensor FP8 repack（`quant_fp8` + `_ae_w_scales`） |
| `infer/native/flashrt_decoder/pipeline.py` | FP8 前向 `decoder_forward` + 标定 `decoder_forward_calibrate` |
| `infer/native/flashrt_decoder/backend.py` | `calibrate` / `save_act_scales` / `load_act_scales` |
| `infer/native/pi05_executor.py` | `_install_flashrt_loop_runtime`（sample_actions 整循环接入 + 标定开关） |
| `config/webui_configs/tensorrt_flashrt_denoise.yaml` | 示例配置（含 `native_flashrt_*` 开关） |
| `scripts/deployment/pi05/build_flashrt_kernels.sh` | Thor 上构建 FP8 kernel |
| `scripts/deployment/pi05/flashrt_decoder_smoke.py` | repack 自检 + FP8 数值对比 |

---

## 6. 修订记录

| 日期 | 说明 |
|---|---|
| 2026-06-01 | 初版：整理 denoise FP8 两条路径 + FlashRT native decoder 启用步骤 + 标定质量改进点 |
| 2026-06-01 | §2.3 标定改进落地（跨步 + 跨样本 max-merge，`native_flashrt_calib_samples`）+ 归属澄清 |
