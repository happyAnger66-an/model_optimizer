# edge-llm: pi05 C++ 推理引擎（Phase 1 + Phase 2）

参考 [TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) 运行时架构
（`EngineExecutor` / `Tensor` / `Alpamayo1ActionRunner` 模式）实现的 pi05 多引擎
C++ 推理管线。

Phase 1（引擎管线 + 对拍）：

- `TrtEngine`：TRT engine 加载 / binding 内省 / `enqueueV3` 薄封装
- `Pi05Runtime`：`embed_prefix → (host mask/position_ids) → llm(prefix KV) → denoise ×10 (Euler)`
- `pi05_compare`：读取 Python 端 dump 的 npy 输入，逐 stage 与 golden 对拍
- `scripts/dump_pi05_io.py`：从 Python TRT 全挂载路径 dump 输入与 golden 输出

Phase 2（完整前后处理，端到端）：

- `PaligemmaTokenizer`：sentencepiece C++（与 openpi 逐 token 对齐，含
  pi0 格式 `[bos]+prompt+"\n"` 与 `discrete_state_input` 离散 state 格式）
- `Pi05Assets`：norm stats（quantile/z-score）+ meta 加载（npy + 纯文本，无 JSON 依赖）
- 前处理：`LiberoInputs`（3 视图，第三视图全黑 / mask=[T,T,F]）→ `Normalize(state)`
  → `TokenizePrompt` → uint8 HWC → fp32 CHW `[-1,1]`（含 `resize_with_pad`）
- 后处理：`Unnormalize`（quantile，pad 维原样保留）→ `LiberoOutputs` 截取前 7 维
- `pi05_infer`：端到端 CLI —— 原始观测（uint8 图像 + state + prompt）→ `[10,7]` actions
- `scripts/prepare_pi05_assets.py`：从 checkpoint 导出部署资产

不在范围（Phase 3-4）：device 端 mask 生成、KV zero-copy 极致优化、CUDA Graph、
FlashRT denoise 后端、服务化。

## 与 Python 实现的等价关系

| C++ | Python 对应 |
|---|---|
| `Pi05Runtime::run` | `PI0Pytorch.sample_actions`（`pi0_pytorch.py` L381-424，TRT 全挂载） |
| host mask 构造 | `make_att_2d_masks` + `_prepare_attention_masks_4d` + `clamp(min=-1e4)`（`pi05_trt_engine_setup.py` L176-187） |
| Euler 循环 | `dt=-1/num_steps`，`t: 1.0 → 0.1`，`x_t += dt * v_t` |
| denoise 双变体 | `timestep` 输入 或 AdaRMS 预计算 `adarms_mod`（`pi05_adarms.py`） |

KV 布局：`past_keys/past_values [18, B, 1, prefix_seq, 256]`，llm 引擎输出直接
零拷贝绑定到 denoise 输入。

## 构建

```bash
cd edge-llm
mkdir -p build && cd build
cmake ..                       # 系统 TensorRT（/usr/include/x86_64-linux-gnu）
# 或 cmake .. -DTRT_PACKAGE_DIR=/path/to/TensorRT
make -j$(nproc)
```

依赖：CUDA Toolkit、TensorRT 10.x（与构建 engine 的版本一致）。
sentencepiece 在 configure 时由 FetchContent 自动拉取并静态构建（首次需联网）。

## 对拍流程

### 1. Python 侧 dump 输入与 golden

在能跑 webui TRT 路径的环境里（golden 取自 Python TRT 全挂载路径，与 C++ 用
同一组 engine，阈值可收紧）：

```bash
python edge-llm/scripts/dump_pi05_io.py \
    --checkpoint /path/to/pytorch_pi05_libero \
    --config-name pi05_libero \
    --engine-path /path/to/trt_engines \
    --embed-prefix-engine embed_prefix.engine \
    --llm-engine llm.engine \
    --denoise-engine denoise.engine \
    --out-dir /tmp/pi05_io --seed 0
# denoise 引擎为 AdaRMS 预计算变体（输入 adarms_mod）时追加 --adarms-precompute
```

生成文件：

```
输入:   image_{i}.npy image_mask_{i}.npy lang_tokens.npy lang_masks.npy noise.npy
        [adarms_mod_step{k}.npy]
golden: prefix_embs.npy prefix_pad_masks.npy past_keys.npy past_values.npy
        v_t_step{0..9}.npy actions.npy
```

### 2. C++ 侧执行并比对

```bash
./build/pi05_compare \
    --engine-dir /path/to/trt_engines \
    --io-dir /tmp/pi05_io \
    --tol 1e-2 \
    [--dump-out /tmp/pi05_cpp_out]   # 可选：落盘 C++ 侧各 stage 结果
```

输出逐 stage `max_abs / mean_abs`，全部 PASS 退出码为 0。
bf16 中间量（prefix_embs / past_keys）建议 `--tol 1e-2`；
v_t / actions（fp32 输出）通常可压到 `1e-3` 量级。

## 端到端推理（Phase 2）

### 1. 导出部署资产（一次性）

```bash
python edge-llm/scripts/prepare_pi05_assets.py \
    --checkpoint /path/to/pytorch_pi05_libero \
    --config-name pi05_libero \
    --out-dir /path/to/pi05_assets
# 输出: meta.txt paligemma_tokenizer.model {state,actions}_{mean,std,q01,q99}.npy
```

### 2. 端到端推理 + golden 对拍

```bash
./build/pi05_infer \
    --engine-dir /path/to/trt_engines \
    --assets-dir /path/to/pi05_assets \
    --obs-dir /tmp/pi05_io \           # dump 脚本生成的 base_image/wrist_image/state/prompt
    --noise /tmp/pi05_io/noise.npy \   # 对拍必须与 Python 共用同一 noise
    --golden /tmp/pi05_io/actions_final.npy --tol 1e-3 \
    --out /tmp/actions.npy
# adarms 预计算引擎：--adarms-dir /tmp/pi05_io（modulation 只依赖固定时间步，可复用）
```

`--obs-dir` 也可以指向自己准备的目录：`base_image.npy` / `wrist_image.npy`
（uint8 HWC）、`state.npy`（fp32）、`prompt.txt`；`--prompt` 可覆盖 prompt.txt。

### 已在本机验证的数值一致性

- tokenizer：与 Python sentencepiece 逐 token 一致（strip / `_`→空格 / bos / 末尾 `"\n"` token）
- 图像 uint8→fp32 CHW `[-1,1]`：与 numpy 参考位级一致（max_abs=0）
- quantile unnormalize + 前 7 维截取：与 numpy 参考位级一致（max_abs=0）

## 验收标准

- [ ] Phase 1：`prefix_embs` / `past_keys` / `past_values` / `v_t_step{k}` / `actions` 对拍 PASS
- [ ] Phase 2：`pi05_infer` 端到端 actions 与 `actions_final.npy`（Python `policy.infer`
      输出）对拍 PASS（同一 noise / 同一 engine）

## 目录结构

```
src/common/    check.h logger tensor(RAII, pinned/device) npy(读写) dtype(转换)
src/runtime/   trtEngine — TRT 薄封装（仿 Edge-LLM EngineExecutor）
src/pi05/      pi05Runtime — 三引擎编排 + Euler 去噪循环
               tokenizer — PaliGemma sentencepiece；assets — norm stats/meta
               preprocess — LiberoInputs/Normalize/图像变换 + Unnormalize/LiberoOutputs
               pi05CompareMain — 逐 stage 对拍 CLI；pi05InferMain — 端到端 CLI
scripts/       dump_pi05_io.py — 输入/golden dump；prepare_pi05_assets.py — 资产导出
```

## 已知限制

- 仅支持 batch=1。
- llm 引擎须输出堆叠 `past_keys/past_values`（`present_key_values.{i}` 变体暂不支持）。
- mask / position_ids 在 host 构造（Phase 3 移到 device kernel）。
- 每步 denoise 的 x_t 更新走 host（[1,10,32] 极小，Phase 3 改 device + CUDA Graph）。
- `attention_mask` 无效位填充默认 `-1e4`，须与 Python 侧
  `trt_attention_mask_neg_cap` 一致（可用 `--neg-cap` 覆盖）。
- 非 224×224 输入图像走 C++ bilinear `resize_with_pad`，与 jax（antialias=True）
  存在微小数值差异；224 输入恒等、完全精确（libero 即 224）。
- adarms 预计算引擎的 `adarms_mod_step{k}.npy` 需由 dump 脚本生成（依赖
  time_mlp 权重）；时间步调度固定，跨请求可复用。
- `pi05_infer` 不传 `--noise` 时用 mt19937 生成噪声，与 numpy 不可比（对拍必须传）。
