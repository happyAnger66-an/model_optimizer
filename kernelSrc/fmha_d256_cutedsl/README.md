# fmha_d256 — CUTLASS 官方 d=256 FMHA prefill 迁移

把 CUTLASS 官方 Blackwell d=256 mixed-input FMHA prefill kernel 迁移到
study_cute 仓库下,方便在 Jetson Thor (sm_110a) 上跑 head_dim=256 的 FMHA。

## 包结构 (对齐 `fmha/`)

```
fmha_d256/
├── __init__.py              # 绑定 device 方法到 config 类 + 导出 run
├── mixed_input_fmha_prefill_d256.py  # 薄 shim (向后兼容 import)
├── runner.py                # run() 测试驱动 / benchmark
├── fmha_d256.py             # CLI 入口
├── bench_sweep.sh           # 批量 benchmark
├── fmha_helpers.py          # CUTLASS helpers (官方)
├── prefill_helpers.py       # load/mma/dequant 流水线 helpers (官方)
├── host/
│   ├── config.py            # MixedInputFusedMultiHeadAttentionPrefillD256 配置
│   ├── launcher.py          # @cute.jit launch (TMA/MMA/SMEM 构建 + kernel 发射)
│   └── torch_ref.py         # create_tensor + torch reference
└── device/
    ├── kernel.py            # @cute.kernel 壳 + pipeline/SMEM 分配 + warp 分发
    ├── warp_load.py         # TMA load warp
    ├── warp_transform.py    # INT8→BF16 dequant warp
    ├── warp_mma.py          # MMA warp + mma_pv
    ├── warp_softmax.py      # softmax warp + softmax_step + store_sum
    └── warp_correction.py   # correction warp + rescale/epilog
```

原先单文件 `mixed_input_fmha_prefill_d256.py` (~2100 行) 已拆成上述模块; 逻辑不变,
仅做结构重组 (与 `fmha/host` + `fmha/device` 相同模式)。

## 文件来源 (CUTLASS 官方源)

| 本地路径 | CUTLASS 官方对应 |
| --- | --- |
| `fmha_helpers.py` | `examples/python/CuTeDSL/helpers/fmha_helpers.py` |
| `prefill_helpers.py` | `examples/python/CuTeDSL/cute/blackwell/kernel/attention/mixed_input_fmha/prefill_helpers.py` |
| `mixed_input_fmha_prefill_d256.py` | 薄 shim → `from fmha_d256 import ...` |
| `fmha_d256.py` | study_cute 自写的轻量 CLI shim |
| `__init__.py` | study_cute 自写,空 (使 `fmha_d256` 变成 Python package) |

`mixed_input_fmha_prefill_d256.py` 改动很小,只是为了兼容 Thor 的 DSL 版本:

| Change | Line | Why |
| --- | --- | --- |
| Import rewrites (`from helpers import ...` → `from fmha_d256 import ...`) | 48-62 | Repo layout 适配 |
| 去掉 `storage.tmem_holding_buf.ptr` / `storage.tmem_dealloc_mbar.ptr` 的 `.ptr` | 666 / 670 | Thor 上 DSL (`nvidia-cutlass-dsl` <= 4.4.2) 中 `storage.<scalar_field>` 已经直接返回 `_Pointer`, 没有 `.ptr` 属性 (新 DSL 才加上的 wrapper) |
| 在 `run()` 末尾追加 benchmark block | `runner.py` | study_cute 增补 |
| 包结构拆分为 `host/` + `device/` | 全目录 | study_cute 重构 (对齐 `fmha/`) |

除上述 patch 外,核心 kernel 逻辑与官方一致。

## 接口契约

| 张量 | dtype | shape | 说明 |
| --- | --- | --- | --- |
| Q | BFloat16 | (B, H_q, S_q, D) | D **必须** 256 |
| K | **Int8** | (B, H_k, S_k, D) | int8 量化, H_q 必须能整除 H_k |
| V | **Int8** | (B, H_k, S_k, D) | int8 量化 |
| O | BFloat16 | (B, H_q, S_q, D) | 输出 |
| scale_k | BFloat16 | (B, H_k, S_k, D//scale_granularity) | per-channel scale |
| scale_v | BFloat16 | (B, H_k, S_k, D//scale_granularity) | per-channel scale |

`scale_granularity` 默认 256 (即每 256 维一个 scale)。也支持 128。

**重要**: 这个 kernel 是 mixed-input 设计 (Q 是 BF16, K/V 是 INT8),不是
study_cute 现有 `fmha/` 模块那种纯 FP16/BF16 接口。如果要接入 LLM prefill
loop (cumulative seqlen + packed cache),需要后续做接口适配 — 见底部"后续
工作"章节。

## 关键设计 (核心差异 vs study_cute `fmha/`)

1. **2-CTA cluster** (`cluster_shape_mn = (2, 1)`, `cta_group = TWO`):
   - `qk_mma_tiler[0] = 256` = 2 × cta_tiler[0]=128, 由 2 个 CTA 协作完成
   - 充分利用 Blackwell 2cta TMA multicast
2. **D=256 在 TMEM 内多 slot 累加**:
   - `iterations_pv = 2`, `tOtO_staged[..., iter]` 不同 iter 用 *不同* TMEM
     ACC slice (O 占 256 cols, S 占 128 cols, 总 384 cols 在 512 限制内)
   - 整个 PV 是 *1 个* acquire+commit, 不需要外层 d_chunk loop 重复整个
     attention
3. **Mixed-input 量化**: 8 个 `transform_warp` 专门做 i8→bf16 dequant; 通过
   独立的 `dequant_kv` pipeline 把转换后的 K/V 喂给 MMA warp

## 在 Jetson Thor (Blackwell, compute capability 11.0) 上的运行命令

> ⚠️ **Thor 的 GPU 实际 arch 是 `sm_110`,不是 `sm_100a`**。CUTLASS DSL 默认
> 不认识 (11, 0) 这个 CC,会 fall through 到 `sm_{major}{minor}` ⇒ `sm_110`。
> 如果你用 `CUTE_DSL_ARCH=sm_100a` 编译,launch 时会报
> `cudaErrorNoKernelImageForDevice (209)` — 这是真实的 ISA 不兼容,不是 bug。
>
> shim 已经内置 *auto-detect*:不设环境变量时,会自动用本地 GPU 的 CC 推出
> `sm_110a` / `sm_100a` / `sm_89` 等。**Thor 上推荐做法是直接什么都不设**,
> 让 shim 自己探测;或者显式 `export CUTE_DSL_ARCH=sm_110`。

```bash
cd <study_cute root>

# (推荐) 让 shim 自动探测 GPU 并 set CUTE_DSL_ARCH
unset CUTE_DSL_ARCH

# (备选) 显式指定 — Thor 上是 sm_110 (不带 a 后缀); B200 上才用 sm_100a
# export CUTE_DSL_ARCH=sm_110

# (1) 基础 smoke test (无 ref check, 最快验证 kernel 能跑通; 不 benchmark)
python3 fmha_d256/fmha_d256.py \
    --q_shape 1,8,256,256 --k_shape 1,8,256,256 \
    --is_persistent --skip_ref_check --no_benchmark

# (2) 带 ref check 的精度验证 + 默认 benchmark (warmup=10, iters=100)
python3 fmha_d256/fmha_d256.py \
    --q_shape 1,8,256,256 --k_shape 1,8,256,256 \
    --is_persistent

# (3) 纯 benchmark (跳过 ref check, 默认 warmup=10/iters=100)
python3 fmha_d256/fmha_d256.py \
    --q_shape 1,8,256,256 --k_shape 1,8,256,256 \
    --is_persistent --skip_ref_check

# (4) 调高 iters 让结果更稳
python3 fmha_d256/fmha_d256.py \
    --q_shape 1,8,1024,256 --k_shape 1,8,1024,256 \
    --is_persistent --skip_ref_check \
    --warmup_iterations 50 --iterations 500

# (5) Causal 模式
python3 fmha_d256/fmha_d256.py \
    --q_shape 1,8,512,256 --k_shape 1,8,512,256 \
    --is_persistent --is_causal

# (6) GQA (H_q != H_k)
python3 fmha_d256/fmha_d256.py \
    --q_shape 1,8,256,256 --k_shape 1,1,256,256 \
    --is_persistent

# (7) 完整 CLI 选项
python3 fmha_d256/fmha_d256.py --help
```

### 一键 sweep (bench_sweep.sh)

跑一组 shape 矩阵,自动解析 + 输出 CSV + 终端 markdown 表格:

```bash
cd <study_cute root>

# (1) 默认 8 个 case (B/H_q/S/causal 覆盖核心场景)
bash fmha_d256/bench_sweep.sh

# (2) 快速 smoke (3 个 case)
bash fmha_d256/bench_sweep.sh --quick

# (3) 完整 16 个 case (大 sweep, ~20 分钟)
bash fmha_d256/bench_sweep.sh --full

# (4) 跳过 ref check, 跑快一点
bash fmha_d256/bench_sweep.sh --no-ref-check

# (5) 自定义 timing
bash fmha_d256/bench_sweep.sh --warmup 50 --iters 500

# (6) 自定义输出 + per-case timeout
bash fmha_d256/bench_sweep.sh --out /tmp/perf.csv --timeout 300
```

输出 CSV 列:`case_id,B,H_q,H_k,S_q,S_k,D,is_causal,is_persistent,warmup,iters,latency_us,tflops,io_bw_gbps,status`

`status` 可能取值:
- `PASS` — kernel 跑通 + (启用时) torch ref 对拍通过
- `REF_FAIL` — kernel 跑通但精度不达 atol
- `FAIL_TIMEOUT` — 超时 (默认 180s),通常是死锁
- `FAIL_rc=N` — Python 异常 (CUDA error, OOM, etc.)
- `FAIL_no_summary` / `FAIL_parse` — 输出格式不对 (升级 shim 后可能出现)

Sweep 跑完后终端会直接 print 一张 markdown 表格,可以直接 paste 到 PR / 文档里。

### Benchmark 输出格式

带默认 `--iterations >= 1` 时,会打印两行 perf summary:

```
[benchmark] avg latency: 12.345 us (warmup=10, iterations=100)
[benchmark] throughput: 78.90 TFLOPS  (B=1, H_q=8, H_k=8, S_q=256, S_k=256, D=256, is_causal=False, is_persistent=True)
[fmha_d256] summary: latency=12.345 us  tflops=78.90  io_bw=10.20 GB/s  shape=(B=1,H_q=8,H_k=8,S_q=256,S_k=256,D=256)
```

- **`latency`** — `cute.testing.benchmark` 用 CUDA Event 测的平均 us
- **`tflops`** — `4 × B × H_q × S_q × S_k × D / latency` (causal 时乘 0.5; 只算 QK + PV 两个 GEMM 的 MAC 部分,softmax 算力忽略)
- **`io_bw`** — 一次 attention 的输入/输出字节数 / latency, 公式:
  `bytes = BF16 Q + INT8 K + INT8 V + BF16 O + BF16 scale_k + BF16 scale_v`

### 已知 Thor 兼容性修复

如果你 pull 了最新 CUTLASS upstream 又遇到下面这种报错:

```
AttributeError: '_Pointer' object has no attribute 'ptr'
  File ".../mixed_input_fmha_prefill_d256.py", line 666, in kernel
    storage.tmem_holding_buf.ptr,
```

说明你引入了新版 DSL 写法。修复:**去掉 line 666 和 line 670 的 `.ptr`**
(`storage.tmem_holding_buf` 在 Thor DSL 里已经是 Pointer 类型),见上面"文件
来源"表格里的 patch 说明。

## 本地非 Blackwell GPU 上的限制

本地 sm_89 (RTX 4070 等) GPU 只能跑 import / argparse 验证 (上面 shim 已经
测过)。`cute.compile` 阶段就会因为 `arch=sm_89 != sm_100a` 报错;
`launch` 阶段更不可能。

也就是说,本目录只能在 Jetson Thor / 其他 Blackwell sm_100 GPU 上真正跑通。

## 与 study_cute 现有 `fmha/` 模块的关系

| 项目 | `fmha/` (现有) | `fmha_d256/` (本目录) |
| --- | --- | --- |
| 来源 | 自行重构 + outer-loop D-chunk 设计 | CUTLASS 官方原样迁移 |
| KV dtype | FP16/BF16 (un-quantized) | INT8 + scale **或** BF16/FP16 同质路径 |
| D 支持 | D≤128 完美; D=256 Round 2 死锁 (见 fmha/STATUS.md) | D=256 原生支持 |
| Cluster | 1-CTA | **2-CTA** |
| LLM prefill 接口 | ✓ 已接入 (cumulative seqlen + packed cache) | ✓ 同质 dtype + `call_llm` |
| 代码量 | ~3500 行 (10+ 模块) | ~2400 行 (3 个官方文件 + 1 个 shim) |
| 推荐用途 | 学习 D≤128 的 outer-loop / pipeline 设计 | 跑 D=256 production 路径 |

## 同质 dtype（不量化）与 LLM prefill

`fmha_d256` 现支持两条编译路径：

| 路径 | KV dtype | 入口 | 用途 |
|------|----------|------|------|
| mixed-input（默认） | INT8 + BF16 scale | `__call__` / `launch` | CUTLASS 官方量化路径 |
| homogeneous | BF16/FP16（与 Q 相同） | `launch_homo` | LLM runner、精度对比 |

**单轮精度验证（同质 BF16）：**

```bash
cd <study_cute root>
unset CUTE_DSL_ARCH
python3 fmha_d256/fmha_d256.py \
    --q_shape 1,8,1024,256 --k_shape 1,8,1024,256 \
    --kv_dtype bf16 --is_persistent --is_causal
```

**多轮 LLM prefill（BSHD + packed KV cache + cum_seqlen_k）：**

```bash
python3 fmha_d256/fmha_d256.py \
    --llm_multi_round --llm_batch 4 --llm_seq_len 128 \
    --llm_rounds 3 --llm_kv_cap 512 \
    --q_shape 1,8,128,256 --k_shape 1,8,128,256 \
    --kv_dtype bf16 --is_persistent --is_causal --iterations 0
```

实现要点：
- `kernel_homo`：独立 sV SMEM、无 scale pipeline
- `transform_k/v`：保留 layout 变换，跳过 dequant
- `_call_llm`：BSHD `(B,S,H,D)` + KV cache `(B,2,H,cap,D)`，与 `fmha/` 契约一致

## 后续工作 (可选)

1. **FP16 端到端** — LLM test 目前用 fp16 存储近似 bf16；完善 cupy bf16 路径
2. **性能对比** — homo vs mixed-input vs `fmha/` D=128 在 Thor 上的 latency
3. **Variable seqlen Q** — 当前 LLM 路径仅 `cum_seqlen_k`，Q 长度来自 tensor shape
