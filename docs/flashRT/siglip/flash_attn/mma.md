# `Sm100FmhaFwdMainloopTmaWarpspecialized::mma` 总结

源码：`third_party/FlashRT/csrc/attention/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp`（约 268–522 行）。

## 角色

- SM100（Blackwell）上 **TMA + warp 专精** 的 FMHA 前向里，**Tensor Core（UMMA）warp** 的主入口。
- 消费 **Load warp** 写入共享内存的 **Q / K / V**，向 **Tensor Memory（TMEM）** 写入 **S（QK  logits）**，从 TMEM 读 **P（softmax 后的概率）** 并与 **smem V** 做 **PV GEMM** 得到 **O（部分输出）**。
- **不负责 softmax**：softmax 由其它 warp 的 `softmax_step` 完成；本函数通过 **`pipeline_s0` / `pipeline_s1`** 与 softmax 交接 **S**，通过 **`pipeline_corr`（`PipelineO`）** 与 correction 交接 **O**。

## 入参与流水线

| 对象 | 含义 |
|------|------|
| `pipeline_q` | Load → MMA，**Q** 的 TMA 异步流水（本函数为 **consumer**，算完 **release**） |
| `pipeline_kv` | Load → MMA，**K / V** 共用流水（按阶段 wait / release，单 stage 时 K/V union 需严格顺序） |
| `pipeline_s0` / `pipeline_s1` | MMA → **两个 softmax warp**，各对应 TMEM 上一块 **S** |
| `pipeline_corr` | MMA → **correction**，对应 TMEM 上的 **O** |
| `storage` | `smem_q`；K/V 为 **union**（`smem_k` / `smem_v`） |

`blk_coord` / `params` 在本函数体内几乎不参与计算；**沿 K 的迭代步数**由 `mask_tile_count = Mask{}.get_trip_count(...)` 决定。

## TMEM 布局（`TmemAllocation`）

- **S0 / S1**：两块 QK 输出（logits），分别交给 softmax warp 0 / 1。
- **P0 / P1**：与 S0/S1 地址区间重叠的 **概率块视图**（softmax 写回后，PV 的 A 侧指向这些偏移）。
- **O0 / O1**：PV 输出槽，供 correction 做缩放与跨步合并。
- **V0 / V1**：与 S0/S1 重叠区域用于 **rowmax 等统计**（在 `softmax_step` 中使用，不在 `mma` 内直接写）。

## 关键对象

- **`mma_qk`**：`CollectiveMmaQK::TiledMma`，**Q×K → S**。
- **`mma_pv_ts`**：`CollectiveMmaPV::TiledMma` 经 `to_tiled_mma_sm100_ts`，**P×V → O**（SM100 TS 路径）。
- **`tSrQ0` / `tSrQ1`**：Q 的两个寄存器/分块视图（是否启用由 **`ThreadShape`** 的 M/K 维是否大于 1 决定）。
- **`tOrP0` / `tOrP1`**：`sP` 布局用 `nullptr` 占位，**真实基址在 TMEM 的 P0/P1**。
- **`pipeline_*_release_state`**：与 consumer 游标对齐，用于在正确时机 **`consumer_release`**，归还 smem stage 给 Load。

## 执行阶段（概要）

### 1. 启动（首块 K / Q）

1. `consumer_wait` **Q**，得到 **`tSrQ0`**。
2. `consumer_wait` **K**，**`producer_acquire(pipeline_s0)`**，**`gemm_zero_acc`：Q1×K1 → `tStS0`**，**`pipeline_s0` commit**。
3. 若 **`ThreadShape` 的 N（K 侧）> 1**：提前 **`release` K**（便于 Load 周转）。
4. 若 **M 或 K 维 `ThreadShape` > 1**：再 wait **Q2**，得 **`tSrQ1`**；若 **N > 1** 再 wait **K**；**`pipeline_s1` acquire**，**Q2×K → `tStS1`**，**`pipeline_s1` commit**。
5. **`release` K**（首块 K 在两路 QK 均用完后释放）。
6. **`consumer_wait` V**；**`pipeline_corr` acquire**（注释：先于 S0 acquire 以减轻关键路径）；**`pipeline_s0` acquire**；**PV：`tOrP0` × V → `tOtO0`**（`gemm_zero_acc`）；**`pipeline_corr` commit**；若 **N > 1** 再 **`release` V**。
7. **`mma_pv_ts.accumulate_ = UMMA::ScaleOut::Zero`**：为后续 **`gemm_reset_zero_acc`** 的累加模式做准备。

### 2. 主循环（`mask_tile_count -= 1` 后 `for (; mask_tile_count > 0; ...)`）

每次迭代沿 mask 再走一个 tile，典型交错为：

- **Wait Ki → Q1×Ki → S0 → `pipeline_s0` commit**；若 **N > 1** → **`release` K**。
- 若 **N > 1**：**Wait V**（上一块/并行支路）；**`pipeline_corr` + `pipeline_s1` acquire** → **`gemm_reset_zero_acc`：P1×V → O1** → **`pipeline_corr` commit** → **`release` V**。
- 若 **N > 1**：再 **Wait Ki**（供 Q2）；**Q2×Ki → S1** → **`pipeline_s1` commit** → **`release` Ki**。
- **Wait Vi**；**`pipeline_corr` acquire**；**`pipeline_s0` acquire** → **`gemm_reset_zero_acc`：P0×Vi → O0** → **`pipeline_corr` commit**；若 **N > 1** → **`release` V**。

**`ThreadShape` 的 N = 1**（默认如 `(2,1,1)` 堆叠 softmax）时，循环内多段 **额外的 K/V wait 与 release** 被 **`if constexpr` 裁掉**，流水更紧。

### 3. 收尾

- **`release` Q** 一次；若 **M 维 `ThreadShape` > 1** 再 **`release` Q** 一次。
- 若 **N > 1**：可能再 **wait V**；最后一次 **PV：P×V → O1**（`tOrP1` / `pipeline_s1`）；**`pipeline_corr` commit**；**`release` V**。
- **`pipeline_s0` / `pipeline_s1` 各再 `producer_commit` 一次**：对齐双槽流水尾部，避免 softmax 侧少握手。

## 与 softmax 的数据关系

- MMA warp：**写 S**（`gemm_zero_acc` + `pipeline_s*` commit）→ softmax：**读 S、写 P 与统计** → MMA warp：**读 P（TMEM）+ smem V，写 O**（`gemm_*` + `pipeline_corr`）。
- 文件末尾注释（约 520–521 行）用时间线概括了 **QK 与 PV 在多 tile 上的交错**。

## 术语对照

| 符号 | 含义 |
|------|------|
| `gemm_zero_acc` | QK 或 PV 上 **累加器清零** 的 GEMM 封装 |
| `gemm_reset_zero_acc` | 与 **`accumulate_`** 配合的 PV GEMM（分块 O 更新） |
| `ThreadShape` | 两个 softmax warp 在 **M/N/K** 上与 Q/K 分块的排布；**N>1** 时 K/V 的 wait/release 更细 |

## 延伸阅读

- 同文件 **`softmax_step`**：TMEM load/store、mask、rowmax、exp2、写 P 与 `pipeline_c` 等。
- **`gemm_zero_acc` / `gemm_reset_zero_acc`** 宏或模板定义通常在 **`fmha_common.hpp`** 或 Cutlass 侧 collective 中；若需对齐到具体 **`warpgroup.mma`** 指令，需在完整 include 树中打开对应实现。
