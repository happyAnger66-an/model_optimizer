# 2-CTA Cluster 算法理解

本文档总结 `fmha_d256_cutedsl` 中 2-CTA cluster 的作用，以及它和 `head_dim=256` 这类大 head dimension FMHA kernel 的关系。

## 整体理解

2-CTA cluster 可以理解成：两个 CTA 被硬件和 CUTLASS/CuTe DSL 组织成一个协作单元，共同完成同一个逻辑 attention tile 中的 Blackwell `tcgen05` MMA 工作。

它不是简单地让：

```text
CTA 0 算一个 head
CTA 1 算另一个 head
```

而更接近：

```text
一个逻辑 attention tile
        │
        ▼
2 个 CTA 组成一个 cluster
        │
        ├── CTA 0：负责 tcgen05 MMA 的一个 slice
        └── CTA 1：负责 tcgen05 MMA 的另一个 slice
        │
        ▼
两个 CTA 通过 cluster-aware pipeline / mbar / TMEM barrier 同步
        │
        ▼
共同完成 QK、softmax、PV、correction
```

也就是说，两个 CTA 不是互不相关地计算两个独立 tile，而是在同一个逻辑 tile 内按 `cluster_layout_vmnk` 分工。

## 和 `head_dim=256` 的关系

可以说 2-CTA cluster 很大程度上是为了让 `D=256` 这类大 head dimension attention tile 更高效、更可控地跑起来，但它不是专门为 `D=256` 发明的补丁。

`head_dim` 变大后，FMHA 的两个核心 GEMM 都会变重：

```text
S = Q @ K^T
O = softmax(S) @ V
```

当 `D=256` 时，沿 head dimension 的归约长度更大，相比 `D=64` 或 `D=128` 会带来更高的计算量、寄存器压力、SMEM 压力、TMEM 压力以及流水线同步压力。

如果只用单 CTA 承担完整 tile，通常会遇到两难：

- tile 做小：资源压力下降，但 tensor core 利用率和吞吐可能下降。
- tile 做大：吞吐潜力更高，但单 CTA 的 SMEM、TMEM、寄存器和同步压力变大。

2-CTA cluster 的价值在于让两个 CTA 协同喂一个更大的 Blackwell `tcgen05` MMA tile，把同一个逻辑 MMA tile 的工作切成两个 CTA slice，从而更容易维持较大的 tile shape、较高 tensor core 利用率和可控的资源占用。

因此，对于 `fmha_d256_cutedsl`：

```text
D=256 让每个 QK/PV tile 变厚
        │
        ▼
单 CTA 扛完整 tile 的资源和吞吐压力变大
        │
        ▼
2-CTA cluster 把同一个逻辑 tile 的 MMA 工作切给两个 CTA
        │
        ▼
更容易维持大 tile、高 tensor core 利用率和可控资源占用
```

一句话总结：

> 2-CTA cluster 很适合解决 `D=256` 这类大 head dimension FMHA 的性能和资源压力问题，但它本质上是 Blackwell 上扩大 MMA tile、提高 tensor core 吞吐、分摊资源压力的协作 CTA 算法，而不只是 “D=256 太大所以算不了” 的 workaround。

## 在 `fmha_d256_cutedsl` 中的代码体现

在 `device/kernel.py` 里，2-CTA cluster 主要体现在以下几个地方。

### CTA 在 cluster 内的 slice 编号

```python
mma_tile_coord_v = bidx % cute.size(qk_tiled_mma.thr_id.shape)
```

`mma_tile_coord_v` 表示当前 CTA 在 2-CTA MMA 中负责哪个 slice，通常可以理解成 `0` 或 `1`。

### CTA 在 cluster 内的 rank 和逻辑坐标

```python
cta_rank_in_cluster = cute.arch.make_warp_uniform(
    cute.arch.block_idx_in_cluster()
)

block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
    cta_rank_in_cluster
)
```

这里拿到当前 CTA 在 cluster 内的 rank，并通过 `cluster_layout_vmnk` 映射成 cluster 逻辑坐标。后续 TMA partition 和 pipeline 都依赖这个坐标来决定当前 CTA 负责哪一部分数据。

注意这里的 `V` 是 CUTLASS/CuTe cluster layout 里的模式维度，不是 attention 中的 Value tensor。

### Cluster-aware pipeline

多个 pipeline 创建时都传入了：

```python
cta_layout_vmnk=cluster_layout_vmnk
```

例如：

```python
load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
    ...
    cta_layout_vmnk=cluster_layout_vmnk,
    ...
).make_participants()
```

这说明这些 producer/consumer 同步关系不是单 CTA 内部的简单 mbar，而是需要理解 cluster 中两个 CTA 的协作布局。

### 2-CTA TMEM 生命周期管理

```python
tmem = utils.TmemAllocator(
    storage.tmem_holding_buf,
    barrier_for_retrieve=tmem_alloc_barrier,
    allocator_warp_id=self.correction_warp_ids[0],
    is_two_cta=True,
    two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar,
)
```

`is_two_cta=True` 表示 TMEM 分配和释放要按 2-CTA 模式处理。因为两个 CTA 会共同使用 TMEM 保存 `S`、`P`、`O_partial` 等中间结果，不能由某一个 CTA 提前释放。

### 2-CTA 对 scale 数据的切分

```python
gScaleK_kdl_ = cute.logical_divide(gScaleK_kdl, (self.scale_k_tiler[0] // 2,))[
    (None, mma_tile_coord_v), None, None
]
```

这里把 scale K 的 leading 维度按 2-CTA 切成两半，并用 `mma_tile_coord_v` 选择当前 CTA 负责的那一半。这是 2-CTA 数据分工的一个直接例子。

## 和 warp-specialized pipeline 的关系

每个 CTA 内部仍然有自己的 warp 角色：

```text
CTA
├── load warp
├── transform warps
├── mma warp
├── softmax warps
└── correction warps
```

2-CTA cluster 后，结构变成：

```text
Cluster
├── CTA 0
│   ├── load / transform / mma / softmax / correction
│   └── 负责 MMA slice 0
│
└── CTA 1
    ├── load / transform / mma / softmax / correction
    └── 负责 MMA slice 1
```

两个 CTA 的 warp 角色结构相同，但它们处理的是同一个逻辑 attention tile 的不同硬件分片。它们通过 cluster-aware pipeline 保证阶段顺序：

```text
数据加载完成
        ▼
transform 可读
        ▼
MMA QK 可读
        ▼
softmax 可读 S
        ▼
MMA PV 可读 P
        ▼
correction 可读 O_partial + row_sum
        ▼
写出最终 O
```

## 最容易误解的点

2-CTA cluster 不是把输出 tile 简单分成两个完全独立 tile。它是把同一个 MMA tile 的工作拆给两个 CTA 协同执行。

因此更准确的理解是：

```text
错误理解：
CTA 0 算一个完整 attention tile
CTA 1 算另一个完整 attention tile

正确理解：
CTA 0 和 CTA 1 共同算同一个逻辑 attention tile，
每个 CTA 负责其中一个 tcgen05 / cluster MMA slice。
```

对于 `D=256` FMHA，这种协作模式的主要收益是：在资源压力可控的前提下继续使用较大的 MMA tile，从而更好地利用 Blackwell tensor core。
