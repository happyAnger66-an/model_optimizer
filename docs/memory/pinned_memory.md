# Pinned Memory 与 Pageable Host 内存

> 面向 Pi0.5 推理链路（`Policy.infer` → ViT TRT）的 host 内存与 H2D 传输说明。  
> 平台背景：**Jetson Thor**（CUDA 13.0+ / L4T，共享 LPDDR5X + 硬件一致 UVM）。

---

## 一、本质区别

### 1.1 Pageable host 内存（普通内存）

通过 `malloc` / `new`、NumPy 默认分配、`torch.empty()`（未 pin）得到的都是 **pageable** 内存。

| 特性 | 说明 |
|------|------|
| OS 管理 | 可被 **换页（swap）**、**迁移物理地址**（compaction、NUMA 迁移等） |
| 物理页 | **不固定**，GPU DMA 引擎 **不能直接** 从这里发起传输 |
| CUDA H2D | 驱动通常先 **拷到内部 staging buffer（本质是 pinned）**，再 DMA 到 GPU → **多一次拷贝** |
| 异步传输 | `cudaMemcpyAsync` / PyTorch `non_blocking=True` 对 pageable 源 **往往无效或退化为同步** |

可以把它理解成：**地址随时可能变，硬件不敢直接搬**。

### 1.2 Pinned Memory（页锁定 / 锁页内存）

通过 `cudaMallocHost`、`cudaHostAlloc`，或 PyTorch 的 `pin_memory=True` / `torch.empty(..., pin_memory=True)` 分配。

| 特性 | 说明 |
|------|------|
| 物理页 | **锁定在 RAM**，不会被 swap、不会被 OS 随意搬家 |
| GPU DMA | **可直接访问**，适合 `cudaMemcpyAsync` |
| 流水线 | 配合 **独立 CUDA stream + `non_blocking=True`**，H2D 可与 GPU 计算 **重叠** |
| 代价 | 占用 **真实物理内存**（不能换出），量大了会挤占系统与其他进程 |

---

## 二、传输路径对比

### 2.1 离散 GPU 上的经典模型

```text
Pageable 源:
  CPU 填数据 → [驱动 staging pinned] → GPU
              ↑ 额外一次 copy，常同步

Pinned 源:
  CPU 填数据 ──async DMA──→ GPU
              ↑ 可与 TRT / 下一帧 CPU 工作重叠
```

### 2.2 与「统一内存 / Zero-copy」的区别

容易混淆的三个概念：

| 类型 | 分配方式 | 典型用途 |
|------|----------|----------|
| Pageable | `malloc` / NumPy | 普通 CPU 数据 |
| Pinned | `cudaMallocHost` / `pin_memory=True` | 高效 H2D/D2H staging |
| Unified (Managed) | `cudaMallocManaged` | CPU/GPU 同址访问，按需 fault/map |

- **Pinned** 解决的是 **「怎么高效搬过去」**。
- **Unified** 解决的是 **「要不要显式搬」**（访问时由 UVM 驱动处理映射）。

---

## 三、Pi0.5 推理链路中的现状

典型路径：

```text
LeRobot uint8 HWC numpy
  → openpi transforms（ResizeImages 224，CPU/JAX）
  → Policy.infer：torch.from_numpy(...).to(cuda)     ← 同步 H2D + pageable
  → Observation.from_dict：uint8→float、HWC→CHW、/255*2-1
  → preprocess_observation_pytorch（可能再 resize）
  → embed_prefix_vit_batched：cat → TRT ViT [V,3,224,224]
  → TRT cuda graph：copy_ → static_inputs → replay
```

关键代码（pageable 路径）：

```python
# third_party/openpi/src/openpi/policies/policy.py
inputs = jax.tree.map(
    lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
    inputs,
)
```

NumPy 分配的是 pageable 内存 → `.to(cuda)` 通常 **无法真正异步重叠**。

ViT TRT 期望输入 **`[N, 3, 224, 224]` float，值域 `[-1, 1]`**。当 ViT TRT 已 ~1ms 时，剩余时间往往在：

- CPU 上 `from_numpy` + normalize + permute
- **pageable host → GPU 的同步拷贝**
- 每帧新建 tensor（allocator 抖动）
- `trt_cuda_graph: true` 时，还要 **copy 进 static input buffer**

---

## 四、Pinned 预处理怎么做

### 4.1 原理

**pinned 源 + `non_blocking=True`** 成对使用才有意义；需 **独立 copy stream**，避免与 TRT / FlashRT 默认 stream 互相等待；**复用 buffer**，不要每帧 `torch.tensor(numpy_array)` 新建。

### 4.2 推荐做法（单帧 / WebUI 推理）

在 `Policy.infer` 入口或 executor 里预分配 buffer，每帧复用：

```python
class GpuImageStaging:
    def __init__(self, device, num_views=3, h=224, w=224):
        self.device = device
        self.pinned = [
            torch.empty((1, 3, h, w), dtype=torch.float32, pin_memory=True)
            for _ in range(num_views)
        ]
        self.gpu = [
            torch.empty((1, 3, h, w), dtype=torch.float32, device=device)
            for _ in range(num_views)
        ]
        self.copy_stream = torch.cuda.Stream(device=device)

    def upload_view(self, i, arr_nhwc_uint8):
        """arr: numpy uint8 [H,W,3]，已在 transforms 里 resize 到 224"""
        t = torch.from_numpy(np.ascontiguousarray(arr_nhwc_uint8))
        x = t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0).mul_(2.0).sub_(1.0)
        self.pinned[i].copy_(x)

    def h2d_all(self):
        with torch.cuda.stream(self.copy_stream):
            for i in range(len(self.pinned)):
                self.gpu[i].copy_(self.pinned[i], non_blocking=True)
        torch.cuda.current_stream().wait_stream(self.copy_stream)
        return self.gpu
```

### 4.3 DataLoader 路径（dataset eval）

```python
loader = DataLoader(
    dataset,
    batch_size=1,
    pin_memory=True,
    num_workers=2,
    persistent_workers=True,
)
for batch in loader:
    images = batch["images"].to(device, non_blocking=True)
```

### 4.4 预期收益

- **不减少** CPU 上 resize/normalize 的时间
- **减少** H2D 阻塞和分配开销
- ViT 已经很快时，端到端通常是 **低～中** 收益

---

## 五、GPU 预处理（与 pinned 配合）

目标：把 **normalize + HWC→CHW +（可选）resize** 放到 GPU，减少 CPU 大图算子和 float32 H2D 体积。

### 5.1 零成本：固定 224，去掉重复 resize

训练 transforms 已有 `ResizeImages(224, 224)`。推理时若分辨率已是 224，应跳过 `preprocess_observation_pytorch` 里的二次 `resize_with_pad_torch`。

### 5.2 推荐：uint8 小拷贝 + GPU normalize

比 CPU 算完 float32 再 H2D 更省带宽（224×224×3 uint8 ≈ 150KB vs float32 ≈ 600KB × 视角数）：

```python
def gpu_preprocess_uint8_nhwc(raw_u8: torch.Tensor, out_bchw: torch.Tensor):
    """raw: [H,W,3] uint8 on GPU; out: [1,3,224,224] float preallocated"""
    x = raw_u8.permute(2, 0, 1).unsqueeze(0).float()
    x.mul_(2.0 / 255.0).sub_(1.0)
    out_bchw.copy_(x)
    return out_bchw
```

H2D 路径：pinned uint8 staging → `non_blocking` 拷到小 GPU tensor → GPU 上 normalize。

### 5.3 与 ViT TRT + CUDA Graph 对接

`trt_torch.py` 在 graph 模式下每帧 `copy_` 到 `static_inputs` 再 `replay()`。理想路径：

```text
GPU preprocess → 直接 copy_ 到 vit_engine static_inputs["pixel_values"] → graph.replay()
```

预处理输出地址固定，避免 cat 再 copy 的双缓冲。

---

## 六、Jetson Thor 平台特性

Thor + **CUDA 13.0 / L4T** 与旧 Jetson（Orin 及更早）差别较大：

| 特性 | 说明 |
|------|------|
| 内存架构 | CPU/GPU **共享** LPDDR5X（如 128GB、~273 GB/s），非 PCIe 离散 GPU |
| UVM + 一致性 | `pageableMemoryAccessUsesHostPageTables = 1` 时，GPU 可通过 host 页表 **直接访问** `malloc`/`mmap` 的 pageable 内存 |
| Managed memory | `concurrentManagedAccess = 1`，`cudaMemPrefetchAsync` 可用 |
| 映射行为 | Tegra 文档：映射后的 unified memory 在 Thor 上 **行为接近 pinned（IO coherent 等）**，但首访仍可能有 page fault |

### 6.1 对 Thor 的含义

- **「pageable 完全不能给 GPU 用」不再成立** — 某些场景 GPU 可直接读 host buffer，甚至减少显式 `cudaMemcpy`。
- 但 **PyTorch 的 `.to(cuda, non_blocking=True)`** 仍通常要求源在 **pinned** 上，才会走真正 async DMA。
- 在 **PyTorch + TensorRT 固定 buffer + CUDA Graph** 链路上，**pinned staging 仍是最稳妥、最好 profile 的优化手段**。

### 6.2 Thor 实践建议

| 场景 | 建议 |
|------|------|
| 追求 H2D 与 TRT 重叠、固定 buffer 复用 | **pinned staging + copy stream + non_blocking** |
| buffer 大、CPU/GPU 反复读同一块 | 评估 **`cudaMallocManaged` + prefetch** |
| 相机/ROS 已是 mmap 共享 buffer | 可探索少一次 copy（直接 GPU 读 pageable），需实测 latency |
| 小帧 uint8、每帧只读一次 | pinned + 小 H2D + GPU normalize 通常最划算 |

### 6.3 Thor 上使用 pinned 的注意点

1. **内存池共享**：pinned 占的是 LPDDR5X 里的一块，锁多了会实打实占 RAM。
2. **别无限预分配**：3 视角 × `[1,3,224,224]` float（pinned + GPU 各一套）通常足够。
3. **首帧 vs 稳态**：UVM/pageable 直接访问可能有 **page fault**；实时推理需 **warmup + prefetch**。
4. **数值路径不变**：pinned 只改传输方式，不改 `(x/255*2-1)` 或 ViT 输入格式。

---

## 七、实施顺序（由易到难）

1. **Profile**：看 `embed_prefix` 里 vision 之外是否还有大量 host 时间。
2. **固定 224 + 去重复 resize**（零成本）。
3. **Pinned buffer + non_blocking + copy stream**（改动小）。
4. **uint8 H2D + GPU normalize**（中等）。
5. **预处理直写 ViT TRT static input**（与 CUDA Graph 对齐，改动大）。

---

## 八、相关文件

| 文件 | 说明 |
|------|------|
| `third_party/openpi/src/openpi/policies/policy.py` | `infer` 入口，当前 pageable H2D |
| `third_party/openpi/src/openpi/models/model.py` | `Observation.from_dict`，uint8→float、HWC→CHW |
| `third_party/openpi/src/openpi/models_pytorch/preprocessing_pytorch.py` | resize / layout |
| `src/model_optimizer/infer/tensorrt/trt_torch.py` | TRT CUDA Graph、`static_inputs.copy_` |
| `src/model_optimizer/infer/tensorrt/pi05_trt_engine_setup.py` | `embed_prefix_vit_batched` |
| `docs/optimizer/optimizer_summary.md` | Pi0.5 整体推理优化总结 |

---

## 九、一句话总结

- **Pageable**：OS 可移动/可换出，GPU 不能可靠 async DMA，H2D 常多一次 staging、易阻塞。
- **Pinned**：物理页固定，async DMA 友好，适合与 TRT ViT / denoise **流水线重叠**。
- **Jetson Thor**：硬件一致 UVM 降低了 pageable 直接给 GPU 用的门槛；但在 PyTorch + TensorRT 固定 buffer 链路上，**pinned staging 仍是首选优化手段**。
