# CPU–GPU 内存模型与传输机制

> 面向 Pi0.5 推理与 **Jetson Thor** 部署的 host/device 内存概念梳理。  
> Pinned 实践与预处理接入见 [`pinned_memory.md`](./pinned_memory.md)；整体优化见 [`../optimizer/optimizer_summary.md`](../optimizer/optimizer_summary.md)。

---

## 一、四种相关概念（不要混为一谈）

| 概念 | 是什么 | 是否显式 `cudaMemcpy` | 典型 API |
|------|--------|----------------------|----------|
| **Pageable** | OS 可换页、可迁移的普通 host 内存 | 需要（且常多一次 staging） | `malloc`、NumPy 默认 |
| **Pinned** | 物理页锁在 RAM，供 DMA 使用 | **默认仍要**（一次 async DMA） | `cudaMallocHost`、`pin_memory=True` |
| **Unified (Managed)** | CPU/GPU 共享分配，UVM 按需 map | **通常不要** | `cudaMallocManaged` |
| **Zero-copy** | 设计目标：少拷一份 buffer | 不要 H2D | Mapped pinned、UVM、NVMM 相机 buffer |

关系：

```text
Pinned     → 解决「怎么高效搬」（DMA 快路径）
Unified    → 解决「要不要搬」（共享物理页）
Zero-copy  → 目标状态（无 Host H2D Memcpy）

Pinned 本身 ≠ zero-copy；Pinned + Mapped → zero-copy 的一种实现
Unified   → zero-copy 的另一种实现
```

---

## 二、Pinned Memory 的主要用途

**核心用途：加快 CPU↔GPU 拷贝，并支持真正的异步 DMA。**

1. **DMA 快路径**：物理地址固定，GPU Copy Engine 可直接访问；pageable 往往要先拷到驱动内部 staging buffer（本质也是 pinned），再 DMA → **多一次拷贝**。
2. **异步传输**：`cudaMemcpyAsync` / PyTorch `non_blocking=True` 要求 H2D 时 **源在 pinned**；否则常退化为同步。

```text
Pinned + Memcpy（主用途）:
  CPU 写 pinned → cudaMemcpyAsync / copy_(non_blocking=True) → GPU device buffer

Pinned + Mapped（扩展用途，见 §五）:
  CPU 写 pinned ←──同一物理页──→ GPU 用 device pointer 直读（无 Memcpy）
```

PyTorch 的 `pin_memory=True` **只做 pin，不做 map**；要 zero-copy 必须额外 `cudaHostGetDevicePointer`。

---

## 三、Zero-copy 指什么

**Zero-copy** 不是单一 CUDA API，而是 **尽量不增加额外 buffer 拷贝** 的目标。

### 3.1 两个层次

| 层次 | 含义 |
|------|------|
| **Host ↔ GPU** | CPU 写完，GPU 直接读，不做 `cudaMemcpy` |
| **整条视觉链** | 相机 → 预处理 → ViT，中间不落地 numpy + `.to(cuda)` |

### 3.2 常见实现路径

```text
路径 A  Pinned + Mapped     cudaHostRegister(MAPPED) + cudaHostGetDevicePointer
路径 B  Unified + Prefetch  cudaMallocManaged + cudaMemPrefetchAsync
路径 C  Thor pageable+页表   malloc/mmap，GPU 走 host 页表（CUDA 13+）
路径 D  相机 NVMM           GStreamer/Argus → GPU 直接消费 surface（机器人量产最彻底）
```

### 3.3 Pi0.5 里 zero-copy 的上限

即使 Host→GPU 零拷贝，TRT CUDA Graph 仍可能有 **GPU 设备内** 拷贝：

```text
prepared_inputs → static_inputs.copy_(...) → graph.replay()
                  ↑ GPU D2D，与 host zero-copy 无关
```

Zero-copy **主要优化「numpy → 第一块 GPU 可见 buffer」**；ViT engine 固定 input 地址时，**static_inputs 写入往往仍在**。

---

## 四、Unified Memory（Managed）的代价

Managed 看起来「一块 buffer、CPU/GPU 都能用、不用 Memcpy」，但 **不是免费午餐**。

| 代价 | 说明 |
|------|------|
| **首访 / Page fault** | 分配后物理页常 lazy map；CPU/GPU 首次访问可能毫秒级尖刺 → 实时推理需 warmup + prefetch |
| **访问性能不确定** | GPU 读 managed/system 页可能走 uncached / coherence 路径，**未必快于 H2D 到 device buffer** |
| **访问模式敏感** | CPU/GPU 交替读写同一页有 coherence 开销；需 `cudaMemAdvise` |
| **不省 RAM** | 仍占 LPDDR 物理页；managed 源 + TRT static buffer 可能 **双份占用** |
| **同步责任** | `concurrentManagedAccess=1` 不等于无竞态；CPU 写与 GPU 读需 stream 顺序 |
| **PyTorch 生态** | 默认 `from_numpy` / `.cuda()` 不走 managed；集成成本高 |
| **Thor 特注** | `cudaMallocManaged` 在 Thor 上 **默认非 GPU-cached**；行为需 profile，不能假设比 pinned 快 |

**结论**：Unified 优化的是 **编程模型与 Memcpy 次数**，不是自动让 ViT 更快；Thor 上值得实验，但 **H2D 到 device + TRT** 仍常是可预测的首选。

---

## 五、Pinned + Mapped：zero-copy 具体怎么做

### 5.1 两步机制

```text
1. Pin   cudaHostAlloc(..., cudaHostAllocMapped)
         或 cudaHostRegister(..., cudaHostRegisterMapped)
2. Map   cudaHostGetDevicePointer(&d_ptr, h_ptr, 0)
         → d_ptr 给 kernel / TRT 使用；CPU 写 h_ptr 侧
```

### 5.2 示例（cuda-python）

```python
from cuda import cudart
import numpy as np

shape = (1, 3, 224, 224)
arr = np.ascontiguousarray(np.empty(shape, dtype=np.float32))

err, = cudart.cudaHostRegister(
    arr.ctypes.data, arr.nbytes,
    cudart.cudaHostRegisterFlags.MAPPED,
)
assert err == cudart.cudaError_t.cudaSuccess

err, d_ptr = cudart.cudaHostGetDevicePointer(arr.ctypes.data, 0)
# CPU：写 arr
# GPU：用 d_ptr 包装成 tensor 或传入 kernel / TRT（需 C++/pybind 或项目内 CudaBuffer 封装）

# 释放前
cudart.cudaHostUnregister(arr.ctypes.data)
```

### 5.3 Pi0.5 推荐数据流

```text
初始化：MappedHostBuffer(3 视角 × [1,3,224,224])

每帧：
  CPU  → normalize/layout 写入 mapped.arr
  sync → 保证 CPU 写完成
  GPU  → 读 mapped d_ptr，或 copy_ 到 vit static_inputs（更稳）→ TRT replay
```

**注意**：PyTorch 原生 API 不便直接从 raw `d_ptr` 建 tensor；生产环境常用 **cuda-python 或小 C++ 扩展**。

---

## 六、底层硬件：GPU 如何访问「注册给它的 CPU 内存」

### 6.1 两种 GPU 访问 host 物理页的路径

| 方式 | 硬件 | 典型 API |
|------|------|----------|
| **显式拷贝** | **Copy Engine（DMA 引擎）** bulk 搬运 | `cudaMemcpy`、`copy_(non_blocking=True)` |
| **Mapped 直读** | **SM global load/store** → GPU MMU → 互联 → DRAM | kernel / TRT 读 mapped `device pointer` |

Mapped zero-copy **不走 Copy Engine API**，而是 **算子直接 load 已映射的物理页**；仍占用 **内存带宽**，只是少占一份 device buffer、少一次 Memcpy 指令。

### 6.2 离散 GPU（x86 + RTX，PCIe）—— 对照

```text
┌──────── CPU ────────┐          PCIe / NVLink          ┌────── GPU ──────┐
│  system RAM         │ ◄──── 读/写事务 ────────────────► │  SM / MMU / L2  │
│  (pinned 物理页)    │                                   │  device VA      │
└─────────────────────┘                                   └─────────────────┘
```

1. `cudaHostRegister(MAPPED)`：锁死物理页（PA 不变）。
2. 驱动 + **IOMMU**：建立 `GPU VA → host PA`。
3. SM `LDG` → MMU → **PCIe Memory Read** → system RAM 返回数据。

**本质**：仍是 GPU 经 **PCIe/NVLink** 访问 **system RAM**；与 CE DMA 同类物理现象，但 SM 散读常不如 CE bulk 高效。

### 6.3 Jetson Thor（SoC 共享 LPDDR）—— 你们平台

```text
┌────────────────────── Thor SoC ──────────────────────┐
│  CPU (Neoverse)              GPU (Blackwell iGPU)    │
│       \                           /                  │
│        └── coherent interconnect ──┘                 │
│                      │                               │
│               LPDDR5X 控制器（128GB 同一池）          │
└──────────────────────────────────────────────────────┘
```

- **无 CPU↔GPU 之间的 PCIe**；pinned/mapped 页就是 **LPDDR 里的物理帧**。
- GPU SM load → SMMU/MMU → **片上 coherent 互联** → 内存控制器 → LPDDR。
- CUDA 13+：**硬件一致 UVM**；`pageableMemoryAccessUsesHostPageTables=1` 时 GPU 甚至可读 **普通 malloc**（按需 fault/map）。

Thor 上 mapped 直读 **不经过 PCIe**，但 **仍占 LPDDR 带宽**；是否比「CE DMA → device buffer」快，**只能 profile**。

### 6.4 术语对照

| 术语 | 含义 | Thor |
|------|------|------|
| **DMA / Copy Engine** | 专用块拷贝硬件 | 有，用于 Memcpy |
| **PCIe** | 板级 CPU↔独显总线 | **CPU↔GPU 不用** |
| **Pin** | 锁物理页，供稳定映射 | 仍需要 |
| **Map** | GPU VA 指向 host PA | 直读前置条件 |
| **Memcpy** | CE 把数据搬到另一块 buffer | 与 mapped 直读二选一或组合 |

```text
Pin   → 「这块 PA 别动」
Map   → 「GPU VA 也能指向这块 PA」
Memcpy → 「CE 搬一份到另一块 buffer」
直读   → 「SM 直接去 PA 读（仍占带宽）」
```

---

## 七、`non_blocking` 在 Pi0.5 里的真实收益

当前 **单帧、同步** `Policy.infer` 路径：

```text
input_transform → 全部 .to(cuda) → preprocess → embed_prefix(ViT) → LLM → denoise
```

**同一帧内**：ViT **必须等** 图像已在 GPU → **H2D 不能与 ViT 并行**。

| 场景 | `non_blocking` 收益 |
|------|---------------------|
| **单帧同步 eval** | **很小**（常 <1ms）；pinned 主要价值是 **少 staging、拷贝更快** |
| 同帧内与其他小 tensor H2D 重叠 | 有，但 state/token 远小于 3 视角图像 |
| **跨帧双缓冲**（采集 ‖ 推理） | **明显**；需异步架构，非改一行 `.to()` |
| TRT `static_inputs.copy_` | GPU **D2D**，且 replay 前仍 sync → 与 host H2D 无关 |

**结论**：`non_blocking` 优化的是 **传输能否与其他工作重叠**；Pi0.5 单帧场景下 **pinned 的价值主要在「拷贝更快」而非「与模型并行」**。要大收益需 **双缓冲流水线** 或 **少做 H2D（zero-copy / uint8 减量）**。

---

## 八、方案对比（Pi0.5 + Thor）

| 方案 | Host H2D | 延迟可预测 | 工程成本 | 备注 |
|------|----------|-----------|----------|------|
| Pageable + `.to(cuda)` | 有，常同步+staging | 中 | 低 | 现状 |
| Pinned + Memcpy async | 有，更快 | **高** | 低 | 首选微优化 |
| Pinned + Mapped | 无 | 中（需 sync） | 中 | GPU 直读可能慢于 device buffer |
| Unified + prefetch | 无 | 低（fault） | 中高 | Thor 可试，需 warmup |
| Thor pageable 直读 | 往往无 | 中 | 中 | 依赖 CUDA 13 + 页表 |
| NVMM 相机 → GPU | 无 | 高（管线成熟时） | 高 | 机器人部署终极路径 |

---

## 九、决策建议（简要）

1. **先 profile**：H2D + preprocess 占 `infer_ms` 比例；若 <5%，不必重押 Unified/Mapped。
2. **优先简单路径**：固定 224、去重复 resize → pinned device buffer → uint8 小 H2D + GPU normalize。
3. **Thor 实验**：Managed 或 mapped 与 pinned+H2D **实测 ViT 输入与端到端 p99**，勿假设 zero-copy 更快。
4. **TRT CUDA Graph**：记住 **static_inputs 的 GPU 内 copy** 与 host 策略独立；zero-copy 省不掉这一段时，整体收益有上限。
5. **机器人在线**：长期看 **NVMM/相机 buffer** 比仅在 `Policy.infer` 换内存模型更值。

---

## 十、相关文档与代码

| 资源 | 说明 |
|------|------|
| [`pinned_memory.md`](./pinned_memory.md) | Pageable/Pinned 区别、预处理接入、Thor 注意点 |
| `third_party/openpi/src/openpi/policies/policy.py` | 当前 pageable H2D 入口 |
| `src/model_optimizer/infer/tensorrt/trt_torch.py` | TRT CUDA Graph、`static_inputs.copy_` |
| [CUDA for Tegra App Note](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/) | Thor UVM / pinned / coherence |
| [CUDA 13.0 for Jetson Thor](https://developer.nvidia.com/blog/whats-new-in-cuda-toolkit-13-0-for-jetson-thor-unified-arm-ecosystem-and-more/) | pageable 页表访问、Managed 特性 |

---

## 十一、一句话总结

- **Pinned**：锁页 + **DMA 快路径**；主用途是 **加快并异步 Memcpy**，本身不是 zero-copy。  
- **Mapped pinned / Unified**：**zero-copy** 的两种实现，省 Host Memcpy，但有 fault、带宽、同步与 TRT static buffer 等代价。  
- **硬件**：Memcpy 走 **Copy Engine**；mapped 直读走 **SM + MMU**；Thor 上两者都访问 **共享 LPDDR**，**不经 PCIe**；zero-copy **不省 DRAM 带宽**。  
- **Pi0.5**：单帧下 `non_blocking` 收益有限；**pinned 加速拷贝** 比 **追求 zero-copy** 更务实，是否再上 Unified/Mapped **以 Thor profile 为准**。
