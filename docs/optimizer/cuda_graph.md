# TensorRT Engine CUDA Graph 分析与落地方案

本文针对 `src/model_optimizer/infer/tensorrt/trt_torch.py` 中 `Engine` 类，分析是否可通过 CUDA Graph 加速，并给出最小可用实现方案。

## 结论

- 可以在 `Engine` 层利用 CUDA Graph 加速。
- 该方案对“同一引擎、重复调用、输入 shape 基本固定”的场景收益较明显（例如 pi05 连续采样推理）。
- 对频繁变化 shape 的场景，仍可运行，但会退化为 eager 路径或命中较低的图缓存。

## 现状瓶颈

当前 `Engine.forward` 每次调用都做了以下工作：

- 逐输入设置地址和必要的 cast/contiguous。
- 根据运行时 shape 动态创建输出 tensor。
- 调用 `execute_async_v3` 后立即 `stream.synchronize()`。

对高频推理而言，Python 调度与 launch 开销不可忽略。CUDA Graph 的价值是把重复执行路径图化并重放，减少这部分开销。

## 适用条件

要稳定使用 CUDA Graph，需要：

- 一次图对应一组固定输入元信息（shape、dtype、device）。
- 图捕获期间执行路径稳定。
- 重放时复用同一批静态 input/output buffer 地址。

因此，建议按输入签名做图缓存，而不是全局只保留一个图。

## 最小可用实现设计

### 1) 增加开关

在 `Engine.__init__` 增加：

- `use_cuda_graph: bool = False`
- `cuda_graph_warmup: int = 1`

默认关闭，保证兼容旧行为。

### 2) 输入签名作为缓存 key

以每个输入的 `(name, shape, dtype, device)` 生成 key，缓存：

- `torch.cuda.CUDAGraph`
- 静态输入 buffer
- 静态输出 buffer
- 专用 graph stream

### 3) 图捕获流程（首次命中 key）

1. 根据当前输入确定 TRT 运行时 shape（处理动态维）。
2. 分配静态输入/输出 buffer，并绑定 tensor address。
3. warmup `execute_async_v3` 若干次。
4. 使用 `torch.cuda.CUDAGraph` 捕获一次执行。

### 4) 图重放流程（后续命中 key）

1. 将用户输入拷贝到静态输入 buffer。
2. `graph.replay()`。
3. 返回静态输出（为避免上层持有导致后续覆盖，按当前实现返回 clone）。

### 5) 回退策略

若输入不完整、环境不满足、或调用方关闭开关，则走原 eager 路径。

## 风险与边界

- 显存增加：每个 shape key 保存一套静态 buffer 与 graph。
- 动态 shape 太多：缓存命中下降，收益减弱。
- 并发场景：单 `execution_context` 下建议串行使用，或为并发 worker 分配独立 context/engine 实例。

## 验证建议

1. 功能一致性：
   - 同输入下 eager 与 graph 输出误差对比（`allclose`）。
2. 性能评估：
   - 预热后对比平均时延与 p95。
   - 区分固定 shape 与多 shape 两种负载。
3. 稳定性：
   - 长时间循环推理检查显存与缓存条目数。

