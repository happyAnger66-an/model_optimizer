# TensorRT-LLM：建网方式与为何不走「先 ONNX」主路径

本文档归纳 TensorRT-LLM（TRT-LLM）在 **engine 构建阶段** 的典型工作方式，及其与 **「PyTorch → ONNX → TensorRT Parser」** 通用流水线的关系与取舍原因。

---

## 1. 典型 build 路径是什么

TRT-LLM 的主流方式是：使用自带的 Python 前端（`Module` / `Tensor` / `functional` 等）**直接在代码里构造 TensorRT 网络**（例如 `add_matrix_multiply`、`add_plugin_v2` 等），再由 **TensorRT Builder** 编译并序列化为 **engine**。

也就是说：

1. **构图（define network）**：在 build 任务中执行模型 `forward`（或等价建网逻辑），向 `INetworkDefinition` 添加层与插件；
2. **Build engine**：基于已定义的网络做优化与编译，得到 `.engine` / plan。

**已生成的 engine 在推理时不会再执行上述 Python 构图逻辑**；插件与融合在 **建网阶段** 已固定进图中。

---

## 2. 是否「绕开先导出 ONNX」

**对主路径而言：是。** TRT-LLM 的常规 LLM build **不依赖**「整网先导出为 ONNX，再经 ONNX Parser 进 TRT」这一通用流程。

| 通用流程 | TRT-LLM 主流流程 |
|----------|------------------|
| 框架 → ONNX → `trtexec` / ONNX Parser → engine | Python 前端 → 直接写 TRT 网络 → Builder → engine |

个别脚本或实验路径仍可能涉及 ONNX 或子图，但 **文档与产品主打的 LLM build** 通常是 **直连 TRT 网络**。

**与 TensorRT-Edge-LLM 的对比（便于区分产品线）**：Edge-LLM 常见路径是 **ONNX 导出 + `trt::` 自定义算子 + 建 engine**；与 TRT-LLM 的默认习惯 **不必混为一谈**。

---

## 3. 为何选择「直连 TensorRT 构图」

### 3.1 LLM 算子与状态难以用「标准 ONNX」干净表达

推理侧大量 **领域专用、带状态** 的能力：KV cache、paged KV、变长 / padding、**FMHA**、**GEMM+SwiGLU** 等融合插件、多卡通信等。若强行走 ONNX，往往需要：

- 大量 **自定义域算子**（仍依赖 TRT Plugin 与 parser 支持），或  
- 图被 **拆碎**，融合边界丢失，后续再靠 pattern match 补回，**脆弱且难维护**。

直连 TRT 可在定义网络时 **显式对齐 TRT-LLM 插件契约与融合边界**。

### 3.2 动态形状与多种 build 选项需要强控制

序列长度、batch、cache 布局、量化尺度、TP/PP 等 **组合空间大**。ONNX 导出对 **动态维度、控制流、导出器版本**  historically 较敏感。TRT-LLM 用统一 Python API **按当前 build 配置生成一张 TRT 图**，对 **shape 推导、plugin 参数、量化常量** 更可控，也更易与 **Builder 优化假设** 一致。

### 3.3 减少「两段式」语义失配

**PyTorch → ONNX** 与 **ONNX → TRT** 是两次转换，易出现算子降级、精度差异、低效子图、Opset 与 Parser **不同步** 等问题。目标若本就是 TRT，**少一层中间 IR** 有利于迭代与对齐版本。

### 3.4 与产品定位一致：LLM 专用栈 + 插件生态

TRT-LLM 面向 **端到端 LLM 推理**（建网、优化、采样、服务等），不是通用「任意 PyTorch → ONNX 部署器」。直连构图便于：

- **版本化** Python 前端与 C++ 插件；  
- 通过 **`plugin_config`** 等开关组合各类融合；  
- 将 **最佳实践** 固化在代码路径中。

### 3.5 ONNX 路线何时仍有价值

- 与 **既有 ONNX 工具链** 或其它推理后端互通；  
- 仅对部分子图 **已有稳定 ONNX**，希望在 TRT 中做局部加速。

但对 **大模型 + 强定制内核 + 随 TensorRT 快速演进** 的主线产品，**直连 TRT 通常总拥有成本更低**。

---

## 4. 小结

| 问题 | 结论 |
|------|------|
| TRT-LLM 是否「先 ONNX 再 build」？ | **主路径不是**；以 **Python 直接构图 + Builder** 为主。 |
| 融合/插件何时进入图？ | **建 engine 过程中的建网阶段**，不是推理时再改图。 |
| 为何选该路线？ | **LLM 语义与融合、动态与量化、版本对齐、产品边界** 等综合考量，专用栈优先 **可控与可演进**。 |

---

*文档主题来自对 TensorRT-LLM 建网方式与 ONNX 通用流程的对比讨论；具体 API 以所用 TRT-LLM 版本文档为准。*
