# torch.onnx.export（dynamo=True）导出流程解读

本文基于本仓库工作区中 `@/home/wujie/sources/opensrc/pytorch` 的 PyTorch 源码（`torch/onnx` 与 `torch/onnx/_internal/exporter`）梳理 `torch.onnx.export` 的实际执行路径，重点解释：

- 为什么会看到 “Obtain model graph … strict=False/strict=True …” 这类日志；
- `dynamic_axes` 与 `dynamic_shapes` 的关系、何时会触发约束冲突；
- 出错时 fallback 到 legacy exporter 的条件与限制；
- 如何把典型报错定位到导出流程的具体阶段。

---

## 总览：`torch.onnx.export` 的两条主路径

入口函数在：

```59:360:/home/wujie/sources/opensrc/pytorch/torch/onnx/__init__.py
def export(..., dynamo: bool = True, dynamic_shapes=None, dynamic_axes=None, fallback: bool = False, ...):
    if dynamo is True or isinstance(model, torch.export.ExportedProgram):
        return _compat.export_compat(...)
    else:
        from ._internal.torchscript_exporter.utils import export
        if dynamic_shapes:
            raise ValueError("... dynamic_axes when dynamo=False.")
        export(...)
        return None
```

- **dynamo=True（默认）**：走新导出器（基于 `torch.export.ExportedProgram`），返回 `torch.onnx.ONNXProgram`。
- **dynamo=False**：走 legacy TorchScript 导出器，不返回 `ONNXProgram`（返回 `None`），动态维仅支持 `dynamic_axes`。

> 你日志中出现 `[torch.onnx] Obtain model graph ... torch.export.export`，说明走的是 **dynamo=True**。

---

## dynamo=True 的兼容层：`_compat.export_compat`

`torch.onnx.export(..., dynamo=True)` 会调用：

```42:245:/home/wujie/sources/opensrc/pytorch/torch/onnx/_internal/exporter/_compat.py
def export_compat(..., dynamic_axes=None, dynamic_shapes=None, fallback=False, legacy_export_kwargs=None):
    ...
    if dynamic_shapes is None and dynamic_axes is not None:
        warnings.warn("# 'dynamic_axes' is not recommended when dynamo=True ...")
        dynamic_shapes, args, kwargs = _dynamic_shapes.from_dynamic_axes_to_dynamic_shapes(...)
    dynamic_shapes_with_export_dim, need_axis_mapping = _dynamic_shapes.convert_str_to_export_dim(dynamic_shapes)
    onnx_program = _core.export(..., dynamic_shapes=dynamic_shapes_with_export_dim, ...)
    ...
    if f is not None:
        onnx_program.save(..., external_data=external_data, ...)
    return onnx_program
```

### 关键点 1：为什么会出现 “dynamic_axes 不推荐” 的 warning

当 `dynamo=True` 且你仍传了 `dynamic_axes` 时，`export_compat` 会尝试把它**转换**为 `dynamic_shapes`，并提示这是不推荐的兼容行为（可能引发约束冲突）。

### 关键点 2：`dynamic_axes -> dynamic_shapes` 的转换机制

转换实现位于：

```20:108:/home/wujie/sources/opensrc/pytorch/torch/onnx/_internal/exporter/_dynamic_shapes.py
def from_dynamic_axes_to_dynamic_shapes(..., dynamic_axes, input_names, output_names):
    ...
    for input_name, axes in dynamic_axes.items():
        if input_name in output_names: continue
        if isinstance(axes, dict):
            dynamic_shapes[input_name] = axes  # 轴->字符串名，后续再转 Dim.DYNAMIC
        elif isinstance(axes, list):
            dynamic_shapes[input_name] = dict.fromkeys(axes, torch.export.Dim.DYNAMIC)
        ...
    ...
    # 关键：为了适配 torch.export.export，按 model signature 重新排序 args/kwargs
    inputs = []
    for idx, param_name in enumerate(sig.parameters):
        if idx < len(args): inputs.append(args[idx])
        elif param_name in kwargs: inputs.append(kwargs[param_name])
    dynamic_shapes = _unflatten_dynamic_shapes_with_inputs_tree(inputs, dynamic_shapes)
    return dynamic_shapes, tuple(inputs), {}
```

要点：

- `dynamic_axes` 的 dict 形式（`{axis: "name"}`）会先保留字符串，后面再统一转换。
- 会把传入的 `args/kwargs` **重排成与 `forward` 参数顺序一致**，并返回 `kwargs={}`。这也是为什么导出时建议尽量用**位置参数 + 明确的 input_names**，减少混乱。

### 关键点 3：字符串动态轴名如何变成 `torch.export.Dim`

```190:232:/home/wujie/sources/opensrc/pytorch/torch/onnx/_internal/exporter/_dynamic_shapes.py
def convert_str_to_export_dim(dynamic_shapes):
    if dynamic_shapes is None or not _any_str_or_dim_in_dynamic_shapes(dynamic_shapes):
        return dynamic_shapes, False
    ...
    if isinstance(dim, str):
        converted_axes_dict[axis] = torch.export.Dim.DYNAMIC
```

也就是说，**`dynamic_axes` 里给的轴名称字符串不会保留下来**；它只用于“标记为动态”，最终变成 `Dim.DYNAMIC`。导出完成后如果需要“重命名动态轴”，会走 `need_axis_mapping` 分支做二次重命名（`onnx_program._rename_dynamic_axes(...)`）。

---

## 核心导出器：`_core.export` 的 3 步（Step 1/2/3）

`export_compat` 最终调用 `_core.export`：

```1279:1476:/home/wujie/sources/opensrc/pytorch/torch/onnx/_internal/exporter/_core.py
def export(...):
    # Step 1: torch.export.export 捕获 ExportedProgram（多策略）
    # Step 2: decomposition + type promotion 等 ONNX 兼容处理
    # Step 3: translate -> ONNX IR -> ONNXProgram
```

### Step 1：捕获 ExportedProgram（你看到的 strict=False/strict=True 日志就在这里）

捕获策略定义在 `_capture_strategies.py`：

```298:305:/home/wujie/sources/opensrc/pytorch/torch/onnx/_internal/exporter/_capture_strategies.py
CAPTURE_STRATEGIES = (
    TorchExportNonStrictStrategy,
    TorchExportStrictStrategy,
)
if _flags.ENABLE_DRAFT_EXPORT:
    CAPTURE_STRATEGIES = (*CAPTURE_STRATEGIES, TorchExportDraftExportStrategy)
```

#### Non-strict 策略（优先）

```210:243:/home/wujie/sources/opensrc/pytorch/torch/onnx/_internal/exporter/_capture_strategies.py
class TorchExportNonStrictStrategy(CaptureStrategy):
    def _capture(...):
        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            try:
                return torch.export.export(..., dynamic_shapes=dynamic_shapes, strict=False, ...)
            except torch._dynamo.exc.UserError as exc:
                new_shapes = torch.export.dynamic_shapes.refine_dynamic_shapes_from_suggested_fixes(exc.msg, dynamic_shapes)
                return torch.export.export(..., dynamic_shapes=new_shapes, strict=False, ...)
```

解释：

- `strict=False` 是默认首选。
- 如果触发 `torch._dynamo.exc.UserError`，会尝试根据 “suggested fixes” 自动细化 dynamic_shapes，再重试一次。

#### Strict 策略（兜底）

```154:188:/home/wujie/sources/opensrc/pytorch/torch/onnx/_internal/exporter/_capture_strategies.py
class TorchExportStrictStrategy(CaptureStrategy):
    def _capture(...):
        with (_patch_dynamo_unsupported_functions(), torch.fx.experimental._config.patch(backed_size_oblivious=True)):
            try:
                return torch.export.export(..., strict=True, ...)
            except torch._dynamo.exc.UserError as exc:
                new_shapes = refine_dynamic_shapes_from_suggested_fixes(...)
                return torch.export.export(..., dynamic_shapes=new_shapes, strict=True, ...)
```

解释：

- strict 模式会打补丁绕过一部分 `torch.export.export` 暂不支持的函数（例如临时替换 `torch.jit.isinstance`）。

#### Draft export（可选）

```265:276:/home/wujie/sources/opensrc/pytorch/torch/onnx/_internal/exporter/_capture_strategies.py
class TorchExportDraftExportStrategy(CaptureStrategy):
    def _capture(...):
        ep = torch.export.draft_export(...)
        report = ep._report
        if not report.successful():
            self._exception = RuntimeError(str(report))  # informational
        return ep
```

是否启用由 `_flags.ENABLE_DRAFT_EXPORT` 控制。

---

## 常见报错定位：为什么会出现 “Dim 冲突：推断静态 256 vs 用户指定动态”

这类报错发生在 **Step 1**（`torch.export.export` 内部的约束推导阶段），典型信息为：

- “用户指定 dynamic dim”
- “trace 推断某维静态（例如 head_dim=256）”
- “conflicts between user-specified ranges and inferred ranges …”

**本质原因**：你把一个天然静态的维度标成了动态（或通过 `dynamic_axes -> dynamic_shapes` 的转换误标到了错误的轴）。

一个常见坑是 KV cache 的形状维度顺序不一致：

- 若张量布局是 `[num_layers, batch, seq_len, head_dim]`，那么 `seq_len` 是 **dim=2**；
- `head_dim`（比如 256）是 **dim=3**，应当保持静态。

当把 `dim=3` 标为动态，就会触发 “推断静态 256 vs 用户指定动态”的冲突。

---

## fallback：新导出失败后如何回退到 legacy

`export_compat` 在捕获/转换失败时，会在 `fallback=True` 的情况下回退：

```170:213:/home/wujie/sources/opensrc/pytorch/torch/onnx/_internal/exporter/_compat.py
except Exception as e:
    if fallback:
        if f is None: raise TypeError("f must be provided when fallback is enabled")
        if dynamic_shapes is not None and dynamic_axes is None:
            if input_names is None: raise ValueError("... input_names or dynamic_axes must be provided ...")
            dynamic_axes = from_dynamic_shapes_to_dynamic_axes(...)
        torch.onnx.utils.export(..., dynamic_axes=dynamic_axes, **legacy_export_kwargs)
        onnx_program = ONNXProgram(ir.load(f), None)
        return onnx_program
    else:
        raise
```

要点：

- **fallback 必须提供 `f`**（需要写文件才能让 legacy exporter 输出 ONNX）。
- 若你只给了 `dynamic_shapes`，fallback 会尝试反向构造 `dynamic_axes`，但这需要 `input_names` 或你直接给 `dynamic_axes`。
- fallback 走的是 `torch.onnx.utils.export`（legacy 路径），其动态形状能力与行为与 dynamo 路径不同。

---

## 实战建议（结合本仓库模型导出）

- **能不用 `dynamic_axes` 就不用（dynamo=True）**：直接提供 `dynamic_shapes`（`torch.export.Dim`）更可控。
- 如果必须用 `dynamic_axes`（历史代码或兼容），要确保：
  - 输入张量的维度语义完全正确（例如 KV 的 `seq_len` 轴到底在 dim2 还是 dim3）。
  - 对天然静态维（如 head_dim）不要标动态。
- 若你在某些模型上 dynamo exporter 不稳定，短期可以：
  - 选择 `dynamo=False`（legacy exporter），并只使用 `dynamic_axes`；
  - 或 `dynamo=True, fallback=True`，但要提供 `f` 且准备好 `input_names/dynamic_axes`。

