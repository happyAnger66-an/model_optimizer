# 废代码分析

基于对代码库的扫描，以下为可考虑删除或清理的废代码。

---

## 1. 可直接删除

### 1.1 `src/parent.py`
- **说明**: 独立的 Parent/Child 示例类，与 model_optimizer 无关
- **依据**: 未被任何 model_optimizer 模块导入
- **建议**: 删除（疑似遗留示例文件）

### 1.2 `quantization/cli.py` 中整块注释 (105-116 行)
```python
#    model.export(args.export_dir)

'''
    if args.model_type == "llm":
        args.dataset = args.calibrate_data
        llm_quantize(args)
    ...
'''
```
- **说明**: 旧量化分支，已被 registry 模式替代
- **影响**: `llm_quantize`、`quantize_onnx` 仅在注释中出现，可考虑移除对应导入
- **建议**: 删注释块；若删除 `llm_quantize` 导入，需确认 webui 是否仍需要

### 1.3 `convert/convert_formt.py` 中整块注释 (115-134 行)
```python
'''
    if model_name.startswith('pi05'):
        ...
    else:
        convert_func = model_convert_methods[...]
    ...
'''
```
- **说明**: 旧 convert 逻辑，当前使用 registry + `model.export()`
- **影响**: `model_convert_methods`、`pt2onnx`、`qwen3_vl2onnx`、`simplifier_model` 仅在此注释或未使用路径中被引用
- **建议**: 删注释块；`model_convert_methods` 等可评估是否仍保留作备用

---

## 2. 未使用/冗余变量

### 2.1 `graph = None`
- **位置**: `models/model.py` L59、`models/pi05/llm.py` L65
- **说明**: 赋值为 `None` 后未使用
- **建议**: 删除该行

### 2.2 `modelopt.torch.opt as mto`
- **位置**: model.py、yolo/__init__.py、quantization/llm_ptq.py、quantization/models/yolo.py、quantization/models/sam.py
- **说明**: 多处 `import mto` 未使用
- **建议**: 删除未使用的 `import modelopt.torch.opt as mto`

---

## 3. 冗余/过时代码块

### 3.1 LLM `_nvfp4_post_processing` 调用参数错误（需修复）
- **位置**: `models/pi05/llm.py` L140
- **问题**: 调用 `self._nvfp4_post_processing(export_dir)` 仅传 1 个参数，但方法签名是 `(self, onnx_path, export_dir)`
- **后果**: 在 NVFP4 量化时会触发 `TypeError`
- **建议**: 改为 `self._nvfp4_post_processing(f"{export_dir}/llm.onnx", export_dir)`

### 3.2 `models/model.py` 中 `quantize`、`get_model_calibrate_loop`
- **说明**: 基类提供默认实现，但 pi05 LLM/Expert/Vit 均覆盖 quantize
- **说明**: YOLO 使用自定义 `quantize` 流程
- **建议**: 保留基类作为兜底，用于未来可能的新模型；若确定不再扩展，可清理

### 3.3 Expert 中大量注释 (66-86 行)
- **说明**: `norm_dense_forward`、`input_layernorm_dense_forward` 等整段注释的 hook 代码
- **建议**: 确认无历史用途后可删除

### 3.4 `quantization/models/sam.py` 中注释块
- 多处注释的 SAM 校准、ONNX 导出逻辑
- **建议**: 若 SAM 功能不再维护，可整体移除或归档

---

## 4. 可能断掉的依赖

### 4.1 `quantization/pi05_ptq.py` → `from ..models.pi0 import Pi05Vit`
- **说明**: `models.pi0` 模块未找到，`Pi05Vit` 可能应为 `pi05.vit.Vit`
- **说明**: 仅在已注释的 `quantize_cli` 分支中调用 `quantize_pi05_vit`
- **建议**: 
  1. 若保留 ViT 量化：修正导入为 `from ..models.pi05.vit import Vit as Pi05Vit`，并补全构造参数
  2. 若废弃 ViT 专用量化：删除 `pi05_ptq.py` 及 cli 中对应注释

### 4.2 `quantization/cli.py` 中 `from .llm_ptq import llm_quantize`
- **说明**: `llm_quantize` 仅在被注释的代码中使用
- **建议**: 若确认不再需要该路径，删除该导入

---

## 5. 可简化的小片段

| 文件 | 内容 | 建议 |
|------|------|------|
| quantization/cli.py L57 | `# parser.add_argument('--config_name'...)` | 删除 |
| models/model.py L101 | `# quantize_model(...)` | 删除 |
| utils/log.py | 若干行内注释 | 可保留作为文档 |
| webui/*.py | 多处 `# print(...)` | 视需要删减 |

---

## 6. 删除/修改建议优先级

| 优先级 | 项 | 风险 |
|--------|-----|------|
| 高 | 删除 `src/parent.py` | 低 |
| 高 | 删除 `graph = None` | 低 |
| 中 | 清理 quantization/cli.py、convert_formt.py 注释块 | 低 |
| 中 | 移除未使用的 `mto` 导入 | 低 |
| 中 | 修正 LLM `_nvfp4_post_processing` 调用参数 | 需回归测试 |
| 低 | 处理 pi05_ptq 与 models.pi0 的导入 | 需确认 ViT 量化需求 |

---

## 7. 建议执行顺序

1. 删除 `src/parent.py`
2. 删除所有 `graph = None`
3. 移除未使用的 `import modelopt.torch.opt as mto`
4. 删除 quantization/cli.py、convert_formt.py 中的注释代码块
5. 检查并修正 LLM 的 `_nvfp4_post_processing` 调用
6. 根据需求决定是否恢复或移除 pi05 ViT 量化相关代码
