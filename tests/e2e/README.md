# E2E 测试

模型优化器端到端 (E2E) 测试框架。

## 目录结构

```
tests/
├── conftest.py          # 根级 pytest 配置
├── e2e/
│   ├── conftest.py      # E2E 专用 fixtures
│   ├── __init__.py
│   ├── README.md
│   ├── test_cli_e2e.py      # CLI 入口测试
│   ├── test_registry_e2e.py # 模型注册表测试
│   └── test_quantize_e2e.py # 量化流程测试（需真实模型）
└── ...
```

## 运行方式

### 安装依赖

```bash
pip install -e ".[dev-test]"
# 或
pip install -e .
pip install pytest
```

### 运行全部 E2E 测试

```bash
# 项目根目录下
pytest tests/e2e -v
```

### 仅运行 E2E 标记的测试

```bash
pytest tests/e2e -m e2e -v
```

### 排除需 GPU 的测试

```bash
pytest tests/e2e -m "e2e and not e2e_gpu" -v
```

### 运行慢速测试

```bash
pytest tests/e2e -m e2e_slow -v
```

## 标记说明

| 标记 | 说明 |
|------|------|
| `e2e` | 端到端测试 |
| `e2e_slow` | 耗时较长的 E2E 测试 |
| `e2e_gpu` | 需要 GPU 的 E2E 测试 |

## CLI 测试说明

- 使用已安装的 `model-optimizer-cli` 或 `model-opt` 时，请先执行 `pip install -e .`
- 未安装时，会回退到 `python -m model_optimizer.cli`，需保证 `src` 在 PYTHONPATH 中

## 可选依赖与跳过

部分测试依赖可选包，缺失时会自动跳过：

| 测试 | 依赖 | 说明 |
|------|------|------|
| test_quantize_help | modelopt | 量化 CLI |
| test_export_help | ultralytics | 导出 CLI |
| TestModelRegistry | jax | 模型注册表（含 pi05） |

## 量化等流程测试

`test_quantize_e2e.py` 中需要真实模型与校准数据的测试默认 `@pytest.mark.skip`。
在本地有模型时，可去掉 skip 并修改路径后运行。
