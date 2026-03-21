# 依赖说明

## 文件说明

| 文件 | 用途 |
|------|------|
| `requirements-base.txt` | 基础依赖：PyTorch、ModelOpt、ONNX、YOLO、WebUI |
| `requirements-pi05.txt` | π0.5 模型：JAX、Flax、LeRobot 等 |
| `requirements-dev.txt` | 开发：含 base + pytest |
| `requirements-docker.txt` | Docker 环境（基镜像已含 PyTorch） |
| `../requirements.txt` | 主入口，引用 base |

## 安装方式

### 1. 使用 pip + requirements

```bash
# 完整安装
pip install -r requirements.txt

# 仅基础
pip install -r requirements/requirements-base.txt
```

### 2. 使用 setup.py extras（推荐）

```bash
# 开发模式 + 完整功能
pip install -e ".[all]"

# 开发模式 + π0.5
pip install -e ".[all,pi05]"

# 开发模式 + 测试
pip install -e ".[all,dev-test]"
```

### 3. 使用安装脚本

```bash
bash scripts/install.sh all    # 完整功能
bash scripts/install.sh pi05   # 含 π0.5
bash scripts/install.sh dev    # 开发环境
```

### 4. 国内镜像

```bash
export PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
bash scripts/install.sh all
```

## π0.5 额外说明

π0.5 模型依赖 **openpi** 源码，需配置 PYTHONPATH：

```bash
export PYTHONPATH=/path/to/openpi/src:/path/to/openpi-client/src:$PYTHONPATH
```
