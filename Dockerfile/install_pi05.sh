#!/usr/bin/env bash
# π0.5 模型依赖安装 (Docker 内使用 uv 管理独立 venv)
# 依赖: 已 COPY requirements/ 到工作目录

set -e

# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

mkdir -p /opt/openpi
chmod -R 777 /opt/openpi

export PATH=$HOME/.local/bin:$PATH

uv venv /opt/openpi
source /opt/openpi/bin/activate

# 优先使用 requirements 目录，兼容旧路径
#PI05_REQ="requirements/requirements-pi05.txt"
[[ -f "requirements_pi05.txt" ]] && PI05_REQ="requirements_pi05.txt"
[[ -f "$PI05_REQ" ]] || { echo "Error: $PI05_REQ not found"; exit 1; }

uv pip install -r "$PI05_REQ" -i https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install nvidia-modelopt[all]==0.39.0 -i https://pypi.tuna.tsinghua.edu.cn/simple