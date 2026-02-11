#! /bin/bash

set -e

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

mkdir -p /opt/openpi
chmod -R 777 /opt/openpi

export PATH=$HOME/.local/bin:$PATH

uv venv /opt/openpi
source /opt/openpi/bin/activate

uv pip install -r requirements_pi05.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
uv pip uninstall torch torchvision torchcodec -y 