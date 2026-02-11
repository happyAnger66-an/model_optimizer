#! /bin/bash
set -e

export PATH=$HOME/.local/bin:$PATH

source /opt/openpi/bin/activate

uv pip uninstall torch torchvision torchcodec 