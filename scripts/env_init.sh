#!/bin/bash

if [ $# -lt 1 ]; then
    echo "usage $0: <your-openpi-code-path>"
    exit 1
fi

codepath=$1
mkdir -p ~/.cache/openpi/big_vision

cp /srcs/.cache/openpi/big_vision/*  ~/.cache/openpi/big_vision/
cp $codepath/src/openpi/models_pytorch/transformers_replace/* \
    /opt/openpi/lib/python3.12/site-packages/transformers/

export PATH=/opt/trt-edge-llm/bin:$PATH
export PYTHONPATH=/opt/trt-edge-llm/lib/python3.12/site-packages/:/usr/local/lib/python3.12/dist-packages/:$PYTHONPATH