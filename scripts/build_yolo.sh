#! /bin/bash
trtexec --onnx=$1 \
    --minShapes=images:1x3x480x640 \
    --maxShapes=images:1x3x1024x1024 \
    --saveEngine=./$2.engine \
    --useCudaGraph \
    --stronglyTyped