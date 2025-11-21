#! /bin/bash
trtexec --onnx=$1 \
    --minShapes=images:1x3x480x640 \
    --maxShapes=images:3x3x480x640 \
    --saveEngine=./yolo_tube.engine \
    --useCudaGraph \
    --stronglyTyped