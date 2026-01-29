import sys
from ultralytics import YOLO

from onnx2pytorch import ConvertModel
import onnx2torch
import onnx
if __name__ == "__main__":
    onnx_model = onnx.load(sys.argv[1])
    pytorch_model = ConvertModel(onnx_model)
    for m in pytorch_model.named_modules():
        print(f'm0: {m[0]} m1: {m[1]}')

    model = YOLO(sys.argv[2], task='segment')
    for m in model.named_modules():
        print(f'm0: {m[0]} m1: {m[1]}')