import os
import argparse
import json

import torch
from torch import nn

from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark
from ultralytics.cfg import TASK2MODEL, TASK2DATA, TASK2METRIC
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--export_dir', type=str, required=True)
    parser.add_argument('--model_task', type=str, required=True)
    parser.add_argument('--jpg_path', type=str)
    args = parser.parse_args()

    model_type = args.model_path.rsplit('.')[-1]
    model = YOLO(args.model_path, task=args.model_task)
    print(f'model task {model.task}')
#    benchmark(model=args.model_path, data='coco8-seg.yaml', imgsz=640, device=0)

    onnx_path = model.export(format="onnx", dynamic=True, simplify=True, device='0')
    os.rename(onnx_path, f"{args.export_dir}")
    import sys
    sys.exit(0)

    imgsz = 640
    metric_model = model
    exported_model = YOLO(onnx_path, task=args.model_task)
    
    metric_model.predict(args.jpg_path, imgsz=imgsz, device=0, verbose=False)

    # Validate
    data = TASK2DATA[model.task]
    print(f'data: {data}')
    results = metric_model.val(
        data=data,
        batch=1,
        imgsz=imgsz,
        plots=False,
        device=0,
#        half=True,
        verbose=False,
        conf=0.001,  # all the pre-set benchmark mAP values are based on conf=0.001
    )
    print(f'results: {results.box.map}')
    metric, speed = results.results_dict[TASK2METRIC[model.task]], results.speed["inference"]
    print(f'metric: {round(metric, 4)}, speed: {round(speed, 2)}')
