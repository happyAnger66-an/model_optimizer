import os
import argparse

from typing import Optional, Any

import shutil

#from ultralytics import YOLO
#from ultralytics.nn.tasks import SegmentationModel

from ..progress.write import write_quantize_progress

def pt2onnx(model_path, export_dir):
    print(f'pt2onnx {model_path} to {export_dir}')
#    model = YOLO(model_path, task='segment')
    model = None

    write_quantize_progress(export_dir, 10, 1, 3, 10, 100) 
    
    model_name = os.path.basename(model_path) 
    onnx_path = model.export(format="onnx", dynamic=True, simplify=True, device='0')

    if not os.path.exists(f'{export_dir}'):
        os.makedirs(f'{export_dir}')

    print(f'pt2onnx export {model_path} to {export_dir}/{model_name}.onnx')
    shutil.copy(onnx_path, f"{export_dir}")
    write_quantize_progress(export_dir, 20, 1, 3, 20, 100) 

model_convert_methods = {
    'pt2onnx': pt2onnx
}

def convert_model(args: Optional[dict[str, Any]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--export_type', type=str, default="onnx")
    parser.add_argument('--export_dir', type=str, required=True)
    print(f'[cli] convert_model args {args[1:]}')
    args = parser.parse_args(args[1:])

    model_name = args.model_name
    if model_name.startswith('pi05'):
        from .pi0 import convert_pi05_model
        convert_pi05_model(args.model_name, args.model_path, args.export_dir)
        return

    name, model_type = os.path.splitext(args.model_name)
    convert_func = model_convert_methods[f'{model_type[1:]}2{args.export_type}']

    convert_func(os.path.join(args.model_path, args.model_name), args.export_dir)

