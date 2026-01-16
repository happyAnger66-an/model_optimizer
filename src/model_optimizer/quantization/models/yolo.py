import argparse
import json

import torch
from torch import nn

from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from model_optimizer.calibrate.yolo_datas import YoLoCalibrationData
from model_optimizer.utils.utils import load_quant_json, normalize_quant_cfg
from model_optimizer.torch_hooks.hooks import hook_module_inputs

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq


# yolo val segment data=coco.yaml batch=1 device=0

def hook_yolo_train_method(mode):
    print(f'hook yolo train called in quantization. Do nothing...')
    pass

def quantize_model_onnx(model, quant_cfg, quant_mode, args):
    from modelopt.onnx.quantization import quantize

    model.eval()
    original_train_method = model.train
    model.train = hook_yolo_train_method
    
    from model_optimizer.collector.collector import YOLOCalibCollector
    yolo_calib_collector = YOLOCalibCollector()
    yolo_calib_collector.start_collect(SegmentationModel, model)

    calib_data = YoLoCalibrationData(args.calib_data)
    for data in calib_data:
#        print(f'data {data}')
        model(data)
    
    yolo_calib_collector.stop_collect()

    print(f'quantize begin {args.onnx_path} {quant_mode} to {args.export_dir}.')
    quantize(
        onnx_path=args.onnx_path,
        quantize_mode=quant_mode,       # fp8, int8, int4 etc.
        calibration_data=yolo_calib_collector.datas,
        calibration_method=quant_cfg["algorithm"],   # max, entropy, awq_clip, rtn_dq etc.
        output_path=f"{args.export_dir}/quant.onnx",
    )
    print(f'quantize done.')

def quantize_model_pt(model, quant_cfg, quant_mode, args):
    print(f'quant cfg {quant_cfg}')
    
    calib_data = YoLoCalibrationData(args.calib_data)

    def calibrate_loop(model):
        for idx, data in enumerate(calib_data):
            if data is None:
                print('calibrate data is None')
                continue

            print(f'calibration idx {idx} data:{data}')
            model(data)

    model.eval()
    original_train_method = model.train
    model.train = hook_yolo_train_method

    def hook_input(model, arg, kwargs):
        print(f'hook input {type(arg)} {len(arg)} {arg[0].shape} {arg[0].dtype} kwargs:{kwargs}')
    hook_module_inputs(model, hook_input, SegmentationModel)

    mtq.quantize(model, mtq.INT8_DEFAULT_CFG,
                 forward_loop=calibrate_loop)
    print(f'quantize summary')
    mtq.print_quant_summary(model)

    model.train = original_train_method
#    dummy_x = torch.randn(1, 3, 480, 640, dtype=torch.float32).cuda()
#    torch.onnx.export(
#        model,
#        (dummy_x),
#        f'{args.export_dir}/yolo.onnx',
#        input_names = ["x"],
#        output_names=['pred_x'],
#        dynamic_axes={'x': {0: 'batch_size'}},
#    )
    model.export(format="onnx", dynamic=True, simplify=True)
    print(f'export to {args.export_dir}/yolo.onnx done.')

if __name__ == "__main__":
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--onnx_path', type=str)
    parser.add_argument('--model_task', type=str, default="segment")
    parser.add_argument('--calib_data', type=str)
    parser.add_argument('--export_dir', type=str, required=True)
    parser.add_argument('--quant_cfg', type=str, required=True)
    args = parser.parse_args()

    # load a pretrain model
    model_type = args.model_path.rsplit('.')[-1]
    model = YOLO(args.model_path, task=args.model_task)
    
    quant_cfg = load_quant_json(args.quant_cfg)
    quant_mode, _ = normalize_quant_cfg(quant_cfg)
    print(f'quant_mode {quant_mode}')
#    print(f'load model {model}')
    print(f'model_type: {model_type}')
    
    print(f'begin quantize model...')
    if args.onnx_path:
        print(f'onnx model quantize')
        time.sleep(1)
        quantize_model_onnx(model, quant_cfg, quant_mode, args)
    else:
        print(f'pt model quantize')
        time.sleep(1)
        quantize_model_pt(model, quant_cfg, quant_mode, args)
    
    print(f'end quantize model...')
