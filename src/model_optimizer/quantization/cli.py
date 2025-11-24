import os
import argparse

import numpy as np

from ..progress.write import write_quantize_progress

def quantize_onnx(model_path, calibrate_data, export_dir, quant_mode, calibrate_method):
    model_name = os.path.basename(model_path)

    calib_datas = np.load(calibrate_data)
    from modelopt.onnx.quantization import quantize
    
    print(f'quantize begin {model_path} mode: {quant_mode} method: {calibrate_method} to {export_dir}.')
    write_quantize_progress(export_dir, 85, 3, 3, 85, 100)
    quantize(onnx_path=model_path,
             quantize_mode=quant_mode,       # fp8, int8, int4 etc.
             calibration_data=calib_datas, # max, entropy, awq_clip, rtn_dq etc.
             calibration_method=calibrate_method,
             output_path=f"{export_dir}/{model_name}_quant.onnx"
             )
    write_quantize_progress(export_dir, 100, 3, 3, 100, 100)
    print(f'quantize done.')

def quantize_cli(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--qformat', type=str, default="fp8")
    parser.add_argument('--calibrate_data', type=str, required=True)
    parser.add_argument('--calibrate_method', type=str, default="entropy")
    parser.add_argument('--export_dir', type=str, required=True)
    args = parser.parse_args(args[1:])
    print(f'[cli] quantize args {args}')

    quantize_onnx(args.model_path,
                  args.calibrate_data, args.export_dir, args.qformat, args.calibrate_method)
