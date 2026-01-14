import os
import argparse

import numpy as np

from ..progress.write import write_quantize_progress, write_running_log
from ..webui.extras.constants import RUNNING_LOG
from .llm_ptq import llm_quantize

def quantize_onnx(model_path, calibrate_data, export_dir, quant_mode, calibrate_method):
    model_name = os.path.basename(model_path)
    export_name = model_name.replace('.', '_')
    calib_datas = np.load(calibrate_data)
    
    content = f'quantize begin {model_path} mode: {quant_mode} method: {calibrate_method} to {export_dir}.'
    write_running_log(export_dir, content)
    write_quantize_progress(export_dir, 85, 3, 3, 85, 100)
    
    from modelopt.onnx.quantization import quantize
    from modelopt.onnx.logging_config import configure_logging
    configure_logging(log_file=f'{export_dir}/{RUNNING_LOG}')
    quant_outfile = f"{export_dir}/{export_name}_quant_{quant_mode}_{calibrate_method}.onnx"
    quantize(onnx_path=model_path,
             quantize_mode=quant_mode,       # fp8, int8, int4 etc.
             calibration_data=calib_datas, # max, entropy, awq_clip, rtn_dq etc.
             calibration_method=calibrate_method,
             output_path=quant_outfile
             )
    
    write_quantize_progress(export_dir, 100, 3, 3, 100, 100)
    content = f'quantize done to: {quant_outfile}'
    write_running_log(export_dir, content)

def quantize_cli(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
#    parser.add_argument('--config_name', type=str, default="pi05_libero")
    parser.add_argument('--model_name', type=str, default="pi05_libero")
    parser.add_argument('--model_type', type=str, default="hf")
    parser.add_argument('--quantize_cfg', type=str, required=True)
    parser.add_argument('--calibrate_data', type=str, required=True)
    parser.add_argument('--calibrate_method', type=str, default="max")
    parser.add_argument('--export_dir', type=str, required=True)
    args = parser.parse_args(args[1:])
    print(f'[cli] quantize args {args}')

    model_name =  args.model_name
    if model_name.startswith('pi05'):
        from ..models.pi05.model_pi05 import Pi05Model
        pi05_model = Pi05Model(model_name, args.model_path)
        pi05_model.load()

    if args.model_type == "llm":
        args.dataset = args.calibrate_data
        llm_quantize(args)
    elif args.model_path.find('pi05_libero/vit'):
        from .pi05_ptq import quantize_pi05_vit
        quantize_pi05_vit(args)
    else:
        quantize_onnx(args.model_path,
                  args.calibrate_data, args.export_dir, args.qformat, args.calibrate_method)
