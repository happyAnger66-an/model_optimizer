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
             # max, entropy, awq_clip, rtn_dq etc.
             calibration_data=calib_datas,
             calibration_method=calibrate_method,
             output_path=quant_outfile
             )

    write_quantize_progress(export_dir, 100, 3, 3, 100, 100)
    content = f'quantize done to: {quant_outfile}'
    write_running_log(export_dir, content)


def get_quant_cfg(quant_cfg_file):
    from model_optimizer.config.config import load_settings
    settings = load_settings(quant_cfg_file)
    return settings.QUANT_CFG

def quantize_cli(args):
    parser = argparse.ArgumentParser(
        description='模型量化工具：支持多种模型格式的量化操作',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例用法:\n'
               '  quantize_cli --model_path model.onnx --quantize_cfg config.json \\\n'
               '               --calibrate_data data.npz --export_dir ./output'
    )
    parser.add_argument('--model_name', type=str, default="pi05_libero",
                       help='模型名称，用于注册表中查找对应的模型类 (默认: pi05_libero)')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径 (必需)')
#    parser.add_argument('--config_name', type=str, default="pi05_libero")
    parser.add_argument('--model_type', type=str, default="hf",
                       help='模型类型，例如: hf, llm 等 (默认: hf)')
    parser.add_argument('--quantize_cfg', type=str, required=True,
                       help='量化配置文件路径，包含量化参数配置 (必需)')
    parser.add_argument('--calibrate_data', type=str, required=True,
                       help='校准数据文件路径，用于量化校准的输入数据 (必需)')
    parser.add_argument('--calibrate_method', type=str, default="max",
                       help='校准方法，可选值: max, entropy, awq_clip, rtn_dq 等 (默认: max)')
    parser.add_argument('--export_dir', type=str, required=True,
                       help='导出目录，量化后的模型将保存到此目录 (必需)')
    args = parser.parse_args(args[1:])
    print(f'[cli] quantize args {args}')

    model_name = args.model_name
    model_path = args.model_path
    
    from ..models.registry import get_model_cls
    model_cls = get_model_cls(model_name)
    model = model_cls.construct_from_name_path(model_name, model_path)

    quant_cfg = get_quant_cfg(args.quantize_cfg)
    print(f'quant_cfg: {quant_cfg}')
    model.quantize(quant_cfg, args.calibrate_data,
                   args.calibrate_method)
    model.export(args.export_dir)


'''
    if args.model_type == "llm":
        args.dataset = args.calibrate_data
        llm_quantize(args)
    elif args.model_path.find('pi05_libero/vit'):
        from .pi05_ptq import quantize_pi05_vit
        quantize_pi05_vit(args)
    else:
        quantize_onnx(args.model_path,
                  args.calibrate_data, args.export_dir, args.qformat, args.calibrate_method)
                  '''
