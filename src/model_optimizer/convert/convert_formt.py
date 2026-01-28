import os
import argparse

import torch

from copy import deepcopy
from typing import Optional, Any

import shutil

from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel

from subprocess import Popen, PIPE, TimeoutExpired, STDOUT

from ..progress.write import write_running_log
from ..webui.extras.constants import RUNNING_LOG

from ..progress.write import write_quantize_progress


def simplifier_model(model_path, output_dir):
    model_name = os.path.basename(model_path)
    simplifier_model_name = model_name.replace('.onnx', '_simplifier.onnx')

    cmd_list = ["onnxsim", f'{model_path}',
                f'{output_dir}/{simplifier_model_name}']
    log_file = f'{output_dir}/{RUNNING_LOG}'

    print(f'simplifier_model cmd {cmd_list}')
    env = deepcopy(os.environ)
    with open(log_file, 'a+') as f:
        build_pipe = Popen(
            cmd_list, env=env,  stdout=f, stderr=STDOUT, text=True)
        try:
            stderr = build_pipe.communicate(timeout=100)[1]
            return_code = build_pipe.returncode
        except TimeoutExpired:
            print(f'simplifier_model {model_path} Timeout')


def pt2onnx(model_path, export_dir, simplifier=True):
    print(f'pt2onnx {model_path} to {export_dir}')
    model = YOLO(model_path, task='segment')

    write_quantize_progress(export_dir, 10, 1, 3, 10, 100)

    model_name = os.path.basename(model_path)
    onnx_path = model.export(
        format="onnx", dynamic=False, simplify=True, device='0')

    if not os.path.exists(f'{export_dir}'):
        os.makedirs(f'{export_dir}')

    export_model_name = os.path.basename(onnx_path)
    export_model_path = f'{export_dir}/{export_model_name}'

    print(f'copy {onnx_path} to {export_dir}')
    shutil.copy(onnx_path, f"{export_dir}")
    write_quantize_progress(export_dir, 20, 1, 3, 20, 100)
    return export_model_path


def qwen3_vl2onnx(model_path, export_dir):
    print(f'qwen3_vl2onnx {model_path} to {export_dir}')
    from transformers import Qwen3VLForConditionalGeneration
    print('loading qwen3_vl model ...')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype="auto", device_map="auto").to(torch.float16)
    print(f'loading qwen3_vl model done {model}')

    from ..models.qwen_vl import QwenVLWrapper
    llm_model = model.model.language_model
    wrapper_model = QwenVLWrapper(llm_model.config, llm_model)
    del model.model.language_model
    print(f'wrapper_model {wrapper_model}')

    from .qwen3_vl import convert_qwen3_vl
    model, export_model_path = convert_qwen3_vl(wrapper_model, export_dir)
    return export_model_path


model_convert_methods = {
    'pt2onnx': pt2onnx,
    'Qwen3-VL': qwen3_vl2onnx,
}


def convert_model(args: Optional[dict[str, Any]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--export_type', type=str, default="onnx")
    parser.add_argument('--export_dir', type=str, required=True)
    parser.add_argument('--simplifier', type=bool, default=True)
    parser.add_argument('--verify_data', type=str, default=None)
    print(f'[cli] convert_model args {args[1:]}')
    args = parser.parse_args(args[1:])

    model_name = args.model_name
    model_path = args.model_path

    from ..models.registry import get_model_cls
    model_cls = get_model_cls(model_name)
    model = model_cls.construct_from_name_path(model_name, model_path)
    export_model_path = model.export(args.export_dir)

    if args.verify_data:
        export_model = model_cls.construct_from_name_path(
            model_name, export_model_path)
        export_model.val(args.verify_data, batch_size=1,
                         output_dir=args.export_dir)


'''
    if model_name.startswith('pi05'):
        from .pi05 import convert_pi05_model
        if '/' in model_name:
            model_name, model_type = model_name.split('/')
        print(f'convert pi05 {model_type}')
        convert_pi05_model(args, model_name, model_type)
    else:
        convert_func = model_convert_methods[f'{model_name}']
        _, export_model_path = convert_func(args.model_path, args.export_dir)
    return
    if args.simplifier:
        simplifier_model(export_model_path, args.export_dir)
    return

    name, model_type = os.path.splitext(args.model_name)
    convert_func = model_convert_methods[f'{model_type[1:]}2{args.export_type}']

    convert_func(args.model_path, args.export_dir)
'''
