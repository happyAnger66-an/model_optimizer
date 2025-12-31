import os
import argparse
from copy import deepcopy
from subprocess import PIPE, Popen, STDOUT, TimeoutExpired

import numpy as np

from ..progress.write import write_quantize_progress, write_running_log
from ..webui.extras.constants import RUNNING_LOG

def profile_onnx(args):
    model_path = args.model_path
    output_dir = args.output_dir
    is_e2e = args.e2e_profile if args.e2e_profile else False
    is_layer = args.layer_profile if args.layer_profile else False
    extra_args = args.extra_args if args.extra_args else None

    model_name = os.path.basename(model_path)
    engine_name = model_name.replace(".onnx", ".engine")
    
    layer_profile = model_name.replace(".onnx", "") + "_layer.profile"
    e2e_profile = model_name.replace(".onnx", "") + "_e2e.profile"

    cmd_list = ["trtexec", f"--onnx={model_path}", "--stronglyTyped",
                f"--saveEngine={output_dir}/{engine_name}"]
    if extra_args:
        cmd_list.extend(extra_args.split())

    if is_e2e:
        cmd_list.extend([f'--exportTimes={output_dir}/{e2e_profile}'])
    
    if is_layer:
        cmd_list.extend([f'--exportProfile={output_dir}/{layer_profile}'])

    print(f'profile cmd_list: {cmd_list}')

    env = deepcopy(os.environ)
    log_file = f'{output_dir}/{RUNNING_LOG}'
    with open(log_file, 'a+') as f:
        build_pipe = Popen(
            cmd_list, env=env,  stdout=f, stderr=STDOUT, text=True)
    
        try:
            stderr = build_pipe.communicate(timeout=100)[1]
            return_code = build_pipe.returncode
        except TimeoutExpired:
            print(f'profile {model_path} Timeout')

        print(f'profile {model_path} return: {return_code}')

def profile_cli(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--e2e_profile', type=bool, default=False)
    parser.add_argument('--layer_profile', type=bool, default=False)
    parser.add_argument('--extra_args', type=str)

    args = parser.parse_args(args[1:])
    print(f'[cli] profile args {args}')

    profile_onnx(args)