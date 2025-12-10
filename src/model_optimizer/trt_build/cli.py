import os
from copy import deepcopy
from subprocess import Popen, PIPE, TimeoutExpired, STDOUT

from ..progress.write import write_running_log
from ..webui.extras.constants import RUNNING_LOG

import argparse

def build_onnx(model_path, output_dir):
    model_name = os.path.basename(model_path)
    engine_name = model_name.replace(".onnx", ".engine")

    cmd_list = ["trtexec", f"--onnx={model_path}", "--stronglyTyped",
                "--minShapes=images:1x3x480x640", "--maxShapes=images:1x3x1024x1024",
                f"--saveEngine={output_dir}/{engine_name}"]
    env = deepcopy(os.environ)

    cmd = ' '.join(cmd_list)
    print(f'build_onnx cmd: {cmd}')

    log_file = f'{output_dir}/{RUNNING_LOG}'
    with open(log_file, 'a+') as f:
        build_pipe = Popen(
            cmd_list, env=env,  stdout=f, stderr=STDOUT, text=True)
    
        try:
            stderr = build_pipe.communicate(timeout=100)[1]
            return_code = build_pipe.returncode
        except TimeoutExpired:
            print(f'build_onnx {model_path} Timeout')

        print(f'build_onnx {model_path} return: {return_code}')

def build_cli(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_shape', type=str, required=True)
    parser.add_argument('--extra_flag', type=str)
    parser.add_argument('--export_dir', type=str, required=True)
    args = parser.parse_args(args[1:])
    print(f'[cli] quantize args {args}')

    build_onnx(args.model_path, args.export_dir)
