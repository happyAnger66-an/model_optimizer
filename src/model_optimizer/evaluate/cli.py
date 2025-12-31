import os
import argparse
from copy import deepcopy

from subprocess import Popen, PIPE, TimeoutExpired, STDOUT

from ..progress.write import write_running_log
from ..webui.extras.constants import RUNNING_LOG

def eval_yolo(model_path, dataset_dir, batch_size, output_dir):
    cmd_list = ["yolo", 'val', 'segment', f"model={model_path}", f"batch={batch_size}",
                f"data={dataset_dir}", 'device=0']
    env = deepcopy(os.environ)

    cmd = ' '.join(cmd_list)
    print(f'eval cmd: {cmd}')

    log_file = f'{output_dir}/{RUNNING_LOG}'
    with open(log_file, 'a+') as f:
        build_pipe = Popen(
            cmd_list, env=env,  stdout=f, stderr=STDOUT, text=True)
        try:
            stderr = build_pipe.communicate(timeout=100)[1]
            return_code = build_pipe.returncode
        except TimeoutExpired:
            print(f'eval {model_path} Timeout')

        print(f'eval {model_path} return: {return_code}')

def eval_cli(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args(args[1:])
    print(f'[cli] eval args {args}')

    eval_yolo(args.model_path, args.dataset_dir, args.batch_size, args.output_dir)
