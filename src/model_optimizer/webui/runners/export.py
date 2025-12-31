import os
from copy import deepcopy
from subprocess import Popen, PIPE
from collections import defaultdict

import pandas as pd
import json

import gradio as gr

from . import CommandRunner
from ..control import get_running_info

class ExportCommand(CommandRunner):
    def __init__(self, manager, data):
        super().__init__(manager, data, "export")
        self.export_format = "onnx"

    def initialize(self):
        self.export_format = self.get_data_elem_by_id('export.export_format')
        self.simplifier = self.get_data_elem_by_id('export.simplifier')
        self.output_path = self.get_data_elem_by_id('export.output_dir')
        self.output_box = self.get_elem_by_id('export.output_box')
        self.progress_bar = self.get_elem_by_id('export.progress_bar')

    def check_inputs(self):
        if self.export_format != "onnx":
            return self.alert('err_export_format')

        return

    def _prepare_cli(self):
        cli_args = []
        model_path = self.model_path

        cli_args.extend(["--model_name", f'{self.model_name}'])
        cli_args.extend(["--model_path", f'{model_path}'])
        cli_args.extend(["--export_dir", f'{self.output_path}'])
        cli_args.extend(["--export_type", f'{self.export_format}'])
        cli_args.extend(["--simplifier", f'{self.simplifier}'])

        return cli_args

    def monitor_phase(self, phase, return_dict):
        pass
    
    def run(self):
        error = self.check_inputs()        
        if error:
            gr.Warning(error)
            yield {self.output_box: error}

        env = deepcopy(os.environ)

        cmd_list = ["model-optimizer-cli", "convert"]
        cmd_list.extend(self._prepare_cli())
        print(f'[export] cmd_list {cmd_list}')

        os.makedirs(self.output_path, exist_ok=True)
        self.exec_process = Popen(
            cmd_list, env=env, stderr=PIPE, text=True)
        yield from self.monitor(finalize=False)