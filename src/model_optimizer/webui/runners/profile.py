import os
from copy import deepcopy
from subprocess import Popen, PIPE
from collections import defaultdict

import pandas as pd
import json

import gradio as gr

from . import CommandRunner
from ..control import get_running_info

class ProfileCommand(CommandRunner):
    def __init__(self, manager, data):
        super().__init__(manager, data, "profile")
        self.e2e_profile, self.layer_profile, self.outbox = None, None, None
        self.layer_prof = None
        self.e2e_prof = None

    def initialize(self):
        pass

    def check_inputs(self):
        e2e_profile = self.get_data_elem_by_id('profile.e2e_profile')
        layer_profile = self.get_data_elem_by_id('profile.layer_profile')
        output_dir = self.get_data_elem_by_id('profile.output_dir')
        if e2e_profile and layer_profile:
            return self.alert('err_profile_conflict')

        self.e2e_profile = e2e_profile
        self.layer_profile = layer_profile 
        self.output_box = self.get_elem_by_id('profile.output_box')
        self.progress_bar = self.get_elem_by_id('profile.progress_bar')
        self.e2e_prof = self.get_elem_by_id('profile.e2e_prof')
        self.layer_prof = self.get_elem_by_id('profile.layer_prof')
        self.output_path = output_dir
        print(f'output_path {self.output_path}')
        return

    def _prepare_cli(self):
        cli_args = []
        model_path = self.model_path

        cli_args.extend(["--model_path", f'{model_path}'])
        cli_args.extend(["--output_dir", f'{self.output_path}'])
        if self.e2e_profile:
            cli_args.extend(["--e2e_profile", f'{self.e2e_profile}'])

        if self.layer_profile:
            cli_args.extend(["--layer_profile", f'{self.layer_profile}'])

        return cli_args

    def get_layer_profile(self):
        model_name = os.path.basename(self.model_path)
        layer_profile = model_name.replace(".onnx", "") + "_layer.profile"
        layer_profile_path = f'{self.output_path}/{layer_profile}'
        print(f'parse layer_profile {layer_profile_path}')

        datas = defaultdict(list)
        with open(layer_profile_path, 'r') as f:
            layer_datas = json.load(f)

        for data in layer_datas:
            for k, v in data.items():
                if k == 'name':
                    datas["name"].append(v)
                elif k == 'averageMs':
                    datas["meanMs"].append(v)
                elif k == 'medianMs':
                    datas["p50Ms"].append(v)
                elif k == 'percentage':
                    datas["%"].append(v)

#        print(f'layer_profile: {datas}')
        return datas

    def monitor_phase(self, phase, return_dict):
        if phase == "finish":
            layer_infos = self.get_layer_profile()
            return_dict[self.layer_prof] = pd.DataFrame(dict(layer_infos))
#            print(f'return dict {return_dict}')
    
    def run(self):
        error = self.check_inputs()        
        if error:
            gr.Warning(error)
            yield {self.output_box: error}

        env = deepcopy(os.environ)

        cmd_list = ["model-optimizer-cli", "profile"]
        cmd_list.extend(self._prepare_cli())
        print(f'[profile] cmd_list {cmd_list}')

        os.makedirs(self.output_path, exist_ok=True)
        self.exec_process = Popen(
            cmd_list, env=env, stderr=PIPE, text=True)
        yield from self.monitor(finalize=False)