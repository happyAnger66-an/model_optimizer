import os
from subprocess import TimeoutExpired
import json
from collections import defaultdict

import pandas as pd

import gradio as gr
from ..extras.misc import is_accelerator_available
from ..locales import ALERTS
from ..control import get_running_info


class CommandRunner:
    def __init__(self, manager, data, cmd_name):
        self.name = cmd_name
        self.manager = manager
        self.data = data
        self.lang, self.model_name, self.model_path = None, None, None
        self.running = False
        self.output_box = None
        self.progress_bar = None
        self.exec_process = None
        self.output_path = None
        self._initialize(self.data)

    def get_elem_by_id(self, elem_id):
        return self.manager.get_elem_by_id(elem_id)
#        print(f'elem_id {elem_id}')

    def get_data_elem_by_id(self, elem_id):
        #        print(f'elem_id {elem_id}')
        return self.data[self.manager.get_elem_by_id(elem_id)]

    def initialize(self):
        raise NotImplementedError("initialize not implement")

    def alert(self, name):
        return ALERTS[name][self.lang]

    def _initialize(self, data):
        def get(elem_id): return data[self.manager.get_elem_by_id(elem_id)]
        lang, self.model_name, self.model_path = get("top.lang"), get(
            "top.model_name"), get("top.model_path")

        self.lang = lang
        if self.running:
            return ALERTS["err_conflict"][self.lang]

        if not self.model_name:
            return ALERTS["err_no_model"][lang]

        if not self.model_path:
            return ALERTS["err_no_path"][lang]

        if not is_accelerator_available():
            gr.Warning(ALERTS["warn_no_cuda"][lang])

        self.initialize()


    def monitor_phase(self, phase, return_dict):
        pass
    
    def monitor(self, finalize=True):
        print('monitor')
        r"""Monitorgit the training progress and logs."""
        self.aborted = False
        self.running = True

        running_log = ""
        return_code = -1
        while return_code == -1:
            if self.aborted:
                yield {
                    self.output_box: ALERTS["info_aborting"][self.lang],
                    self.progress_bar: gr.Slider(visible=False),
                }
            else:
                running_log, running_progress, running_info = get_running_info(
                    self.lang, self.output_path)
                return_dict = {
                    self.output_box: running_log,
                    self.progress_bar: running_progress,
                }

                self.monitor_phase('running', return_dict)
                yield return_dict

            try:
                stderr = self.exec_process.communicate(timeout=10)[1]
                return_code = self.exec_process.returncode
            except TimeoutExpired:
                continue

        if return_code == 0 or self.aborted:
            finish_info = ALERTS["info_finished"][self.lang]

            running_log, running_progress, running_info = get_running_info(
                self.lang, self.output_path)
            return_dict = {
                self.output_box: running_log,
                self.progress_bar: running_progress,
            }

            self.monitor_phase('return', return_dict)
            yield return_dict

            finish_log = ALERTS["info_finished"][self.lang] + \
                    "\n\n" + running_log
        else:
            finish_info = ALERTS["err_failed"][self.lang]
            finish_log = ALERTS["err_failed"][self.lang] + \
                f" Exit code: {return_code}\n\n```\n{stderr}\n```\n"

        if finalize:
            self._finalize(self.lang, finish_info)
        return_dict = {self.output_box: finish_log,
                       self.progress_bar: gr.Slider(visible=False)}
        self.monitor_phase('finish', return_dict)
        yield return_dict
