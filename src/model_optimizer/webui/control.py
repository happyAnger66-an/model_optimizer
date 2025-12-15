import os
from typing import Any

import json
import gradio as gr

from .extras.constants import RUNNING_LOG, PROGRESS_LOG, QUANTIZE_LOG

def get_running_info(lang: str, output_path: os.PathLike) -> tuple[str, "gr.Slider", dict[str, Any]]:
    r"""Get running infomation for monitor.
    """
    running_log = ""
    running_progress = gr.Slider(visible=False)
    running_info = {}

    running_log_path = os.path.join(output_path, RUNNING_LOG)
#    print(f'running_log {running_log_path}')
    if os.path.isfile(running_log_path):
        with open(running_log_path, encoding="utf-8") as f:
            running_log = "```\n" + f.read()[-20000:] + "\n```\n"  # avoid lengthy log
            #print(f'running_log {running_log}')

    progress_log_path = os.path.join(output_path, PROGRESS_LOG)
    if os.path.isfile(progress_log_path):
        progress_log: list[dict[str, Any]] = []
        with open(progress_log_path, encoding="utf-8") as f:
            for line in f:
                progress_log.append(json.loads(line))

        if len(progress_log) != 0:
            latest_log = progress_log[-1]
            percentage = latest_log["percentage"]
            label = "Running {:d}/{:d}: {} < {}".format(
                latest_log["current_steps"],
                latest_log["total_steps"],
                latest_log["elapsed_time"],
                latest_log["remaining_time"],
            )
            running_progress = gr.Slider(label=label, value=percentage, visible=True)
    
    return running_log, running_progress, running_info


def get_quantize_info(lang: str, output_path: os.PathLike, do_quantize: bool) -> tuple[str, "gr.Slider", dict[str, Any]]:
    r"""Get training infomation for monitor.

    If do_quantize is True:
        Inputs: top.lang, train.output_path
        Outputs: train.output_box, train.progress_bar, train.loss_viewer, train.swanlab_link
    If do_quantize is False:
        Inputs: top.lang, eval.output_path
        Outputs: eval.output_box, eval.progress_bar, None, None
    """
    running_log = ""
    running_progress = gr.Slider(visible=False)
    running_info = {}

    running_log_path = os.path.join(output_path, RUNNING_LOG)
#    print(f'running_log {running_log_path}')
    if os.path.isfile(running_log_path):
        with open(running_log_path, encoding="utf-8") as f:
            running_log = "```\n" + f.read()[-20000:] + "\n```\n"  # avoid lengthy log
            #print(f'running_log {running_log}')

    quantize_log_path = os.path.join(output_path, QUANTIZE_LOG)
    if os.path.isfile(quantize_log_path):
        quantize_log: list[dict[str, Any]] = []
        with open(quantize_log_path, encoding="utf-8") as f:
            for line in f:
                quantize_log.append(json.loads(line))

        if len(quantize_log) != 0:
            latest_log = quantize_log[-1]
            percentage = latest_log["percentage"]
            label = "Running {:d}/{:d}: {} < {}".format(
                latest_log["current_steps"],
                latest_log["total_steps"],
                latest_log["elapsed_time"],
                latest_log["remaining_time"],
            )
            running_progress = gr.Slider(label=label, value=percentage, visible=True)

    return running_log, running_progress, running_info