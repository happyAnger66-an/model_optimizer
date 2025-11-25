import os
import json

from ..webui.extras.constants import QUANTIZE_LOG, RUNNING_LOG

def write_running_log(log_dir, content):
    file_path = os.path.join(log_dir, RUNNING_LOG)
    with open(file_path, 'a+') as f:
        f.write(content + '\n')

def write_quantize_progress(log_dir,
                   percentage, current_steps, total_steps, 
                   elapsed_time, remaining_time):
    content = {
        "percentage": percentage,
        "current_steps": current_steps,
        "total_steps": total_steps,
        "elapsed_time": elapsed_time,
        "remaining_time": remaining_time
    }

    file_path = os.path.join(log_dir, QUANTIZE_LOG)
    with open(file_path, 'a+') as f:
        f.write(json.dumps(content) + '\n')
