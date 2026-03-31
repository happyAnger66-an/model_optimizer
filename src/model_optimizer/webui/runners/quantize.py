import os
import time
from copy import deepcopy
from subprocess import STDOUT, Popen

import gradio as gr

from . import CommandRunner
from ..control import get_running_info
from ..extras.constants import PROGRESS_LOG, RUNNING_LOG


class QuantizeCommand(CommandRunner):
    def __init__(self, manager, data):
        super().__init__(manager, data, "quantize")
        self.quantize_cfg: str | None = None
        self.calibrate_data: str | None = None
        self.export_dir: str | None = None

    def initialize(self):
        return

    def check_inputs(self):
        self.output_box = self.get_elem_by_id("quantize.output_box")
        self.progress_bar = self.get_elem_by_id("quantize.progress_bar")
        self.quantize_cfg = self.get_data_elem_by_id("quantize.quantize_cfg")
        self.calibrate_data = self.get_data_elem_by_id("quantize.calibrate_data")
        self.export_dir = self.get_data_elem_by_id("quantize.export_dir")

        if not self.model_name:
            return self.alert("err_no_model")
        if not self.model_path:
            return self.alert("err_no_path")
        if not self.quantize_cfg:
            return "请提供量化配置文件路径（quantize_cfg）。"
        if not self.calibrate_data:
            return "请提供校准数据路径（calibrate_data）。"
        if not self.export_dir:
            return self.alert("err_no_output_dir")
        return

    def _prepare_cli(self) -> list[str]:
        return [
            "--model_name",
            str(self.model_name),
            "--model_path",
            str(self.model_path),
            "--quantize_cfg",
            str(self.quantize_cfg),
            "--calibrate_data",
            str(self.calibrate_data),
            "--export_dir",
            str(self.export_dir),
        ]

    def run(self):
        error = self.check_inputs()
        if error:
            gr.Warning(error)
            yield (error, gr.Slider(visible=False))
            return

        export_dir = str(self.export_dir)
        os.makedirs(export_dir, exist_ok=True)
        running_log_path = os.path.join(export_dir, RUNNING_LOG)
        progress_path = os.path.join(export_dir, PROGRESS_LOG)
        for p in (running_log_path, progress_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        env = deepcopy(os.environ)
        cmd_list = ["model-opt", "quantize"]
        cmd_list.extend(self._prepare_cli())
        cmd_md = "### 执行命令\n\n```bash\n" + " ".join(cmd_list) + "\n```"

        log_f = open(running_log_path, "a+", encoding="utf-8")
        try:
            log_f.write("[webui] quantize started\n")
            log_f.write("[webui] cmd: " + " ".join(cmd_list) + "\n")
            log_f.flush()
        except Exception:
            pass

        self.exec_process = Popen(
            cmd_list,
            env=env,
            stdout=log_f,
            stderr=STDOUT,
            text=True,
        )
        try:
            log_f.write(f"[webui] pid: {self.exec_process.pid}\n")
            log_f.flush()
        except Exception:
            pass

        start_ts = time.time()
        return_code: int | None = None
        while return_code is None:
            running_log, running_progress, _ = get_running_info(self.lang, export_dir)
            if not os.path.exists(progress_path):
                pct = min(95.0, 3.0 + (time.time() - start_ts) * 1.2)
                running_progress = gr.Slider(
                    label=f"量化进行中（{pct:.1f}%）",
                    value=pct,
                    minimum=0,
                    maximum=100,
                    visible=True,
                    interactive=False,
                )
            yield (cmd_md + "\n\n" + (running_log or ""), running_progress)
            time.sleep(10)
            return_code = self.exec_process.poll()

        ok = (return_code == 0)
        if ok:
            gr.Info("量化已完成。")
        else:
            gr.Warning(f"量化失败（exit code={return_code}），请查看日志。")
        running_log, running_progress, _ = get_running_info(self.lang, export_dir)
        yield (cmd_md + "\n\n" + (running_log or ""), running_progress)
        try:
            log_f.close()
        except Exception:
            pass

