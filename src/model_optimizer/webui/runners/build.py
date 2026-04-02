import os
import time
from copy import deepcopy
from subprocess import STDOUT, Popen

import gradio as gr

from . import CommandRunner
from ..control import get_running_info
from ..extras.constants import PROGRESS_LOG, RUNNING_LOG
from ...config.config import load_settings
from ...trt_build.build import validate_precision_matches_onnx


class BuildCommand(CommandRunner):
    def __init__(self, manager, data):
        super().__init__(manager, data, "compile")
        self.build_cfg: str | None = None

    def initialize(self):
        # 组件在 check_inputs 里读取（便于校验失败时也能返回正确 output）
        return

    def check_inputs(self):
        self.output_box = self.get_elem_by_id("compile.output_box")
        self.progress_bar = self.get_elem_by_id("compile.progress_bar")
        # compile.export_dir 在 UI 语义上是“导出文件路径”（不是目录）
        self.output_path = self.get_data_elem_by_id("compile.export_dir")
        self.build_cfg = self.get_data_elem_by_id("compile.build_cfg")

        if not self.model_path:
            return self.alert("err_no_path")
        if not str(self.model_path).strip().lower().endswith(".onnx"):
            return "模型路径必须是一个 `.onnx` 文件。"
        if not self.output_path:
            return self.alert("err_no_output_dir")
        if not str(self.output_path).strip().lower().endswith(".engine"):
            return "导出文件必须是一个 `.engine` 文件名（以 `.engine` 结尾）。"
        if not self.build_cfg:
            return "请提供编译配置文件路径（build_cfg）。"
        return

    def _prepare_cli(self) -> list[str]:
        return [
            "--model_path",
            str(self.model_path),
            "--build_cfg",
            str(self.build_cfg),
            "--export_dir",
            str(self.output_path),
        ]

    def run(self):
        error = self.check_inputs()
        if error:
            gr.Warning(error)
            yield (error, gr.Slider(visible=False))
            return

        # Web 侧预检查：精度与 ONNX dtype 不一致时直接提示，避免用户等很久才发现输出塌缩
        try:
            cfg_mod = load_settings(str(self.build_cfg))
            precision = getattr(cfg_mod, "build_cfg", {}).get("precision", "bf16")
            validate_precision_matches_onnx(str(self.model_path), precision)
        except Exception as e:
            gr.Warning(str(e))
            yield (str(e), gr.Slider(visible=False))
            return

        export_file_path = str(self.output_path)
        # 只创建父目录：export_dir 是文件路径时，不能当目录创建
        log_dir = os.path.dirname(export_file_path) or "."
        os.makedirs(log_dir, exist_ok=True)
        running_log_path = os.path.join(log_dir, RUNNING_LOG)
        progress_path = os.path.join(log_dir, PROGRESS_LOG)
        for p in (running_log_path, progress_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        env = deepcopy(os.environ)
        cmd_list = ["model-opt", "build"]
        cmd_list.extend(self._prepare_cli())
        cmd_md = "### 执行命令\n\n```bash\n" + " ".join(cmd_list) + "\n```"

        log_f = open(running_log_path, "a+", encoding="utf-8")
        try:
            log_f.write("[webui] build started\n")
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
            running_log, running_progress, _ = get_running_info(self.lang, log_dir)
            # 若没有 progress.jsonl，用轻量“时间进度”避免看起来卡住
            if not os.path.exists(progress_path):
                pct = min(95.0, 3.0 + (time.time() - start_ts) * 1.5)
                running_progress = gr.Slider(
                    label=f"编译进行中（{pct:.1f}%）",
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
            gr.Info("编译已完成。")
        else:
            gr.Warning(f"编译失败（exit code={return_code}），请查看日志。")
        running_log, running_progress, _ = get_running_info(self.lang, log_dir)
        yield (cmd_md + "\n\n" + (running_log or ""), running_progress)
        try:
            log_f.close()
        except Exception:
            pass

