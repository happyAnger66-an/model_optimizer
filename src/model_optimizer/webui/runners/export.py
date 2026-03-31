import os
import time
from copy import deepcopy
from subprocess import STDOUT, Popen

import gradio as gr

from . import CommandRunner
from ..control import get_running_info
from ..extras.constants import RUNNING_LOG

class ExportCommand(CommandRunner):
    def __init__(self, manager, data):
        super().__init__(manager, data, "export")
        self.export_format = "onnx"
        self.steps_box = None
        self.cmd_box = None
        self._start_ts: float | None = None
        self._cmd_list: list[str] | None = None

    def initialize(self):
        self.export_format = self.get_data_elem_by_id('export.export_format')
        self.simplifier = self.get_data_elem_by_id('export.simplifier')
        self.output_path = self.get_data_elem_by_id('export.output_dir')
        self.output_box = self.get_elem_by_id('export.output_box')
        self.steps_box = self.get_elem_by_id('export.steps_box')
        self.cmd_box = self.get_elem_by_id('export.cmd_box')
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
        # 若 export CLI 已写入 progress.jsonl，则进度条由 control.get_running_info 的读取结果提供，
        # 这里不要用“伪进度”覆盖真实进度；仅补充步骤与命令展示。
        try:
            from ..extras.constants import PROGRESS_LOG
            has_real_progress = (
                self.output_path is not None
                and os.path.isfile(os.path.join(self.output_path, PROGRESS_LOG))
            )
        except Exception:
            has_real_progress = False

        if has_real_progress:
            if phase == "running":
                return_dict[self.steps_box] = "\n".join(
                    [
                        "### 导出步骤",
                        "- **1) 参数校验**：已完成",
                        "- **2) 启动导出进程**：已完成",
                        "- **3) 导出中**：进行中（可在下方查看日志）",
                    ]
                )
            elif phase in ("return", "finish"):
                ok = (self.exec_process is not None and self.exec_process.returncode == 0) or getattr(self, "aborted", False)
                return_dict[self.steps_box] = "\n".join(
                    [
                        "### 导出步骤",
                        "- **1) 参数校验**：已完成",
                        "- **2) 启动导出进程**：已完成",
                        f"- **3) 导出中**：{'已完成' if ok else '失败/中断'}",
                    ]
                )
            return

        # fallback：没有 progress.jsonl 时，用“分步骤 + 伪进度”的方式提升体验
        if self._start_ts is None:
            self._start_ts = time.time()
        elapsed_s = max(0.0, time.time() - self._start_ts)

        if phase == "running":
            # 0~90%：随时间缓慢增长，避免长任务看起来“卡住”
            pct = min(90.0, 5.0 + elapsed_s * 2.0)
            steps_md = "\n".join(
                [
                    "### 导出步骤",
                    "- **1) 参数校验**：已完成",
                    "- **2) 启动导出进程**：已完成",
                    "- **3) 导出中**：进行中（可在下方查看日志）",
                ]
            )
            return_dict[self.steps_box] = steps_md
            return_dict[self.progress_bar] = gr.Slider(
                label=f"导出进行中（{pct:.1f}%）",
                value=pct,
                minimum=0,
                maximum=100,
                visible=True,
                interactive=False,
            )
        elif phase in ("return", "finish"):
            # return：拿到最后一轮日志；finish：收尾隐藏进度条。这里把步骤补齐。
            ok = (self.exec_process is not None and self.exec_process.returncode == 0) or getattr(self, "aborted", False)
            steps_md = "\n".join(
                [
                    "### 导出步骤",
                    "- **1) 参数校验**：已完成",
                    "- **2) 启动导出进程**：已完成",
                    f"- **3) 导出中**：{'已完成' if ok else '失败/中断'}",
                ]
            )
            return_dict[self.steps_box] = steps_md
            if phase == "return":
                return_dict[self.progress_bar] = gr.Slider(
                    label="导出完成" if ok else "导出结束（失败/中断）",
                    value=100.0 if ok else 0.0,
                    minimum=0,
                    maximum=100,
                    visible=True,
                    interactive=False,
                )
    
    def run(self):
        error = self.check_inputs()        
        if error:
            gr.Warning(error)
            # 注意：export 前端期望 4 个输出（steps/cmd/log/progress）
            yield (
                "### 导出步骤\n- **参数校验**：失败",
                "",
                error,
                gr.Slider(visible=False),
            )
            return

        env = deepcopy(os.environ)

        cmd_list = ["model-optimizer-cli", "export"]
        cmd_list.extend(self._prepare_cli())
        self._cmd_list = cmd_list
        self._start_ts = time.time()
        print(f'[export] cmd_list {cmd_list}')

        os.makedirs(self.output_path, exist_ok=True)
        cmd_md = "### 执行命令\n\n```bash\n" + " ".join(cmd_list) + "\n```"
        # 关键：将子进程 stdout/stderr 统一写入 running_log.txt，保证 WebUI 可持续读取日志更新
        running_log_path = os.path.join(self.output_path, RUNNING_LOG)
        log_f = open(running_log_path, "a+", encoding="utf-8")
        self.exec_process = Popen(
            cmd_list,
            env=env,
            stdout=log_f,
            stderr=STDOUT,
            text=True,
        )
        steps_md = "### 导出步骤\n- **1) 参数校验**：已完成\n- **2) 启动导出进程**：已完成\n- **3) 导出中**：进行中（可在下方查看日志）"

        # Gradio 在不同版本下对“按组件返回 dict”的支持差异较大，这里固定返回 4 元组，
        # 与 export 页 outputs=[steps_box, cmd_box, output_box, progress_bar] 严格对齐。
        running_log = ""
        running_progress: gr.Slider = gr.Slider(visible=False)
        return_code = -1
        while return_code == -1:
            running_log, running_progress, _ = get_running_info(self.lang, self.output_path)
            yield (steps_md, cmd_md, running_log, running_progress)
            # stdout/stderr 已重定向到文件，这里只需要轮询进程状态
            time.sleep(10)
            return_code = self.exec_process.poll()

        ok = (return_code == 0)
        if ok:
            steps_end = "### 导出步骤\n- **1) 参数校验**：已完成\n- **2) 启动导出进程**：已完成\n- **3) 导出中**：已完成"
        else:
            steps_end = "### 导出步骤\n- **1) 参数校验**：已完成\n- **2) 启动导出进程**：已完成\n- **3) 导出中**：失败"

        # 最后再刷新一轮日志与进度（若 progress.jsonl 存在，会显示 100%）
        running_log, running_progress, _ = get_running_info(self.lang, self.output_path)
        yield (steps_end, cmd_md, running_log, running_progress)
        try:
            log_f.close()
        except Exception:
            pass