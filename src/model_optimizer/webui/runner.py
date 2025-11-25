import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional
from subprocess import PIPE, Popen, TimeoutExpired
from collections.abc import Generator

import shutil
import json

from .commom import abort_process, save_cmd, get_save_dir, load_eval_results
from .locales import ALERTS
from .extras.misc import is_accelerator_available, torch_gc
from .control import get_quantize_info

import gradio as gr

if TYPE_CHECKING:
    from gradio.components import Component

    from .manager import Manager


class Runner:
    r"""A class to manage the running status of the trainers."""

    def __init__(self, manager: "Manager", demo_mode: bool = False) -> None:
        r"""Init a runner."""
        self.manager = manager
        self.demo_mode = demo_mode
        """ Resume """
        self.quantizer: Optional[Popen] = None
        self.do_quantize = True
        self.running_data: dict[Component, Any] = None
        """ State """
        self.aborted = False
        self.running = False

    def set_abort(self) -> None:
        self.aborted = True
        if self.quantizer is not None:
            abort_process(self.quantizer.pid)

    def run_eval(self, data):
        yield from self._launch(data, do_quantize=False)

    def _parse_eval_args(self, data: dict["Component", Any]) -> dict[str, Any]:
        r"""Build and validate the evaluation arguments."""
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        model_name, model_path = get("top.model_name"), get("top.model_path")

        args = dict(
            stage="sft",
            model_path=get("top.model_path"),
            preprocessing_num_workers=16,
            dataset_dir=get("eval.dataset_dir"),
            output_dir=get_save_dir(model_name, model_path, get("eval.output_dir")),
        )

        if get("eval.predict"):
            args["do_predict"] = True
        else:
            args["do_eval"] = True

        return args

    def _initialize(self, data: dict["Component", Any], do_quantize: bool, from_preview: bool) -> str:
        r"""Validate the configuration."""
        def get(elem_id): return data[self.manager.get_elem_by_id(elem_id)]
        lang, model_name, model_path = get("top.lang"), get(
            "top.model_name"), get("top.model_path")
        dataset_dir = get("quantize.dataset_dir") if do_quantize else get("eval.dataset_dir")

        if self.running:
            return ALERTS["err_conflict"][lang]

        if not model_name:
            return ALERTS["err_no_model"][lang]

        if not model_path:
            return ALERTS["err_no_path"][lang]

        if not dataset_dir:
            return ALERTS["err_no_dataset"][lang]

        if do_quantize:
            if not get("quantize.output_dir"):
                return ALERTS["err_no_output_dir"][lang]
        else:
            if not get("eval.output_dir"):
                return ALERTS["err_no_output_dir"][lang]

        if not is_accelerator_available():
            gr.Warning(ALERTS["warn_no_cuda"][lang])

        return ""

    def _finalize(self, lang: str, finish_info: str) -> None:
        r"""Clean the cached memory and resets the runner."""
        finish_info = ALERTS["info_aborted"][lang] if self.aborted else finish_info
        gr.Info(finish_info)
        self.quantizer = None
        self.aborted = False
        self.running = False
        self.running_data = None
        torch_gc()

    def _parse_quantize_args(self, data: dict["Component", Any]) -> dict[str, Any]:
        r"""Build and validate the training arguments."""
        def get(elem_id): return data[self.manager.get_elem_by_id(elem_id)]
        model_name, model_path = get("top.model_name"), get("top.model_path")
#        user_config = load_config()

        args = dict(
            do_quantizer=True,
            model_name=get("top.model_name"),
            model_path=get("top.model_path"),
            dataset_dir=get("quantize.dataset_dir"),
#            dataset=",".join(get("quantize.dataset")),
            output_dir=get("quantize.output_dir"),
        )

        # quantization
        if get("quantize.quantization_bit") != "none":
            args["quantize_bit"] = get("quantize.quantization_bit")
            args["calibrate_method"] = get("quantize.calibrate_method")

        print(f' quantize args: {args}')
        return args
    
    def _prepare_eval_cli(self, args):
        cli_args = []
        model_name = args["model_path"]

        model_name = model_name.replace('.onnx', '.engine')
        cli_args.extend(["--model_path", model_name])
        cli_args.extend(["--dataset_dir", f'{args["dataset_dir"]}'])
        cli_args.extend(["--output_dir", f'{args["output_dir"]}'])

        return cli_args

    def _prepare_build_cli(self, args):
        cli_args = []
        model_name = args["model_path"]

        cli_args.extend(["--model_path", model_name])
        cli_args.extend(["--export_dir", f'{args["output_dir"]}'])

        return cli_args

    def _prepare_quantize_cli(self, args):
        cli_args = []
        model_name = args["model_name"]

        onnx_name = model_name.replace('.pt', '.onnx')
        cli_args.extend(["--model_path", f'{os.path.join(args["output_dir"], onnx_name)}'])
        cli_args.extend(["--qformat", f'{args["quantize_bit"]}'])
        cli_args.extend(["--export_dir", f'{args["output_dir"]}'])

        calibrate_data = os.path.join(args['output_dir'], 'calibrate.npy')
        cli_args.extend(["--calibrate_data", f'{calibrate_data}'])
        cli_args.extend(["--calibrate_method", f'{args["calibrate_method"]}'])

        return cli_args
    
    def _prepare_calibrate_cli(self, args):
        cli_args = []
        model_name = args["model_name"]
        model_path = args["model_path"]

        cli_args.extend(["--model_name", f'{model_name}'])
        cli_args.extend(["--model_path", f'{model_path}'])
        cli_args.extend(["--dataset_dir", f'{args["dataset_dir"]}'])
        cli_args.extend(["--export_dir", f'{args["output_dir"]}'])

        return cli_args

    def _prepare_convert_cli(self, args):
        cli_args = []
        model_name = args["model_name"]
        model_path = args["model_path"]

        cli_args.extend(["--model_name", f'{model_name}'])
        cli_args.extend(["--model_path", f'{model_path}'])
        cli_args.extend(["--export_type", f'onnx'])
        cli_args.extend(["--export_dir", f'{args["output_dir"]}'])

        return cli_args

    def _launch(self, data: dict["Component", Any], do_quantize: bool) -> Generator[dict["Component", Any], None, None]:
        r"""Start the training process."""
        output_box = self.manager.get_elem_by_id(
            "{}.output_box".format("quantize" if do_quantize else "eval"))
        error = self._initialize(data, do_quantize, from_preview=False)
        if error:
            gr.Warning(error)
            yield {output_box: error}
        else:
            self.do_quantize, self.running_data = do_quantize, data
            args = self._parse_quantize_args(data) if do_quantize else self._parse_eval_args(data)

            if args["output_dir"] and len(args["output_dir"]) > 2:
                if os.path.exists(args["output_dir"]) and do_quantize:
                    shutil.rmtree(args["output_dir"])
            os.makedirs(args["output_dir"], exist_ok=True)

            env = deepcopy(os.environ)

            if do_quantize:
                # NOTE: DO NOT USE shell=True to avoid security risk
                cmd_list = ["model-optimizer-cli", "convert"]
                cmd_list.extend(self._prepare_convert_cli(args))
                print(f'quantize [convert] cmd_list {cmd_list}')
                self.quantizer = Popen(
                    cmd_list, env=env, stderr=PIPE, text=True)
                yield from self.monitor(finalize=False)
                
                cmd_list = ["model-optimizer-cli", "calibrate"]
                cmd_list.extend(self._prepare_calibrate_cli(args))
                print(f'quantize [calibrate] cmd_list {cmd_list}')
                self.quantizer = Popen(
                    cmd_list, env=env, stderr=PIPE, text=True)
                yield from self.monitor(finalize=False)
                
                cmd_list = ["model-optimizer-cli", "quantize"]
                cmd_list.extend(self._prepare_quantize_cli(args))
                print(f'quantize [quantize] cmd_list {cmd_list}')
                self.quantizer = Popen(
                    cmd_list, env=env, stderr=PIPE, text=True)
                yield from self.monitor()
            else:
                model_name = args["model_path"]
                if model_name.endswith('.onnx'):
                    cmd_list = ["model-optimizer-cli", "build"]
                    cmd_list.extend(self._prepare_build_cli(args))
                    print(f'eval [build] cmd_list {cmd_list}')
                    self.quantizer = Popen(
                        cmd_list, env=env, stderr=PIPE, text=True)
                    yield from self.monitor(finalize=False)
                
                cmd_list = ["model-optimizer-cli", "eval"]
                cmd_list.extend(self._prepare_eval_cli(args))
                print(f'eval [eval] cmd_list {cmd_list}')
                self.quantizer = Popen(
                    cmd_list, env=env, stderr=PIPE, text=True)
                yield from self.monitor()


    def run_quantize(self, data):
        yield from self._launch(data, do_quantize=True)

    def monitor(self, finalize=True):
        r"""Monitorgit the training progress and logs."""
        self.aborted = False
        self.running = True

        def get(
            elem_id): return self.running_data[self.manager.get_elem_by_id(elem_id)]
        lang, model_name, model_path = get("top.lang"), get(
            "top.model_name"), get("top.model_path")
        output_dir = get("{}.output_dir".format("quantize" if self.do_quantize else "eval"))
        output_path = output_dir
#        output_path = get_save_dir(model_name, model_path, output_dir)

        output_box = self.manager.get_elem_by_id(
            "{}.output_box".format("quantize" if self.do_quantize else "eval"))
        progress_bar = self.manager.get_elem_by_id(
            "{}.progress_bar".format("quantize" if self.do_quantize else "eval"))

        running_log = ""
        return_code = -1
        while return_code == -1:
            if self.aborted:
                yield {
                    output_box: ALERTS["info_aborting"][lang],
                    progress_bar: gr.Slider(visible=False),
                }
            else:
                running_log, running_progress, running_info = get_quantize_info(
                    lang, output_path, self.do_quantize)
                return_dict = {
                    output_box: running_log,
                    progress_bar: running_progress,
                }

                yield return_dict

            try:
                stderr = self.quantizer.communicate(timeout=2)[1]
                return_code = self.quantizer.returncode
            except TimeoutExpired:
                continue

        if return_code == 0 or self.aborted:
            finish_info = ALERTS["info_finished"][lang]

            running_log, running_progress, running_info = get_quantize_info(
                    lang, output_path, self.do_quantize)
            return_dict = {
                    output_box: running_log,
                    progress_bar: running_progress,
                }

            yield return_dict

            if self.do_quantize:
                finish_log = ALERTS["info_finished"][lang] + \
                    "\n\n" + running_log
            else:
                finish_log = load_eval_results(os.path.join(output_path, "all_results.json")) + "\n\n" + running_log
        else:
            print(stderr)
            finish_info = ALERTS["err_failed"][lang]
            finish_log = ALERTS["err_failed"][lang] + \
                f" Exit code: {return_code}\n\n```\n{stderr}\n```\n"

        if finalize:
            self._finalize(lang, finish_info)
        return_dict = {output_box: finish_log,
                       progress_bar: gr.Slider(visible=False)}
        yield return_dict
