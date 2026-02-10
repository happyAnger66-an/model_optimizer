# Copyright 2025 the model_optimizer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
from copy import deepcopy


USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   model_optimizer-cli quantize: quantize a model |\n"
    + "|   model_optimizer-cli export: export a model format |\n"
    + "|   model_optimizer-cli profile: profile a model |\n"
    + "|   model_optimizer-cli calibrate: calibrate a model |\n"
    + "|   model_optimizer-cli build -h: build a onnx model to engine |\n"
    + "|   model_optimizer-cli eval -h: eval model |\n"
    + "|   model_optimizer-cli webui: launch webui                        |\n"
    + "|   model_optimizer-cli download: download a model                      |\n"
    + "|   model_optimizer-cli compare: compare data                      |\n"
    + "|   model_optimizer-cli version: show version info                      |\n"
    + "| Hint: You can use `moc` as a shortcut for `model_optimizer-cli`.      |\n"
    + "-" * 70
)


def launch():

    VERSION = "0.0.1.dev"
    WELCOME = (
        "-" * 58
        + "\n"
        + f"| Welcome to Model optimizer, version {VERSION}"
        + " " * (21 - len(VERSION))
        + "|\n|"
        + " " * 56
        + "|\n"
        + "| Project page: https://github.com/happyAnger66-an/model_optimizer.git |\n"
        + "-" * 58
    )

    command = sys.argv.pop(1) if len(sys.argv) > 1 else "help"
    if command == "webui":
        from .webui.interface import run_web_ui

        run_web_ui()
    elif command == "version":
        print(WELCOME)
    elif command == "quantize":
        from .quantization.cli import quantize_cli
        quantize_cli(sys.argv)
    elif command == "profile":
        from .profile.cli import profile_cli
        profile_cli(sys.argv)
    elif command == "export":
        from .convert.convert_formt import convert_model
        convert_model(sys.argv)
    elif command == "calibrate":
        from .calibrate.cli import calibrate_cli
        calibrate_cli(sys.argv)
    elif command == "build":
        from .trt_build.cli import build_cli
        build_cli(sys.argv)
    elif command == "eval":
        from .evaluate.cli import eval_cli
        eval_cli(sys.argv)
    elif command == "datasets":
        from .datasets.cli import eval_datasets
        eval_datasets(sys.argv)
    elif command == "download":
        from .download.cli import download_cli
        download_cli(sys.argv)
    elif command == "compare":
        from .compare.cli import compare_cli
        compare_cli(sys.argv)
    else:
        print(USAGE)
