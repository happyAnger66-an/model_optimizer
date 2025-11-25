# Copyright 2025 the modelfactory team.
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
    + "|   modelfactory-cli api -h: launch an OpenAI-style API server       |\n"
    + "|   modelfactory-cli chat -h: launch a chat interface in CLI         |\n"
    + "|   modelfactory-cli export -h: merge LoRA adapters and export model |\n"
    + "|   modelfactory-cli train -h: train models                          |\n"
    + "|   modelfactory-cli webchat -h: launch a chat interface in Web UI   |\n"
    + "|   modelfactory-cli webui: launch LlamaBoard                        |\n"
    + "|   modelfactory-cli env: show environment info                      |\n"
    + "|   modelfactory-cli version: show version info                      |\n"
    + "| Hint: You can use `lmf` as a shortcut for `modelfactory-cli`.      |\n"
    + "-" * 70
)


def launch():

    VERSION = "0.0.1.dev"
    WELCOME = (
        "-" * 58
        + "\n"
        + f"| Welcome to Model Factory, version {VERSION}"
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
    elif command == "convert":
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
    else:
        print(USAGE)
