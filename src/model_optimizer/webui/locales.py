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

LOCALES = {
    "title": {
        "en": {
            "value": "<h1><center>Model Optimizer: Unified Model Optimization and Deployment Toolkit</center></h1>",
        },
        "zh": {
            "value": "<h1><center>Model Optimizer: 模型高效优化平台</center></h1>",
        },
    },
    "subtitle": {
        "en": {
            "value": (
                "<h3><center>Visit <a href='https://github.com/happyAnger66-an/model_optimizer' target='_blank'>"
                "GitHub Page</a></center></h3>"
            ),
        },
        "zh": {
            "value": (
                "<h3><center>访问 <a href='https://github.com/happyAnger66-an/model_optimizer' target='_blank'>"
                "GitHub 主页</a></center></h3>"
            ),
        },
    },
    "lang": {
        "en": {
            "label": "Language",
        },
        "zh": {
            "label": "语言",
        },
    },
    "model_name": {
        "en": {
            "label": "Model name",
            "info": "Input the initial name to search for the model.",
        },
        "zh": {
            "label": "模型名称",
            "info": "输入首单词以检索模型。",
        },
    },
    "model_path": {
        "en": {
            "label": "Model path",
            "info": "Path to pretrained model or model identifier from Hugging Face.",
        },
        "zh": {
            "label": "模型路径",
            "info": "本地模型的文件路径或 Hugging Face 的模型标识符。",
        },
    },
    "quantization_bit": {
        "en": {
            "label": "Quantization bit",
            "info": "Quantization bit width.",
        },
        "zh": {
            "label": "量化等级",
            "info": "量化bit（FP only）。",
        },
    },
    "simplifier": {
        "en": {
            "label": "simplifier model",
            "info": "simplifier model (use onnx_simplifier).",
        },
        "zh": {
            "label": "优化模型",
            "info": "优化模型(使用onnx_simplifier).",
        },
    },
    "export_format": {
        "en": {
            "label": "export_format",
            "info": "export_format (onnx).",
        },
        "zh": {
            "label": "导出格式",
            "info": "导出格式(只支持onnx).",
        },
    },
    "quantization_method": {
        "en": {
            "label": "Quantization method",
            "info": "Quantization algorithm to use.",
        },
        "zh": {
            "label": "量化方法",
            "info": "使用的校准算法。",
        },
    },
    "calibrate_method": {
        "en": {
            "label": "Calibrate method",
            "info": "Calibrate algorithm to use.",
        },
        "zh": {
            "label": "校准方法",
            "info": "使用的校准算法。",
        },
    },
    "quantize_cfg": {
        "en": {
            "label": "Quantize config path",
            "info": "Path to quantization configuration file.",
        },
        "zh": {
            "label": "量化配置文件路径",
            "info": "量化配置文件路径（quantize_cfg）。",
        },
    },
    "calibrate_data": {
        "en": {
            "label": "Calibrate data path",
            "info": "Path to calibration data for quantization.",
        },
        "zh": {
            "label": "校准数据路径",
            "info": "校准数据路径（calibrate_data）。",
        },
    },
    "progress_bar": {
        "en": {
            "label": "Progress bar",
            "info": "Progress bar to show the quantize process.",
        },
        "zh": {
            "label": "进度条",
            "info": "进度条。",
        },
    },
    "do_perf": {
        "en": {
            "label": "Do performance",
            "info": "Do performance evaluation.",
        },
        "zh": {
            "label": "性能评估",
            "info": "是否进行性能评估。",
        },
    },
    "build_cfg": {
        "en": {
            "label": "Build config path",
            "info": "Path to build configuration file.",
        },
        "zh": {
            "label": "编译配置文件路径",
            "info": "编译配置文件路径（build_cfg）。",
        },
    },
    "export_dir": {
        "en": {
            "label": "Export dir",
            "info": "Directory to save exported model.",
        },
        "zh": {
            "label": "导出目录",
            "info": "保存导出模型的文件夹路径。",
        },
    },
    "e2e_profile": {
        "en": {
            "label": "Do e2e performance",
            "info": "Do e2e performance evaluation.",
        },
        "zh": {
            "label": "进行e2e性能评估",
            "info": "是否进行e2e性能评估。",
        },
    },
    "layer_profile": {
        "en": {
            "label": "Do layer performance",
            "info": "Do layer performance evaluation.",
        },
        "zh": {
            "label": "进行layer性能评估",
            "info": "是否进行layer性能评估。",
        },
    },
    "e2e_prof": {
        "en": {
            "label": "e2e performance datas",
        },
        "zh": {
            "label": "e2e性能数据",
        },
    },
    "layer_prof": {
        "en": {
            "label": "layer performance datas",
        },
        "zh": {
            "label": "layer性能数据",
        },
    },
    "extra_compile_args": {
        "en": {
            "label": "Extra compile args",
            "info": "Extra compile arguments. (eg:--noDataTransfers --useCudaGraph)",
        },
        "zh": {
            "label": "额外编译参数",
            "info": "额外的编译参数 (例如:--noDataTransfers --useCudaGraph)",
        },
    },
    "shapes": {
        "en": {
            "label": "Shapes",
            "info": "Shapes of the model (e.g., name1:1x3x224x224,name2:1x1000).",
        },
        "zh": {
            "label": "模型形状",
            "info": "模型的形状 (例如:name1:1x3x224x224,name2:1x1000)",
        },
    },
    "dataset_dir": {
        "en": {
            "label": "Data dir",
            "info": "Path to the data directory.",
        },
        "zh": {
            "label": "数据路径",
            "info": "数据文件夹的路径。",
        },
    },
    "dataset": {
        "en": {
            "label": "Dataset",
        },
        "zh": {
            "label": "数据集",
        },
    },
    "start_btn": {
        "en": {
            "value": "Start",
        },
        "zh": {
            "value": "开始",
        },
    },
    "stop_btn": {
        "en": {
            "value": "Abort",
        },
        "zh": {
            "value": "中断",
        },
    },
    "output_dir": {
        "en": {
            "label": "Output dir",
            "info": "Directory for saving results.",
        },
        "zh": {
            "label": "输出目录",
            "info": "保存结果的路径。",
        },
    },
    "output_box": {
        "en": {
            "value": "Ready.",
        },
        "zh": {
            "value": "准备就绪。",
        },
    },
    "predict": {
        "en": {
            "label": "Save predictions",
        },
        "zh": {
            "label": "保存预测结果",
        },
    },
    "export_btn": {
        "en": {
            "value": "Export",
        },
        "zh": {
            "value": "开始导出",
        },
    },
    "device_memory": {
        "en": {
            "label": "Device memory",
            "info": "Current memory usage of the device (GB).",
        },
        "zh": {
            "label": "设备显存",
            "info": "当前设备的显存（GB）。",
        },
    },
}


ALERTS = {
    "err_conflict": {
        "en": "A process is in running, please abort it first.",
        "zh": "任务已存在，请先中断任务.",
    },
    "err_profile_conflict": {
        "en": "e2e and layer profile simultaneously are not supported.",
        "zh": "不能同时对e2e和layer进行profile."
    },
    "err_profile_model_type": {
        "en": "support onnx model only.",
        "zh": "目前只支持onnx模型."
    },
    "err_no_model": {
        "en": "Please select a model.",
        "zh": "请选择模型。",
    },
    "err_no_path": {
        "en": "Model not found.",
        "zh": "模型未找到。",
    },
    "err_no_dataset": {
        "en": "Please choose a dataset.",
        "zh": "请选择数据集。",
    },
    "err_no_output_dir": {
        "en": "Please provide output dir.",
        "zh": "请填写输出目录。",
    },
    "err_failed": {
        "en": "Failed.",
        "zh": "执行出错。",
    },
    "warn_no_cuda": {
        "en": "CUDA environment was not detected.",
        "zh": "未检测到 CUDA 环境。",
    },
    "info_aborting": {
        "en": "Aborted, wait for terminating...",
        "zh": "任务中断，正在等待进程结束……",
    },
    "info_aborted": {
        "en": "Ready.",
        "zh": "准备就绪。",
    },
    "info_finished": {
        "en": "Finished.",
        "zh": "执行完毕。",
    },
}
