# Copyright 2025 the LlamaOptimizer team.
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
            "value": "<h1><center>Model Optimizer: Unified Efficient Fine-Tuning of 100+ LLMs</center></h1>",
        },
        "zh": {
            "value": "<h1><center>Model Optimizer: 模型高效优化平台</center></h1>",
        },
    },
    "subtitle": {
        "en": {
            "value": (
                "<h3><center>Visit <a href='https://github.com/hiyouga/Model-Optimizer' target='_blank'>"
                "GitHub Page</a> <a href='https://llamafactory.readthedocs.io/en/latest/' target='_blank'>"
                "Documentation</a></center></h3>"
            ),
        },
        "zh": {
            "value": (
                "<h3><center>访问 <a href='https://github.com/hiyouga/Model-Optimizer' target='_blank'>"
                "GitHub 主页</a> <a href='https://llamafactory.readthedocs.io/zh-cn/latest/' target='_blank'>"
                "官方文档</a></center></h3>"
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
    "hub_name": {
        "en": {
            "label": "Hub name",
            "info": "Choose the model download source.",
        },
        "zh": {
            "label": "模型下载源",
            "info": "选择模型下载源。（网络受限环境推荐使用 ModelScope）",
        },
    },
    "checkpoint_path": {
        "en": {
            "label": "Checkpoint path",
        },
    },
    "quantization_bit": {
        "en": {
            "label": "Quantization bit",
            "info": "Enable quantization (QLoRA).",
        },
        "zh": {
            "label": "量化等级",
            "info": "量化bit（FP only）。",
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
    "template": {
        "en": {
            "label": "Chat template",
            "info": "The chat template used in constructing prompts.",
        },
        "zh": {
            "label": "对话模板",
            "info": "构建提示词时使用的模板。",
        },
    },
    "rope_scaling": {
        "en": {
            "label": "RoPE scaling",
            "info": "RoPE scaling method to use.",
        },
        "zh": {"label": "RoPE 插值方法", "info": "RoPE 插值时使用的方法。"},
    },
    "booster": {
        "en": {
            "label": "Booster",
            "info": "Approach used to boost training speed.",
        },
        "zh": {"label": "加速方式", "info": "使用的加速方法。"},
    },
    "training_stage": {
        "en": {
            "label": "Stage",
            "info": "The stage to perform in training.",
        },
        "zh": {
            "label": "训练阶段",
            "info": "目前采用的训练方式。",
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
    "data_preview_btn": {
        "en": {
            "value": "Preview dataset",
        },
        "zh": {
            "value": "预览数据集",
        },
    },
    "preview_count": {
        "en": {
            "label": "Count",
        },
        "zh": {
            "label": "数量",
        },
    },
    "page_index": {
        "en": {
            "label": "Page",
        },
        "zh": {
            "label": "页数",
        },
    },
    "prev_btn": {
        "en": {
            "value": "Prev",
        },
        "zh": {
            "value": "上一页",
        },
    },
    "next_btn": {
        "en": {
            "value": "Next",
        },
        "zh": {
            "value": "下一页",
        },
    },
    "close_btn": {
        "en": {
            "value": "Close",
        },
        "zh": {
            "value": "关闭",
        },
    },
    "preview_samples": {
        "en": {
            "label": "Samples",
        },
        "zh": {
            "label": "样例",
        },
    },
    "batch_size": {
        "en": {
            "label": "Batch size",
            "info": "Number of samples processed on each GPU.",
        },
        "zh": {
            "label": "批处理大小",
            "info": "每个 GPU 处理的样本数量。",
        },
    },
    "warmup_steps": {
        "en": {
            "label": "Warmup steps",
            "info": "Number of steps used for warmup.",
        },
        "zh": {
            "label": "预热步数",
            "info": "学习率预热采用的步数。",
        },
    },
    "extra_args": {
        "en": {
            "label": "Extra arguments",
            "info": "Extra arguments passed to the trainer in JSON format.",
        },
        "zh": {
            "label": "额外参数",
            "info": "以 JSON 格式传递给训练器的额外参数。",
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
    "infer_backend": {
        "en": {
            "label": "Inference engine",
        },
        "zh": {
            "label": "推理引擎",
        },
    },
    "infer_dtype": {
        "en": {
            "label": "Inference data type",
        },
        "zh": {
            "label": "推理数据类型",
        },
    },
    "load_btn": {
        "en": {
            "value": "Load model",
        },
        "zh": {
            "value": "加载模型",
        },
    },
    "info_box": {
        "en": {
            "value": "Model unloaded, please load a model first.",
        },
        "zh": {
            "value": "模型未加载，请先加载模型。",
        },
    },
    "query": {
        "en": {
            "placeholder": "Input...",
        },
        "zh": {
            "placeholder": "输入...",
        },
    },
    "submit_btn": {
        "en": {
            "value": "Submit",
        },
        "zh": {
            "value": "提交",
        },
    },
    "max_length": {
        "en": {
            "label": "Maximum length",
        },
        "zh": {
            "label": "最大长度",
        },
    },
    "clear_btn": {
        "en": {
            "value": "Clear history",
        },
        "zh": {
            "value": "清空历史",
        },
    },
    "export_size": {
        "en": {
            "label": "Max shard size (GB)",
            "info": "The maximum size for a model file.",
        },
        "zh": {
            "label": "最大分块大小（GB）",
            "info": "单个模型文件的最大大小。",
        },
    },
    "export_quantization_bit": {
        "en": {
            "label": "Export quantization bit.",
            "info": "Quantizing the exported model.",
        },
        "zh": {
            "label": "导出量化等级",
            "info": "量化导出模型。",
        },
    },
    "export_quantization_dataset": {
        "en": {
            "label": "Export quantization dataset",
            "info": "The calibration dataset used for quantization.",
        },
        "zh": {
            "label": "导出量化数据集",
            "info": "量化过程中使用的校准数据集。",
        },
    },
    "export_device": {
        "en": {
            "label": "Export device",
            "info": "Which device should be used to export model.",
        },
        "zh": {
            "label": "导出设备",
            "info": "导出模型使用的设备类型。",
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
        "zh": "任务已存在，请先中断训练.",
    },
    "err_profile_conflict": {
        "en": "e2e and layer profile simultaneously are not supported.",
        "zh": "不能同时对e2e和layer进行profile."
    },
    "err_exists": {
        "en": "You have loaded a model, please unload it first.",
        "zh": "模型已存在，请先卸载模型。",
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
    "err_no_adapter": {
        "en": "Please select an adapter.",
        "zh": "请选择适配器。",
    },
    "err_no_output_dir": {
        "en": "Please provide output dir.",
        "zh": "请填写输出目录。",
    },
    "err_no_reward_model": {
        "en": "Please select a reward model.",
        "zh": "请选择奖励模型。",
    },
    "err_no_export_dir": {
        "en": "Please provide export dir.",
        "zh": "请填写导出目录。",
    },
    "err_gptq_lora": {
        "en": "Please merge adapters before quantizing the model.",
        "zh": "量化模型前请先合并适配器。",
    },
    "err_failed": {
        "en": "Failed.",
        "zh": "量化出错。",
    },
    "err_demo": {
        "en": "Training is unavailable in demo mode, duplicate the space to a private one first.",
        "zh": "展示模式不支持训练，请先复制到私人空间。",
    },
    "err_tool_name": {
        "en": "Tool name not found.",
        "zh": "工具名称未找到。",
    },
    "err_json_schema": {
        "en": "Invalid JSON schema.",
        "zh": "Json 格式错误。",
    },
    "err_config_not_found": {
        "en": "Config file is not found.",
        "zh": "未找到配置文件。",
    },
    "warn_no_cuda": {
        "en": "CUDA environment was not detected.",
        "zh": "未检测到 CUDA 环境。",
    },
    "warn_output_dir_exists": {
        "en": "Output dir already exists, will resume training from here.",
        "zh": "输出目录已存在，将从该断点恢复训练。",
    },
    "warn_no_instruct": {
        "en": "You are using a non-instruct model, please fine-tune it first.",
        "zh": "您正在使用非指令模型，请先对其进行微调。",
    },
    "info_aborting": {
        "en": "Aborted, wait for terminating...",
        "zh": "训练中断，正在等待进程结束……",
    },
    "info_aborted": {
        "en": "Ready.",
        "zh": "准备就绪。",
    },
    "info_finished": {
        "en": "Finished.",
        "zh": "量化完毕。",
    },
    "info_config_saved": {
        "en": "Arguments have been saved at: ",
        "zh": "训练参数已保存至：",
    },
    "info_config_loaded": {
        "en": "Arguments have been restored.",
        "zh": "训练参数已载入。",
    },
    "info_loading": {
        "en": "Loading model...",
        "zh": "加载中……",
    },
    "info_unloading": {
        "en": "Unloading model...",
        "zh": "卸载中……",
    },
    "info_loaded": {
        "en": "Model loaded, now you can chat with your model!",
        "zh": "模型已加载，可以开始聊天了！",
    },
    "info_unloaded": {
        "en": "Model unloaded.",
        "zh": "模型已卸载。",
    },
    "info_exporting": {
        "en": "Exporting model...",
        "zh": "正在导出模型……",
    },
    "info_exported": {
        "en": "Model exported.",
        "zh": "模型导出完成。",
    },
}
