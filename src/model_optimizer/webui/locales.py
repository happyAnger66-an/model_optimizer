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
            "value": "<h1><center>🦙🏭Model Optimizer: Unified Efficient Fine-Tuning of 100+ LLMs</center></h1>",
        },
        "zh": {
            "value": "<h1><center>🦙🏭Model Optimizer: 模型高效优化平台</center></h1>",
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
        "ru": {
            "value": "Начать",
        },
        "zh": {
            "value": "开始",
        },
        "ko": {
            "value": "시작",
        },
        "ja": {
            "value": "開始",
        },
    },
    "stop_btn": {
        "en": {
            "value": "Abort",
        },
        "ru": {
            "value": "Прервать",
        },
        "zh": {
            "value": "中断",
        },
        "ko": {
            "value": "중단",
        },
        "ja": {
            "value": "中断",
        },
    },
    "output_dir": {
        "en": {
            "label": "Output dir",
            "info": "Directory for saving results.",
        },
        "ru": {
            "label": "Выходной каталог",
            "info": "Каталог для сохранения результатов.",
        },
        "zh": {
            "label": "输出目录",
            "info": "保存结果的路径。",
        },
        "ko": {
            "label": "출력 디렉토리",
            "info": "결과를 저장할 디렉토리.",
        },
        "ja": {
            "label": "出力ディレクトリ",
            "info": "結果を保存するパス。",
        },
    },
    "output_box": {
        "en": {
            "value": "Ready.",
        },
        "ru": {
            "value": "Готово.",
        },
        "zh": {
            "value": "准备就绪。",
        },
        "ko": {
            "value": "준비 완료.",
        },
        "ja": {
            "value": "準備完了。",
        },
    },
    "loss_viewer": {
        "en": {
            "label": "Loss",
        },
        "ru": {
            "label": "Потери",
        },
        "zh": {
            "label": "损失",
        },
        "ko": {
            "label": "손실",
        },
        "ja": {
            "label": "損失",
        },
    },
    "predict": {
        "en": {
            "label": "Save predictions",
        },
        "ru": {
            "label": "Сохранить предсказания",
        },
        "zh": {
            "label": "保存预测结果",
        },
        "ko": {
            "label": "예측 결과 저장",
        },
        "ja": {
            "label": "予測結果を保存",
        },
    },
    "infer_backend": {
        "en": {
            "label": "Inference engine",
        },
        "ru": {
            "label": "Инференс движок",
        },
        "zh": {
            "label": "推理引擎",
        },
        "ko": {
            "label": "추론 엔진",
        },
        "ja": {
            "label": "推論エンジン",
        },
    },
    "infer_dtype": {
        "en": {
            "label": "Inference data type",
        },
        "ru": {
            "label": "Тип данных для вывода",
        },
        "zh": {
            "label": "推理数据类型",
        },
        "ko": {
            "label": "추론 데이터 유형",
        },
        "ja": {
            "label": "推論データタイプ",
        },
    },
    "load_btn": {
        "en": {
            "value": "Load model",
        },
        "ru": {
            "value": "Загрузить модель",
        },
        "zh": {
            "value": "加载模型",
        },
        "ko": {
            "value": "모델 불러오기",
        },
        "ja": {
            "value": "モデルを読み込む",
        },
    },
    "unload_btn": {
        "en": {
            "value": "Unload model",
        },
        "ru": {
            "value": "Выгрузить модель",
        },
        "zh": {
            "value": "卸载模型",
        },
        "ko": {
            "value": "모델 언로드",
        },
        "ja": {
            "value": "モデルをアンロード",
        },
    },
    "info_box": {
        "en": {
            "value": "Model unloaded, please load a model first.",
        },
        "ru": {
            "value": "Модель не загружена, загрузите модель сначала.",
        },
        "zh": {
            "value": "模型未加载，请先加载模型。",
        },
        "ko": {
            "value": "모델이 언로드되었습니다. 모델을 먼저 불러오십시오.",
        },
        "ja": {
            "value": "モデルがロードされていません。最初にモデルをロードしてください。",
        },
    },
    "role": {
        "en": {
            "label": "Role",
        },
        "ru": {
            "label": "Роль",
        },
        "zh": {
            "label": "角色",
        },
        "ko": {
            "label": "역할",
        },
        "ja": {
            "label": "役割",
        },
    },
    "system": {
        "en": {
            "placeholder": "System prompt (optional)",
        },
        "ru": {
            "placeholder": "Системный запрос (по желанию)",
        },
        "zh": {
            "placeholder": "系统提示词（非必填）",
        },
        "ko": {
            "placeholder": "시스템 프롬프트 (선택 사항)",
        },
        "ja": {
            "placeholder": "システムプロンプト（オプション）",
        },
    },
    "tools": {
        "en": {
            "placeholder": "Tools (optional)",
        },
        "ru": {
            "placeholder": "Инструменты (по желанию)",
        },
        "zh": {
            "placeholder": "工具列表（非必填）",
        },
        "ko": {
            "placeholder": "툴 (선택 사항)",
        },
        "ja": {
            "placeholder": "ツールリスト（オプション）",
        },
    },
    "image": {
        "en": {
            "label": "Image (optional)",
        },
        "ru": {
            "label": "Изображение (по желанию)",
        },
        "zh": {
            "label": "图像（非必填）",
        },
        "ko": {
            "label": "이미지 (선택 사항)",
        },
        "ja": {
            "label": "画像（オプション）",
        },
    },
    "video": {
        "en": {
            "label": "Video (optional)",
        },
        "ru": {
            "label": "Видео (по желанию)",
        },
        "zh": {
            "label": "视频（非必填）",
        },
        "ko": {
            "label": "비디오 (선택 사항)",
        },
        "ja": {
            "label": "動画（オプション）",
        },
    },
    "query": {
        "en": {
            "placeholder": "Input...",
        },
        "ru": {
            "placeholder": "Ввод...",
        },
        "zh": {
            "placeholder": "输入...",
        },
        "ko": {
            "placeholder": "입력...",
        },
        "ja": {
            "placeholder": "入力...",
        },
    },
    "submit_btn": {
        "en": {
            "value": "Submit",
        },
        "ru": {
            "value": "Отправить",
        },
        "zh": {
            "value": "提交",
        },
        "ko": {
            "value": "제출",
        },
        "ja": {
            "value": "送信",
        },
    },
    "max_length": {
        "en": {
            "label": "Maximum length",
        },
        "ru": {
            "label": "Максимальная длина",
        },
        "zh": {
            "label": "最大长度",
        },
        "ko": {
            "label": "최대 길이",
        },
        "ja": {
            "label": "最大長",
        },
    },
    "max_new_tokens": {
        "en": {
            "label": "Maximum new tokens",
        },
        "ru": {
            "label": "Максимальное количество новых токенов",
        },
        "zh": {
            "label": "最大生成长度",
        },
        "ko": {
            "label": "응답의 최대 길이",
        },
        "ja": {
            "label": "最大生成長",
        },
    },
    "top_p": {
        "en": {
            "label": "Top-p",
        },
        "ru": {
            "label": "Лучшие-p",
        },
        "zh": {
            "label": "Top-p 采样值",
        },
        "ko": {
            "label": "Top-p",
        },
        "ja": {
            "label": "Top-p",
        },
    },
    "temperature": {
        "en": {
            "label": "Temperature",
        },
        "ru": {
            "label": "Температура",
        },
        "zh": {
            "label": "温度系数",
        },
        "ko": {
            "label": "온도",
        },
        "ja": {
            "label": "温度",
        },
    },
    "skip_special_tokens": {
        "en": {
            "label": "Skip special tokens",
        },
        "ru": {
            "label": "Пропустить специальные токены",
        },
        "zh": {
            "label": "跳过特殊 token",
        },
        "ko": {
            "label": "스페셜 토큰을 건너뛰기",
        },
        "ja": {
            "label": "スペシャルトークンをスキップ",
        },
    },
    "escape_html": {
        "en": {
            "label": "Escape HTML tags",
        },
        "ru": {
            "label": "Исключить HTML теги",
        },
        "zh": {
            "label": "转义 HTML 标签",
        },
        "ko": {
            "label": "HTML 태그 이스케이프",
        },
        "ja": {
            "label": "HTML タグをエスケープ",
        },
    },
    "clear_btn": {
        "en": {
            "value": "Clear history",
        },
        "ru": {
            "value": "Очистить историю",
        },
        "zh": {
            "value": "清空历史",
        },
        "ko": {
            "value": "기록 지우기",
        },
        "ja": {
            "value": "履歴をクリア",
        },
    },
    "export_size": {
        "en": {
            "label": "Max shard size (GB)",
            "info": "The maximum size for a model file.",
        },
        "ru": {
            "label": "Максимальный размер фрагмента (ГБ)",
            "info": "Максимальный размер файла модели.",
        },
        "zh": {
            "label": "最大分块大小（GB）",
            "info": "单个模型文件的最大大小。",
        },
        "ko": {
            "label": "최대 샤드 크기 (GB)",
            "info": "모델 파일의 최대 크기.",
        },
        "ja": {
            "label": "最大シャードサイズ（GB）",
            "info": "単一のモデルファイルの最大サイズ。",
        },
    },
    "export_quantization_bit": {
        "en": {
            "label": "Export quantization bit.",
            "info": "Quantizing the exported model.",
        },
        "ru": {
            "label": "Экспорт бита квантования",
            "info": "Квантование экспортируемой модели.",
        },
        "zh": {
            "label": "导出量化等级",
            "info": "量化导出模型。",
        },
        "ko": {
            "label": "양자화 비트 내보내기",
            "info": "내보낸 모델의 양자화.",
        },
        "ja": {
            "label": "量子化ビットをエクスポート",
            "info": "エクスポートするモデルを量子化します。",
        },
    },
    "export_quantization_dataset": {
        "en": {
            "label": "Export quantization dataset",
            "info": "The calibration dataset used for quantization.",
        },
        "ru": {
            "label": "Экспорт набора данных для квантования",
            "info": "Набор данных калибровки, используемый для квантования.",
        },
        "zh": {
            "label": "导出量化数据集",
            "info": "量化过程中使用的校准数据集。",
        },
        "ko": {
            "label": "양자화 데이터셋 내보내기",
            "info": "양자화에 사용되는 교정 데이터셋.",
        },
        "ja": {
            "label": "量子化データセットをエクスポート",
            "info": "量子化プロセスで使用されるキャリブレーションデータセット。",
        },
    },
    "export_device": {
        "en": {
            "label": "Export device",
            "info": "Which device should be used to export model.",
        },
        "ru": {
            "label": "Экспорт устройство",
            "info": "Какое устройство следует использовать для экспорта модели.",
        },
        "zh": {
            "label": "导出设备",
            "info": "导出模型使用的设备类型。",
        },
        "ko": {
            "label": "내보낼 장치",
            "info": "모델을 내보내는 데 사용할 장치.",
        },
        "ja": {
            "label": "エクスポートデバイス",
            "info": "モデルをエクスポートするために使用するデバイスタイプ。",
        },
    },
    "export_legacy_format": {
        "en": {
            "label": "Export legacy format",
            "info": "Do not use safetensors to save the model.",
        },
        "ru": {
            "label": "Экспорт в устаревший формат",
            "info": "Не использовать safetensors для сохранения модели.",
        },
        "zh": {
            "label": "导出旧格式",
            "info": "不使用 safetensors 格式保存模型。",
        },
        "ko": {
            "label": "레거시 형식 내보내기",
            "info": "모델을 저장하는 데 safetensors를 사용하지 않습니다.",
        },
        "ja": {
            "label": "レガシーフォーマットをエクスポート",
            "info": "safetensors フォーマットを使用せずにモデルを保存します。",
        },
    },
    "export_dir": {
        "en": {
            "label": "Export dir",
            "info": "Directory to save exported model.",
        },
        "ru": {
            "label": "Каталог экспорта",
            "info": "Каталог для сохранения экспортированной модели.",
        },
        "zh": {
            "label": "导出目录",
            "info": "保存导出模型的文件夹路径。",
        },
        "ko": {
            "label": "내보내기 디렉토리",
            "info": "내보낸 모델을 저장할 디렉토리.",
        },
        "ja": {
            "label": "エクスポートディレクトリ",
            "info": "エクスポートしたモデルを保存するフォルダのパス。",
        },
    },
    "export_hub_model_id": {
        "en": {
            "label": "HF Hub ID (optional)",
            "info": "Repo ID for uploading model to Hugging Face hub.",
        },
        "ru": {
            "label": "HF Hub ID (опционально)",
            "info": "Идентификатор репозитория для загрузки модели на Hugging Face hub.",
        },
        "zh": {
            "label": "HF Hub ID（非必填）",
            "info": "用于将模型上传至 Hugging Face Hub 的仓库 ID。",
        },
        "ko": {
            "label": "HF 허브 ID (선택 사항)",
            "info": "모델을 Hugging Face 허브에 업로드하기 위한 레포 ID.",
        },
        "ja": {
            "label": "HF Hub ID（オプション）",
            "info": "Hugging Face Hub にモデルをアップロードするためのリポジトリ ID。",
        },
    },
    "export_btn": {
        "en": {
            "value": "Export",
        },
        "ru": {
            "value": "Экспорт",
        },
        "zh": {
            "value": "开始导出",
        },
        "ko": {
            "value": "내보내기",
        },
        "ja": {
            "value": "エクスポート",
        },
    },
    "device_memory": {
        "en": {
            "label": "Device memory",
            "info": "Current memory usage of the device (GB).",
        },
        "ru": {
            "label": "Память устройства",
            "info": "Текущая память на устройстве (GB).",
        },
        "zh": {
            "label": "设备显存",
            "info": "当前设备的显存（GB）。",
        },
        "ko": {
            "label": "디바이스 메모리",
            "info": "지금 사용 중인 기기 메모리 (GB).",
        },
        "ja": {
            "label": "デバイスメモリ",
            "info": "現在のデバイスのメモリ（GB）。",
        },
    },
}


ALERTS = {
    "err_conflict": {
        "en": "A process is in running, please abort it first.",
        "ru": "Процесс уже запущен, пожалуйста, сначала прервите его.",
        "zh": "任务已存在，请先中断训练。",
        "ko": "프로세스가 실행 중입니다. 먼저 중단하십시오.",
        "ja": "プロセスが実行中です。最初に中断してください。",
    },
    "err_exists": {
        "en": "You have loaded a model, please unload it first.",
        "ru": "Вы загрузили модель, сначала разгрузите ее.",
        "zh": "模型已存在，请先卸载模型。",
        "ko": "모델이 로드되었습니다. 먼저 언로드하십시오.",
        "ja": "モデルがロードされています。最初にアンロードしてください。",
    },
    "err_no_model": {
        "en": "Please select a model.",
        "ru": "Пожалуйста, выберите модель.",
        "zh": "请选择模型。",
        "ko": "모델을 선택하십시오.",
        "ja": "モデルを選択してください。",
    },
    "err_no_path": {
        "en": "Model not found.",
        "ru": "Модель не найдена.",
        "zh": "模型未找到。",
        "ko": "모델을 찾을 수 없습니다.",
        "ja": "モデルが見つかりません。",
    },
    "err_no_dataset": {
        "en": "Please choose a dataset.",
        "ru": "Пожалуйста, выберите набор данных.",
        "zh": "请选择数据集。",
        "ko": "데이터 세트를 선택하십시오.",
        "ja": "データセットを選択してください。",
    },
    "err_no_adapter": {
        "en": "Please select an adapter.",
        "ru": "Пожалуйста, выберите адаптер.",
        "zh": "请选择适配器。",
        "ko": "어댑터를 선택하십시오.",
        "ja": "アダプターを選択してください。",
    },
    "err_no_output_dir": {
        "en": "Please provide output dir.",
        "ru": "Пожалуйста, укажите выходную директорию.",
        "zh": "请填写输出目录。",
        "ko": "출력 디렉토리를 제공하십시오.",
        "ja": "出力ディレクトリを入力してください。",
    },
    "err_no_reward_model": {
        "en": "Please select a reward model.",
        "ru": "Пожалуйста, выберите модель вознаграждения.",
        "zh": "请选择奖励模型。",
        "ko": "리워드 모델을 선택하십시오.",
        "ja": "報酬モデルを選択してください。",
    },
    "err_no_export_dir": {
        "en": "Please provide export dir.",
        "ru": "Пожалуйста, укажите каталог для экспорта.",
        "zh": "请填写导出目录。",
        "ko": "Export 디렉토리를 제공하십시오.",
        "ja": "エクスポートディレクトリを入力してください。",
    },
    "err_gptq_lora": {
        "en": "Please merge adapters before quantizing the model.",
        "ru": "Пожалуйста, объедините адаптеры перед квантованием модели.",
        "zh": "量化模型前请先合并适配器。",
        "ko": "모델을 양자화하기 전에 어댑터를 병합하십시오.",
        "ja": "モデルを量子化する前にアダプターをマージしてください。",
    },
    "err_failed": {
        "en": "Failed.",
        "ru": "Ошибка.",
        "zh": "量化出错。",
        "ko": "실패했습니다.",
        "ja": "失敗しました。",
    },
    "err_demo": {
        "en": "Training is unavailable in demo mode, duplicate the space to a private one first.",
        "ru": "Обучение недоступно в демонстрационном режиме, сначала скопируйте пространство в частное.",
        "zh": "展示模式不支持训练，请先复制到私人空间。",
        "ko": "데모 모드에서는 훈련을 사용할 수 없습니다. 먼저 프라이빗 레포지토리로 작업 공간을 복제하십시오.",
        "ja": "デモモードではトレーニングは利用できません。最初にプライベートスペースに複製してください。",
    },
    "err_tool_name": {
        "en": "Tool name not found.",
        "ru": "Имя инструмента не найдено.",
        "zh": "工具名称未找到。",
        "ko": "툴 이름을 찾을 수 없습니다.",
        "ja": "ツール名が見つかりません。",
    },
    "err_json_schema": {
        "en": "Invalid JSON schema.",
        "ru": "Неверная схема JSON.",
        "zh": "Json 格式错误。",
        "ko": "잘못된 JSON 스키마입니다.",
        "ja": "JSON スキーマが無効です。",
    },
    "err_config_not_found": {
        "en": "Config file is not found.",
        "ru": "Файл конфигурации не найден.",
        "zh": "未找到配置文件。",
        "ko": "Config 파일을 찾을 수 없습니다.",
        "ja": "設定ファイルが見つかりません。",
    },
    "warn_no_cuda": {
        "en": "CUDA environment was not detected.",
        "ru": "Среда CUDA не обнаружена.",
        "zh": "未检测到 CUDA 环境。",
        "ko": "CUDA 환경이 감지되지 않았습니다.",
        "ja": "CUDA 環境が検出されませんでした。",
    },
    "warn_output_dir_exists": {
        "en": "Output dir already exists, will resume training from here.",
        "ru": "Выходной каталог уже существует, обучение будет продолжено отсюда.",
        "zh": "输出目录已存在，将从该断点恢复训练。",
        "ko": "출력 디렉토리가 이미 존재합니다. 위 출력 디렉토리에 저장된 학습을 재개합니다.",
        "ja": "出力ディレクトリが既に存在します。このチェックポイントからトレーニングを再開します。",
    },
    "warn_no_instruct": {
        "en": "You are using a non-instruct model, please fine-tune it first.",
        "ru": "Вы используете модель без инструкции, пожалуйста, primeros выполните донастройку этой модели.",
        "zh": "您正在使用非指令模型，请先对其进行微调。",
        "ko": "당신은 지시하지 않은 모델을 사용하고 있습니다. 먼저 이를 미세 조정해 주세요.",
        "ja": "インストラクションモデルを使用していません。まずモデルをアダプターに適合させてください。",
    },
    "info_aborting": {
        "en": "Aborted, wait for terminating...",
        "ru": "Прервано, ожидание завершения...",
        "zh": "训练中断，正在等待进程结束……",
        "ko": "중단되었습니다. 종료를 기다리십시오...",
        "ja": "トレーニングが中断されました。プロセスの終了を待っています...",
    },
    "info_aborted": {
        "en": "Ready.",
        "ru": "Готово.",
        "zh": "准备就绪。",
        "ko": "준비되었습니다.",
        "ja": "準備完了。",
    },
    "info_finished": {
        "en": "Finished.",
        "ru": "Завершено.",
        "zh": "量化完毕。",
        "ko": "완료되었습니다.",
        "ja": "トレーニングが完了しました。",
    },
    "info_config_saved": {
        "en": "Arguments have been saved at: ",
        "ru": "Аргументы были сохранены по адресу: ",
        "zh": "训练参数已保存至：",
        "ko": "매개변수가 저장되었습니다: ",
        "ja": "トレーニングパラメータが保存されました: ",
    },
    "info_config_loaded": {
        "en": "Arguments have been restored.",
        "ru": "Аргументы были восстановлены.",
        "zh": "训练参数已载入。",
        "ko": "매개변수가 복원되었습니다.",
        "ja": "トレーニングパラメータが読み込まれました。",
    },
    "info_loading": {
        "en": "Loading model...",
        "ru": "Загрузка модели...",
        "zh": "加载中……",
        "ko": "모델 로딩 중...",
        "ja": "モデルをロード中...",
    },
    "info_unloading": {
        "en": "Unloading model...",
        "ru": "Выгрузка модели...",
        "zh": "卸载中……",
        "ko": "모델 언로딩 중...",
        "ja": "モデルをアンロード中...",
    },
    "info_loaded": {
        "en": "Model loaded, now you can chat with your model!",
        "ru": "Модель загружена, теперь вы можете общаться с вашей моделью!",
        "zh": "模型已加载，可以开始聊天了！",
        "ko": "모델이 로드되었습니다. 이제 모델과 채팅할 수 있습니다!",
        "ja": "モデルがロードされました。チャットを開始できます！",
    },
    "info_unloaded": {
        "en": "Model unloaded.",
        "ru": "Модель выгружена.",
        "zh": "模型已卸载。",
        "ko": "모델이 언로드되었습니다.",
        "ja": "モデルがアンロードされました。",
    },
    "info_thinking": {
        "en": "🌀 Thinking...",
        "ru": "🌀 Думаю...",
        "zh": "🌀 思考中...",
        "ko": "🌀 생각 중...",
        "ja": "🌀 考えています...",
    },
    "info_thought": {
        "en": "✅ Thought",
        "ru": "✅ Думать закончено",
        "zh": "✅ 思考完成",
        "ko": "✅ 생각이 완료되었습니다",
        "ja": "✅ 思考完了",
    },
    "info_exporting": {
        "en": "Exporting model...",
        "ru": "Экспорт модели...",
        "zh": "正在导出模型……",
        "ko": "모델 내보내기 중...",
        "ja": "モデルをエクスポート中...",
    },
    "info_exported": {
        "en": "Model exported.",
        "ru": "Модель экспортирована.",
        "zh": "模型导出完成。",
        "ko": "모델이 내보내졌습니다.",
        "ja": "モデルのエクスポートが完了しました。",
    },
    "info_swanlab_link": {
        "en": "### SwanLab Link\n",
        "ru": "### SwanLab ссылка\n",
        "zh": "### SwanLab 链接\n",
        "ko": "### SwanLab 링크\n",
        "ja": "### SwanLab リンク\n",
    },
}
