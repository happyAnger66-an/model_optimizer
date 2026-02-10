<div align="center">

model optimizer
===========================

<div align="left">

## 1. Overview

model optimizer is a tool to make model quantize, optimize, deploy easier. including functions:

- quantize model
- optimize model
- deploy model


## 2. Getting Started

The code was tested in the following environments

- Ubuntu 24.04.03 LTS
- NVIDIA GPU 4070/5090/Thor driver version 580.95.05 with CUDA 13.0
- Docker nvcr.io/nvidia/tensorrt:25.10-py3

### 2.1 Installation

#### 2.1.1 use tensorrt docker image
```bash
$ docker run -it --gpus -v$(pwd):/srcs all nvcr.io/nvidia/tensorrt:25.10-py3

# cd /srcs

# pip install -r requirements.txt

# python setup.py install

# model-opt webui
```

## 3. Supported Models

|**Architecture**|**Model**||
|-|-|-|
|YOLO11|yolo||
|π0.5|pi05_libero||
|π0.5|pi05_libero/vit||
|π0.5|pi05_libero/llm||
|π0.5|pi05_libero/expert||


## 4. Usage

### 4.1 CLI (Product Ready)

#### 4.1.1 `π0.5` inference
```shell
python scripts/deployment/pi05/standalone_inference_script.py --model_path /openpi/pytorch_pi05_libero/ --inference_mode tensorrt  --perf 
```

`output`:
```shell
Inference time: 0.4213 seconds
e2e 419.43 ± 4.14 ms (shared)
action 25.14 ± 0.77 ms (shared)
vit 62.79 ± 0.49 ms (shared)
llm 100.47 ± 1.55 ms (shared)
```

+ `model_path`: pi05 model path, must a `pytorch` model
+ `inference_mode`: `tensorrt` or `pytorch`, choose tensorrt or pytorch as inference backend.
+ `--perf`: perf stats

#### 4.2 For more cli usages
[ALL CLI usage docs](./docs/cli.md)

### 4.2 WebPage (Not all ready yet.)
use it in the web browser: `http://ip:7860`

![index.png](img/index.png)

## 5. Contributing

## 6. License

## 7. Contact

+ `email:` `happyAnger66@163.com`