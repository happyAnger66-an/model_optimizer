<div align="center">

model optimizer
===========================

<div align="left">

## Overview

model optimizer is a tool to make model quantize, optimize, deploy easier. including functions:

- quantize model
- optimize model
- deploy model


## Getting Started

The code was tested in the following environments

- Ubuntu 24.04.03 LTS
- NVIDIA GPU 4070/5090/Thor driver version 580.95.05 with CUDA 13.0 support
- Docker nvcr.io/nvidia/tensorrt:25.10-py3

### Installation

#### use tensorrt docker image
```bash
$ docker run -it --gpus -v$(pwd):/srcs all nvcr.io/nvidia/tensorrt:25.10-py3

# cd /srcs

# pip install -r requirements.txt

# python setup.py install

# model-opt webui
```

### Usage

use it in the web browser: `http://ip:7860`

![index.png](img/index.png)

## Contributing

## License

## Contact