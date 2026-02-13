# π0.5 model quantize

This page describes how to use model-opt to quantize `pi05 model`.


## 1. gemma LLM model quantize

### 1.1 quant_cfg

```py
from modelopt.torch.quantization import INT8_DEFAULT_CFG, INT8_SMOOTHQUANT_CFG, FP8_DEFAULT_CFG
QUANT_CFG = INT8_DEFAULT_CFG

QUANT_CFG["algorithm"] = "max"
QUANT_CFG["mode"] = "int8"
QUANT_CFG["quant_cfg"]["input_layernorm"] = {"enable": False}
QUANT_CFG["quant_cfg"]["post_attention_layernorm"] = {"enable": False}
QUANT_CFG["quant_cfg"]["norm"] = {"enable": False}
```

+ To ensure the quantization effect, the following layers will not be quantized

`input_layernorm`, `post_attention_layernorm`, `norm`

### 1.2 calibrate

```shell
model-opt quantize --model_name pi05_libero/llm --model_path /srcs/openpi/pytorch_pi05_libero/ --quantize_cfg config/quant/llm_quant_cfg.py --calibrate_data /tmp/caliba/pi05/pi05_llm_calib_data.pt --export_dir /tmp/quantize/pi05
```