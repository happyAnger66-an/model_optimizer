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

### 1.2 collector calibrate datas
We collect some mock data with `standalone_inference_script.py`, you should get some real data in product.

```shell
python scripts/deployment/pi05/standalone_inference_script.py --model-path /srcs/openpi/pytorch_pi05_libero/ --inference-mode tensorrt --calib-save-path /tmp/calibrate/pi0
```

+ output
```shell
collectd 40 datas
root@thor02:/srcs/codes/model_optimizer# 
root@thor02:/srcs/codes/model_optimizer# 
root@thor02:/srcs/codes/model_optimizer# ls -l /tmp/calibrate/pi05/
total 301628
-rw-r--r-- 1 root root 308866743 Feb 14 09:31 pi05_llm_calib_data.p
```

### 1.3 quantize

```shell
model-opt quantize --model_name pi05_libero/llm --model_path /srcs/openpi/pytorch_pi05_libero/ --quantize_cfg config/quant/llm_quant_cfg.py --calibrate_data /tmp/caliba/pi05/pi05_llm_calib_data.pt --export_dir /tmp/quantize/pi05
```

+ You can modify the config/quant/llm_quant_cfg.py as your need.

+ output:
```shell
378 TensorQuantizers found in model
Start LLM export onnx...
Loading extension modelopt_cuda_ext...

Loaded extension modelopt_cuda_ext in 36.0 seconds
LLM export onnx done to /tmp/quantize/pi05/ cost:55.919289112091064s
```

### 1.4 Run the test using the model that was just quantized

```shell
python scripts/deployment/pi05/standalone_inference_script.py --model-path /srcs/openpi/pytorch_pi05_libero/ --inference-mode tensorrt --trt_engine_path /tmp/build/pi05/ --llm_engine llm_int8q.engine --perf
```


## 2. action expert quantize


#### You may get bellow erros:

```shell
  File "/opt/openpi/lib/python3.12/site-packages/transformers/models/gemma/modeling_gemma.py", line 88, in forward
    modulation = self.dense(cond)
                 ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1787, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1766, in _slow_forward
    result = self.forward(*input, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/torch/quantization/nn/modules/quant_module.py", line 161, in forward
    return super().forward(input, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/modelopt/torch/quantization/nn/modules/quant_module.py", line 113, in forward
    output = super().forward(input, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/linear.py", line 134, in forward
    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float
```