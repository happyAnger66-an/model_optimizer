# 1. Usage

## 1.1 CLI

```shell
----------------------------------------------------------------------
| Usage:                                                             |
|   model_optimizer-cli quantize: quantize a model |
|   model_optimizer-cli export: export a model format |
|   model_optimizer-cli profile: profile a model |
|   model_optimizer-cli calibrate: calibrate a model |
|   model_optimizer-cli build -h: build a onnx model to engine |
|   model_optimizer-cli eval -h: eval model |
|   model_optimizer-cli webui: launch webui                        |
|   model_optimizer-cli download: download a model                      |
|   model_optimizer-cli version: show version info                      |
| Hint: You can use `moc` as a shortcut for `model_optimizer-cli`.      |
----------------------------------------------------------------------
```

### 1.1.1 inference

+ use `pytorch` + `action Dit tensorrt engine`

```
python scripts/deployment/pi05/standalone_inference_script.py --model_path /openpi/pytorch_pi05_libero/ --inference_mode tensorrt  --perf --trt_engine_path /tmp/build --expert_engine expert_bf16.engine
```

+ `--trt_engine_path`: 指定tensorrt engine存放path
+ `--expert_engine`: 指定专家`Dit tensorrt engine`名字