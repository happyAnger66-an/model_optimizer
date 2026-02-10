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

##### use `pytorch` + `action Dit tensorrt engine`

```
python scripts/deployment/pi05/standalone_inference_script.py --model_path /openpi/pytorch_pi05_libero/ --inference_mode tensorrt  --perf --trt_engine_path /tmp/build --expert_engine expert_bf16.engine
```

+ `--trt_engine_path`: 指定tensorrt engine存放path
+ `--expert_engine`: 指定专家`Dit tensorrt engine`名字


##### save infer results
```shell
python scripts/deployment/pi05/standalone_inference_script.py --model_path /openpi/pytorch_pi05_libero/ --inference_mode tensorrt  --perf --trt_engine_path /tmp/build --expert_engine expert_bf16.engine  --save_output_path /tmp/output/action_engine/
```

+ `--save_output_path`: save infer results path


##### compare infer results

```shell
model-opt compare --data_path1 /tmp/pi05/output_data/inputs.npz --data_path2 /tmp/output/action_engine/outputs.npz
```

+ `--data_path1`: infer results data path 1 `
+ `--data_path2`: infer results data path 2 `

`output:`
```shell
=== Prediction Comparison ===

actions:
Cosine Similarity (PyTorch/TensorRT):    0.99251389503479
L1 Mean/Max Distance (PyTorch/TensorRT): 0.0299/0.1872
Max Output Values (PyTorch/TensorRT):    0.9996/0.9988
Mean Output Values (PyTorch/TensorRT):   0.1286/0.1266
Min Output Values (PyTorch/TensorRT):    -0.4451/-0.5092
Skipping policy_timing because it is in the filter keys


=== Prediction Comparison ===

actions:
Cosine Similarity (PyTorch/TensorRT):    0.9909465312957764
L1 Mean/Max Distance (PyTorch/TensorRT): 0.0294/0.2820
Max Output Values (PyTorch/TensorRT):    1.0039/1.0033
Mean Output Values (PyTorch/TensorRT):   0.1227/0.1264
Min Output Values (PyTorch/TensorRT):    -0.4976/-0.4405
```