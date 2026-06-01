# WebUI A/B Test Configs

用于 `scripts/deployment/pi05/lerobot_eval_webui_server.py` 的 full-loop CUDA Graph A/B 测试配置。

## 文件

- `native_full_loop_ab_baseline.yaml`：`native_full_loop_graph=false`
- `native_full_loop_ab_enabled.yaml`：`native_full_loop_graph=true`

## 使用

先把 YAML 里的 `checkpoint` 改成你的真实路径，然后运行：

```bash
python scripts/deployment/pi05/lerobot_eval_webui_server.py --webui-config webui_configs/native_full_loop_ab_baseline.yaml
python scripts/deployment/pi05/lerobot_eval_webui_server.py --webui-config webui_configs/native_full_loop_ab_enabled.yaml
```

若需要覆盖单个参数（例如样本数），可在命令行追加：

```bash
python scripts/deployment/pi05/lerobot_eval_webui_server.py \
  --webui-config webui_configs/native_full_loop_ab_enabled.yaml \
  --num-samples 500
```

## 建议对比日志

```bash
rg -n "\\[summary\\] e2e/chunk|\\[summary\\] predict_ms|NativeDenoiseLoopRunnerV2 summary|path_stats|captured full-loop|pt_cuda_graph_full_loop_replay" /path/to/log
```
