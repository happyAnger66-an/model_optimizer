## ptq layer collect.

```bash
python scripts/deployment/pi05/lerobot_eval_webui_server.py --checkpoint /srcs/openpi/pytorch_pi05_libero/ --config pi05_libero  --dataset-root /srcs/.cache/huggingface/lerobot/physical-intelligence/libero/  --host 0.0.0.0 --port 8765 --num-samples 100 --ptq-compare  --ptq-quant-cfg config/quant/llm_quant_nvfp4_cfg.py --ptq-calib-dir /tmp/calib/lerbot/ --ptq-parts llm --ptq-layer-report-path /tmp/ptq_report/
```