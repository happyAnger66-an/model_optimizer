# Pi0.5 WebUI：PyTorch 浮点 vs PTQ 对比与分层误差

本文说明 `scripts/deployment/pi05/lerobot_eval_webui_server.py`（包 `lerobot_eval_webui`）中 **`--ptq-compare` 模式**：在 LeRobot 离线评估 WebUI 里同时跑 **浮点 PyTorch 策略** 与 **同 checkpoint 第二份策略上的选择性 PTQ（ModelOpt fake quant）**，流式推送两路预测差异，并可选生成 **各量化层输出相对浮点的误差报告**。

## 与前序设计的关系

实现与先前方案一致要点：

- **双 policy**：`create_trained_policy` 加载两次，互不共享权重；仅第二份执行 `quantize_model`。
- **子系统粒度**：与 `Vit` / `LLM` / `Expert` / `Pi05DenoiseStep`（`denoise`）的 `quantize` 相同，通过 **包装同一组 `nn.Module` 引用** 对 `vision_tower + multi_modal_projector`、`language_model`、`gemma_expert.model`、以及 **denoise 路径上的 expert + `action_in_proj` / `time_mlp_*` / `action_out_proj`** **就地**插入量化节点。
- **与 TensorRT 对比互斥**：`--compare-mode`（PyTorch vs TRT）与 `--ptq-compare` 不可同时开启。
- **分层误差**：在 PTQ 后模型上枚举 `modelopt` 的 QuantLinear（若不可用则降级为名称启发式），按 `named_modules` 路径在 **FP 与 PTQ 的同路径模块** 上注册 forward hook，依次 `infer` 收集输出，汇总 MSE / MAE / max abs（跨样本平均）。

## 命令行参数（`lerobot_eval_webui.config.Args`）

| 参数 | 含义                                                         |
|------|--------------------------------------------------------------|
| `--ptq-compare` | 开启双路 PyTorch（FP + PTQ）                                  |
| `--ptq-trt-compare` | 开启双路：**PyTorch PTQ（fake quant）** vs **TensorRT engine**（PTQ−TRT 对比） |
| `--ptq-quant-cfg` | ModelOpt 量化配置 JSON（可含顶层 `quant_mode`，会经 `normalize_quant_cfg` 处理） |
| `--ptq-calib-dir` | 与 `open_pi05_calib_for_quantize` 一致：含 `pi05_vit` / `pi05_llm` / `pi05_expert` / `pi05_denoise` 的 manifest 或分片目录 |
| `--ptq-parts` | 可多选：`vit`、`llm`、`expert`、`denoise`（tyro：`--ptq-parts llm denoise`） |
| `--ptq-layer-report-path` | 可选：分层 JSON 报告输出路径                                |
| `--ptq-layer-report-samples` | layer report 使用 `[start_index, start_index+N)` 连续帧，默认 32 |
| `--ptq-layer-report-histogram` | 报告每层附带 FP 激活 subsample 直方图（默认开启；`--no-ptq-layer-report-histogram` 可关以减小 JSON/meta） |
| `--ptq-layer-report-hist-bins` | 直方图 bin 数，默认 40 |
| `--ptq-layer-report-hist-max-elems` | 每层每次 forward 参与直方图的最大元素数，默认 100000 |

约束：

- 必须 **`--inference-mode pytorch`**。
- **`--ptq-parts` 非空**；`--ptq-quant-cfg` 为存在的文件；`--ptq-calib-dir` 为存在的目录。
- `--ptq-trt-compare` 需要额外提供 TensorRT 引擎参数：`--engine-path` 与各 `--{vit,llm,expert,...}_engine`（与 `--compare-mode` 相同）。

## 管线行为

1. **bundle 加载**（`bundle.load_infer_bundle`）：创建 `policy`（浮点），再创建 `policy_ptq`，按 `ptq_parts` 调用 `ptq_compare.apply_selective_ptq`（内部对 `Vit`/`LLM`/`Expert` 包装调用 `quantize_model` + `set_dynamic_quant(..., "fp16")`，**不导出 ONNX**）。
2. **可选 layer report**：在得到 `dataset` 与 `repack_fn` 后，调用 `write_ptq_layer_report`，对每条样本先后 `policy.infer` 与 `policy_ptq.infer`（带 hook），JSON 写入 `ptq_layer_report_path`；并在 meta 中带上该路径（若配置）。报告每层可含 **`fp_activation_histogram`**：`bin_edges`/`counts`、`underflow`/`overflow`（落在首帧按 0.1%–99.9% 分位确定的边界外）、以及 subsample 上的 **min/max/mean/std**；WebUI 中点击带直方图的表行可查看条形图。
3. **chunk 推理**（`chunk_infer.process_infer_chunk`）：与 TRT 对比类似，对同 obs 双路 `infer`，step JSON 中增加 `pred_action_ptq` 与 `mse_ptq` / `mae_pt_ptq` 等字段；累计 `running_*_ptq` 统计供前端按维表使用。

### ptq_trt_compare（PTQ vs TRT）管线差异

- **两路**：pred1 为 *同一份 PyTorch policy 就地应用 PTQ（fake quant）*；pred2 为 *另一份 policy 挂载 TensorRT engine*。
- **指标**：meta 中会携带 `pred1_name=PTQ`、`pred2_name=TRT`、`pair_name=PTQ−TRT`，前端会据此把 `pred_pt/pred_trt` 标签替换为 `pred_ptq/pred_trt` 并显示 `PTQ−TRT`。

## WebSocket / 前端

- **meta**：`ptq_compare`、`ptq_parts`、`ptq_quant_cfg`、`ptq_calib_dir`、`ptq_layer_report_path`（可选）、`backend` 为 `pytorch+ptq`。
- **step**：`metrics` 中含 `mse_pct_dim_mean_ptq`、`rel_p99_dim_ptq`、`mse_pt_ptq_dim_mean` 等与 TRT 侧 `*_trt` 平行的键；第三路动作用 `pred_action_ptq`。
- **浏览器**（`webui_client/app.js`）：`ptq_compare` 与 `compare_mode` 互斥；第三路曲线仍沿用 “pred_trt” 样式名，数据来自 PTQ；对比表/误差曲线标题在 PTQ 下显示为 PT−PTQ。

## 校准数据

各子系统使用与训练侧一致的组件名：

- `vit` → `component="pi05_vit"`
- `llm` → `component="pi05_llm"`
- `expert` → `component="pi05_expert"`

可用与 **`--calib-save-path`** 相同流程先收数据，再将该目录传给 `--ptq-calib-dir`。

## Layer 报告 JSON 格式

顶层字段包括：`parts`、`indices_start`、`indices_used`、`layer_count`、`layers`。`layers` 为按 `mse_mean` 降序的列表，每项含：

- `module`：`named_modules` 完整路径  
- `mse_mean` / `mae_mean` / `max_abs_mean`  
- `samples`：该层有效累计次数（缺 hook 或 shape 不一致的步会跳过）

若未找到任何 QuantLinear，则 `layers` 为空并带说明字段。

## 相关代码入口

| 文件 | 作用 |
|------|------|
| `lerobot_eval_webui/ptq_compare.py` | 加载 quant cfg、选择性 PTQ、写 layer report |
| `lerobot_eval_webui/bundle.py` | 第二 policy、校验、meta、bundle 统计槽位 |
| `lerobot_eval_webui/chunk_infer.py` | 双路 infer 与 step metrics |
| `lerobot_eval_webui/protocol.py` | `StepEvent.pred_action_ptq` |
| `src/model_optimizer/models/pi05/{vit,llm,expert}.py` | 参考 `quantize` 与 `get_calibrate_dataset` |

## 资源与注意

- **显存**：双份全量权重 + 量化状态，约为单 policy 的两倍左右（随配置变化）。
- **NVFP4**：与现网 `quantize` 一致会走 `set_dynamic_quant`；本模式**不**跑 ONNX NVFP4 后处理。
- **日志**：`Expert.forward` 等处若为 `INFO` 会刷屏，评估时可按需调高 logger 级别。
