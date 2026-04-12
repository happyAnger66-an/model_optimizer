#!/usr/bin/env python3
"""
LeRobot 离线评估 WebUI（client-server）：

- Server 通过 WebSocket **流式**推送“时间对齐后的 step 事件”：一次推理产生 action_horizon 个 steps，
  每个 step 对齐到数据集 label 的同一行（idx+k）。
- 事件包含：episode_id/global_index/k_in_chunk、gt/pred action、误差指标、prompt、RGB 图像（默认 base，可选 wrist）。
- Client 侧用浏览器订阅展示（见 ./webui_client/）。

协议（v1）：

- ``type="meta"``：连接建立后 server 先发 1 条元数据
- ``type="step"``：按时间顺序持续推送 step
- ``type="done"``：本 run 评估区间内 step 已全部推送；随后 server 关闭 WebSocket 并 **进程退出**
- ``type="log"``：可选日志事件
- ``type="control"``（client→server）：``{"type":"control","action":"pause"|"resume"}``，在**下一个 chunk 推理前**暂停/继续；server 回复 ``type=control_ack``

可选 **PTQ 对比**（``--ptq-compare``）：双路 PyTorch（浮点 vs 选择性 PTQ），协议字段见 ``docs/quantize_ptq_compare.md``。

可选 **TensorRT vs ONNX Runtime**（``--trt-ort-compare``）：双路引擎对比，需同时 ``--engine-path`` 与 ``--ort-engine-path``；与其它 compare 标志互斥。

实现已拆分为 ``lerobot_eval_webui`` 包（同目录下），本文件仅作兼容入口。
"""

from __future__ import annotations

from lerobot_eval_webui import main

if __name__ == "__main__":
    main()
