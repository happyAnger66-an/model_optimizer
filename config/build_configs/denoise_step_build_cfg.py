# Pi05DenoiseStep（denoise.onnx）TensorRT / model-opt build 配置。
# 须与导出 ONNX 的 input_names、张量秩一致；层数须与 checkpoint 中 gemma expert 一致（此处按 18 层示例）。
#
# 输入语义（与 src/model_optimizer/models/pi05/dit.py forward 一致）：
#   prefix_pad_masks  (B, L_prefix)     与 LLM prefix 有效 token 对齐；ONNX 常为 bool，引擎侧按实际类型绑定
#   past_keys         (num_layers, B, L_prefix, head_dim)
#   past_values       同 past_keys
#   x_t               (B, action_horizon, action_dim)  当前噪声动作；默认 float32
#   timestep          (B,)                           flow 时间标量；默认 float32
#
# 默认形状对应 pi05_libero 常见设置：L_prefix=968，action_horizon=10，action_dim=32，head_dim=256。
# 若模型 config 不同，请同步修改下列元组。

_NUM_LAYERS = 18
_PREFIX_LEN = 968
_ACTION_HORIZON = 10
_ACTION_DIM = 32
_HEAD_DIM = 256

build_cfg = {
    "precision": "bf16",
    "strongly_typed_network": True,
    "workspace_mb": 8192,
    # Do not set layer_precision_overrides for /action_in_proj to fp32: ONNX keeps MatMul activations in
    # bf16 while bias may stay fp32; TRT then fails on Add(MatMul, bias) with "types Float and BFloat16".
    "min_shapes": {
        "prefix_pad_masks": (1, _PREFIX_LEN),
        "past_keys": (_NUM_LAYERS, 1, _PREFIX_LEN, _HEAD_DIM),
        "past_values": (_NUM_LAYERS, 1, _PREFIX_LEN, _HEAD_DIM),
        "x_t": (1, _ACTION_HORIZON, _ACTION_DIM),
        "timestep": (1,),
    },
    "opt_shapes": {
        "prefix_pad_masks": (1, _PREFIX_LEN),
        "past_keys": (_NUM_LAYERS, 1, _PREFIX_LEN, _HEAD_DIM),
        "past_values": (_NUM_LAYERS, 1, _PREFIX_LEN, _HEAD_DIM),
        "x_t": (1, _ACTION_HORIZON, _ACTION_DIM),
        "timestep": (1,),
    },
    "max_shapes": {
        "prefix_pad_masks": (1, _PREFIX_LEN),
        "past_keys": (_NUM_LAYERS, 1, _PREFIX_LEN, _HEAD_DIM),
        "past_values": (_NUM_LAYERS, 1, _PREFIX_LEN, _HEAD_DIM),
        "x_t": (1, _ACTION_HORIZON, _ACTION_DIM),
        "timestep": (1,),
    },
}
