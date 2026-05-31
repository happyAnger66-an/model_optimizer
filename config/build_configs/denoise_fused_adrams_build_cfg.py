# Pi05DenoiseStep（denoise.onnx）TensorRT / model-opt build 配置：
#   AdaRMS Dense 预计算 + fused MLP（gate/up 合并）。
#
# 重要：fused_mlp 只把每层 gate/up 两个 MatMul 合并为一个更大的 gate_up_proj MatMul，
# **不改变 ONNX 的输入/输出签名**，因此本配置的 I/O 形状与
# ``denoise_static_adrams_build_cfg.py`` 完全一致；二者可互换使用。单列一份仅为命名清晰。
#
# 输入语义（与 src/model_optimizer/models/pi05/dit.py forward 一致）：
#   prefix_pad_masks  (B, L_prefix)
#   past_keys         (num_layers, B, L_prefix, head_dim)
#   past_values       同 past_keys
#   x_t               (B, action_horizon, action_dim)            当前噪声动作；float32
#   adarms_mod        (num_norms, B, dim*3)                       预计算 AdaRMS modulation；float32
#
#   - num_norms = 2 * num_layers + 1 = 2*18+1 = 37
#   - dim*3     = expert hidden_size(1024) * 3 = 3072
#   这两个值会在 export 日志打印：``[adarms] precompute export: N norms × dim*3=D``，请据实核对。

_NUM_LAYERS = 18
_PREFIX_LEN = 818
_ACTION_HORIZON = 10
_ACTION_DIM = 32
_HEAD_DIM = 256
_NUM_NORMS = 2 * _NUM_LAYERS + 1  # 37，须与 export 日志的 N 一致
_DIM3 = 1024 * 3                  # 3072，= expert width(1024) * 3，须与 export 日志的 D 一致


def _shapes():
    return {
        "prefix_pad_masks": (1, _PREFIX_LEN),
        "past_keys": (_NUM_LAYERS, 1, _PREFIX_LEN, _HEAD_DIM),
        "past_values": (_NUM_LAYERS, 1, _PREFIX_LEN, _HEAD_DIM),
        "x_t": (1, _ACTION_HORIZON, _ACTION_DIM),
        "adarms_mod": (_NUM_NORMS, 1, _DIM3),
    }


build_cfg = {
    "precision": "bf16",
    "strongly_typed_network": True,
    "workspace_mb": 8192,
    # 勿对 /action_in_proj 设 fp32 layer_precision_overrides：ONNX 中其 MatMul 激活为 bf16、
    # bias 也保持 bf16，二者匹配；强行 fp32 会触发 Add(MatMul, bias) 的类型冲突。
    "min_shapes": _shapes(),
    "opt_shapes": _shapes(),
    "max_shapes": _shapes(),
}
