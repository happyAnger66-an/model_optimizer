# Pi0.5 LLM KV-only ONNX TensorRT / model-opt build 配置。
#
# 配套 feature config：
#   config/feature_configs/llm_kv_only.json
#
# 该 feature config 默认：
#   - export.llm_kv_only=true      ：ONNX 只输出 past_keys / past_values，不输出 last_hidden_state
#   - export.llm_static_shape=true : ONNX 输入序列维为静态
#   - export.llm_seq_len=968       : prefix = 3 * 256 image tokens + 200 language tokens
#
# 因此 TensorRT profile 的 min/opt/max 必须与 ONNX 静态维完全一致。
# 若继续使用 llm_build_cfg.py（SEQ_LEN=818），会触发：
#   Input tensor inputs_embeds has static dimensions that don't match kMIN dimensions
#   Input dimensions are [1,968,2048] but profile dimensions are [1,818,2048]

SEQ_LEN = 968

build_cfg = {
    "precision": "bf16",
    "workspace_mb": 8192,
    "min_shapes": {
        "inputs_embeds": (1, SEQ_LEN, 2048),
        "attention_mask": (1, 1, SEQ_LEN, SEQ_LEN),
        "position_ids": (1, SEQ_LEN),
    },
    "opt_shapes": {
        "inputs_embeds": (1, SEQ_LEN, 2048),
        "attention_mask": (1, 1, SEQ_LEN, SEQ_LEN),
        "position_ids": (1, SEQ_LEN),
    },
    "max_shapes": {
        "inputs_embeds": (1, SEQ_LEN, 2048),
        "attention_mask": (1, 1, SEQ_LEN, SEQ_LEN),
        "position_ids": (1, SEQ_LEN),
    },
}
