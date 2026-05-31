# TensorRT-Edge-LLM AttentionPlugin 在 ONNX 中多为 FP16；若 bf16 建引擎报
# "doesn't report any supported format combinations"，请改为 "fp16"。
SEQ_LEN = 818
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
    #"plugin_lib_paths": [
    #    "/srcs/codes/llmOnEdge/build-fmha/libNvInfer_edgellm_plugin.so"
    #],

    # Debug：把中间 tensor 也标为 engine 输出（便于做 TRT layer report / 对齐 PyTorch）。
    # 1) 先把 debug_dump_tensor_names=True 跑一次 build，在日志里找到你要的 tensor 名；
    # 2) 把它们填到 debug_output_tensors；重新 build 得到带额外 outputs 的 engine。
    # 注意：outputs 越多，engine 越慢/越占显存；仅用于定位问题。
    # "debug_dump_tensor_names": True,
    # "debug_output_tensors": [
    #     "some_tensor_name_in_network",
    # ],
}