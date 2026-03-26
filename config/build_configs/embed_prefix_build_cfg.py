# Pi05EmbedPrefix（embed_prefix.onnx）TensorRT / model-opt build 配置。
# 须与 src/model_optimizer/models/pi05/embed_prefix.py 导出时的 input_names、秩一致。
#
# 输入语义（与 Pi05EmbedPrefix.forward 一致，LIBERO 默认 3 路相机）：
#   image_{i}       (B, 3, 224, 224)  float ，与 openpi 预处理一致
#   image_mask_{i}  (B,)              bool ，该路图像是否有效
#   lang_tokens     (B, L)            int64，token 序列
#   lang_masks      (B, L)            bool ，与 prompt padding 对齐
#
# 输出（仅解析 ONNX 时用，profile 一般只设输入）：
#   prefix_embs / prefix_pad_masks / prefix_att_masks  序列维约 768 + L（SigLIP patch × 路数 + L）。
#
# 默认 L 与 pi05_libero 的 max_token_len=200 对齐；单 batch 前缀总长 968（3×256+200）。
# 若改 checkpoint / dataset 的路数或 max_token_len，请同步修改常量与下列元组。

_NUM_VIEWS = 3
_C, _H, _W = 3, 224, 224

# 与 openpi Pi0Config（pi05）默认 max_token_len 一致
_MAX_TOKEN_LEN = 200

# 动态 batch / token 维：无 multi-batch 需求时可令 MIN=OPT=MAX
_BATCH_MIN = 1
_BATCH_OPT = 1
_BATCH_MAX = 4

_TOKEN_MIN = 1
_TOKEN_OPT = _MAX_TOKEN_LEN
_TOKEN_MAX = _MAX_TOKEN_LEN


def _image(b: int):
    return (b, _C, _H, _W)


def _img_mask(b: int):
    return (b,)


def _build_io_dict(batch_min: int, batch_opt: int, batch_max: int, tok_min: int, tok_opt: int, tok_max: int):
    d = {}
    for i in range(_NUM_VIEWS):
        d[f"image_{i}"] = _image(batch_min)
        d[f"image_mask_{i}"] = _img_mask(batch_min)
    d["lang_tokens"] = (batch_min, tok_min)
    d["lang_masks"] = (batch_min, tok_min)

    d_opt = {}
    for i in range(_NUM_VIEWS):
        d_opt[f"image_{i}"] = _image(batch_opt)
        d_opt[f"image_mask_{i}"] = _img_mask(batch_opt)
    d_opt["lang_tokens"] = (batch_opt, tok_opt)
    d_opt["lang_masks"] = (batch_opt, tok_opt)

    d_max = {}
    for i in range(_NUM_VIEWS):
        d_max[f"image_{i}"] = _image(batch_max)
        d_max[f"image_mask_{i}"] = _img_mask(batch_max)
    d_max["lang_tokens"] = (batch_max, tok_max)
    d_max["lang_masks"] = (batch_max, tok_max)

    return d, d_opt, d_max


_min, _opt, _max = _build_io_dict(
    _BATCH_MIN,
    _BATCH_OPT,
    _BATCH_MAX,
    _TOKEN_MIN,
    _TOKEN_OPT,
    _TOKEN_MAX,
)

build_cfg = {
    "precision": "bf16",
    "workspace_mb": 8192,
    "min_shapes": _min,
    "opt_shapes": _opt,
    "max_shapes": _max,
}
