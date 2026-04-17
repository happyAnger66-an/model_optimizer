# Pi0.5 整图 ONNX（ONNXWrapper → sample_actions）TensorRT / model-opt build 配置。
# 须与 ``src/model_optimizer/models/pi05/utils/whole_export.py`` 中
# ``export_whole_model_to_onnx`` 的 ``input_names`` / ``output_names``、张量秩一致。
#
# ONNX I/O（与 ``torch.onnx.export`` 一致）：
#   images      (B, 9, 224, 224)   float ，9 = len(IMAGE_KEYS)*3，与 ``create_dummy_inputs`` 堆叠格式一致
#   img_masks   (B, 3)            bool ，每路相机是否有效
#   lang_tokens (B, L)            int64
#   lang_masks  (B, L)            bool
#   state       (B, action_dim)   float
#   noise       (B, action_horizon, action_dim)  float
#
#   actions     (B, action_horizon, action_dim)   float （profile 一般只配输入；此处不写入 build_cfg）
#
# 默认 L / action 维与 openpi ``TrainConfig(name="pi05_libero")`` 的 ``Pi0Config(pi05=True, …)`` 一致：
#   max_token_len=200，action_horizon=10，action_dim=32。
# 若 checkpoint 修改了 ``model.config`` 中对应字段，请同步修改下列常量与各元组。
#
# 关于动态轴：导出侧对 ``lang_*`` 标记了 ``seq_len`` 动态轴；若实际 ONNX 将 L 固化为常数，
# TensorRT 要求 profile 在该维上 min==opt==max（与 ``embed_prefix_build_cfg.py`` 说明相同）。
# 这里将 token 维固定为 200；若你导出的 ONNX 上 L 真正动态，再为 ``lang_tokens`` / ``lang_masks``
# 设置不同的 min/opt/max。
#
# 精度：当前 ``export_whole_model_to_onnx`` 使用 ``compute_dtype=torch.float16`` 打补丁导出；
# 若你改为 bf16 导出，请把 ``precision`` 改为 ``bf16`` 并与 ONNX 对齐。

_NUM_VIEWS = 3
_NUM_IMAGE_CH = _NUM_VIEWS * 3  # 与 whole_export 中按相机拼接的通道数一致
_H, _W = 224, 224

_MAX_TOKEN_LEN = 200
_ACTION_HORIZON = 10
_ACTION_DIM = 32

_BATCH_MIN = 1
_BATCH_OPT = 1
_BATCH_MAX = 4

_TOKEN_MIN = _MAX_TOKEN_LEN
_TOKEN_OPT = _MAX_TOKEN_LEN
_TOKEN_MAX = _MAX_TOKEN_LEN


def _images(b: int):
    return (b, _NUM_IMAGE_CH, _H, _W)


def _img_masks(b: int):
    return (b, _NUM_VIEWS)


def _lang(b: int, tok: int):
    return (b, tok)


def _state(b: int):
    return (b, _ACTION_DIM)


def _noise_or_actions(b: int):
    return (b, _ACTION_HORIZON, _ACTION_DIM)


def _profile(batch_min: int, batch_opt: int, batch_max: int, tok_min: int, tok_opt: int, tok_max: int):
    return {
        "images": _images(batch_min),
        "img_masks": _img_masks(batch_min),
        "lang_tokens": _lang(batch_min, tok_min),
        "lang_masks": _lang(batch_min, tok_min),
        "state": _state(batch_min),
        "noise": _noise_or_actions(batch_min),
    }, {
        "images": _images(batch_opt),
        "img_masks": _img_masks(batch_opt),
        "lang_tokens": _lang(batch_opt, tok_opt),
        "lang_masks": _lang(batch_opt, tok_opt),
        "state": _state(batch_opt),
        "noise": _noise_or_actions(batch_opt),
    }, {
        "images": _images(batch_max),
        "img_masks": _img_masks(batch_max),
        "lang_tokens": _lang(batch_max, tok_max),
        "lang_masks": _lang(batch_max, tok_max),
        "state": _state(batch_max),
        "noise": _noise_or_actions(batch_max),
    }


_min, _opt, _max = _profile(
    _BATCH_MIN,
    _BATCH_OPT,
    _BATCH_MAX,
    _TOKEN_MIN,
    _TOKEN_OPT,
    _TOKEN_MAX,
)

build_cfg = {
    "precision": "fp16",
    "workspace_mb": 8192,
    "min_shapes": _min,
    "opt_shapes": _opt,
    "max_shapes": _max,
}
