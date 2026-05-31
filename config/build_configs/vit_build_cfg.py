# Pi0.5 SigLIP ViT（vit.onnx）TensorRT / model-opt build 配置。
#
# 多视角 batching：把 N 个相机视角堆成 batch 维一次过引擎（SigLIP 各图独立、无跨图注意力，
# 数值等价于逐视角）。需配合 vit.py 导出放开 pixel_values 的 batch 动态轴，以及执行器开关
# config.vit_batch_views=True。
#   - min batch=1   兼容单视角调试；
#   - opt/max batch=_NUM_VIEWS  按真实工作点选 tactic（pi05_libero 默认 2 视角）。
# 若部署视角数不同，改 _NUM_VIEWS 即可。

_NUM_VIEWS = 3

build_cfg = {
    "precision": "bf16",
    "workspace_mb": 8192,
    "min_shapes": {
        "pixel_values": (1, 3, 224, 224),
    },
    "opt_shapes": {
        "pixel_values": (_NUM_VIEWS, 3, 224, 224),
    },
    "max_shapes": {
        "pixel_values": (_NUM_VIEWS, 3, 224, 224),
    },
}
