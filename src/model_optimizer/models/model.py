import torch.nn as nn


class Model:
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path

    def load(self, config):
        raise NotImplementedError

    def export_onnx(self, *args, **kwargs):
        raise NotImplementedError

    def quantize(self, quant_cfg, calib_data, calib_method):
        from model_optimizer.quantization.quantization_utils import quantize_model
        quantize_model(self, quant_cfg, calib_data, calib_method)
