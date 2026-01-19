from functools import partial
import torch.nn as nn

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq


class Model:
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path

    def simplifier_model(self, model_path, output_dir):
        raise NotImplementedError

    def load(self, config):
        raise NotImplementedError

    @classmethod
    def construct_from_name_path(cls, model_name, model_path):
        return cls(model_name, model_path)

    def export_onnx(self, *args, **kwargs):
        raise NotImplementedError

    def quantize_start(self, quant_cfg, calib_data, calib_method):
        pass

    def quantize_end(self):
        pass

    def quantize(self, quant_cfg, calib_data, calib_method):
        self.quantize_start(quant_cfg, calib_data, calib_method)

        calibrate_loop = self.get_model_calibrate_loop(calib_data)
        mto.quantize(self.model, quant_cfg,
                     forward_loop=calibrate_loop)
        print(f'quantize summary')
        mto.print_quant_summary(self.model)

        self.quantize_end()

        # from model_optimizer.quantization.quantization_utils import quantize_model
        # quantize_model(self, quant_cfg, calib_data, calib_method)

    def get_model_calibrate_loop(self, calib_data):
        datasets = self.get_calibrate_dataset(calib_data)

        def calibrate_loop(model):
            for idx, data in enumerate(datasets):
                if data is None:
                    print('calibrate data is None')
                    continue

                print(f'calibration idx {idx} data:{data}')
                model(data)

        return partial(calibrate_loop, self.model)
