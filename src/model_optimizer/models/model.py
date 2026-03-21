import torch
from functools import partial
import torch.nn as nn

import modelopt.torch.quantization as mtq

from modelopt.onnx.quantization.qdq_utils import fp4qdq_to_2dq
from model_optimizer.utils.utils import is_fp4_quantized
import time
import os
from termcolor import colored
import onnx

class Model:
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path
        self.val_datas_before = []
        self.val_datas_after = []
        self.is_quantized = False

    def simplifier_model(self, model_path, output_dir):
        raise NotImplementedError

    def get_calibrate_dataset(self, calib_data_file):
        datas = torch.load(calib_data_file)
        return datas

    def load(self, config):
        raise NotImplementedError

    def val(self, dataset, batch_size, max_data = 100, output_dir=None):
        raise NotImplementedError(
            f'{self.model_name} val method is not implemented')

    @classmethod
    def construct_from_name_path(cls, model_name, model_path):
        return cls(model_name, model_path)

    def export_onnx(self, *args, **kwargs):
        raise NotImplementedError

    def quantize_start(self, quant_cfg, calib_data, calib_method):
        pass

    def quantize_end(self, export_dir):
        pass

    def _nvfp4_post_processing(self, onnx_path, export_dir):
        with torch.inference_mode():
            self.model.save_pretrained(export_dir)

#        onnx_path = f"{export_dir}/llm.onnx"
        if is_fp4_quantized(self):
            t1 = time.time()
            onnx.shape_inference.infer_shapes_path(onnx_path)
            onnx_model = onnx.load(onnx_path)
            graph = None

            print(
                colored(
                    "NVFP4 quantization detected in the model, \
                        compressing some weights to NVFP4", "green")
            )
            onnx_model = fp4qdq_to_2dq(onnx_model)
            print(
                colored(
                    "Removing all the files in the output directory except for .json files",
                    "green"
                )
            )
            for file in os.listdir(export_dir):
                if file.endswith(".json"):
                    continue
                os.remove(os.path.join(export_dir, file))
            onnx.save_model(onnx_model,
                            onnx_path,
                            save_as_external_data=True,
                            all_tensors_to_one_file=True,
                            location="onnx_model.data",
                            convert_attribute=True)
            t2 = time.time()
            print(
                colored(
                    f"NVFP4 quantization post processing cost:{t2 - t1}s", "green"
                )
            )

    def quantize(self, quant_cfg, calib_data, export_dir):
        self.quantize_start(quant_cfg, calib_data, None)

        calibrate_loop = self.get_model_calibrate_loop(calib_data)
        mtq.quantize(self.model, quant_cfg,
                     forward_loop=calibrate_loop)
        print(f'quantize summary')
        mtq.print_quant_summary(self.model)

        self.quantize_end(export_dir)

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
