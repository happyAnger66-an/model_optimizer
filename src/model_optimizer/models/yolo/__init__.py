import os
import shutil

from ultralytics import YOLO

from ..model import Model

from ultralytics.nn.tasks import SegmentationModel
from model_optimizer.calibrate.yolo_datas import YoLoCalibrationData
from model_optimizer.utils.utils import load_quant_json, normalize_quant_cfg
from model_optimizer.torch_hooks.hooks import hook_module_inputs

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

from model_optimizer.evaluate.metrics.yolo_metric import YoloMetric
from onnx2pytorch import ConvertModel

def hook_yolo_train_method(model):
    print(f'hook yolo train called in quantization. Do nothing...')
    return model


class YoloModel(Model):
    def __init__(self, model_name, model_path):
        super().__init__(model_name, model_path)
        self.is_onnx = False
        if self.model_path.endswith('.onnx'):
            self.is_onnx = True
            pt_model = ConvertModel(self.model_path)
            print(f'{pt_model.named_modules()}')
        self.model = YOLO(model_path, task="segment")
        self.original_train_method = None
        self.onnx_path = None
        self.onnx_quantize_path = None
        self.is_quantized = False

    def load(self, config):
        self.model = self._get_yolo_model(self.model_name, self.model_path)

    def _get_yolo_model(self):
        return self.model

    def quantize(self, quant_cfg, calib_data, calib_method, export_dir, input_shapes=None):
        # super().quantize(quant_cfg, calib_data, calib_method)
        if self.model_path.endswith('.onnx'):
            self.onnx_path = self.model_path
        else:
            self.onnx_path = self.model.export(
                format="onnx", dynamic=False, simplify=True)
            print(f'export onnx model {self.onnx_path} from {self.model_path}')
        self.onnx_quantize(quant_cfg, calib_data, export_dir, input_shapes)

    def val(self, val_data, batch_size, output_dir):
#        print(f'val {val_data}')
        if self.is_quantized:
            return self.val_onnx(val_data, batch_size, output_dir)
        else:
            # kwargs = {"save_dir": output_dir, "save_txt": True}
            kwargs = {"save_dir": output_dir}
            metrics = self.model.val(data=val_data, imgsz=640, **kwargs)
            return YoloMetric(metrics)

    def val_onnx(self, val_data, batch_size, output_dir):
        onnx_model = YOLO(self.onnx_quantize_path, task="segment")
        kwargs = {"save_dir": output_dir}
        metrics = onnx_model.val(data=val_data, imgsz=640, **kwargs)
        return YoloMetric(metrics)

    def get_calibrate_dataset(self, calib_data):
        return YoLoCalibrationData(calib_data)

    def quantize_start(self, quant_cfg, quant_mode, calib_data):
        self.model.eval()
        self.original_train_method = self.model.train
        self.model.train = hook_yolo_train_method

    def quantize_start(self, quant_cfg, quant_mode, calib_data, export_dir):
        self.model.eval()
        self.original_train_method = self.model.train
        self.model.train = hook_yolo_train_method
        
        def hook_input(model, arg, kwargs):
            print(
                f'hook input {type(arg)} {len(arg)} {arg[0].shape} {arg[0].dtype} kwargs:{kwargs}')
        hook_module_inputs(self.model, hook_input, SegmentationModel)

    def quantize_end(self, export_dir):
        self.model.train = self.original_train_method

    def export(self, export_dir):
        save_path = self.model.export(
            format="onnx", dynamic=False, simplify=True)
        export_model_name = os.path.basename(self.model_path).split('.')[0]
        export_model_path = f"{export_dir}/{export_model_name}.onnx"
        shutil.move(save_path, export_model_path)
        return export_model_path

    def onnx_quantize(self, quant_cfg, calib_data, export_dir, input_shapes=None):
        from modelopt.onnx.quantization import quantize
        model = self.model
        model.eval()
        original_train_method = model.train
        model.train = hook_yolo_train_method

        from model_optimizer.calibrate.collector.collector import YOLOCalibCollector
        yolo_calib_collector = YOLOCalibCollector(input_shapes)
        yolo_calib_collector.start_collect(SegmentationModel, model)

        calib_data = YoLoCalibrationData(calib_data)
        for data in calib_data:
            model(data)

        yolo_calib_collector.stop_collect()

        print(
            f'onnx quantize {self.onnx_path} {quant_cfg["mode"]} {quant_cfg["algorithm"]} to {export_dir}')
        self.onnx_quantize_path = f"{export_dir}/{self.model_name}_quant.onnx"
        quantize(
            onnx_path=self.onnx_path,
            quantize_mode=quant_cfg["mode"],       # fp8, int8, int4 etc.
            calibration_data=yolo_calib_collector.datas,
            # max, entropy, awq_clip, rtn_dq etc.
            calibration_method=quant_cfg["algorithm"],
            output_path=self.onnx_quantize_path
        )
        self.is_quantized = True
        print(f'quantize done.')
