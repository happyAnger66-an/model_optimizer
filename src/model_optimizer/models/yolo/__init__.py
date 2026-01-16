from ultralytics import YOLO

from ..model import Model

from ultralytics.nn.tasks import SegmentationModel
from model_optimizer.calibrate.yolo_datas import YoLoCalibrationData
from model_optimizer.utils.utils import load_quant_json, normalize_quant_cfg
from model_optimizer.torch_hooks.hooks import hook_module_inputs

def hook_yolo_train_method(model):
    print(f'hook yolo train called in quantization. Do nothing...')
    return model

class YoloModel(Model):
    def __init__(self, model_name, model_path):
        super().__init__(model_name, model_path)
        self.model = YOLO(model_path)

    def load(self, config):
        self.model = self._get_yolo_model(self.model_name, self.model_path)

    def _get_yolo_model(self):
        return self.model

    def quantize(self, quant_cfg, calib_data, calib_method):
        pass

    def onnx_quantize(self, quant_cfg, quant_mode, calib_data, export_dir):
        from modelopt.onnx.quantization import quantize
        model = self.model
        model.eval()
        original_train_method = model.train
        model.train = hook_yolo_train_method
        
        from model_optimizer.collector.collector import YOLOCalibCollector
        yolo_calib_collector = YOLOCalibCollector()
        yolo_calib_collector.start_collect(SegmentationModel, model)

        calib_data = YoLoCalibrationData(calib_data)
        for data in calib_data:
            model(data)
        
        yolo_calib_collector.stop_collect()

        quantize(
            onnx_path=self.model_path,
            quantize_mode=quant_mode,       # fp8, int8, int4 etc.
            calibration_data=yolo_calib_collector.datas,
            calibration_method=quant_cfg["algorithm"],   # max, entropy, awq_clip, rtn_dq etc.
            output_path=f"{export_dir}/{self.model_name}_quant.onnx",
        )
        print(f'quantize done.')