import argparse

import torch
from torch import nn

from ultralytics import SAM
from ultralytics.models.sam.predict import SAM2Predictor

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

from model_optimizer.calibrate.yolo_datas import ImageDatasets
from model_optimizer.collector.collector import SamCalibCollector

#from export_onnx_model import export_onnx_model

def hook_sam_train_method(mode):
    print(f'hook sam train called in quantization. Do nothing...')
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--calib_data', type=str, required=True)
    parser.add_argument('--export_dir', type=str, required=True)
    args = parser.parse_args()

    # load a pretrain model
    model_type = args.model_path.rsplit('.')[-1]
    model = SAM(args.model_path)
    print(f'init model {model}')

#    sam_calib_collector = SamCalibCollector()
#    sam_calib_collector.start_collect(SAM2Predictor, model.model)

#    for data in calib_data:
#        print(f'data {data}')
#        model(data)
    
#    sam_calib_collector.stop_collect()

    calib_data = ImageDatasets(args.calib_data)
    def calibrate_loop(model):
        for idx, data in enumerate(calib_data):
            if data is None:
                print('calibrate data is None')
                continue

            print(f'calibration idx {idx} data:{data}')
            model(data)

    original_train_method = model.train
    model.train = hook_sam_train_method
    mtq.quantize(model, mtq.INT8_DEFAULT_CFG,
                 forward_loop=calibrate_loop)
    print(f'quantize summary')
    mtq.print_quant_summary(model)

    dummy_x = torch.randn(1, 3, 1024, 1024, dtype=torch.float32).cuda()
    model.train = original_train_method
    model.export(format="onnx", dynamic=True, simplify=True)
#    torch.onnx.export(
#        model,
#        dummy_x,
#        f'{args.export_dir}/sam.onnx',
#        input_names = ["x"],
#        output_names=['pred_x'],
#        dynamic_axes={'x': {0: 'batch_size'}},
#        dynamo=True
#    )
    print(f'export to {args.export_dir}/sam.onnx done.')