import os
import argparse

from ..progress.write import write_quantize_progress

def calibrate(model_path, dataset_dir, export_dir):
    from ultralytics import YOLO
    model = YOLO(model_path, task="segment")

    model.eval()
    original_train_method = model.train
    model.train = hook_yolo_train_method

    from ..collector.collector import YOLOCalibCollector
    from .yolo_datas import YoLoCalibrationData
    from ultralytics.nn.tasks import SegmentationModel
    yolo_calib_collector = YOLOCalibCollector()
    yolo_calib_collector.start_collect(SegmentationModel, model)

    percent = 20
    write_quantize_progress(export_dir, percent, 2, 3, percent, 100)
    calib_data = YoLoCalibrationData(dataset_dir)
    for idx, data in enumerate(calib_data):
        print(f'calibrate: {idx}')
        model(data)
        percent+=1
        write_quantize_progress(export_dir, percent, 2, 3, percent, 100)

    yolo_calib_collector.stop_collect()

    import numpy as np
    np.save(f'{export_dir}/calibrate.npy', yolo_calib_collector.datas["images"])
    print(f'calibrate datas saved to {export_dir}/calibrate.npy')
    write_quantize_progress(export_dir, 80, 2, 3, 80, 100)



def calibrate_cli(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--export_dir', type=str, required=True)
    args = parser.parse_args(args[1:])
    print(f'[cli] calibrate args {args}')

    calibrate(os.path.join(args.model_path, args.model_name),
              args.dataset_dir, args.export_dir)


def hook_yolo_train_method(mode):
    print(f'hook yolo train called in quantization. Do nothing...')
    pass
