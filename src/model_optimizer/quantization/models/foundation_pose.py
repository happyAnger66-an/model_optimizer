from torch.utils.data import DataLoader, Dataset
import os
import logging

import numpy as np
import torch

from omegaconf import OmegaConf
from learning.models.score_network import ScoreNetMultiPair
from learning.models.refine_network import RefineNet

import modelopt.torch.quantization as mtq


def get_cfg(model_dir, model_name='model_best.pth'):
    ckpt_dir = f'{model_dir}/{model_name}'

    cfg = OmegaConf.load(f'{model_dir}/config.yml')

    cfg['ckpt_dir'] = ckpt_dir
    cfg['enable_amp'] = True

    # Defaults, to be backward compatible
    if 'use_normal' not in cfg:
        cfg['use_normal'] = False
    if 'use_BN' not in cfg:
        cfg['use_BN'] = False
    if 'zfar' not in cfg:
        cfg['zfar'] = np.inf
    if 'c_in' not in cfg:
        cfg['c_in'] = 4
    if 'normalize_xyz' not in cfg:
        cfg['normalize_xyz'] = False
    if 'crop_ratio' not in cfg or cfg['crop_ratio'] is None:
        cfg['crop_ratio'] = 1.2

    print(f"cfg: \n {OmegaConf.to_yaml(cfg)}")
    return cfg, ckpt_dir


class FoundationPoseCalibrationData(Dataset):
    def __init__(self, calib_file,
                 calib_size: int = 300):
        self.calib_inputs = torch.load(calib_file)
        self.calib_size = min(calib_size, len(self.calib_inputs))

    def __len__(self):
        return self.calib_size

    def __getitem__(self, idx):
        return self.calib_inputs[idx]

def export_onnx(model, save_dir):
    dummy_A = torch.randn((1, 6, 160, 160), dtype=torch.float32).cuda()
    dummy_B = torch.randn((1, 6, 160, 160), dtype=torch.float32).cuda()
    torch.onnx.export(
        model,
        (dummy_A, dummy_B),
        f'{save_dir}/model_origin.onnx',
        input_names=['A', 'B'],
        output_names=['pred_xyz'],
        dynamic_axes={'A': {0: 'batch_size'},
                    'B': {0: 'batch_size'},
                    'pred_xyz': {0: 'batch_size'}},
    )

def quantize_model(model, calib_data_file, save_dir):
    quant_cfg = mtq.INT8_DEFAULT_CFG
    # Disable Conv to avoid accuracy degradation.
#    quant_cfg["quant_cfg"]["nn.Conv2d"] = {"*": {"enable": True}}
#    quant_cfg["quant_cfg"]["nn.MultiheadAttention"] = {"*": {"enable": False}}
#    quant_cfg["quant_cfg"]["nn.Linear"] = {"*": {"enable": False}}
    print(f'quant cfg {quant_cfg}')

    calib_data = FoundationPoseCalibrationData(calib_data_file)

    def calibrate_loop(model):
        for idx, data in enumerate(calib_data):
            if data is None:
                print('calibrate data is None')
                continue

            print(f'calibration idx {idx}')
            A, B = data
            A = torch.from_numpy(A).cuda()
            B = torch.from_numpy(B).cuda()
            model(A, B)

#    with torch.no_grad():
    mtq.quantize(model, quant_cfg,
                 forward_loop=calibrate_loop)
    print(f'quantize summary')
    mtq.print_quant_summary(model)

    os.makedirs(save_dir, exist_ok=True)
    dummy_A = torch.randn((1, 6, 160, 160), dtype=torch.float32).cuda()
    dummy_B = torch.randn((1, 6, 160, 160), dtype=torch.float32).cuda()
#    with torch.no_grad():
    torch.onnx.export(
        model,
        (dummy_A, dummy_B),
        f'{save_dir}/model_quantized.onnx',
        input_names=['A', 'B'],
        output_names=['pred_xyz'],
        dynamic_axes={'A': {0: 'batch_size'},
                      'B': {0: 'batch_size'},
                      'pred_xyz': {0: 'batch_size'}},
    )


if __name__ == "__main__":
    import sys
    net_type = int(sys.argv[2])
    cfg, ckpt_dir = get_cfg(sys.argv[1])

    if net_type == 1:
        model = ScoreNetMultiPair(cfg, c_in=cfg['c_in']).cuda()
    else:
        model = RefineNet(cfg=cfg, c_in=cfg['c_in']).cuda()

    ckpt = torch.load(ckpt_dir)
    if 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt)

    dummy_A = torch.randn((1, 6, 160, 160), dtype=torch.float32).cuda()
    dummy_B = torch.randn((1, 6, 160, 160), dtype=torch.float32).cuda()
    dummy_L = 1

    if net_type == 1:
        torch.onnx.export(
            model,
            (dummy_A, dummy_B, dummy_L),
            "./score.onnx",
            input_names=['A', 'B', 'L']
        )
    else:
        torch.onnx.export(
            model,
            (dummy_A, dummy_B),
            "./refine.onnx",
            input_names=['A', 'B']
        )
    print(f'begin quantize model...')
    quantize_model(model)
    print(f'end quantize model...')
    
    print(f'------------- quantize summary: -----------------')
    print(f'-------------------------------------------------')
    mtq.print_quant_summary(model)
    print(f'-------------------------------------------------')
    print(f'------------- quantize summary: -----------------')
