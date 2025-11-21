import torch
import numpy as np


def max_abs_error(origin, quantized):
    return np.max(np.abs(origin - quantized))

def cmp_datas(file1, file2, cmp_list):
    datas1 = torch.load(file1)
    datas2 = torch.load(file2)
    assert len(datas1) == len(datas2)
    
    for data1, data2 in zip(datas1, datas2):
        for i, data in enumerate(data1):
            for cmp_func in cmp_list:
                print(f'{cmp_func.__name__} {i}: {cmp_func(data, data2[i])}')
        print('------------------')

def mse_error(original, quantized):
    """均方误差"""
    return np.mean((original - quantized) ** 2)
            
def psnr(original, quantized, max_val=None):
    """峰值信噪比 - 值越大表示质量越好"""
    mse = mse_error(original, quantized)
    if max_val is None:
        max_val = np.max(np.abs(original))
    return 20 * np.log10(max_val) - 10 * np.log10(mse)


if __name__ == "__main__":
    import sys
    cmp_datas(sys.argv[1], sys.argv[2], [psnr, max_abs_error])