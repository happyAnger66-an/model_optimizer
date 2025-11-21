import os
from pathlib import Path

from torch.utils.data import DataLoader, Dataset
import numpy as np

class ImageDatasets(Dataset):
    def __init__(self, data_dir,
                 calib_size=1000):
        self.data_dir = data_dir
        self.images_dir = Path(os.path.join(self.data_dir))
        self.images = sorted(list(self.images_dir.rglob('*.jpg')))
        self.calib_size = min(calib_size, len(self.images))

    def __len__(self):
        return self.calib_size

    def __getitem__(self, idx):
        # Use sequential indices directly
        return self.images[idx]

class YoLoCalibrationData(Dataset):
    def __init__(self, data_dir,
                 calib_size=1000):
        self.data_dir = data_dir
        self.images_dir = Path(os.path.join(self.data_dir))
        self.images = sorted(list(self.images_dir.rglob('*.jpg')))
        self.calib_size = min(calib_size, len(self.images))

    def __len__(self):
        return self.calib_size

    def __getitem__(self, idx):
        # Use sequential indices directly
        return self.images[idx]


if __name__ == "__main__":
    import sys
    datas = YoLoCalibrationData(sys.argv[1])
    print(datas.images)
    for data in datas:
        print(f'data: {data}')