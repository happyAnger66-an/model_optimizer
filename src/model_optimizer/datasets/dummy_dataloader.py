import torch
from torch.utils.data import Dataset


class LLMDummyDataLoader(Dataset):
    def __init__(self, calib_size):
        self.calib_size = calib_size
        self.datas = []
        for i in range(self.calib_size):
            inputs_ids = torch.randint(1000, (1, 512)).cuda()
            data = {"input_ids":  inputs_ids}
            self.datas.append(data)

    def __len__(self):
        return self.calib_size

    def __getitem__(self, idx):
        return self.datas[idx]
