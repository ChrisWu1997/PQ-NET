import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ShapeCodesDataset(Dataset):
    def __init__(self, data_root):
        super(ShapeCodesDataset, self).__init__()
        self.code_dir = data_root # os.path.join(CODE_ROOT)
        self.all_items = sorted(os.listdir(self.code_dir))

    def __getitem__(self, index):
        code_path = os.path.join(self.code_dir, self.all_items[index])
        shape_code = np.load(code_path)[:, 0, :]
        shape_code = np.concatenate([shape_code[0], shape_code[1]], axis=0)  # (1024,)
        shape_code = torch.tensor(shape_code, dtype=torch.float32)
        return shape_code # {"shape_code": shape_code}

    def __len__(self):
        return len(self.all_items)
