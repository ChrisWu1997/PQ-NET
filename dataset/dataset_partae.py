from torch.utils.data import Dataset
import torch
import numpy as np
import os
import json
import random
from dataset.data_utils import collect_data_id, load_from_hdf5_by_part


# Part AE dataset
######################################################
class PartAEDataset(Dataset):
    def __init__(self, phase, data_root, class_name, points_batch_size, all_points=False, resolution=64):
        super(PartAEDataset, self).__init__()
        self.data_root = os.path.join(data_root, class_name)
        self.class_name = class_name
        self.parts_info = self.load_part_data_info(phase)
        self.phase = phase
        self.points_batch_size = points_batch_size

        self.all_points = all_points
        self.resolution = resolution

    def load_part_data_info(self, phase):
        shape_names = collect_data_id(self.class_name, phase)
        with open('data/{}_info.json'.format(self.class_name), 'r') as fp:
            nparts_dict = json.load(fp)
        parts_info = []
        for name in shape_names:
            shape_h5_path = os.path.join(self.data_root, name + '.h5')
            if not os.path.exists(shape_h5_path):  # check file existence
                continue
            parts_info.extend([(shape_h5_path, x) for x in range(nparts_dict[name])])

        return parts_info

    def __getitem__(self, index):
        shape_path, part_idx = self.parts_info[index]
        n_parts, parts_voxel, data_points, data_values = load_from_hdf5_by_part(shape_path, part_idx, self.resolution)

        # shuffle selected points
        if not self.all_points and len(data_points) > self.points_batch_size:
            indices = np.arange(len(data_points))
            random.shuffle(indices)
            # np.random.shuffle(indices)
            indices = indices[:self.points_batch_size]
            data_points = data_points[indices]
            data_values = data_values[indices]

        batch_voxels = torch.tensor(parts_voxel.astype(np.float), dtype=torch.float32).unsqueeze(0)  # (1, dim, dim, dim)
        batch_points = torch.tensor(data_points, dtype=torch.float32)  # (points_batch_size, 3)
        batch_values = torch.tensor(data_values, dtype=torch.float32)  # (points_batch_size, 1)
        # batch_affine = torch.tensor(np.concatenate([scale, translation]), dtype=torch.float32)

        return {"vox3d": batch_voxels,
                "points": batch_points,
                "values": batch_values,
                "n_parts": n_parts,
                "part_idx": part_idx,
                # "affine": batch_affine,
                "path": shape_path}

    def __len__(self):
        return len(self.parts_info)


if __name__ == "__main__":
    pass
