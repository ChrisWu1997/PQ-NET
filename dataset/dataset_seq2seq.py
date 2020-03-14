from torch.utils.data import Dataset
import torch
import os
import json
from dataset.data_utils import collect_data_id, load_from_hdf5_seq


# Seq2Seq dataset
######################################################
class Seq2SeqDataset(Dataset):
    def __init__(self, phase, data_root, class_name, max_n_parts):
        self.data_root = os.path.join(data_root, class_name)
        self.class_name = class_name

        self.max_n_parts = max_n_parts
        self.shape_names = self.load_shape_names(phase, self.max_n_parts)

        self.all_paths = [os.path.join(self.data_root, name + '.h5') for name in self.shape_names]
        self.phase = phase

    def load_shape_names(self, phase, max_n_parts, min_n_parts=2):
        shape_names = collect_data_id(self.class_name, phase)
        with open('data/{}_info.json'.format(self.class_name), 'r') as fp:
            nparts_dict = json.load(fp)

        filtered_shape_names = []
        for name in shape_names:
            shape_h5_path = os.path.join(self.data_root, name + '.h5')
            if not os.path.exists(shape_h5_path):  # check file existence
                continue

            if min_n_parts <= nparts_dict[name] <= max_n_parts:
                filtered_shape_names.append(name)

        return filtered_shape_names

    def __getitem__(self, index):
        path = self.all_paths[index]
        data_dict = load_from_hdf5_seq(path, self.max_n_parts, return_numpy=True)
        n_parts = data_dict['n_parts']
        parts_vox3d = torch.tensor(data_dict['vox3d'], dtype=torch.float32).unsqueeze(1)  # (n_parts, 1, dim, dim, dim)

        stop_sign = torch.zeros((n_parts, 1), dtype=torch.float32)
        stop_sign[-1] = 1

        mask = torch.ones((n_parts, 1), dtype=torch.float32)

        cond = torch.tensor(data_dict['cond'], dtype=torch.float32)

        batch_affine = torch.tensor(data_dict['affine'], dtype=torch.float32)
        batch_affine_target = batch_affine.clone()

        return {'vox3d': parts_vox3d, 'n_parts': n_parts, 'path': path, 'sign': stop_sign,
                'mask': mask, 'cond': cond,
                'affine_input': batch_affine, 'affine_target': batch_affine_target}

    def __len__(self):
        return len(self.shape_names)


def pad_collate_fn_for_dict(batch):
    n_parts_batch = [d['n_parts'] for d in batch]
    max_n_parts = max(n_parts_batch)
    # n_parts_batch = [torch.LongTensor(x) for x in n_parts_batch]
    name_batch = [d['path'] for d in batch]
    vox3d_batch = [d['vox3d'] for d in batch]
    vox3d_batch = list(map(lambda x: pad_tensor(x, pad=max_n_parts, dim=0), vox3d_batch))
    vox3d_batch = torch.stack(vox3d_batch, dim=0)

    sign_batch = [d['sign'] for d in batch]
    sign_batch = list(map(lambda x: pad_tensor(x, pad=max_n_parts, dim=0), sign_batch))
    sign_batch = torch.stack(sign_batch, dim=1)
    mask_batch = [d['mask'] for d in batch]
    mask_batch = list(map(lambda x: pad_tensor(x, pad=max_n_parts, dim=0), mask_batch))
    mask_batch = torch.stack(mask_batch, dim=1)
    affine_input = [d['affine_input'] for d in batch]
    affine_input = list(map(lambda x: pad_tensor(x, pad=max_n_parts, dim=0), affine_input))
    affine_input = torch.stack(affine_input, dim=1)
    affine_target = [d['affine_target'] for d in batch]
    affine_target = list(map(lambda x: pad_tensor(x, pad=max_n_parts, dim=0), affine_target))
    affine_target = torch.stack(affine_target, dim=1)

    cond_batch = torch.stack([d['cond'] for d in batch], dim=0)
    return {'vox3d': vox3d_batch, 'n_parts': n_parts_batch, 'path': name_batch, 'sign': sign_batch,
            'mask': mask_batch, 'cond': cond_batch,
            'affine_input': affine_input, 'affine_target': affine_target}


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


if __name__ == "__main__":
    pass
