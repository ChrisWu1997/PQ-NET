import os
import json
import h5py
import numpy as np
import torch


SPLIT_DIR = "data/train_val_test_split"


def collect_data_id(classname, phase):
    filename = os.path.join(SPLIT_DIR, "{}.{}.json".format(classname, phase))
    if not os.path.exists(filename):
        raise ValueError("Invalid filepath: {}".format(filename))

    all_ids = []
    with open(filename, 'r') as fp:
        info = json.load(fp)
    for item in info:
        all_ids.append(item["anno_id"])

    return all_ids


def n_parts_map(n):
    return n - 2


def load_from_hdf5_by_part(path, idx, resolution=64, rescale=True):
    """load part voxel and sampled sdf

    :param path: filepath to h5 data
    :param idx: part id
    :param resolution: resolution for sampled sdf
    :param rescale: rescale points value range to 0~1
    :return:
    """
    with h5py.File(path, 'r') as data_dict:
        n_parts = data_dict.attrs['n_parts']
        parts_voxel = data_dict['parts_voxel_scaled64'][idx].astype(np.float)
        # scale = data_dict['scales'][idx]
        # translation = data_dict['translations'][idx]
        data_points = data_dict['points_{}'.format(resolution)][idx]
        data_values = data_dict['values_{}'.format(resolution)][idx]
    if rescale:
        data_points = data_points / resolution
    return n_parts, parts_voxel, data_points, data_values  # , scale, translation


def load_from_hdf5_seq(path, max_n_parts, return_numpy=False, rescale_affine=True):
    """load part data for seq2seq training

    :param path: filepath to h5 data
    :param max_n_parts: max number of parts for this category
    :param return_numpy: return numpy version
    :param rescale_affine: rescale value range to 0~1
    :return:
    """
    with h5py.File(path, 'r') as data_dict:
        n_parts = data_dict.attrs['n_parts']
        # model_voxel = data_dict['voxel64'][:]
        parts_voxel = data_dict['parts_voxel_scaled64'][:].astype(np.float)
        data_points64 = data_dict['points_64'][:]
        data_values64 = data_dict['values_64'][:]
        translation = data_dict['translations'][:]
        size = data_dict['size'][:]
        affine = np.concatenate([translation, size], axis=1)
        cond = np.zeros((n_parts_map(max_n_parts) + 1, ))
        cond[n_parts_map(n_parts)] = 1
    if rescale_affine is True:
        # rescale translation values to 0~1
        affine = affine / parts_voxel.shape[-1]
        data_points64 = data_points64 / parts_voxel.shape[-1]  # FIXME: scale points coordinates to 0~1
    if return_numpy:
        return {"vox3d": parts_voxel,
                "points": data_points64,
                "values": data_values64,
                "affine": affine,
                "n_parts": n_parts,
                "cond": cond,
                "path": path}
    else:
        batch_voxels = torch.tensor(parts_voxel, dtype=torch.float32).unsqueeze(1)  # (n_parts, 1, dim, dim, dim)
        batch_points = torch.tensor(data_points64, dtype=torch.float32)  # (n_parts, points_batch_size, 3)
        batch_values = torch.tensor(data_values64, dtype=torch.float32)  # (n_parts, points_batch_size, 1)
        # translation = translation / parts_voxel.shape[-1]
        batch_affine = torch.tensor(affine, dtype=torch.float32)  # (n_parts, 6)
        batch_cond = torch.tensor(cond, dtype=torch.float32)
        return {"vox3d": batch_voxels,
                "points": batch_points,
                "values": batch_values,
                "affine": batch_affine,
                "n_parts": n_parts,
                "cond": batch_cond,
                "path": path}
