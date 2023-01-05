import h5py
from skimage import morphology
import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse

VOX_DIM = 64


def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def nearest_in_line(line, p):
    surface_points = np.where(line >= 1)[0]
    dists = np.abs(surface_points - p)
    shorest_idx = np.argsort(dists)[0]
    return surface_points[shorest_idx], dists[shorest_idx]


def nearest_surface_lines(vox3d, points):
    dim = vox3d.shape[0]
    candidates = []
    # x axis
    line = vox3d[:, points[1], points[2]]
    x_p, x_dist = nearest_in_line(line, points[0])
    candidates.append((x_p, points[1], points[2]))

    # y axis
    line = vox3d[points[0], :, points[2]]
    y_p, y_dist = nearest_in_line(line, points[1])
    candidates.append((points[0], y_p, points[2]))

    # z axis
    line = vox3d[points[0], points[1], :]
    z_p, z_dist = nearest_in_line(line, points[2])
    candidates.append((points[0], points[1], z_p))

    idx = np.argsort([x_dist, y_dist, z_dist])[0]
    return candidates[idx]


def nearest_surface_cubic(points, surface_points):
    dists = np.sum((surface_points - points.reshape(1, 3)) ** 2, axis=1)
    shorest_idx = np.argmin(dists)
    return surface_points[shorest_idx]


def sparse2solid(shape_name, class_dir, out_class_dir):
    vox_dim = dim_voxel = VOX_DIM
    # path = "/home/wurundi/PycharmProjects/PartGen/scripts/sampling/1282.h5"
    path = os.path.join(class_dir, shape_name)
    with h5py.File(path, 'r') as data_dict:
        n_parts = data_dict.attrs['n_parts']
        shape_parts_voxel = data_dict['shape_voxel{}'.format(dim_voxel)][:]

    d = 1

    shape_bin_voxel = np.zeros_like(shape_parts_voxel, dtype=np.uint8)
    surface_points = np.stack(np.where(shape_parts_voxel >= 1), axis=1)
    shape_bin_voxel[surface_points[:, 0], surface_points[:, 1], surface_points[:, 2]] = 1

    # warp one layer around
    shape_bin_voxel_large = np.zeros((dim_voxel + d * 2, dim_voxel + d * 2, dim_voxel + d * 2), dtype=np.uint8)
    shape_bin_voxel_large[d:d + dim_voxel, d:d + dim_voxel, d:d + dim_voxel] = shape_bin_voxel

    # find connected inner area
    labels, num_labels = morphology.label(shape_bin_voxel_large, background=1, connectivity=1, return_num=True)

    outside_label = labels[0][0][0]

    # remove outer warped layer
    labels = labels[d:d + dim_voxel, d:d + dim_voxel, d:d + dim_voxel]

    # mask out surface points
    labels[surface_points[:, 0], surface_points[:, 1], surface_points[:, 2]] = outside_label

    inner_points = np.stack(np.where(labels != outside_label), axis=1)

    solid_parts_voxel = np.copy(shape_parts_voxel)
    for i in range(inner_points.shape[0]):
        points = inner_points[i]
        nearest = nearest_surface_lines(shape_parts_voxel, points)
        # nearest = nearest_surface_cubic(points, surface_points)
        solid_parts_voxel[points[0], points[1], points[2]] = shape_parts_voxel[nearest[0], nearest[1], nearest[2]]
        # results[points[0], points[1], points[2]] = 1

    if len(np.unique(solid_parts_voxel)) - 1 != n_parts:
        print("n_parts mismatch! ID: {}, ori: {}, now: {}".format(shape_name, n_parts, len(np.unique(solid_parts_voxel)) - 1))
        return 0, 0
        # raise ValueError("n_parts mismatch")

    out_shape_hdf5_path = os.path.join(out_class_dir, shape_name)

    with h5py.File(out_shape_hdf5_path, 'w') as fp:
        fp.attrs['n_parts'] = n_parts
        fp.create_dataset("shape_voxel{}".format(vox_dim), shape=(vox_dim, vox_dim, vox_dim),
                          dtype=np.uint8, data=solid_parts_voxel, compression=9)

        parts_voxel = []
        for i in range(1, n_parts + 1, 1):
            part_voxel = np.zeros((vox_dim, vox_dim, vox_dim), dtype=np.bool)
            part_voxel[np.where(solid_parts_voxel == i)] = True
            if np.sum(part_voxel) == 0:
                print(out_shape_hdf5_path, i)
            parts_voxel.append(part_voxel)
        fp.create_dataset('parts_voxel{}'.format(vox_dim), shape=(n_parts, vox_dim, vox_dim, vox_dim),
                          dtype=np.bool, data=parts_voxel, compression=9)
        fp.attrs['name'] = shape_name.encode('utf-8')

    return solid_parts_voxel, n_parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='source directory')
    parser.add_argument('--out', type=str, help='output directory for solid voxel h5 files')
    args = parser.parse_args()

    class_dir = args.src
    out_class_dir = args.out

    ensure_dir(out_class_dir)

    shape_names = sorted(os.listdir(class_dir))
    shape_names = [x for x in shape_names if x[-3:] == '.h5']

    Parallel(n_jobs=-1, verbose=2)(delayed(sparse2solid)(name, class_dir, out_class_dir) for name in shape_names)

if __name__ == '__main__':
    main()
