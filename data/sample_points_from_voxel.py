import numpy as np
import os
import h5py
import argparse
from joblib import Parallel, delayed


def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def safe_minmax(x, limit):
    if len(x) == 0:
        return 0, limit
    xmin, xmax = np.min(x), np.max(x)
    xmin = min(xmin, max(xmax - 1, 0))
    xmax = max(xmax, min(xmin + 1, limit))
    return xmin, xmax


def find_bounding_box(voxel_model):
    x, y, z = np.where(voxel_model >= 1)
    xmin, xmax = safe_minmax(x, voxel_model.shape[0])
    ymin, ymax = safe_minmax(y, voxel_model.shape[1])
    zmin, zmax = safe_minmax(z, voxel_model.shape[2])
    return [xmin, xmax, ymin, ymax, zmin, zmax]


def sample_points_from_vox3d(voxel_model_64, dim_voxel, batch_size, d=2, sigma=0.1):
    """sample points from voxel surface

    :param voxel_model_64: voxel model at 64^3 resolution
    :param dim_voxel: target dimension of sampled points
    :param batch_size: number of points to be sampled
    :param d: size of neighbor window
    :param sigma: sigma for normal distribution
    :return:
    """
    ori_dim = voxel_model_64.shape[0]

    # downsample to given resolution
    # wrap several layers around the cube surface in order to sample near the cube border.
    multiplier = int(ori_dim // dim_voxel)
    voxel_model = np.zeros((dim_voxel + d * 2, dim_voxel + d * 2, dim_voxel + d * 2), dtype=np.uint8)
    # voxel_model[d:d+dim_voxel, d:d+dim_voxel, d:d+dim_voxel] = voxel_model_64[::multiplier, ::multiplier, ::multiplier]
    voxel_model_fullfill = np.zeros((dim_voxel, dim_voxel, dim_voxel), dtype=np.uint8)
    for i in range(dim_voxel):
        for j in range(dim_voxel):
            for k in range(dim_voxel):
                voxel_model_fullfill[i, j, k] = np.max(
                    voxel_model_64[i * multiplier:(i + 1) * multiplier, j * multiplier:(j + 1) * multiplier,
                    k * multiplier:(k + 1) * multiplier])
    voxel_model[d:d+dim_voxel, d:d+dim_voxel, d:d+dim_voxel] = voxel_model_fullfill
    del voxel_model_fullfill

    # bounding box
    bbox = find_bounding_box(voxel_model)

    # statistics
    exceed = 0

    # sample points near surface
    sample_points = np.zeros([batch_size, 3], np.float)
    sample_values = np.zeros([batch_size, 1], np.uint8)
    batch_size_counter = 0
    voxel_model_flag = np.zeros_like(voxel_model, dtype=np.uint8)
    # np.zeros([dim_voxel, dim_voxel, dim_voxel], np.uint8)
    positive = 0
    for i in range(max(bbox[0] - d, d), min(bbox[1] + d, dim_voxel + d)):
        for j in range(max(bbox[2] - d, d), min(bbox[3] + d, dim_voxel + d)):
            for k in range(max(bbox[4] - d, d), min(bbox[5] + d, dim_voxel + d)):
                if batch_size_counter >= batch_size:
                    break
                neighbor_cube = voxel_model[i - d:i + d + 1, j - d:j + d + 1, k - d:k + d + 1]
                if np.max(neighbor_cube) != np.min(neighbor_cube):
                    sample_points[batch_size_counter, 0] = i
                    sample_points[batch_size_counter, 1] = j
                    sample_points[batch_size_counter, 2] = k
                    sample_values[batch_size_counter, 0] = voxel_model[i, j, k]
                    if voxel_model[i, j, k] >= 1:
                        positive += 1
                    voxel_model_flag[i, j, k] = 1
                    batch_size_counter += 1

    positive = 0
    if batch_size_counter >= batch_size:
        # print("Batch_size exceeded! Desired {}, but got {}.".format(batch_size, batch_size_counter))
        exceed += 1
        batch_size_counter = 0
        voxel_model_flag = np.zeros_like(voxel_model, dtype=np.uint8)
        for i in range(max(bbox[0] - d, d), min(bbox[1] + d, dim_voxel + d), 2):
            for j in range(max(bbox[2] - d, d), min(bbox[3] + d, dim_voxel + d), 2):
                for k in range(max(bbox[4] - d, d), min(bbox[5] + d, dim_voxel + d), 2):
                    if batch_size_counter >= batch_size:
                        break
                    neighbor_cube = voxel_model[i - d:i + d + 1, j - d:j + d + 1, k - d:k + d + 1]
                    if np.max(neighbor_cube) != np.min(neighbor_cube):
                        sample_points[batch_size_counter, 0] = i
                        sample_points[batch_size_counter, 1] = j
                        sample_points[batch_size_counter, 2] = k
                        sample_values[batch_size_counter, 0] = voxel_model[i, j, k]
                        if voxel_model[i, j, k] >= 1:
                            positive += 1
                        voxel_model_flag[i, j, k] = 1
                        batch_size_counter += 1
    if batch_size_counter == 0:
        raise RuntimeError("no occupied! {}".format(np.sum(voxel_model_64)))

    # fill remaining slots
    if batch_size_counter < batch_size:
        # fill other slots with random points
        repeat = batch_size // batch_size_counter
        remain = batch_size % batch_size_counter
        for i in range(1, repeat, 1):
            sample_points[batch_size_counter*i:batch_size_counter*(i + 1), :] = \
                sample_points[:batch_size_counter, :] + np.random.uniform(low=-sigma, high=sigma, size=(batch_size_counter, 3))
            sample_values[batch_size_counter*i:batch_size_counter*(i + 1), :] = sample_values[:batch_size_counter]

        if remain > 0:
            indices = np.arange(batch_size_counter)
            np.random.shuffle(indices)
            indices = indices[:remain]
            sample_points[-remain:] = sample_points[indices] + np.random.uniform(low=-sigma, high=sigma, size=(remain, 3))
            sample_values[-remain:] = sample_values[indices]

    # translate coordinates back
    sample_points = sample_points - d  # 3
    sample_points = np.clip(sample_points, 0, dim_voxel - 1)
    if np.sum(sample_points < 0) > 0 or np.sum(sample_points > dim_voxel - 1) > 0:
        raise RuntimeError("Out range.")

    nr_points_in_bbox = np.sum(voxel_model_flag[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]])
    size_bbox = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2]) * (bbox[5] - bbox[4])
    bbox_ratio = nr_points_in_bbox / size_bbox

    return sample_points, sample_values, exceed, bbox_ratio


def process_one(src_shape_hdf5_path):
    # read source voxel data
    with h5py.File(src_shape_hdf5_path, 'r') as fp:
        # if 'points_64' in fp:
        #     is_processed = True
        parts_voxel = fp['parts_voxel_scaled{}'.format(64)][:]
        n_parts = fp.attrs['n_parts']

    # sample points at resolution 64x64x64
    dim_voxel = 64
    batch_size = 32 * 32 * 32
    parts_points = []
    parts_values = []
    bbox_ratios = 0

    try:
        for i in range(n_parts):
            sample_points, sample_values, exceed, bbox_ratio = sample_points_from_vox3d(
                parts_voxel[i], dim_voxel, batch_size)

            parts_points.append(sample_points)
            parts_values.append(sample_values)
            bbox_ratios += bbox_ratio
    except RuntimeError as e:
        print(e, 'shape_name:{}'.format(src_shape_hdf5_path.split('/')[-1]), dim_voxel)
        return

    parts_points = np.stack(parts_points, axis=0)
    parts_values = np.stack(parts_values, axis=0)

    with h5py.File(src_shape_hdf5_path, 'a') as fp:
        try:
            del fp['points_64']
            del fp['values_64']
        except:
            pass
        fp.create_dataset("points_{}".format(dim_voxel), [n_parts, batch_size, 3], np.float, compression=9, data=parts_points)
        fp.create_dataset("values_{}".format(dim_voxel), [n_parts, batch_size, 1], np.uint8, compression=9, data=parts_values)

    # sample points at resolution 32x32x32
    dim_voxel = 32
    batch_size = 16 * 16 * 16 * 2
    parts_points = []
    parts_values = []
    bbox_ratios = 0

    try:
        for i in range(n_parts):
            sample_points, sample_values, exceed, bbox_ratio = sample_points_from_vox3d(
                parts_voxel[i], dim_voxel, batch_size)

            parts_points.append(sample_points)
            parts_values.append(sample_values)
            bbox_ratios += bbox_ratio
    except RuntimeError as e:
        print(e, 'shape_name:{}'.format(src_shape_hdf5_path.split('/')[-1]), dim_voxel)
        return

    parts_points = np.stack(parts_points, axis=0)
    parts_values = np.stack(parts_values, axis=0)

    with h5py.File(src_shape_hdf5_path, 'a') as fp:
        try:
            del fp['points_32']
            del fp['values_32']
        except:
            pass
        fp.create_dataset("points_{}".format(dim_voxel), [n_parts, batch_size, 3], np.float, compression=9, data=parts_points)
        fp.create_dataset("values_{}".format(dim_voxel), [n_parts, batch_size, 1], np.uint8, compression=9, data=parts_values)

    # sample points at resolution 16x16x16
    dim_voxel = 16
    batch_size = 16 * 16 * 16
    parts_points = []
    parts_values = []
    bbox_ratios = 0

    try:
        for i in range(n_parts):
            sample_points, sample_values, exceed, bbox_ratio = sample_points_from_vox3d(
                parts_voxel[i], dim_voxel, batch_size)

            parts_points.append(sample_points)
            parts_values.append(sample_values)
            bbox_ratios += bbox_ratio
    except RuntimeError as e:
        print(e, 'shape_name:{}'.format(src_shape_hdf5_path.split('/')[-1]), dim_voxel)
        return

    parts_points = np.stack(parts_points, axis=0)
    parts_values = np.stack(parts_values, axis=0)

    with h5py.File(src_shape_hdf5_path, 'a') as fp:
        try:
            del fp['points_16']
            del fp['values_16']
        except:
            pass
        fp.create_dataset("points_{}".format(dim_voxel), [n_parts, batch_size, 3], np.float, compression=9, data=parts_points)
        fp.create_dataset("values_{}".format(dim_voxel), [n_parts, batch_size, 1], np.uint8, compression=9, data=parts_values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='data', help="file path to source data")
    parser.add_argument('--category', type=str, required=True, help="shape category")
    parser.add_argument('-P', '--process', type=int, default=10, help="number of threads to parallel")
    args = parser.parse_args()

    src_root = args.src
    class_name = args.category

    class_dir = os.path.join(src_root, class_name)

    shape_names = sorted(os.listdir(class_dir))
    shape_names = [x for x in shape_names if x[-3:] == '.h5']

    paths = [os.path.join(class_dir, name) for name in shape_names]

    Parallel(n_jobs=args.process, verbose=2)(delayed(process_one)(path) for path in paths)


if __name__ == '__main__':
    main()
