import h5py
from scipy import ndimage
import os
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from joblib import Parallel, delayed

VOX_DIM = 64

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


def scale_parts(path):
    dim_voxel = vox_dim =  VOX_DIM
    # path = "/home/wurundi/PycharmProjects/PartGen/scripts/sampling/1282.h5"
    with h5py.File(path, 'r') as data_dict:
        n_parts = data_dict.attrs['n_parts']
        parts_voxel = data_dict['parts_voxel{}'.format(vox_dim)][:]

    n_parts = parts_voxel.shape[0]
    # dim_voxel = vox_dim =  128
    d = 1

    parts_voxel_scaled = []
    scales = []
    translations = []
    parts_size = []
    for i in range(n_parts):
        ori_voxel = parts_voxel[i]
        bbox = find_bounding_box(ori_voxel)
        # warp
        mins = np.asarray(list(map(lambda x: max(x - d, 0), bbox[::2])))
        maxs = np.asarray(list(map(lambda x: min(x + d, dim_voxel - 1), bbox[1::2])))

        axis_lengths = maxs - mins + 1
        parts_size.append(axis_lengths)

        bbox_voxel = ori_voxel[mins[0]:maxs[0] + 1, mins[1]:maxs[1] + 1, mins[2]:maxs[2] + 1]

        midpoint = (mins + maxs) / 2.0

        scale = dim_voxel / np.max(axis_lengths)
        # bbox_voxel_scaled = ndimage.zoom(bbox_voxel, scale, mode='nearest')
        bbox_voxel_scaled = resize(bbox_voxel, axis_lengths * scale, mode='constant')
        bbox_voxel_scaled = np.asarray(bbox_voxel_scaled >= 0.5, dtype=np.bool)
        x_len, y_len, z_len = bbox_voxel_scaled.shape

        voxel_scaled64 = np.zeros((dim_voxel, dim_voxel, dim_voxel), dtype=np.bool)
        center = dim_voxel // 2
        new_mins = (center - x_len // 2, center - y_len // 2, center - z_len // 2)
        voxel_scaled64[new_mins[0]:new_mins[0] + x_len,
                       new_mins[1]:new_mins[1] + y_len,
                       new_mins[2]:new_mins[2] + z_len] = bbox_voxel_scaled

        parts_voxel_scaled.append(voxel_scaled64)
        scales.append(1.0 / scale)
        translations.append(midpoint)

    parts_voxel_scaled = np.stack(parts_voxel_scaled, axis=0)
    scales = np.stack(scales, axis=0).reshape(-1, 1)
    translations = np.stack(translations, axis=0)
    parts_size = np.stack(parts_size, axis=0)
    # return parts_voxel_scaled, scales, translations, parts_size

    with h5py.File(path, 'a') as fp:
        # del fp['parts_voxel_scaled{}'.format(vox_dim)]
        # del fp['scales']
        # del fp['translations']
        # del fp['size']
        fp.create_dataset('parts_voxel_scaled{}'.format(vox_dim), shape=(n_parts, vox_dim, vox_dim, vox_dim),
                          dtype=np.bool, data=parts_voxel_scaled, compression=9)
        fp.create_dataset('scales'.format(vox_dim), shape=(n_parts, 1),
                          dtype=np.float, data=scales, compression=9)
        fp.create_dataset('translations'.format(vox_dim), shape=(n_parts, 3),
                          dtype=np.float, data=translations, compression=9)
        fp.create_dataset('size'.format(vox_dim), shape=(n_parts, 3),
                          dtype=np.int, data=parts_size, compression=9)
        # fp['parts_voxel_scaled64'][:] = parts_voxel_scaled
        # fp['scales'][:] = scales
        # fp['translations'][:] = translations
        # fp['size'][:] = parts_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='source directory')
    args = parser.parse_args()

    class_dir = args.src # "/dev/data/partnet_data/wurundi/voxelized_solid"

    shape_names = sorted(os.listdir(class_dir))
    shape_names = [x for x in shape_names if x[-3:] == '.h5']

    paths = [os.path.join(class_dir, shape_name) for shape_name in shape_names]
    Parallel(n_jobs=-1, verbose=2)(delayed(scale_parts)(path) for path in paths)

if __name__ == '__main__':
    main()
