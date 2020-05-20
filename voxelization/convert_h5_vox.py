import h5py
import os
import glob
import numpy as np
from tqdm import tqdm
import json


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='source directory')
    parser.add_argument('--out', type=str, help='output directory')
    args = parser.parse_args()

    src_root = args.src # "/home/megaBeast/Desktop/partnet_data/voxelized/Table2"
    tgt_root = args.out # "/dev/data/partnet_data/wurundi/voxelized/Table2"
    ensure_dir(tgt_root)

    vox_dim = 64
    class_info = {}
    shape_names = sorted(os.listdir(src_root))
    total_valid_nums = 0
    for name in tqdm(shape_names):
        shape_dir = os.path.join(src_root, name)
        part_paths = sorted(glob.glob(os.path.join(shape_dir, 'object_*.h5')))
        # u_part_paths = sorted(glob.glob(os.path.join(shape_dir, 'u_object_*.h5')))
        n_parts = len(part_paths)

        save_path = os.path.join(tgt_root, name + '.h5')

        # whole shape
        # with h5py.File(os.path.join(shape_dir, 'shape.h5'), 'r') as fp:
        #     shape_voxel = fp['tensor'][0]

        shape_voxel = np.zeros((vox_dim, vox_dim, vox_dim))

        # parts voxel in origin shape
        incomplete = False
        parts_voxel_origin = []
        for i in range(n_parts):
            path = os.path.join(shape_dir, 'object_{}.h5'.format(i + 1))
            with h5py.File(path, 'r') as fp:
                part_voxel = fp['tensor'][0]
            parts_voxel_origin.append(part_voxel)

            part_points = np.where(part_voxel >= 1)
            part_points = np.stack(part_points, axis=0)
            ori_voxel_on_points = shape_voxel[part_points[0], part_points[1], part_points[2]]
            zero_positions = np.where(ori_voxel_on_points == 0)[0]
            part_points = part_points[:, zero_positions]

            if part_points.shape[1] == 0:
                print("ID {} part {} not shown.".format(name, i))
                incomplete = True
                break
            shape_voxel[part_points[0], part_points[1], part_points[2]] = i + 1

        if incomplete:
            continue

        parts_voxel_origin = np.stack(parts_voxel_origin, axis=0)

        # parts voxel in its own scaled shape
        # parts_voxel_scaled = []
        # for i in range(n_parts):
        #     with h5py.File(u_part_paths[i], 'r') as fp:
        #         part_voxel = fp['tensor'][0]
        #     parts_voxel_scaled.append(part_voxel)
        # parts_voxel_scaled = np.stack(parts_voxel_scaled, axis=0)

        if len(np.unique(shape_voxel)) - 1 != n_parts:
            print("n_parts mismatch! ID: {}, ori: {}, now: {}".format(name, n_parts,
                                                                      len(np.unique(shape_voxel)) - 1))
        # save
        with h5py.File(save_path, 'w') as fp:
            fp.create_dataset('shape_voxel{}'.format(vox_dim), shape=(vox_dim, vox_dim, vox_dim),
                              dtype=np.uint8, data=shape_voxel, compression=9)
            fp.create_dataset('parts_voxel{}'.format(vox_dim), shape=(n_parts, vox_dim, vox_dim, vox_dim),
                              dtype=np.bool, data=parts_voxel_origin, compression=9)
            # fp.create_dataset('parts_voxel_scaled{}'.format(vox_dim), shape=(n_parts, vox_dim, vox_dim, vox_dim),
            #                   dtype=np.bool, data=parts_voxel_scaled, compression=9)
            fp.attrs['n_parts'] = n_parts
            fp.attrs['name'] = name.encode('utf-8')

        class_info.update({name: n_parts})
        total_valid_nums += 1

    with open(os.path.join(tgt_root + "_info.json"), 'w') as f:
        json.dump(class_info, f)

    print(total_valid_nums)

if __name__ == '__main__':
    main()
