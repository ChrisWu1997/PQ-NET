import h5py
import os
import glob
import numpy as np
from tqdm import tqdm
import json
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='source directory')
    parser.add_argument('--out', type=str, help='output directory')
    args = parser.parse_args()

    src_root = args.src # "/home/megaBeast/Desktop/partnet_data/voxelized/Table2"
    tgt_root = args.out # "/dev/data/partnet_data/wurundi/voxelized/Table2"
    os.makedirs(tgt_root, exist_ok=True)

    class_info = {}
    shape_names = sorted(os.listdir(src_root))
    shape_names = [name for name in shape_names if name.endswith('.h5')]
    total_valid_nums = 0
    for name in tqdm(shape_names):
        path = os.path.join(src_root, name)
        with h5py.File(path, "r") as fp:
            part_voxels = fp["tensor"][:] # (n_parts, reso, reso, reso)

        shape_voxel = np.zeros(part_voxels.shape[1:], dtype=np.uint8)
        incomplete = False
        for i in range(part_voxels.shape[0]):
            mask = part_voxels[i] > 0
            if np.sum(mask) == 0:
                print("ID {} part {} not shown. skip this shape.".format(name, i))
                incomplete = True
                break
            shape_voxel[part_voxels[i] > 0] = i + 1
        
        if incomplete:
            continue
        
        vox_dim = shape_voxel.shape[0]
        n_parts = part_voxels.shape[0]
        save_path = os.path.join(tgt_root, name)
        shape_id = name.split(".")[0]
        with h5py.File(save_path, 'w') as fp:
            fp.create_dataset('shape_voxel{}'.format(vox_dim), shape=(vox_dim, vox_dim, vox_dim),
                              dtype=np.uint8, data=shape_voxel, compression=9)
            fp.create_dataset('parts_voxel{}'.format(vox_dim), shape=(n_parts, vox_dim, vox_dim, vox_dim),
                              dtype=np.bool, data=part_voxels.astype(np.bool), compression=9)
            fp.attrs['n_parts'] = n_parts
            fp.attrs['name'] = shape_id.encode('utf-8')

        class_info.update({shape_id: n_parts})
        total_valid_nums += 1

    with open(os.path.join(tgt_root + "_info.json"), 'w') as f:
        json.dump(class_info, f)

    print(total_valid_nums)


if __name__ == '__main__':
    main()
