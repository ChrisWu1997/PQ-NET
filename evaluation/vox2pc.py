import numpy as np
import mcubes as libmcubes
import trimesh
import argparse
import glob
import os
from tqdm import tqdm
import h5py
import json
from trimesh.sample import sample_surface
from pc_utils import write_ply

N_PTS = 2048
SPLIT_DIR = "../data/train_val_test_split"

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str)
parser.add_argument("--category", type=str)
parser.add_argument("--test_data", action="store_true", help="process test data")
args = parser.parse_args()

h5_paths = sorted(glob.glob(os.path.join(args.src, "*.h5")))
save_dir = args.src + "_pc"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def collect_data_id(split_dir, classname, phase):
    filename = os.path.join(split_dir, "{}.{}.json".format(classname, phase))
    if not os.path.exists(filename):
        raise ValueError("Invalid filepath: {}".format(filename))

    all_ids = []
    with open(filename, 'r') as fp:
        info = json.load(fp)
    for item in info:
        all_ids.append(item["anno_id"])

    return all_ids

test_names = None if not args.test_data else collect_data_id(SPLIT_DIR, args.category, 'test')

cnt = 0
for path in tqdm(h5_paths):
    data_id = path.split('/')[-1].split('.')[0]
    if test_names is not None and data_id not in test_names:
        continue

    with h5py.File(path, "r") as fp:
        shape_voxel = fp['voxel'][:] if not args.test_data else fp['shape_voxel64'][:]

    vertices, triangles = libmcubes.marching_cubes(shape_voxel, 0)
    shape_mesh = trimesh.Trimesh(vertices, triangles)
    points, _ = sample_surface(shape_mesh, N_PTS)

    save_path = os.path.join(save_dir, data_id + '.ply')
    write_ply(points, save_path)
    cnt += 1

print("total:", cnt)
