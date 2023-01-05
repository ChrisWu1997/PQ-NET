import os
import trimesh
import argparse
from tqdm import tqdm


def obj2off_folder(input_folder, output_folder, reso=64):
    os.makedirs(output_folder, exist_ok=True)
    for i, file in enumerate(sorted(os.listdir(input_folder))):
        if file.endswith(".obj"):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file.replace(".obj", ".off"))
            obj2off(input_path, output_path, reso=reso)


def obj2off(input_file, output_file, reso=64):
    mesh = trimesh.load(input_file)
    # scale mesh from [-1, 1] to [0, reso]
    mesh.vertices = mesh.vertices * reso / 2 + reso / 2
    mesh.export(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='class input folder')
    parser.add_argument('-o', '--output', type=str, help='class output folder')
    parser.add_argument('--reso', type=int, default=64, help='resolution')
    args = parser.parse_args()

    shape_names = sorted(os.listdir(args.input))
    for name in tqdm(shape_names):
        obj2off_folder(os.path.join(args.input, name), os.path.join(args.output, name), reso=args.reso)
