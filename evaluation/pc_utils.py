import numpy as np
import random
from plyfile import PlyData, PlyElement


def project_pc_to_image(points, resolution=64):
    """

    :param points: (n, 3) range(-1, 1)
    :return: binary image
    """
    img = []
    for i in range(3):
        canvas = np.zeros((resolution, resolution))
        axis = [0, 1, 2]
        axis.remove(i)
        proj_points = (points[:, axis] + 1) / 2 * resolution
        proj_points = proj_points.astype(np.int)
        canvas[proj_points[:, 0], proj_points[:, 1]] = 1
        img.append(canvas)
    img = np.concatenate(img, axis=1)
    return img


def read_ply(path):
    with open(path, 'rb') as f:
        plydata = PlyData.read(f)
        x = np.array(plydata['vertex']['x'])
        y = np.array(plydata['vertex']['y'])
        z = np.array(plydata['vertex']['z'])
        vertex = np.stack([x, y, z], axis=1)
    return vertex


def write_ply(points, filename, text=False):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)


def rotate_point_cloud(points, transformation_mat):

    new_points = np.dot(transformation_mat, points.T).T

    return new_points


def downsample_point_cloud(points, n_pts):
    """downsample points by random choice

    :param points: (n, 3)
    :param n_pts: int
    :return:
    """
    p_idx = random.sample(list(range(points.shape[0])), k=n_pts)
    return points[p_idx]


def upsample_point_cloud(points, n_pts):
    """upsample points by random choice

    :param points: (n, 3)
    :param n_pts: int, > n
    :return:
    """
    p_idx = random.sample(list(range(points.shape[0])), k=n_pts - points.shape[0])
    dup_points = points[p_idx]
    points = np.concatenate([points, dup_points], axis=0)
    return points


def sample_point_cloud_by_n(points, n_pts):
    if n_pts > points.shape[0]:
        return upsample_point_cloud(points, n_pts)
    elif n_pts < points.shape[0]:
        return downsample_point_cloud(points, n_pts)
    else:
        return points
