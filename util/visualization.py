import numpy as np
from itertools import product, combinations
import mcubes as libmcubes
import trimesh


def project_voxel_along_xyz(voxels, concat=False):
    img1 = np.clip(np.amax(voxels, axis=0)*256, 0, 255).astype(np.uint8)
    img2 = np.clip(np.amax(voxels, axis=1)*256, 0, 255).astype(np.uint8)
    img3 = np.clip(np.amax(voxels, axis=2)*256, 0, 255).astype(np.uint8)
    if concat:
        dim = img1.shape[0]
        line = np.zeros((dim, 2), dtype=np.uint8)
        whole_img = np.concatenate([img1, line, img2, line, img3], axis=1)
        return whole_img
    else:
        return img1, img2, img3


def sdf2voxel(points, values, vox_dim=64):
    points = np.round(points).astype(int)
    voxels = np.zeros([vox_dim, vox_dim, vox_dim], np.uint8)
    discrete_values = np.zeros_like(values, dtype=np.uint8)
    discrete_values[np.where(values > 0.5)] = 1
    # print("SDF nonzero ratio: {}".format(np.sum(values > 0.5) / values.size))
    voxels[points[:, 0], points[:, 1], points[:, 2]] = np.reshape(discrete_values, [-1])
    return voxels


def visualize_sdf(points, values, concat=False, vox_dim=64):
    voxels = sdf2voxel(points, values, vox_dim)
    return project_voxel_along_xyz(voxels, concat=concat)


def draw_voxel_model(voxels, is_show=True, save_path=None):
    import matplotlib.pyplot as plt
    if not is_show:
        import matplotlib
        matplotlib.use('Agg')
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors='b', edgecolor='k')

    if is_show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path, transparent=True)


def draw_parts_sdf(points, values, ax, limit=64):
    n_parts = values.shape[0]
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    # points = np.round(points).astype(np.int)
    for idx in range(n_parts):
        part_points = points[idx]
        # part_points = np.round(points[idx]).astype(np.int)
        part_values = values[idx]
        postive_points = part_points[np.where(part_values >= 0.5)[0]]
        # negative_points = part_points[np.where(part_values < 0.5)[0]]

        # voxels = np.zeros((64, 64, 64), dtype=np.bool)
        # voxels[postive_points[:, 0], postive_points[:, 1], postive_points[:, 2]] = True
        # ax.voxels(voxels, facecolors=colors[idx % len(colors)], edgecolor='k')

        ax.scatter(postive_points[:, 0], postive_points[:, 1], postive_points[:, 2], c=colors[idx % len(colors)])

    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_zlim(0, limit)


def partsdf2voxel(points, values, vox_dim=64, by_part=True):
    """

    :param points: (n_parts, n_points, 3) or [(n_points1, 3), (n_points2, 3), ...]
    :param values: (n_parts, n_points, 1) or [(n_points1, 1), (n_points2, 1), ...]
    :return: voxel: (vox_dim, vox_dim, vox_dim)
    """
    n_parts = len(points)
    # points = np.round(points).astype(int)
    voxels = np.zeros([vox_dim, vox_dim, vox_dim], np.uint8)
    for idx in range(n_parts):
        part_points = np.round(points[idx]).astype(int)
        part_values = values[idx]
        postive_points = part_points[np.where(part_values >= 0.5)[0]]
        voxels[postive_points[:, 0], postive_points[:, 1], postive_points[:, 2]] = idx + 1
    if not by_part:
        voxels[np.where(voxels >= 1)] = 1
    return voxels


def partsdf2mesh(points, values, affine=None, vox_dim=64, by_part=True):
    """

    :param points: (n_parts, n_points, 3) or [(n_points1, 3), (n_points2, 3), ...]
    :param values: (n_parts, n_points, 1) or [(n_points1, 1), (n_points2, 1), ...]
    :param affine: (n_parts, 1, 4)
    :param vox_dim: int
    :return:
    """
    # if vox_dim is None:
    #     vox_dim = vox_dim
    if not by_part:
        shape_voxel = partsdf2voxel(points, values, vox_dim=vox_dim, by_part=False)
        vertices, triangles = libmcubes.marching_cubes(shape_voxel, 0)
        shape_mesh = trimesh.Trimesh(vertices, triangles)
        return shape_mesh

    n_parts = len(points)
    colors = [[0, 0, 255, 255],      # blue
              [0, 255, 0, 255],      # green
              [255, 0, 0, 255],      # red
              [255, 255, 0, 255],    # yellow
              [0, 255, 255, 255],    # cyan
              [255, 0, 255, 255],    # Magenta
              [160, 32, 240, 255],   # purple
              [255, 255, 240, 255]]  # ivory
    shape_mesh = []
    for idx in range(n_parts):
        part_voxel = partsdf2voxel(np.asarray(points[idx:idx+1]), np.asarray(values[idx:idx+1]), vox_dim)
        vertices, triangles = libmcubes.marching_cubes(part_voxel, 0)
        if affine is not None:
            vertices = vertices * affine[idx, :, :1] + affine[idx, :, 1:] * vox_dim
        part_mesh = trimesh.Trimesh(vertices, triangles, face_colors=colors[idx % len(colors)])
        shape_mesh.append(part_mesh)
        # print(trimesh.visual.random_color())
    shape_mesh = trimesh.util.concatenate(shape_mesh)
    return shape_mesh


def affine2boxmesh(affines):
    """

    :param affines: (n_parts, 6), range (0, 1)
    :return:
    """
    from trimesh.path.creation import box_outline
    from trimesh.path.util import concatenate
    n_parts = len(affines)
    colors = [[0, 0, 255, 255],      # blue
              [0, 255, 0, 255],      # green
              [255, 0, 0, 255],      # red
              [255, 255, 0, 255],    # yellow
              [0, 255, 255, 255],    # cyan
              [255, 0, 255, 255],    # Magenta
              [160, 32, 240, 255],   # purple
              [255, 255, 240, 255]]  # ivory
    shape_box = []
    for idx in range(n_parts):
        part_trans = affines[idx, :3]
        part_size = affines[idx, 3:]
        trans_mat = np.eye(4)
        # translate to center of axis aligned bounds
        trans_mat[:3, 3] = part_trans
        part_box = box_outline(transform=trans_mat,
                               extents=part_size
                               )
        shape_box.append(part_box)
    shape_box = concatenate(shape_box)
    return shape_box


def minmax2points(minmax):
    minp = minmax[:3]
    maxp = minmax[3:]
    P = np.asarray([minp,
         [maxp[0], minp[1], minp[2]],
         [maxp[0], minp[1], maxp[2]],
         [minp[0], minp[1], maxp[2]],
         [minp[0], maxp[1], minp[2]],
         [maxp[0], maxp[1], minp[2]],
         maxp,
         [minp[0], maxp[1], maxp[2]],
         ])
    P = P[:, [2, 1, 0]]
    return P


def points2verts(Z):
    verts = [[Z[0], Z[1], Z[2], Z[3]],
             [Z[4], Z[5], Z[6], Z[7]],
             [Z[0], Z[1], Z[5], Z[4]],
             [Z[2], Z[3], Z[7], Z[6]],
             [Z[1], Z[2], Z[6], Z[5]],
             [Z[4], Z[7], Z[3], Z[0]]]
    return verts


def draw_parts_bbox(bboxes, ax, limit=64, transparency=0.25):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    n_parts = bboxes.shape[0]
    colors = [[0, 0, 255, 255],      # blue
              [0, 255, 0, 255],      # green
              [255, 0, 0, 255],      # red
              [255, 255, 0, 255],    # yellow
              [0, 255, 255, 255],    # cyan
              [255, 0, 255, 255],    # Magenta
              [160, 32, 240, 255],   # purple
              [255, 255, 240, 255]]  # ivory
    colors = np.asarray(colors) / 255
    # points = np.round(points).astype(np.int)
    for idx in range(n_parts):
        bbox = bboxes[idx]
        points = minmax2points(bbox)
        verts = points2verts(points)

        pc = Poly3DCollection(verts, linewidths=0.5, edgecolors='k', alpha=transparency)
        pc.set_facecolor(colors[idx % len(colors)])
        ax.add_collection3d(pc)

        # size = (bbox[3:] - bbox[:3]).tolist()
        # for s, e in combinations(np.array(list(product(bbox[[2, 5]], bbox[[1, 4]], bbox[[0, 3]]))), 2):
        #     if np.sum(np.abs(s - e)) in size:
        #         ax.plot3D(*zip(s, e), color=colors[idx % len(colors)], alpha=transparency)

    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_zlim(0, limit)


def affine2bboxes(affine, limit=64):
    mins = (affine[:, :3] - affine[:, 3:6] / 2) * limit
    maxs = (affine[:, :3] + affine[:, 3:6] / 2) * limit
    bboxes = np.clip(np.round(np.concatenate([mins, maxs], axis=1)).astype(int), 0, limit - 1)
    return bboxes


def draw_parts_bbox_voxel(affine=None, bboxes=None, limit=64, proj=True):
    if bboxes is None:
        bboxes = affine2bboxes(affine, limit)
    n_parts = len(bboxes)
    voxel = np.zeros((limit, limit, limit), dtype=np.uint8)
    for idx in range(n_parts):
        bbox = bboxes[idx]
        size = (bbox[3:] - bbox[:3]).tolist()
        for s, e in combinations(np.array(list(product(bbox[[0, 3]], bbox[[1, 4]], bbox[[2, 5]]))), 2):
            if np.sum(np.abs(s - e)) in size:
                if s[0] != e[0]:
                    voxel[s[0]:e[0], s[1], s[2]] = 1
                elif s[1] != e[1]:
                    voxel[s[0], s[1]:e[1], s[2]] = 1
                else:
                    voxel[s[0], s[1], s[2]:e[2]] = 1
    if proj:
        return project_voxel_along_xyz(voxel, concat=True)
    return voxel


if __name__ == "__main__":
    pass
