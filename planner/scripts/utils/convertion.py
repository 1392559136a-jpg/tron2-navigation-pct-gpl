import numpy as np


def transTrajGrid2Map(grid_dim, center, resolution, traj_grid, z_offset=0.5):
    offset = np.array([grid_dim[1] // 2, grid_dim[0] // 2, 0])
    center_ = np.array([center[1], center[0], float(z_offset)])

    traj_grid = (traj_grid - offset) * resolution + center_

    traj_map = np.stack([traj_grid[:, 1], traj_grid[:, 0], traj_grid[:, 2]], axis=1)

    return traj_map
