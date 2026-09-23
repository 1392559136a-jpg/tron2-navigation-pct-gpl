"""Portable XYZ PCD I/O with Open3D and pypcd4 backends."""

from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None

try:
    from pypcd4 import Encoding, PointCloud
except ImportError:
    Encoding = None
    PointCloud = None


def _missing_backend_error():
    return RuntimeError(
        "PCD I/O requires Open3D or pypcd4; install open3d or pypcd4"
    )


def load_xyz(path):
    """Load finite XYZ points from an ASCII, binary, or compressed PCD."""
    path = Path(path)
    if o3d is not None:
        points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    elif PointCloud is not None:
        points = PointCloud.from_path(path).numpy(("x", "y", "z"))
    else:
        raise _missing_backend_error()

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("PCD contains no XYZ points: {}".format(path))
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] == 0:
        raise ValueError("PCD contains no finite XYZ points: {}".format(path))
    return np.ascontiguousarray(points, dtype=np.float32)


def write_xyz(path, points):
    """Write XYZ points as a binary-compressed PCD."""
    path = Path(path)
    points = np.ascontiguousarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("XYZ points must have shape (N, 3)")

    if o3d is not None:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(
            points.astype(np.float64, copy=False)
        )
        if not o3d.io.write_point_cloud(
            str(path), cloud, write_ascii=False, compressed=True
        ):
            raise RuntimeError("Open3D failed to write {}".format(path))
    elif PointCloud is not None:
        PointCloud.from_xyz_points(points).save(
            path, encoding=Encoding.BINARY_COMPRESSED
        )
    else:
        raise _missing_backend_error()


def nearest_neighbor_distances(reference, target):
    """Return each reference point's distance to its nearest target point."""
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    target = np.ascontiguousarray(target, dtype=np.float32)
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("reference points must have shape (N, 3)")
    if target.ndim != 2 or target.shape[1] != 3 or target.shape[0] == 0:
        raise ValueError("target points must have nonempty shape (N, 3)")

    if o3d is not None:
        reference_cloud = o3d.geometry.PointCloud()
        reference_cloud.points = o3d.utility.Vector3dVector(reference)
        target_cloud = o3d.geometry.PointCloud()
        target_cloud.points = o3d.utility.Vector3dVector(target)
        return np.asarray(
            reference_cloud.compute_point_cloud_distance(target_cloud)
        )

    try:
        from scipy.spatial import cKDTree
    except ImportError as error:
        raise RuntimeError(
            "SciPy is required for PCD comparison without Open3D"
        ) from error
    distances, _ = cKDTree(target).query(reference, k=1, workers=-1)
    return np.asarray(distances)
