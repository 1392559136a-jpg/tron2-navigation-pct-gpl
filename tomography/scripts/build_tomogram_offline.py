#!/usr/bin/env python3
"""Build a PCT tomogram from a PCD file without requiring ROS."""

import argparse
import hashlib
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
TOMOGRAPHY_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(TOMOGRAPHY_ROOT) not in sys.path:
    sys.path.insert(0, str(TOMOGRAPHY_ROOT))

from config.scene_map import SceneMap  # noqa: E402
from pcd_io import load_xyz  # noqa: E402

GPU_BASE_BUFFER_COUNT = 6
FLOAT32_BYTES = np.dtype(np.float32).itemsize
DEFAULT_GPU_MEMORY_FRACTION = 0.65


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a PCT tomogram offline using scene_map.py defaults."
    )
    parser.add_argument("--pcd", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resolution", type=float, default=None)
    parser.add_argument(
        "--ground-h",
        type=float,
        default=None,
        help="Override scene_map.py automatic 0.02-percentile ground height.",
    )
    parser.add_argument("--slice-dh", type=float, default=None)
    parser.add_argument("--interval-min", type=float, default=None)
    parser.add_argument("--interval-free", type=float, default=None)
    parser.add_argument("--step-max", type=float, default=None)
    parser.add_argument("--slope-max", type=float, default=None)
    parser.add_argument("--kernel-size", type=int, default=None)
    parser.add_argument("--standable-ratio", type=float, default=None)
    parser.add_argument("--safe-margin", type=float, default=None)
    parser.add_argument("--inflation", type=float, default=None)
    parser.add_argument(
        "--xy-dilation-cells",
        type=int,
        default=0,
        help="Replicate each point over an XY grid-radius to fill voxel-map holes.",
    )
    parser.add_argument(
        "--profile-name",
        choices=("scene_map.py", "nx-large"),
        default="scene_map.py",
        help="Record the selected processing profile in the pickle metadata.",
    )
    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=DEFAULT_GPU_MEMORY_FRACTION,
        help=(
            "Maximum fraction of total GPU memory permitted for the six base "
            "tomogram buffers (default: %(default)s)."
        ),
    )
    return parser.parse_args()


def validate_args(args):
    if not args.pcd.is_file():
        raise FileNotFoundError("PCD file does not exist: {}".format(args.pcd))
    positive_values = (
        ("resolution", args.resolution),
        ("slice-dh", args.slice_dh),
        ("interval-min", args.interval_min),
        ("interval-free", args.interval_free),
        ("step-max", args.step_max),
        ("slope-max", args.slope_max),
    )
    for name, value in positive_values:
        if value is not None and (not np.isfinite(value) or value <= 0.0):
            raise ValueError("{} must be finite and positive".format(name))
    nonnegative_values = (
        ("safe-margin", args.safe_margin),
        ("inflation", args.inflation),
    )
    for name, value in nonnegative_values:
        if value is not None and (not np.isfinite(value) or value < 0.0):
            raise ValueError("{} must be finite and nonnegative".format(name))
    if args.ground_h is not None and not np.isfinite(args.ground_h):
        raise ValueError("ground-h must be finite")
    if args.xy_dilation_cells < 0 or args.xy_dilation_cells > 4:
        raise ValueError("xy-dilation-cells must be in [0, 4]")
    if args.kernel_size is not None and (
        args.kernel_size < 3 or args.kernel_size % 2 == 0
    ):
        raise ValueError("kernel-size must be an odd integer >= 3")
    if args.standable_ratio is not None and not (
        0.0 < args.standable_ratio <= 1.0
    ):
        raise ValueError("standable-ratio must be in (0, 1]")
    if not np.isfinite(args.gpu_memory_fraction) or not (
        0.0 < args.gpu_memory_fraction <= 1.0
    ):
        raise ValueError("gpu-memory-fraction must be finite and in (0, 1]")


def load_points(pcd_path):
    return load_xyz(pcd_path)


def configure_scene(args):
    scene = SceneMap()
    overrides = (
        (scene.map, "resolution", args.resolution, float),
        (scene.map, "ground_h", args.ground_h, float),
        (scene.map, "slice_dh", args.slice_dh, float),
        (scene.trav, "interval_min", args.interval_min, float),
        (scene.trav, "interval_free", args.interval_free, float),
        (scene.trav, "step_max", args.step_max, float),
        (scene.trav, "slope_max", args.slope_max, float),
        (scene.trav, "kernel_size", args.kernel_size, int),
        (scene.trav, "standable_ratio", args.standable_ratio, float),
        (scene.trav, "safe_margin", args.safe_margin, float),
        (scene.trav, "inflation", args.inflation, float),
    )
    for owner, name, value, converter in overrides:
        if value is not None:
            setattr(owner, name, converter(value))
    return scene


def dilate_points_xy(points, resolution, radius_cells):
    if radius_cells == 0:
        return points
    offsets = np.arange(-radius_cells, radius_cells + 1, dtype=np.float32)
    offset_x, offset_y = np.meshgrid(offsets, offsets, indexing="ij")
    xy_offsets = np.column_stack(
        (offset_x.ravel(), offset_y.ravel(), np.zeros(offset_x.size))
    )
    xy_offsets *= np.float32(resolution)
    expanded = points[:, np.newaxis, :] + xy_offsets[np.newaxis, :, :]
    return np.ascontiguousarray(expanded.reshape(-1, 3), dtype=np.float32)


def calculate_grid_geometry(points, scene, xy_padding=0.0):
    points_min = np.min(points, axis=0)
    points_max = np.max(points, axis=0)
    if not np.isfinite(xy_padding) or xy_padding < 0.0:
        raise ValueError("xy_padding must be finite and nonnegative")
    if xy_padding:
        points_min = points_min.copy()
        points_max = points_max.copy()
        points_min[:2] -= xy_padding
        points_max[:2] += xy_padding
    resolution = float(scene.map.resolution)
    slice_dh = float(scene.map.slice_dh)
    if scene.map.ground_h is None:
        low_z = float(np.percentile(points[:, 2], 0.02))
        ground_h = float(np.floor(low_z / slice_dh) * slice_dh)
    else:
        ground_h = float(scene.map.ground_h)

    if points_max[2] <= ground_h:
        raise ValueError(
            "ground-h {:.3f} is not below point-cloud max Z {:.3f}".format(
                ground_h, float(points_max[2])
            )
        )

    map_dim_x = int(np.ceil((points_max[0] - points_min[0]) / resolution)) + 4
    map_dim_y = int(np.ceil((points_max[1] - points_min[1]) / resolution)) + 4
    n_slice_init = max(
        1, int(np.ceil((points_max[2] - ground_h) / slice_dh))
    )
    center = ((points_max[:2] + points_min[:2]) / 2.0).astype(np.float32)
    slice_h0 = ground_h + slice_dh

    cells = int(map_dim_x) * int(map_dim_y) * int(n_slice_init)
    base_buffer_bytes = cells * GPU_BASE_BUFFER_COUNT * FLOAT32_BYTES
    return {
        "points_min": points_min,
        "points_max": points_max,
        "resolution": resolution,
        "slice_dh": slice_dh,
        "ground_h": ground_h,
        "map_dim_x": map_dim_x,
        "map_dim_y": map_dim_y,
        "n_slice_init": n_slice_init,
        "center": center,
        "slice_h0": slice_h0,
        "cells": cells,
        "base_buffer_bytes": base_buffer_bytes,
    }


def format_gib(byte_count):
    return "{:.3f} GiB".format(float(byte_count) / (1024.0 ** 3))


def check_gpu_memory(geometry, memory_fraction, memory_info=None):
    if memory_info is None:
        import cupy as cp

        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    else:
        free_bytes, total_bytes = memory_info
    free_bytes = int(free_bytes)
    total_bytes = int(total_bytes)
    total_fraction_budget = int(total_bytes * memory_fraction)
    current_free_budget = int(free_bytes * 0.90)
    safe_budget = min(total_fraction_budget, current_free_budget)
    required_bytes = int(geometry["base_buffer_bytes"])

    print(
        "INITIAL_GRID={}x{}x{}".format(
            geometry["map_dim_x"],
            geometry["map_dim_y"],
            geometry["n_slice_init"],
        ),
        flush=True,
    )
    print("INITIAL_GRID_CELLS={}".format(geometry["cells"]), flush=True)
    print(
        "GPU_BASE_BUFFERS={}xfloat32={}".format(
            GPU_BASE_BUFFER_COUNT, format_gib(required_bytes)
        ),
        flush=True,
    )
    print(
        "GPU_MEMORY_FREE={} TOTAL={} SAFE_BUDGET={}".format(
            format_gib(free_bytes),
            format_gib(total_bytes),
            format_gib(safe_budget),
        ),
        flush=True,
    )

    if required_bytes > safe_budget:
        raise MemoryError(
            "tomogram base buffers need at least {}, exceeding the safe GPU "
            "budget {}. The actual peak is higher. Use the nx-large profile "
            "(0.2 m resolution with bounded keyframe range/Z), reduce the map "
            "bounds, or run on a larger GPU; refusing allocation before the "
            "system is killed.".format(
                format_gib(required_bytes), format_gib(safe_budget)
            )
        )

    return {
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "safe_budget_bytes": safe_budget,
        "memory_fraction": float(memory_fraction),
    }


def build_tomogram(points, scene, memory_fraction=DEFAULT_GPU_MEMORY_FRACTION):
    # Keep geometry/profile inspection usable without importing CuPy. The GPU
    # implementation is needed only after the allocation preflight passes.
    from tomogram import Tomogram

    geometry = calculate_grid_geometry(points, scene)
    gpu_memory = check_gpu_memory(geometry, memory_fraction)

    builder = Tomogram(scene)
    builder.initMappingEnv(
        geometry["center"],
        geometry["map_dim_x"],
        geometry["map_dim_y"],
        geometry["n_slice_init"],
        geometry["slice_h0"],
    )
    started = time.monotonic()
    layers_t, trav_gx, trav_gy, layers_g, layers_c, timing = builder.point2map(
        points
    )
    elapsed = time.monotonic() - started
    stacked = np.stack((layers_t, trav_gx, trav_gy, layers_g, layers_c))

    if stacked.ndim != 4 or stacked.shape[1] == 0:
        raise RuntimeError("tomogram generation returned an invalid array")
    if not np.isfinite(layers_t).all():
        raise RuntimeError("traversability layer contains non-finite values")

    metadata = {
        "data": stacked.astype(np.float16),
        "resolution": geometry["resolution"],
        "center": geometry["center"],
        "slice_h0": geometry["slice_h0"],
        "slice_dh": geometry["slice_dh"],
        "ground_h": geometry["ground_h"],
    }
    stats = {
        "points": int(points.shape[0]),
        "point_min": geometry["points_min"],
        "point_max": geometry["points_max"],
        "map_dim_x": geometry["map_dim_x"],
        "map_dim_y": geometry["map_dim_y"],
        "n_slice_init": geometry["n_slice_init"],
        "initial_grid_cells": geometry["cells"],
        "gpu_base_buffer_bytes": geometry["base_buffer_bytes"],
        "gpu_memory": gpu_memory,
        "n_slice_output": int(stacked.shape[1]),
        "elapsed": elapsed,
        "timing": timing,
    }
    return metadata, stats


def atomic_pickle_dump(data, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(output_path))
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    validate_args(args)
    scene = configure_scene(args)
    resolution = float(scene.map.resolution)
    source_pcd_sha256 = sha256_file(args.pcd)
    source_points = load_points(args.pcd)
    points = dilate_points_xy(
        source_points, resolution, args.xy_dilation_cells
    )
    metadata, stats = build_tomogram(
        points, scene, memory_fraction=args.gpu_memory_fraction
    )
    metadata["source_pcd_sha256"] = source_pcd_sha256
    metadata["source_pcd_name"] = args.pcd.name
    metadata["source_point_count"] = int(source_points.shape[0])
    metadata["xy_dilation_cells"] = int(args.xy_dilation_cells)
    metadata["build_parameters"] = {
        "profile": args.profile_name,
        "kernel_size": int(scene.trav.kernel_size),
        "interval_min": float(scene.trav.interval_min),
        "interval_free": float(scene.trav.interval_free),
        "standable_ratio": float(scene.trav.standable_ratio),
        "step_max": float(scene.trav.step_max),
        "slope_max": float(scene.trav.slope_max),
        "cost_barrier": float(scene.trav.cost_barrier),
        "safe_margin": float(scene.trav.safe_margin),
        "inflation": float(scene.trav.inflation),
    }
    atomic_pickle_dump(metadata, args.output)

    print("PCD={}".format(args.pcd.resolve()))
    print("PCD_SHA256={}".format(source_pcd_sha256))
    print("OUTPUT={}".format(args.output.resolve()))
    print("SOURCE_POINTS={}".format(source_points.shape[0]))
    print("PROCESSED_POINTS={}".format(stats["points"]))
    print("XY_DILATION_CELLS={}".format(args.xy_dilation_cells))
    print("POINT_MIN={}".format(np.round(stats["point_min"], 3).tolist()))
    print("POINT_MAX={}".format(np.round(stats["point_max"], 3).tolist()))
    print("INITIAL_GRID_CELLS={}".format(stats["initial_grid_cells"]))
    print(
        "GPU_BASE_BUFFER_BYTES={}".format(stats["gpu_base_buffer_bytes"])
    )
    print(
        "GPU_SAFE_BUDGET_BYTES={}".format(
            stats["gpu_memory"]["safe_budget_bytes"]
        )
    )
    print("OUTPUT_LAYERS={}".format(stats["n_slice_output"]))
    print("CENTER={}".format(np.round(metadata["center"], 3).tolist()))
    print("RESOLUTION={}".format(metadata["resolution"]))
    print("GROUND_H={}".format(metadata["ground_h"]))
    print("SLICE_H0={}".format(metadata["slice_h0"]))
    print("SLICE_DH={}".format(metadata["slice_dh"]))
    print("BUILD_PARAMETERS={}".format(metadata["build_parameters"]))
    print("GPU_TIME_MS={}".format(stats["timing"]))
    print("ELAPSED_SEC={:.3f}".format(stats["elapsed"]))


if __name__ == "__main__":
    main()
