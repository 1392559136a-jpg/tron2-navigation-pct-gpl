#!/usr/bin/env python3
"""Select a safe NX PCT processing profile before rebuilding a map."""

import argparse
import os
from pathlib import Path

from build_tomogram_offline import (
    DEFAULT_GPU_MEMORY_FRACTION,
    calculate_grid_geometry,
    format_gib,
    load_points,
)
from config.scene_map import SceneMap


GIB = 1024 ** 3
DEFAULT_MAX_STANDARD_BASE_GIB = 8.0
DEFAULT_HEADROOM = 1.10


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the standard PCT grid from result.pcd and choose "
            "standard or nx-large for a 16 GiB NX."
        )
    )
    parser.add_argument("--pcd", required=True, type=Path)
    parser.add_argument("--xy-dilation-cells", type=int, default=1)
    parser.add_argument(
        "--max-standard-base-gib",
        type=float,
        default=float(
            os.environ.get(
                "PCT_AUTO_STANDARD_MAX_GIB", DEFAULT_MAX_STANDARD_BASE_GIB
            )
        ),
    )
    parser.add_argument("--headroom", type=float, default=DEFAULT_HEADROOM)
    return parser.parse_args()


def read_system_memory():
    values = {}
    with Path("/proc/meminfo").open("r", encoding="utf-8") as stream:
        for line in stream:
            key, value = line.split(":", 1)
            fields = value.split()
            if fields:
                values[key] = int(fields[0]) * 1024
    return values["MemTotal"], values.get("MemAvailable", values["MemFree"])


def safe_standard_budget(total_bytes, available_bytes, maximum_bytes):
    return min(
        int(total_bytes * DEFAULT_GPU_MEMORY_FRACTION),
        int(available_bytes * 0.90),
        int(maximum_bytes),
    )


def choose_profile(required_bytes, budget_bytes, headroom=DEFAULT_HEADROOM):
    if headroom < 1.0:
        raise ValueError("headroom must be at least 1.0")
    adjusted_required = int(required_bytes * headroom)
    return (
        "standard" if adjusted_required <= int(budget_bytes) else "nx-large"
    )


def main():
    args = parse_args()
    if not args.pcd.is_file():
        raise FileNotFoundError("PCD file does not exist: {}".format(args.pcd))
    if args.xy_dilation_cells < 0 or args.xy_dilation_cells > 4:
        raise ValueError("xy-dilation-cells must be in [0, 4]")
    if args.max_standard_base_gib <= 0.0:
        raise ValueError("max-standard-base-gib must be positive")

    points = load_points(args.pcd)
    scene = SceneMap()
    xy_padding = (
        args.xy_dilation_cells * float(scene.map.resolution)
    )
    geometry = calculate_grid_geometry(
        points, scene, xy_padding=xy_padding
    )
    total_bytes, available_bytes = read_system_memory()
    budget_bytes = safe_standard_budget(
        total_bytes,
        available_bytes,
        args.max_standard_base_gib * GIB,
    )
    profile = choose_profile(
        geometry["base_buffer_bytes"], budget_bytes, args.headroom
    )

    print(
        "AUTO_PROFILE_STANDARD_GRID={}x{}x{}".format(
            geometry["map_dim_x"],
            geometry["map_dim_y"],
            geometry["n_slice_init"],
        )
    )
    print(
        "AUTO_PROFILE_STANDARD_BASE_BUFFERS={}".format(
            format_gib(geometry["base_buffer_bytes"])
        )
    )
    print(
        "AUTO_PROFILE_STANDARD_BUDGET={}".format(format_gib(budget_bytes))
    )
    print("AUTO_PROFILE_HEADROOM={:.2f}".format(args.headroom))
    print("SELECTED_PROFILE={}".format(profile))


if __name__ == "__main__":
    main()