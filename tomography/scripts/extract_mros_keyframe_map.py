#!/usr/bin/env python3
"""Extract a dense map PCD from an MROS/ROS1 keyframe bag."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from pcd_io import write_xyz


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge /keyframe_pcd using /keyframe_pose from an MROS ROS1 bag."
        )
    )
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--base-transform",
        choices=("rs-fairy", "livox-mid360", "identity"),
        default="rs-fairy",
        help="Transform prepended by the map saver before writing result.pcd.",
    )
    parser.add_argument(
        "--min-relative-z",
        type=float,
        default=None,
        help="Keep points at or above this Z in each keyframe sensor frame.",
    )
    parser.add_argument(
        "--max-relative-z",
        type=float,
        default=None,
        help="Keep points at or below this Z in each keyframe sensor frame.",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pointcloud2_xyz(message):
    if message.height <= 0 or message.width <= 0 or message.point_step <= 0:
        raise ValueError("PointCloud2 has invalid dimensions or point_step")
    fields = {field.name: field for field in message.fields}
    missing = {"x", "y", "z"}.difference(fields)
    if missing:
        raise ValueError("PointCloud2 is missing fields: {}".format(sorted(missing)))
    if any(fields[name].datatype != 7 for name in ("x", "y", "z")):
        raise ValueError("PointCloud2 XYZ fields must use FLOAT32 datatype")

    endian = ">" if message.is_bigendian else "<"
    dtype = np.dtype(
        {
            "names": ("x", "y", "z"),
            "formats": (endian + "f4", endian + "f4", endian + "f4"),
            "offsets": tuple(int(fields[name].offset) for name in ("x", "y", "z")),
            "itemsize": int(message.point_step),
        }
    )
    expected = int(message.row_step) * int(message.height)
    raw = np.asarray(message.data, dtype=np.uint8)
    if raw.nbytes < expected:
        raise ValueError(
            "PointCloud2 data is truncated: {} < {}".format(raw.nbytes, expected)
        )

    rows = []
    row_points = int(message.width)
    packed_row_bytes = row_points * int(message.point_step)
    for row in range(int(message.height)):
        start = row * int(message.row_step)
        row_data = raw[start : start + packed_row_bytes]
        structured = np.frombuffer(row_data, dtype=dtype, count=row_points)
        rows.append(
            np.column_stack((structured["x"], structured["y"], structured["z"]))
        )
    points = np.concatenate(rows, axis=0).astype(np.float32, copy=False)
    return np.ascontiguousarray(points[np.isfinite(points).all(axis=1)])


def quaternion_matrix(quaternion):
    q = np.array(
        [quaternion.w, quaternion.x, quaternion.y, quaternion.z],
        dtype=np.float64,
    )
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("Pose contains an invalid quaternion")
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_matrix(message):
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_matrix(message.pose.orientation)
    matrix[:3, 3] = (
        message.pose.position.x,
        message.pose.position.y,
        message.pose.position.z,
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Pose contains non-finite values")
    return matrix


def base_transform(name):
    matrix = np.eye(4, dtype=np.float64)
    if name == "rs-fairy":
        yaw = np.deg2rad(-60.0)
        matrix[:3, :3] = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    elif name == "livox-mid360":
        matrix[1, 1] = -1.0
        matrix[2, 2] = -1.0
    return matrix


def filter_relative_height(points, minimum_z=None, maximum_z=None):
    if minimum_z is not None and maximum_z is not None:
        if minimum_z > maximum_z:
            raise ValueError("min-relative-z must not exceed max-relative-z")
    keep = np.ones(points.shape[0], dtype=bool)
    if minimum_z is not None:
        keep &= points[:, 2] >= minimum_z
    if maximum_z is not None:
        keep &= points[:, 2] <= maximum_z
    return np.ascontiguousarray(points[keep], dtype=np.float32)


def load_keyframes(path):
    typestore = get_typestore(Stores.ROS1_NOETIC)
    clouds = []
    poses = []
    with Reader(path) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in ("/keyframe_pcd", "/keyframe_pose")
        ]
        for connection, timestamp, rawdata in reader.messages(
            connections=connections
        ):
            message = typestore.deserialize_ros1(rawdata, connection.msgtype)
            record = (int(timestamp), message)
            if connection.topic == "/keyframe_pcd":
                clouds.append(record)
            else:
                poses.append(record)

    if not clouds or len(clouds) != len(poses):
        raise ValueError(
            "bag has {} clouds and {} poses".format(len(clouds), len(poses))
        )
    clouds.sort(key=lambda item: item[0])
    poses.sort(key=lambda item: item[0])
    for index, (cloud, pose) in enumerate(zip(clouds, poses)):
        if cloud[0] != pose[0]:
            raise ValueError(
                "keyframe {} timestamp mismatch: {} != {}".format(
                    index, cloud[0], pose[0]
                )
            )
    return clouds, poses


def write_pcd_atomic(points, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp" + output.suffix)
    try:
        write_xyz(temporary, points)
        os.replace(str(temporary), str(output))
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    args = parse_args()
    if not args.bag.is_file():
        raise FileNotFoundError("bag does not exist: {}".format(args.bag))
    if args.output.suffix.lower() != ".pcd":
        raise ValueError("output must use the .pcd extension")
    if args.min_relative_z is not None and not np.isfinite(args.min_relative_z):
        raise ValueError("min-relative-z must be finite")
    if args.max_relative_z is not None and not np.isfinite(args.max_relative_z):
        raise ValueError("max-relative-z must be finite")
    if (
        args.min_relative_z is not None
        and args.max_relative_z is not None
        and args.min_relative_z > args.max_relative_z
    ):
        raise ValueError("min-relative-z must not exceed max-relative-z")

    clouds, poses = load_keyframes(args.bag)
    fixed_transform = base_transform(args.base_transform)
    merged = []
    input_points = 0
    for (_, cloud_message), (_, pose_message) in zip(clouds, poses):
        points = pointcloud2_xyz(cloud_message)
        input_points += int(points.shape[0])
        points = filter_relative_height(
            points, args.min_relative_z, args.max_relative_z
        )
        if points.shape[0] == 0:
            continue
        transform = fixed_transform.dot(pose_matrix(pose_message))
        transformed = points.astype(np.float64).dot(transform[:3, :3].T)
        transformed += transform[:3, 3]
        merged.append(transformed.astype(np.float32))

    if not merged:
        raise ValueError("relative-height filter removed every keyframe point")

    points = np.ascontiguousarray(np.concatenate(merged, axis=0), dtype=np.float32)
    write_pcd_atomic(points, args.output)
    metadata = {
        "source_bag": args.bag.name,
        "source_bag_sha256": sha256_file(args.bag),
        "base_transform": args.base_transform,
        "min_relative_z": args.min_relative_z,
        "max_relative_z": args.max_relative_z,
        "keyframe_count": len(clouds),
        "input_point_count": input_points,
        "point_count": int(points.shape[0]),
        "point_min": np.min(points, axis=0).astype(float).tolist(),
        "point_max": np.max(points, axis=0).astype(float).tolist(),
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print("BAG={}".format(args.bag.resolve()))
    print("BAG_SHA256={}".format(metadata["source_bag_sha256"]))
    print("KEYFRAMES={}".format(metadata["keyframe_count"]))
    print("INPUT_POINTS={}".format(input_points))
    print("OUTPUT_POINTS={}".format(metadata["point_count"]))
    print("MIN_RELATIVE_Z={}".format(args.min_relative_z))
    print("MAX_RELATIVE_Z={}".format(args.max_relative_z))
    print("POINT_MIN={}".format(metadata["point_min"]))
    print("POINT_MAX={}".format(metadata["point_max"]))
    print("OUTPUT={}".format(args.output.resolve()))


if __name__ == "__main__":
    main()
