#!/usr/bin/env python3
"""Publish a PCT tomogram pickle as an aligned, transient ROS 2 point cloud."""

import argparse
import os
import pickle
import time
from collections import deque

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Empty
from visualization_msgs.msg import Marker

from pct_goal_editor import PCTGoalEditor


MAX_TOMOGRAM_BYTES = 1_000_000_000
MAX_VISUALIZATION_POINTS = 5_000_000


def remove_small_components(mask, minimum_cells):
    """Keep 8-connected mask components containing at least minimum_cells."""
    if minimum_cells <= 1:
        return mask

    dim_x, dim_y = mask.shape
    remaining = set(int(index) for index in np.flatnonzero(mask))
    kept = np.zeros(mask.size, dtype=bool)
    while remaining:
        seed = remaining.pop()
        component = [seed]
        pending = deque([seed])
        while pending:
            index = pending.popleft()
            x_idx, y_idx = divmod(index, dim_y)
            for offset_x in (-1, 0, 1):
                neighbor_x = x_idx + offset_x
                if neighbor_x < 0 or neighbor_x >= dim_x:
                    continue
                for offset_y in (-1, 0, 1):
                    if offset_x == 0 and offset_y == 0:
                        continue
                    neighbor_y = y_idx + offset_y
                    if neighbor_y < 0 or neighbor_y >= dim_y:
                        continue
                    neighbor = neighbor_x * dim_y + neighbor_y
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.append(neighbor)
                        pending.append(neighbor)
        if len(component) >= minimum_cells:
            kept[component] = True
    return kept.reshape(mask.shape)


def load_tomogram(path):
    with open(path, 'rb') as stream:
        payload = pickle.load(stream)

    if not isinstance(payload, dict):
        raise ValueError('tomogram payload must be a dictionary')
    required = {'data', 'resolution', 'center', 'slice_h0', 'slice_dh'}
    missing = required.difference(payload)
    if missing:
        raise ValueError('tomogram is missing keys: %s' % sorted(missing))

    data = np.asarray(payload['data'])
    if data.ndim != 4 or data.shape[0] < 5 or data.shape[1] < 1:
        raise ValueError(
            'tomogram data must have shape [>=5, layers, dim_x, dim_y]'
        )
    if data.nbytes > MAX_TOMOGRAM_BYTES:
        raise ValueError(
            'tomogram data is too large: %d bytes > %d bytes'
            % (data.nbytes, MAX_TOMOGRAM_BYTES)
        )
    if not np.issubdtype(data.dtype, np.floating):
        raise ValueError('tomogram data must use a floating-point dtype')

    resolution = float(payload['resolution'])
    center = np.asarray(payload['center'], dtype=np.float64).reshape(-1)
    slice_h0 = float(payload['slice_h0'])
    slice_dh = float(payload['slice_dh'])
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError('tomogram resolution must be finite and positive')
    if center.size != 2 or not np.isfinite(center).all():
        raise ValueError('tomogram center must contain two finite values')
    if not np.isfinite(slice_h0):
        raise ValueError('tomogram slice_h0 must be finite')
    if not np.isfinite(slice_dh) or slice_dh <= 0.0:
        raise ValueError('tomogram slice_dh must be finite and positive')

    return data, resolution, center, slice_h0, slice_dh


def calculate_z_alignment(
    data,
    resolution,
    center,
    slice_h0,
    slice_dh,
    pose,
    path_height_offset,
    cost_threshold,
    max_z_alignment,
):
    dim_x, dim_y = data.shape[2], data.shape[3]
    grid_xy = (
        np.round((pose[:2] - center) / resolution).astype(np.int64)
        + np.array([dim_x // 2, dim_y // 2], dtype=np.int64)
    )
    x_idx, y_idx = int(grid_xy[0]), int(grid_xy[1])
    if x_idx < 0 or x_idx >= dim_x or y_idx < 0 or y_idx >= dim_y:
        raise ValueError(
            'localization (%.3f, %.3f) is outside tomogram bounds'
            % (pose[0], pose[1])
        )

    costs = np.asarray(data[0, :, x_idx, y_idx], dtype=np.float64)
    heights = np.asarray(data[3, :, x_idx, y_idx], dtype=np.float64)
    valid = np.isfinite(costs) & np.isfinite(heights)
    valid &= costs <= cost_threshold
    valid &= heights > -100.0
    candidates = np.flatnonzero(valid)
    if candidates.size == 0:
        raise ValueError(
            'no traversable tomogram layer at localization (%.3f, %.3f)'
            % (pose[0], pose[1])
        )

    nominal_layer = (pose[2] - slice_h0) / slice_dh
    expected_ground = pose[2] - path_height_offset
    scores = np.abs(candidates - nominal_layer)
    scores += (
        0.25
        * np.abs(heights[candidates] - expected_ground)
        / max(slice_dh, 1.0e-6)
    )
    scores += 1.0e-3 * costs[candidates]
    layer = int(candidates[np.argmin(scores)])
    ground_z = float(heights[layer])
    z_alignment = float(pose[2] - (ground_z + path_height_offset))
    if not np.isfinite(z_alignment) or abs(z_alignment) > max_z_alignment:
        raise ValueError(
            'required Z alignment %.3fm exceeds %.3fm'
            % (z_alignment, max_z_alignment)
        )
    return z_alignment, layer, ground_z


def build_visualization_points(
    data,
    resolution,
    center,
    slice_dh,
    z_alignment,
    grid_stride=1,
    cost_threshold=None,
    minimum_component_cells=1,
    minimum_z=None,
    maximum_z=None,
    hide_terminal_envelope=False,
):
    if grid_stride < 1:
        raise ValueError('grid_stride must be at least 1')
    if minimum_component_cells < 1:
        raise ValueError('minimum_component_cells must be at least 1')
    if minimum_z is not None and maximum_z is not None:
        if minimum_z > maximum_z:
            raise ValueError('minimum_z must not exceed maximum_z')
    traversability = np.array(data[0], copy=True)
    ground = np.array(data[3], copy=True)
    dim_x, dim_y = ground.shape[1], ground.shape[2]
    chunks = []
    point_count = 0

    for layer in range(ground.shape[0]):
        if layer + 1 < ground.shape[0]:
            covered = (ground[layer + 1] - ground[layer]) < slice_dh
            ground[layer, covered] = np.nan
            traversability[layer + 1, covered] = np.minimum(
                traversability[layer, covered],
                traversability[layer + 1, covered],
            )

        # The terminal PCT layer is the cumulative top envelope: every XY
        # column keeps its highest return below the terminal slice. Keep it in
        # the de-duplication pass above, but optionally omit it from the display
        # so roofs and ceilings do not obscure lower navigable layers.
        if hide_terminal_envelope and layer + 1 == ground.shape[0]:
            continue

        aligned_ground = ground[layer] + z_alignment
        valid = np.isfinite(ground[layer]) & np.isfinite(traversability[layer])
        if cost_threshold is not None:
            valid &= traversability[layer] <= cost_threshold
        if minimum_z is not None:
            valid &= aligned_ground >= minimum_z
        if maximum_z is not None:
            valid &= aligned_ground <= maximum_z
        valid = remove_small_components(valid, minimum_component_cells)
        x_idx, y_idx = np.nonzero(valid)
        if grid_stride > 1:
            keep = (x_idx % grid_stride == 0) & (y_idx % grid_stride == 0)
            x_idx = x_idx[keep]
            y_idx = y_idx[keep]
        if x_idx.size == 0:
            continue
        point_count += int(x_idx.size)
        if point_count > MAX_VISUALIZATION_POINTS:
            raise ValueError(
                'visualization contains too many points: %d > %d'
                % (point_count, MAX_VISUALIZATION_POINTS)
            )

        points = np.empty((x_idx.size, 4), dtype=np.float32)
        points[:, 0] = (
            (x_idx.astype(np.float32) - 0.5 * dim_x) * resolution
            + center[0]
        )
        points[:, 1] = (
            (y_idx.astype(np.float32) - 0.5 * dim_y) * resolution
            + center[1]
        )
        points[:, 2] = aligned_ground[x_idx, y_idx].astype(np.float32)
        points[:, 3] = traversability[layer, x_idx, y_idx].astype(np.float32)
        chunks.append(points)

    if not chunks:
        raise ValueError('tomogram has no finite visualization points')
    return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype='<f4')


def create_cloud(points, frame_id, stamp):
    message = PointCloud2()
    message.header.frame_id = frame_id
    message.header.stamp = stamp
    message.height = 1
    message.width = int(points.shape[0])
    message.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(
            name='intensity', offset=12, datatype=PointField.FLOAT32, count=1
        ),
    ]
    message.is_bigendian = False
    message.point_step = 16
    message.row_step = message.point_step * message.width
    message.data = points.tobytes(order='C')
    message.is_dense = True
    return message


def create_cloud_chunks(points, frame_id, chunk_points):
    if chunk_points < 1:
        raise ValueError('chunk_points must be at least 1')
    chunk_count = (points.shape[0] + chunk_points - 1) // chunk_points
    chunks = []
    for chunk_index in range(chunk_count):
        start = chunk_index * chunk_points
        stop = min(points.shape[0], start + chunk_points)
        # This is a private transport topic. The timestamp carries compact
        # framing metadata while each PointCloud2 remains structurally valid:
        # sec=total chunks, nanosec=zero-based chunk index.
        stamp = Time(sec=int(chunk_count), nanosec=int(chunk_index))
        chunks.append(create_cloud(points[start:stop], frame_id, stamp))
    return chunks


class TomogramPublisher(Node):
    def __init__(self, args):
        super().__init__('pct_tomogram_visualizer')
        self.args = args
        self.started = False
        self.cloud = None
        self.cloud_chunks = []
        self.chunk_publish_cursor = None
        self.goal_editor = None
        self.latest_pose = None
        self.next_start_attempt = 0.0
        self.scan_path_relay_count = 0
        self.last_subscription_count = 0
        self.last_chunk_subscription_count = 0
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publisher = (
            self.create_publisher(PointCloud2, args.topic, qos)
            if args.topic
            else None
        )
        chunk_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=512,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.chunk_publisher = (
            self.create_publisher(PointCloud2, args.chunk_topic, chunk_qos)
            if args.chunk_topic
            else None
        )
        self.chunk_request_subscription = (
            self.create_subscription(
                Empty,
                args.chunk_request_topic,
                self.chunk_request_callback,
                QoSProfile(
                    history=HistoryPolicy.KEEP_LAST,
                    depth=10,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.VOLATILE,
                ),
            )
            if args.chunk_request_topic
            else None
        )
        scan_visualization_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.scan_path_publisher = self.create_publisher(
            Path, args.scan_path_topic, scan_visualization_qos
        )
        self.scan_marker_subscription = self.create_subscription(
            Marker,
            args.scan_marker_source_topic,
            self.scan_marker_callback,
            scan_visualization_qos,
        )
        self.pose_subscription = self.create_subscription(
            PoseStamped, args.pose_topic, self.pose_callback, qos_profile_sensor_data
        )
        self.subscription_timer = self.create_timer(
            0.25, self.publish_for_new_subscribers
        )
        self.chunk_timer = self.create_timer(
            args.chunk_period, self.publish_next_chunk
        )
        if args.publish_before_localization:
            self.prepare_prelocalization_preview()
            self.get_logger().info(
                'Published map-frame preview; waiting for %s to apply final '
                'Z alignment'
                % args.pose_topic
            )
        else:
            self.get_logger().info(
                'Waiting for %s before publishing tomogram %s'
                % (args.pose_topic, os.path.abspath(args.tomogram))
            )

    def update_visualization_outputs(self, points, description):
        if self.publisher is not None:
            self.cloud = create_cloud(
                points, self.args.frame_id, self.get_clock().now().to_msg()
            )
            self.publisher.publish(self.cloud)
            self.get_logger().info(
                'Published %d %s PCT cells on %s'
                % (points.shape[0], description, self.args.topic)
            )
        if self.chunk_publisher is not None:
            self.cloud_chunks = create_cloud_chunks(
                points, self.args.frame_id, self.args.chunk_points
            )
            if self.chunk_publisher.get_subscription_count() > 0:
                self.schedule_chunk_publish()
            self.get_logger().info(
                'Prepared %d %s PCT cells as %d chunks on %s'
                % (
                    points.shape[0],
                    description,
                    len(self.cloud_chunks),
                    self.args.chunk_topic,
                )
            )

    def prepare_prelocalization_preview(self):
        try:
            data, resolution, center, _, slice_dh = load_tomogram(
                self.args.tomogram
            )
            points = build_visualization_points(
                data,
                resolution,
                center,
                slice_dh,
                0.0,
                self.args.grid_stride,
                self.args.visualization_cost_threshold,
                self.args.minimum_component_cells,
                hide_terminal_envelope=self.args.hide_terminal_envelope,
            )
            self.update_visualization_outputs(
                points, 'pre-localization preview'
            )
        except Exception as error:
            self.get_logger().error(
                'Cannot publish pre-localization PCT preview: %s' % error
            )

    def schedule_chunk_publish(self):
        if self.chunk_publisher is None or not self.cloud_chunks:
            return
        self.chunk_publish_cursor = 0

    def chunk_request_callback(self, _message):
        if not self.cloud_chunks:
            return
        if self.chunk_publish_cursor is not None:
            return
        self.schedule_chunk_publish()
        self.get_logger().info(
            'Received PCT map resend request on %s'
            % self.args.chunk_request_topic
        )

    def publish_next_chunk(self):
        if self.chunk_publish_cursor is None:
            return
        if self.chunk_publish_cursor >= len(self.cloud_chunks):
            self.get_logger().info(
                'Published %d PCT map chunks on %s'
                % (len(self.cloud_chunks), self.args.chunk_topic)
            )
            self.chunk_publish_cursor = None
            return
        self.chunk_publisher.publish(
            self.cloud_chunks[self.chunk_publish_cursor]
        )
        self.chunk_publish_cursor += 1

    def scan_marker_callback(self, message):
        if message.type != Marker.LINE_STRIP:
            return
        path = Path()
        path.header.frame_id = self.args.frame_id
        path.header.stamp = self.get_clock().now().to_msg()
        for point in message.points:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position = point
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.scan_path_publisher.publish(path)
        self.scan_path_relay_count += 1
        if self.scan_path_relay_count == 1:
            self.get_logger().info(
                "Converting SCAN trajectory Marker %s -> Path %s "
                "with frame '%s' -> '%s'"
                % (
                    self.args.scan_marker_source_topic,
                    self.args.scan_path_topic,
                    message.header.frame_id,
                    self.args.frame_id,
                )
            )

    def pose_callback(self, message):
        if message.header.frame_id.lstrip('/') != self.args.frame_id:
            self.get_logger().error(
                "Ignoring pose frame '%s'; expected '%s'"
                % (message.header.frame_id, self.args.frame_id)
            )
            return
        pose = np.array(
            [
                message.pose.position.x,
                message.pose.position.y,
                message.pose.position.z,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(pose).all():
            self.get_logger().error('Ignoring localization with non-finite position')
            return
        self.latest_pose = pose
        if self.goal_editor is not None:
            self.goal_editor.update_pose(pose)
        if self.started:
            return

        now = time.monotonic()
        if now < self.next_start_attempt:
            return
        self.next_start_attempt = now + 5.0
        try:
            data, resolution, center, slice_h0, slice_dh = load_tomogram(
                self.args.tomogram
            )
            z_alignment, layer, ground_z = calculate_z_alignment(
                data,
                resolution,
                center,
                slice_h0,
                slice_dh,
                pose,
                self.args.path_height_offset,
                self.args.cost_threshold,
                self.args.max_z_alignment,
            )
            minimum_z = (
                None
                if self.args.display_min_z_offset is None
                else pose[2] + self.args.display_min_z_offset
            )
            maximum_z = (
                None
                if self.args.display_max_z_offset is None
                else pose[2] + self.args.display_max_z_offset
            )
            points = build_visualization_points(
                data,
                resolution,
                center,
                slice_dh,
                z_alignment,
                self.args.grid_stride,
                self.args.visualization_cost_threshold,
                self.args.minimum_component_cells,
                minimum_z,
                maximum_z,
                self.args.hide_terminal_envelope,
            )
            if not self.args.disable_goal_editor:
                editor_data = data
                editor_resolution = resolution
                editor_center = center
                editor_slice_h0 = slice_h0
                editor_slice_dh = slice_dh
                editor_z_alignment = z_alignment
                if self.args.goal_editor_tomogram:
                    (
                        editor_data,
                        editor_resolution,
                        editor_center,
                        editor_slice_h0,
                        editor_slice_dh,
                    ) = load_tomogram(self.args.goal_editor_tomogram)
                    editor_z_alignment, _, _ = calculate_z_alignment(
                        editor_data,
                        editor_resolution,
                        editor_center,
                        editor_slice_h0,
                        editor_slice_dh,
                        pose,
                        self.args.path_height_offset,
                        self.args.cost_threshold,
                        self.args.max_z_alignment,
                    )
                self.goal_editor = PCTGoalEditor(
                    self,
                    editor_data,
                    editor_resolution,
                    editor_center,
                    editor_slice_h0,
                    editor_slice_dh,
                    editor_z_alignment,
                    self.args.frame_id,
                    self.args.clicked_topic,
                    self.args.goal_topic,
                    self.args.marker_topic,
                    self.args.marker_namespace,
                    self.args.path_height_offset,
                    self.args.cost_threshold,
                    self.args.layer_search_radius,
                )
                self.goal_editor.update_pose(pose)
                if self.args.goal_editor_tomogram:
                    self.get_logger().info(
                        'Goal editor uses full tomogram %s (Z offset=%+.3fm)'
                        % (
                            os.path.abspath(self.args.goal_editor_tomogram),
                            editor_z_alignment,
                        )
                    )
            self.update_visualization_outputs(points, 'aligned')
            self.get_logger().info(
                'Tomogram alignment: layer=%d, ground=%.3fm, Z offset=%+.3fm'
                % (layer, ground_z, z_alignment)
            )
            self.get_logger().info(
                'Visualization filters: cost<=%s, component>=%d cells, '
                'Z=[%s, %s]m, terminal envelope=%s'
                % (
                    self.args.visualization_cost_threshold,
                    self.args.minimum_component_cells,
                    '-inf' if minimum_z is None else '%.3f' % minimum_z,
                    '+inf' if maximum_z is None else '%.3f' % maximum_z,
                    'hidden' if self.args.hide_terminal_envelope else 'shown',
                )
            )
            self.started = True
        except Exception as error:
            self.get_logger().error(
                'Cannot publish PCT tomogram; retrying after 5s: %s' % error
            )

    def publish_for_new_subscribers(self):
        if self.publisher is not None and self.cloud is not None:
            subscription_count = self.publisher.get_subscription_count()
            if subscription_count > self.last_subscription_count:
                self.cloud.header.stamp = self.get_clock().now().to_msg()
                self.publisher.publish(self.cloud)
                self.get_logger().info(
                    'Republished PCT tomogram for %d subscriber(s)'
                    % subscription_count
                )
            self.last_subscription_count = subscription_count

        if self.chunk_publisher is not None:
            subscription_count = self.chunk_publisher.get_subscription_count()
            if (
                self.cloud_chunks
                and subscription_count > self.last_chunk_subscription_count
            ):
                self.schedule_chunk_publish()
                self.get_logger().info(
                    'Sending PCT map chunks to %d subscriber(s)'
                    % subscription_count
                )
            self.last_chunk_subscription_count = subscription_count


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--tomogram', required=True)
    parser.add_argument('--goal-editor-tomogram', default='')
    parser.add_argument('--topic', default='/pct_tomogram_aligned')
    parser.add_argument('--chunk-topic', default='')
    parser.add_argument('--chunk-request-topic', default='/pct_tomogram_request')
    parser.add_argument('--chunk-points', type=int, default=512)
    parser.add_argument('--chunk-period', type=float, default=0.02)
    parser.add_argument('--pose-topic', default='/pose_stamped')
    parser.add_argument(
        '--publish-before-localization',
        action='store_true',
        help=(
            'Publish an unshifted map-frame preview before the first pose; '
            'replace it with the aligned map after localization.'
        ),
    )
    parser.add_argument('--frame-id', default='map')
    parser.add_argument('--path-height-offset', type=float, default=0.5)
    parser.add_argument('--cost-threshold', type=float, default=20.0)
    parser.add_argument('--max-z-alignment', type=float, default=3.0)
    parser.add_argument('--grid-stride', type=int, default=1)
    parser.add_argument('--visualization-cost-threshold', type=float, default=None)
    parser.add_argument('--minimum-component-cells', type=int, default=1)
    parser.add_argument('--display-min-z-offset', type=float, default=None)
    parser.add_argument('--display-max-z-offset', type=float, default=None)
    parser.add_argument(
        '--hide-terminal-envelope',
        action='store_true',
        help=(
            'Hide the final cumulative top-envelope layer from visualization '
            'without changing the planner tomogram.'
        ),
    )
    parser.add_argument(
        '--initial-pose',
        type=float,
        nargs=3,
        metavar=('X', 'Y', 'Z'),
        default=None,
        help='Inject one map-frame pose for an offline visualization preview.',
    )
    parser.add_argument('--clicked-topic', default='/clicked_point')
    parser.add_argument('--goal-topic', default='/goal_pose')
    parser.add_argument('--marker-topic', default='/pct_waypoints')
    parser.add_argument('--marker-namespace', default='/pct_waypoint_editor')
    parser.add_argument('--layer-search-radius', type=float, default=0.5)
    parser.add_argument('--disable-goal-editor', action='store_true')
    parser.add_argument('--scan-marker-source-topic', default='/optimal_list')
    parser.add_argument('--scan-path-topic', default='/optimal_list_path')
    return parser.parse_known_args(args=args)


def main(args=None):
    cli_args, ros_args = parse_args(args)
    if not os.path.isfile(cli_args.tomogram):
        raise FileNotFoundError(cli_args.tomogram)
    if cli_args.goal_editor_tomogram:
        if not os.path.isfile(cli_args.goal_editor_tomogram):
            raise FileNotFoundError(cli_args.goal_editor_tomogram)
    if not cli_args.topic and not cli_args.chunk_topic:
        raise ValueError('--topic or --chunk-topic must be provided')
    if cli_args.chunk_points < 1:
        raise ValueError('--chunk-points must be at least 1')
    if cli_args.chunk_period <= 0.0:
        raise ValueError('--chunk-period must be positive')
    if cli_args.minimum_component_cells < 1:
        raise ValueError('--minimum-component-cells must be at least 1')
    if (
        cli_args.display_min_z_offset is not None
        and cli_args.display_max_z_offset is not None
        and cli_args.display_min_z_offset > cli_args.display_max_z_offset
    ):
        raise ValueError(
            '--display-min-z-offset must not exceed --display-max-z-offset'
        )
    if cli_args.initial_pose is not None:
        if not np.isfinite(cli_args.initial_pose).all():
            raise ValueError('--initial-pose values must be finite')

    rclpy.init(args=ros_args)
    node = TomogramPublisher(cli_args)
    if cli_args.initial_pose is not None:
        message = PoseStamped()
        message.header.frame_id = cli_args.frame_id
        message.pose.position.x = cli_args.initial_pose[0]
        message.pose.position.y = cli_args.initial_pose[1]
        message.pose.position.z = cli_args.initial_pose[2]
        message.pose.orientation.w = 1.0
        node.pose_callback(message)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.goal_editor is not None:
            node.goal_editor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()