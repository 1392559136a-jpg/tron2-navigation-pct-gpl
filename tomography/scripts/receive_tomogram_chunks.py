#!/usr/bin/env python3
"""Reassemble MTU-friendly PCT map chunks into a local PointCloud2 topic."""

import argparse
import copy
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
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

from pct_goal_editor import PCTGoalEditor, PointCloudLayerIndex


MAX_CHUNKS = 100_000
MAX_POINT_BYTES = 200_000_000


def extract_xyzi(message):
    fields = {field.name: field for field in message.fields}
    required = ('x', 'y', 'z', 'intensity')
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError('PointCloud2 is missing fields: %s' % missing)

    offsets = []
    for name in required:
        field = fields[name]
        if field.datatype != PointField.FLOAT32 or field.count != 1:
            raise ValueError('%s field must be one FLOAT32 value' % name)
        if field.offset < 0 or field.offset + 4 > message.point_step:
            raise ValueError('%s field has an invalid offset' % name)
        offsets.append(int(field.offset))

    point_count = int(message.width) * int(message.height)
    expected_bytes = point_count * int(message.point_step)
    if len(message.data) != expected_bytes:
        raise ValueError('PointCloud2 has an inconsistent byte count')

    byte_order = '>' if message.is_bigendian else '<'
    dtype = np.dtype(
        {
            'names': list(required),
            'formats': [byte_order + 'f4'] * len(required),
            'offsets': offsets,
            'itemsize': int(message.point_step),
        }
    )
    records = np.frombuffer(bytes(message.data), dtype=dtype, count=point_count)
    return np.column_stack([records[name] for name in required])


class TomogramChunkReceiver(Node):
    def __init__(self, args):
        super().__init__('pct_tomogram_receiver')
        self.args = args
        self.expected_chunks = None
        self.chunks = {}
        self.template = None
        self.output_cloud = None
        self.last_subscription_count = 0
        self.latest_pose = None
        self.latest_pose_message = None
        self.goal_editor = None
        self.last_ground_pose_publish_time = 0.0
        self.last_ground_pose_layer = None
        self.last_ground_pose_z = None
        self.last_chunk_time = None
        self.last_request_time = 0.0
        self.request_count = 0

        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=512,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publisher = self.create_publisher(
            PointCloud2, args.output_topic, output_qos
        )
        self.ground_pose_publisher = self.create_publisher(
            PoseStamped, args.ground_pose_topic, output_qos
        )
        self.subscription = self.create_subscription(
            PointCloud2, args.chunk_topic, self.chunk_callback, input_qos
        )
        self.request_publisher = self.create_publisher(
            Empty,
            args.chunk_request_topic,
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        self.pose_subscription = None
        if not args.disable_goal_editor:
            self.pose_subscription = self.create_subscription(
                PoseStamped,
                args.pose_topic,
                self.pose_callback,
                qos_profile_sensor_data,
            )
        self.subscription_timer = self.create_timer(
            0.25, self.publish_for_new_subscribers
        )
        self.request_timer = self.create_timer(0.5, self.request_map_if_needed)
        self.get_logger().info(
            'Waiting for PCT map chunks on %s; output=%s'
            % (args.chunk_topic, args.output_topic)
        )

    def pose_callback(self, message):
        if message.header.frame_id.lstrip('/') != self.args.frame_id:
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
            return
        self.latest_pose = pose
        self.latest_pose_message = copy.deepcopy(message)
        if self.goal_editor is not None:
            self.goal_editor.update_pose(pose)
            self.publish_ground_pose(message)

    def publish_ground_pose(self, message, force=False):
        if self.goal_editor is None:
            return
        now = time.monotonic()
        if not force and now - self.last_ground_pose_publish_time < 0.1:
            return
        body_position = np.array(
            [
                message.pose.position.x,
                message.pose.position.y,
                message.pose.position.z,
            ],
            dtype=np.float64,
        )
        ground_position, layer = self.goal_editor.project_start_to_ground(
            body_position, log=False
        )
        if layer < 0:
            return

        ground_pose = copy.deepcopy(message)
        ground_pose.header.stamp = self.get_clock().now().to_msg()
        ground_pose.pose.position.z = float(ground_position[2])
        self.ground_pose_publisher.publish(ground_pose)
        self.last_ground_pose_publish_time = now
        if (
            layer != self.last_ground_pose_layer
            or self.last_ground_pose_z is None
            or abs(ground_position[2] - self.last_ground_pose_z) > 0.05
        ):
            self.get_logger().info(
                'Robot ground projection: body z=%.3fm -> floor z=%.3fm, L%d; '
                'visualization only'
                % (body_position[2], ground_position[2], layer)
            )
            self.last_ground_pose_layer = layer
            self.last_ground_pose_z = float(ground_position[2])

    @staticmethod
    def fields_signature(message):
        return tuple(
            (field.name, field.offset, field.datatype, field.count)
            for field in message.fields
        )

    def reset_transfer(self, message, total_chunks):
        self.expected_chunks = total_chunks
        self.chunks = {}
        self.template = copy.deepcopy(message)
        self.output_cloud = None
        self.get_logger().info(
            'Receiving %d PCT map chunks' % total_chunks
        )

    def chunk_callback(self, message):
        self.last_chunk_time = time.monotonic()
        total_chunks = int(message.header.stamp.sec)
        chunk_index = int(message.header.stamp.nanosec)
        if not 1 <= total_chunks <= MAX_CHUNKS:
            self.get_logger().error(
                'Ignoring chunk with invalid total %d' % total_chunks
            )
            return
        if not 0 <= chunk_index < total_chunks:
            self.get_logger().error(
                'Ignoring chunk index %d/%d' % (chunk_index, total_chunks)
            )
            return
        if message.height != 1 or message.width < 1 or message.point_step < 1:
            self.get_logger().error('Ignoring malformed PointCloud2 chunk')
            return
        expected_bytes = int(message.width) * int(message.point_step)
        if message.row_step != expected_bytes or len(message.data) != expected_bytes:
            self.get_logger().error(
                'Ignoring chunk %d with inconsistent byte count' % chunk_index
            )
            return

        if chunk_index == 0:
            self.reset_transfer(message, total_chunks)
        elif self.expected_chunks is None:
            return
        elif total_chunks != self.expected_chunks:
            self.get_logger().warning(
                'Ignoring chunk from a different transfer (%d != %d)'
                % (total_chunks, self.expected_chunks)
            )
            return

        if (
            message.point_step != self.template.point_step
            or message.is_bigendian != self.template.is_bigendian
            or self.fields_signature(message) != self.fields_signature(self.template)
        ):
            self.get_logger().error(
                'Ignoring chunk %d with incompatible point layout' % chunk_index
            )
            return

        self.chunks[chunk_index] = (int(message.width), bytes(message.data))
        if chunk_index == 0 or (chunk_index + 1) % 25 == 0:
            self.get_logger().info(
                'Received PCT map chunk %d/%d'
                % (chunk_index + 1, self.expected_chunks)
            )
        if len(self.chunks) == self.expected_chunks:
            self.assemble_cloud()

    def assemble_cloud(self):
        ordered = [self.chunks[index] for index in range(self.expected_chunks)]
        width = sum(item[0] for item in ordered)
        data = b''.join(item[1] for item in ordered)
        if len(data) > MAX_POINT_BYTES:
            raise RuntimeError(
                'assembled PCT map is too large: %d bytes' % len(data)
            )

        cloud = PointCloud2()
        cloud.header.frame_id = self.template.header.frame_id
        cloud.header.stamp = self.get_clock().now().to_msg()
        cloud.height = 1
        cloud.width = width
        cloud.fields = copy.deepcopy(self.template.fields)
        cloud.is_bigendian = self.template.is_bigendian
        cloud.point_step = self.template.point_step
        cloud.row_step = width * cloud.point_step
        cloud.data = data
        cloud.is_dense = self.template.is_dense
        self.output_cloud = cloud
        self.publisher.publish(cloud)
        self.get_logger().info(
            'Assembled and published %d PCT cells (%d bytes) on %s'
            % (width, len(data), self.args.output_topic)
        )
        if not self.args.disable_goal_editor:
            self.configure_goal_editor(cloud)

    def configure_goal_editor(self, cloud):
        try:
            points = extract_xyzi(cloud)
            if self.goal_editor is None:
                self.goal_editor = PCTGoalEditor.from_point_cloud(
                    self,
                    points,
                    self.args.frame_id,
                    self.args.clicked_topic,
                    self.args.goal_topic,
                    self.args.marker_topic,
                    self.args.marker_namespace,
                    self.args.path_height_offset,
                    self.args.cost_threshold,
                    self.args.layer_search_radius,
                    self.args.floor_tolerance,
                )
                if self.latest_pose is not None:
                    self.goal_editor.update_pose(self.latest_pose)
                if self.latest_pose_message is not None:
                    self.publish_ground_pose(
                        self.latest_pose_message, force=True
                    )
                self.get_logger().info(
                    'Local Foxy PCT goal editor is ready; '
                    'InteractiveMarker messages stay on this computer'
                )
            else:
                index = PointCloudLayerIndex(
                    points,
                    self.args.cost_threshold,
                    self.args.layer_search_radius,
                    self.args.floor_tolerance,
                )
                self.goal_editor.replace_layer_index(index)
        except Exception as error:
            self.get_logger().error(
                'Cannot initialize local PCT goal editor: %s' % error
            )

    def request_map_if_needed(self):
        if self.output_cloud is not None:
            return
        now = time.monotonic()
        if now - self.last_request_time < self.args.request_period:
            return
        if (
            self.last_chunk_time is not None
            and now - self.last_chunk_time < self.args.request_stall_timeout
        ):
            return
        self.request_publisher.publish(Empty())
        self.last_request_time = now
        self.request_count += 1
        received = len(self.chunks)
        expected = self.expected_chunks or 0
        self.get_logger().warning(
            'Requesting PCT map transfer #%d (received %d/%d chunks)'
            % (self.request_count, received, expected)
        )

    def publish_for_new_subscribers(self):
        subscription_count = self.publisher.get_subscription_count()
        if (
            self.output_cloud is not None
            and subscription_count > self.last_subscription_count
        ):
            self.output_cloud.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(self.output_cloud)
            self.get_logger().info(
                'Republished local PCT map for %d subscriber(s)'
                % subscription_count
            )
        self.last_subscription_count = subscription_count


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk-topic', default='/pct_tomogram_chunks')
    parser.add_argument('--chunk-request-topic', default='/pct_tomogram_request')
    parser.add_argument('--output-topic', default='/pct_tomogram_aligned')
    parser.add_argument('--request-period', type=float, default=3.0)
    parser.add_argument('--request-stall-timeout', type=float, default=2.0)
    parser.add_argument('--pose-topic', default='/pose_stamped')
    parser.add_argument('--ground-pose-topic', default='/pose_stamped_ground')
    parser.add_argument('--frame-id', default='map')
    parser.add_argument('--clicked-topic', default='/clicked_point')
    parser.add_argument('--goal-topic', default='/goal_pose')
    parser.add_argument('--marker-topic', default='/pct_waypoints')
    parser.add_argument('--marker-namespace', default='/pct_waypoint_editor')
    parser.add_argument('--path-height-offset', type=float, default=0.5)
    parser.add_argument('--cost-threshold', type=float, default=20.0)
    parser.add_argument('--layer-search-radius', type=float, default=0.5)
    parser.add_argument('--floor-tolerance', type=float, default=0.2)
    parser.add_argument('--disable-goal-editor', action='store_true')
    return parser.parse_known_args(args=args)


def main(args=None):
    cli_args, ros_args = parse_args(args)
    if cli_args.request_period <= 0.0:
        raise ValueError('--request-period must be positive')
    if cli_args.request_stall_timeout <= 0.0:
        raise ValueError('--request-stall-timeout must be positive')
    rclpy.init(args=ros_args)
    node = TomogramChunkReceiver(cli_args)
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
