#!/usr/bin/env python3

import argparse
import copy
import os
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Empty, String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLANNER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PLANNER_ROOT not in sys.path:
    sys.path.insert(0, PLANNER_ROOT)

from config import Config
from planner_wrapper import TomogramPlanner
from utils import traj2ros


SCENE_TOMOGRAMS = {
    'Map': 'map',
    'Spiral': 'spiral0.3_2',
    'Building': 'building2_9',
    'Plaza': 'plaza3_10',
}


def reliable_qos(depth=10, transient_local=False):
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=(
            DurabilityPolicy.TRANSIENT_LOCAL
            if transient_local
            else DurabilityPolicy.VOLATILE
        ),
    )


class PCTNavigationNode(Node):
    """ROS 2 wrapper preserving the validated PCT-to-SCAN navigation contract."""

    def __init__(self, scene, tomogram_override=''):
        super().__init__('pct_planner')

        self.frame_id = self.declare_parameter('frame_id', 'map').value.lstrip('/')
        self.pose_topic = self.declare_parameter('pose_topic', '/pose_stamped').value
        self.goal_topic = self.declare_parameter('goal_topic', '/goal_pose').value
        self.legacy_goal_topic = self.declare_parameter(
            'legacy_goal_topic', '/move_base_simple/goal'
        ).value
        self.path_topic = self.declare_parameter('path_topic', '/pct_path').value
        self.cancel_input_topic = self.declare_parameter(
            'cancel_input_topic', '/navigation/cancel'
        ).value
        self.cancel_output_topic = self.declare_parameter(
            'cancel_output_topic', '/scan_planner/cancel'
        ).value
        self.preempt_output_topic = self.declare_parameter(
            'preempt_output_topic', '/scan_planner/preempt'
        ).value
        self.status_topic = self.declare_parameter(
            'status_topic', '/scan_planner/goal_status'
        ).value
        self.local_status_topic = self.declare_parameter(
            'local_status_topic', '/scan_planner/local_goal_status'
        ).value
        self.pose_timeout = max(
            0.05, float(self.declare_parameter('pose_timeout', 1.0).value)
        )
        self.start_layer = int(self.declare_parameter('start_layer', -1).value)
        self.goal_layer = int(self.declare_parameter('goal_layer', -1).value)
        self.keep_goal_on_start_layer = bool(
            self.declare_parameter('keep_goal_on_start_layer', False).value
        )
        self.align_path_z_to_pose = bool(
            self.declare_parameter('align_path_z_to_pose', True).value
        )
        self.max_path_z_alignment = max(
            0.0,
            float(self.declare_parameter('max_path_z_alignment', 3.0).value),
        )
        self.path_height_offset = float(
            self.declare_parameter('path_height_offset', 0.5).value
        )

        default_tomogram = tomogram_override or SCENE_TOMOGRAMS.get(scene, scene)
        self.tomogram = self.declare_parameter('tomogram', default_tomogram).value

        self._latest_pose = None
        self._pose_received_ns = None
        self._request_generation = 0

        self.path_pub = self.create_publisher(
            Path, self.path_topic, reliable_qos(depth=1, transient_local=True)
        )
        self.cancel_pub = self.create_publisher(
            Empty, self.cancel_output_topic, reliable_qos(depth=2)
        )
        self.preempt_pub = self.create_publisher(
            Empty, self.preempt_output_topic, reliable_qos(depth=2)
        )
        self.status_pub = self.create_publisher(
            String, self.status_topic, reliable_qos(depth=1, transient_local=True)
        )

        config = Config()
        config.wrapper.path_height_offset = self.path_height_offset
        self.planner = TomogramPlanner(config)
        self.get_logger().info('Loading tomogram: %s' % self.tomogram)
        self.planner.loadTomogram(self.tomogram)

        self.pose_sub = self.create_subscription(
            PoseStamped, self.pose_topic, self.pose_callback, qos_profile_sensor_data
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, self.goal_topic, self.goal_callback, reliable_qos(depth=1)
        )
        self.legacy_goal_sub = None
        if self.legacy_goal_topic and self.legacy_goal_topic != self.goal_topic:
            self.legacy_goal_sub = self.create_subscription(
                PoseStamped,
                self.legacy_goal_topic,
                self.goal_callback,
                reliable_qos(depth=1),
            )
        self.cancel_sub = self.create_subscription(
            Empty,
            self.cancel_input_topic,
            self.cancel_callback,
            reliable_qos(depth=2),
        )
        self.local_status_sub = None
        if self.local_status_topic and self.local_status_topic != self.status_topic:
            self.local_status_sub = self.create_subscription(
                String,
                self.local_status_topic,
                self.local_status_callback,
                reliable_qos(depth=10),
            )

        self.publish_status('NO_GOAL')
        self.get_logger().info(
            'Ready: pose=%s goals=[%s, %s] path=%s frame=%s map=%s'
            % (
                self.pose_topic,
                self.goal_topic,
                self.legacy_goal_topic,
                self.path_topic,
                self.frame_id,
                self.tomogram,
            )
        )

    def frame_matches(self, frame_id):
        return bool(frame_id) and frame_id.lstrip('/') == self.frame_id

    def publish_status(self, status):
        self.status_pub.publish(String(data=status))

    def local_status_callback(self, msg):
        if msg.data in ('GOAL_RUNNING', 'GOAL_REACHED', 'GOAL_FAILED', 'GOAL_CANCEL'):
            self.publish_status(msg.data)
        else:
            self.get_logger().warning('Ignoring unknown local status: %s' % msg.data)

    def pose_callback(self, msg):
        if not self.frame_matches(msg.header.frame_id):
            self.get_logger().error(
                "Pose frame '%s' does not match '%s'; no implicit TF is applied"
                % (msg.header.frame_id, self.frame_id)
            )
            return

        values = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        if not np.isfinite(values).all():
            self.get_logger().error('Ignoring localization pose with non-finite values')
            return

        self._latest_pose = copy.deepcopy(msg)
        self._pose_received_ns = self.get_clock().now().nanoseconds

    def cancel_callback(self, _msg):
        self._request_generation += 1
        self.cancel_pub.publish(Empty())
        self.publish_empty_path()
        self.publish_status('GOAL_CANCEL')
        self.get_logger().warning('Navigation goal cancelled')

    def publish_empty_path(self):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.frame_id
        self.path_pub.publish(path)

    def goal_callback(self, goal):
        if not self.frame_matches(goal.header.frame_id):
            self.get_logger().error(
                "Goal frame '%s' does not match '%s'; no implicit TF is applied"
                % (goal.header.frame_id, self.frame_id)
            )
            self.publish_status('GOAL_FAILED')
            return

        self._request_generation += 1
        generation = self._request_generation
        pose = copy.deepcopy(self._latest_pose)
        now_ns = self.get_clock().now().nanoseconds
        pose_age = float('inf')
        if self._pose_received_ns is not None:
            pose_age = max(0.0, (now_ns - self._pose_received_ns) * 1.0e-9)

        # Stop the previous local trajectory before the potentially slow global plan.
        self.preempt_pub.publish(Empty())

        if pose is None or pose_age > self.pose_timeout:
            self.get_logger().error(
                'No fresh localization pose (age=%.3fs, limit=%.3fs)'
                % (pose_age, self.pose_timeout)
            )
            self.publish_status('GOAL_FAILED')
            return

        start_pos = np.array(
            [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z],
            dtype=np.float64,
        )
        goal_pos = np.array(
            [goal.pose.position.x, goal.pose.position.y, goal.pose.position.z],
            dtype=np.float64,
        )
        requested_start_layer = self.start_layer if self.start_layer >= 0 else None
        requested_goal_layer = self.goal_layer if self.goal_layer >= 0 else None

        self.get_logger().info(
            'Planning (%.2f, %.2f, %.2f) -> (%.2f, %.2f, %.2f)'
            % (*start_pos, *goal_pos)
        )
        try:
            trajectory = self.planner.plan(
                start_pos,
                goal_pos,
                start_layer=requested_start_layer,
                end_layer=requested_goal_layer,
                keep_goal_on_start_layer=self.keep_goal_on_start_layer,
            )
        except Exception as exc:
            self.get_logger().error('Planning rejected: %s' % exc)
            trajectory = None

        if generation != self._request_generation:
            self.get_logger().warning('Discarding obsolete plan')
            return
        if trajectory is None or len(trajectory) == 0:
            self.get_logger().error('No path found')
            self.publish_status('GOAL_FAILED')
            return

        trajectory = np.asarray(trajectory, dtype=np.float64)
        if (
            trajectory.ndim != 2
            or trajectory.shape[1] < 3
            or not np.isfinite(trajectory).all()
        ):
            self.get_logger().error('Planner returned a malformed trajectory')
            self.publish_status('GOAL_FAILED')
            return

        if self.align_path_z_to_pose:
            z_alignment = float(start_pos[2] - trajectory[0, 2])
            if abs(z_alignment) > self.max_path_z_alignment:
                self.get_logger().error(
                    'Required path Z alignment %.3fm exceeds %.3fm'
                    % (z_alignment, self.max_path_z_alignment)
                )
                self.publish_status('GOAL_FAILED')
                return
            trajectory = trajectory.copy()
            trajectory[:, 2] += z_alignment

        # The first path point is the exact localization snapshot used to plan.
        trajectory[0, :3] = start_pos
        path = traj2ros(
            trajectory,
            frame_id=self.frame_id,
            stamp=self.get_clock().now().to_msg(),
        )
        self.path_pub.publish(path)
        self.publish_status('GOAL_RUNNING')
        self.get_logger().info(
            'Published %d points, layers %d -> %d'
            % (
                len(path.poses),
                self.planner.last_start_layer,
                self.planner.last_end_layer,
            )
        )


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scene',
        default='Map',
        help='Tomogram alias or file stem: Map, Spiral, Building or Plaza',
    )
    parser.add_argument(
        '--tomogram',
        default='',
        help='Optional absolute pickle path or tomogram file stem',
    )
    cli_args, ros_args = parser.parse_known_args(args=args)

    rclpy.init(args=ros_args)
    node = None
    try:
        node = PCTNavigationNode(cli_args.scene, cli_args.tomogram)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()