"""Interactive PCT goal marker used by the local Foxy RViz operator UI."""

import numpy as np
from scipy.spatial import cKDTree
from geometry_msgs.msg import PointStamped, PoseStamped
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from interactive_markers.menu_handler import MenuHandler
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
    Marker,
    MarkerArray,
)


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


class TomogramLayerIndex:
    """Read-only traversable-layer lookup independent of the native planner."""

    INVALID_ELEVATION = -100.0

    def __init__(
        self,
        data,
        resolution,
        center,
        slice_h0,
        slice_dh,
        cost_threshold,
        search_radius,
    ):
        self.traversability = np.array(data[0], copy=True)
        self.ground = np.array(data[3], copy=True)
        self.resolution = float(resolution)
        self.center = np.asarray(center, dtype=np.float64)
        self.slice_h0 = float(slice_h0)
        self.slice_dh = float(slice_dh)
        self.cost_threshold = float(cost_threshold)
        self.search_radius = float(search_radius)
        self.dim_x = int(data.shape[2])
        self.dim_y = int(data.shape[3])
        self.offset = np.array(
            [self.dim_x // 2, self.dim_y // 2], dtype=np.int64
        )

    def world_to_grid(self, position):
        position = np.asarray(position, dtype=np.float64).reshape(-1)
        if position.size < 2 or not np.isfinite(position[:2]).all():
            return None
        grid = (
            np.round((position[:2] - self.center) / self.resolution)
            .astype(np.int64)
            + self.offset
        )
        x_idx, y_idx = int(grid[0]), int(grid[1])
        if not (0 <= x_idx < self.dim_x and 0 <= y_idx < self.dim_y):
            return None
        return x_idx, y_idx

    def find_nearest(self, position, layer):
        grid = self.world_to_grid(position)
        if grid is None or not 0 <= layer < self.ground.shape[0]:
            return None
        x_idx, y_idx = grid
        radius = max(0, int(np.ceil(self.search_radius / self.resolution)))
        min_x = max(0, x_idx - radius)
        max_x = min(self.dim_x, x_idx + radius + 1)
        min_y = max(0, y_idx - radius)
        max_y = min(self.dim_y, y_idx + radius + 1)

        elevations = self.ground[layer, min_x:max_x, min_y:max_y]
        costs = self.traversability[layer, min_x:max_x, min_y:max_y]
        valid = (
            np.isfinite(elevations)
            & np.isfinite(costs)
            & (elevations > self.INVALID_ELEVATION)
            & (costs <= self.cost_threshold)
        )
        candidates = np.argwhere(valid)
        if candidates.size == 0:
            return None

        candidate_x = candidates[:, 0] + min_x
        candidate_y = candidates[:, 1] + min_y
        world_x = (
            (candidate_x - self.offset[0]) * self.resolution + self.center[0]
        )
        world_y = (
            (candidate_y - self.offset[1]) * self.resolution + self.center[1]
        )
        distances = np.hypot(
            world_x - float(position[0]), world_y - float(position[1])
        )
        inside = distances <= self.search_radius + 1.0e-9
        if not np.any(inside):
            return None

        candidates = candidates[inside]
        candidate_x = candidate_x[inside]
        candidate_y = candidate_y[inside]
        world_x = world_x[inside]
        world_y = world_y[inside]
        distances = distances[inside]
        candidate_costs = costs[candidates[:, 0], candidates[:, 1]]
        best = int(np.lexsort((candidate_costs, distances))[0])
        local_x, local_y = candidates[best]
        return {
            'layer_idx': int(layer),
            'elevation': float(elevations[local_x, local_y]),
            'traversability_cost': float(costs[local_x, local_y]),
            'nearest_position': np.array(
                [world_x[best], world_y[best]], dtype=np.float64
            ),
            'spatial_distance': float(distances[best]),
        }

    def find_available_layers(self, position):
        available = []
        for layer in range(self.ground.shape[0]):
            item = self.find_nearest(position, layer)
            if item is not None:
                item['height'] = self.slice_h0 + layer * self.slice_dh
                available.append(item)

        available.sort(key=lambda item: item['elevation'])
        physical_layers = []
        tolerance = max(0.15, 2.0 * self.resolution)
        for item in available:
            if (
                not physical_layers
                or abs(item['elevation'] - physical_layers[-1]['elevation'])
                > tolerance
            ):
                physical_layers.append(item)
                continue
            current = physical_layers[-1]
            if (
                item['traversability_cost'],
                item['spatial_distance'],
                item['layer_idx'],
            ) < (
                current['traversability_cost'],
                current['spatial_distance'],
                current['layer_idx'],
            ):
                physical_layers[-1] = item
        return physical_layers


class PointCloudLayerIndex:
    """Traversable-floor lookup reconstructed from the received XYZI cloud."""

    def __init__(
        self,
        points,
        cost_threshold,
        search_radius,
        floor_tolerance,
    ):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 4:
            raise ValueError('point-cloud layer index requires Nx4 XYZI points')
        if not np.isfinite(search_radius) or search_radius <= 0.0:
            raise ValueError('search radius must be finite and positive')
        if not np.isfinite(floor_tolerance) or floor_tolerance <= 0.0:
            raise ValueError('floor tolerance must be finite and positive')

        valid = np.isfinite(points[:, :4]).all(axis=1)
        valid &= points[:, 3] <= float(cost_threshold)
        if not np.any(valid):
            raise ValueError('point cloud has no traversable finite points')
        self.points = (
            points[:, :4]
            if np.all(valid)
            else np.ascontiguousarray(points[valid, :4])
        )
        self.cost_threshold = float(cost_threshold)
        self.search_radius = float(search_radius)
        self.floor_tolerance = float(floor_tolerance)
        self.tree = cKDTree(self.points[:, :2])

    def find_available_layers(self, position):
        position = np.asarray(position, dtype=np.float64).reshape(-1)
        if position.size < 2 or not np.isfinite(position[:2]).all():
            return []
        indices = self.tree.query_ball_point(
            position[:2], self.search_radius
        )
        if not indices:
            return []

        candidates = self.points[np.asarray(indices, dtype=np.int64)]
        order = np.argsort(candidates[:, 2], kind='stable')
        candidates = candidates[order]
        split_points = np.flatnonzero(
            np.diff(candidates[:, 2]) > self.floor_tolerance
        ) + 1
        groups = np.split(candidates, split_points)

        available = []
        for layer_idx, group in enumerate(groups):
            distances = np.hypot(
                group[:, 0] - float(position[0]),
                group[:, 1] - float(position[1]),
            )
            best = int(np.lexsort((group[:, 3], distances))[0])
            available.append(
                {
                    'layer_idx': layer_idx,
                    'height': float(group[best, 2]),
                    'elevation': float(group[best, 2]),
                    'traversability_cost': float(group[best, 3]),
                    'nearest_position': np.asarray(
                        group[best, :2], dtype=np.float64
                    ),
                    'spatial_distance': float(distances[best]),
                }
            )
        return available


class PCTGoalEditor:
    """Restore the old Publish Point -> orange ball -> floor menu workflow."""

    MARKER_NAME = 'pending_waypoint'

    def __init__(
        self,
        node,
        data,
        resolution,
        center,
        slice_h0,
        slice_dh,
        z_alignment,
        frame_id,
        clicked_topic,
        goal_topic,
        marker_topic,
        server_namespace,
        path_height_offset,
        cost_threshold,
        search_radius,
    ):
        index = TomogramLayerIndex(
            data,
            resolution,
            center,
            slice_h0,
            slice_dh,
            cost_threshold,
            search_radius,
        )
        self._initialize(
            node,
            index,
            z_alignment,
            frame_id,
            clicked_topic,
            goal_topic,
            marker_topic,
            server_namespace,
            path_height_offset,
        )

    @classmethod
    def from_point_cloud(
        cls,
        node,
        points,
        frame_id,
        clicked_topic,
        goal_topic,
        marker_topic,
        server_namespace,
        path_height_offset,
        cost_threshold,
        search_radius,
        floor_tolerance,
    ):
        editor = cls.__new__(cls)
        index = PointCloudLayerIndex(
            points,
            cost_threshold,
            search_radius,
            floor_tolerance,
        )
        editor._initialize(
            node,
            index,
            0.0,
            frame_id,
            clicked_topic,
            goal_topic,
            marker_topic,
            server_namespace,
            path_height_offset,
        )
        return editor

    def _initialize(
        self,
        node,
        index,
        z_alignment,
        frame_id,
        clicked_topic,
        goal_topic,
        marker_topic,
        server_namespace,
        path_height_offset,
    ):
        self.node = node
        self.frame_id = frame_id
        self.z_alignment = float(z_alignment)
        self.path_height_offset = float(path_height_offset)
        self.forced_layer = None
        self.latest_pose = None
        self.menu = None
        self.index = index
        self.goal_publisher = node.create_publisher(
            PoseStamped, goal_topic, reliable_qos(depth=1)
        )
        self.marker_publisher = node.create_publisher(
            MarkerArray,
            marker_topic,
            reliable_qos(depth=1, transient_local=True),
        )
        self.server = InteractiveMarkerServer(node, server_namespace)
        self.clicked_subscription = node.create_subscription(
            PointStamped,
            clicked_topic,
            self.clicked_callback,
            reliable_qos(depth=5),
        )
        self.cancel_subscription = node.create_subscription(
            Empty,
            '/navigation/cancel',
            self.cancel_navigation_callback,
            reliable_qos(depth=2),
        )
        node.get_logger().info(
            'PCT goal editor ready: Publish Point -> orange ball -> '
            'Interact/right-click -> select floor or confirm'
        )

    def replace_layer_index(self, index):
        if not hasattr(index, 'find_available_layers'):
            raise TypeError('layer index must implement find_available_layers')
        self.index = index
        self.forced_layer = None
        self.server.erase(self.MARKER_NAME)
        self.server.applyChanges()

    def update_pose(self, pose):
        self.latest_pose = np.asarray(pose, dtype=np.float64).copy()

    def clicked_callback(self, message):
        if message.header.frame_id.lstrip('/') != self.frame_id:
            self.node.get_logger().error(
                "Ignoring clicked point frame '%s'; expected '%s'"
                % (message.header.frame_id, self.frame_id)
            )
            return
        point = np.array(
            [message.point.x, message.point.y, message.point.z],
            dtype=np.float64,
        )
        if not np.isfinite(point).all():
            self.node.get_logger().error('Ignoring non-finite clicked point')
            return
        self.seed(point)

    def seed(self, point):
        available = self.index.find_available_layers(point[:2])
        if not available:
            self.node.get_logger().warning(
                'No traversable PCT layer within %.2fm of (%.2f, %.2f)'
                % (
                    self.index.search_radius,
                    point[0],
                    point[1],
                )
            )
            return

        selected = min(
            available,
            key=lambda item: (
                abs(
                    item['elevation']
                    + self.z_alignment
                    - float(point[2])
                ),
                item['spatial_distance'],
                item['traversability_cost'],
            ),
        )
        marker = InteractiveMarker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.name = self.MARKER_NAME
        marker.description = '右键选择楼层或确认目标'
        marker.scale = 1.0
        marker.pose.position.x = float(selected['nearest_position'][0])
        marker.pose.position.y = float(selected['nearest_position'][1])
        marker.pose.position.z = float(
            selected['elevation'] + self.z_alignment
        )
        marker.pose.orientation.w = 1.0

        visual = InteractiveMarkerControl()
        visual.name = 'waypoint_menu'
        visual.always_visible = True
        visual.interaction_mode = InteractiveMarkerControl.MENU
        sphere = Marker()
        sphere.type = Marker.SPHERE
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.35
        sphere.color.r = 1.0
        sphere.color.g = 0.55
        sphere.color.b = 0.0
        sphere.color.a = 1.0
        visual.markers.append(sphere)
        marker.controls.append(visual)

        move_z = InteractiveMarkerControl()
        move_z.name = 'move_z'
        move_z.orientation.w = 1.0
        move_z.orientation.y = 1.0
        move_z.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        marker.controls.append(move_z)

        self.forced_layer = None
        self.server.erase(self.MARKER_NAME)
        self.server.insert(marker, feedback_callback=self.feedback_callback)

        self.menu = MenuHandler()
        self.menu.insert('确认目标 / Navigate here', callback=self.commit_callback)
        floor_menu = self.menu.insert('选择楼层 / Select floor')
        for item in available:
            title = 'L%d  z=%.2fm  cost=%.1f' % (
                item['layer_idx'],
                item['elevation'] + self.z_alignment,
                item['traversability_cost'],
            )
            self.menu.insert(
                title,
                parent=floor_menu,
                callback=self.layer_callback(item['layer_idx']),
            )
        self.menu.insert('取消 / Cancel', callback=self.cancel_callback)
        self.menu.apply(self.server, self.MARKER_NAME)
        self.server.applyChanges()
        self.node.get_logger().info(
            'Pending PCT goal at (%.2f, %.2f); available floors: %s'
            % (
                marker.pose.position.x,
                marker.pose.position.y,
                ', '.join(
                    'L%d(z=%.2f)'
                    % (
                        item['layer_idx'],
                        item['elevation'] + self.z_alignment,
                    )
                    for item in available
                ),
            )
        )

    def layer_callback(self, layer):
        def callback(feedback):
            available = self.index.find_available_layers(
                [feedback.pose.position.x, feedback.pose.position.y]
            )
            selected = next(
                (
                    item
                    for item in available
                    if item['layer_idx'] == int(layer)
                ),
                None,
            )
            if selected is None:
                self.node.get_logger().warning(
                    'PCT layer %d is no longer available here' % layer
                )
                return
            pose = feedback.pose
            pose.position.x = float(selected['nearest_position'][0])
            pose.position.y = float(selected['nearest_position'][1])
            pose.position.z = float(
                selected['elevation'] + self.z_alignment
            )
            self.forced_layer = int(layer)
            self.server.setPose(self.MARKER_NAME, pose)
            self.server.applyChanges()

        return callback

    def feedback_callback(self, feedback):
        if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
            self.forced_layer = None

    def select_feedback_layer(self, feedback):
        available = self.index.find_available_layers(
            [feedback.pose.position.x, feedback.pose.position.y]
        )
        if not available:
            return None
        if self.forced_layer is not None:
            selected = next(
                (
                    item
                    for item in available
                    if item['layer_idx'] == self.forced_layer
                ),
                None,
            )
            if selected is not None:
                return selected
        return min(
            available,
            key=lambda item: (
                abs(
                    item['elevation']
                    + self.z_alignment
                    - feedback.pose.position.z
                ),
                item['spatial_distance'],
                item['traversability_cost'],
            ),
        )

    def commit_callback(self, feedback):
        selected = self.select_feedback_layer(feedback)
        if selected is None:
            self.node.get_logger().error('Pending PCT goal has no valid floor')
            return
        goal = PoseStamped()
        goal.header.frame_id = self.frame_id
        goal.header.stamp = self.node.get_clock().now().to_msg()
        goal.pose.position.x = float(selected['nearest_position'][0])
        goal.pose.position.y = float(selected['nearest_position'][1])
        goal.pose.position.z = float(
            selected['elevation']
            + self.path_height_offset
            + self.z_alignment
        )
        goal.pose.orientation.w = 1.0

        self.server.erase(self.MARKER_NAME)
        self.server.applyChanges()
        self.publish_confirmed_markers(goal, selected['layer_idx'])
        self.goal_publisher.publish(goal)
        self.node.get_logger().info(
            'Confirmed PCT goal (%.2f, %.2f, %.2f), layer L%d'
            % (
                goal.pose.position.x,
                goal.pose.position.y,
                goal.pose.position.z,
                selected['layer_idx'],
            )
        )

    def publish_confirmed_markers(self, goal, goal_layer):
        message = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        message.markers.append(clear)
        if self.latest_pose is not None:
            start_position, _ = self.project_start_to_ground(
                self.latest_pose
            )
            self.add_labeled_sphere(
                message,
                0,
                start_position,
                '#1 START GROUND',
                (0.0, 1.0, 0.2),
            )
        self.add_labeled_sphere(
            message,
            2,
            np.array(
                [
                    goal.pose.position.x,
                    goal.pose.position.y,
                    goal.pose.position.z,
                ],
                dtype=np.float64,
            ),
            '#2 GOAL L%d' % goal_layer,
            (0.1, 0.45, 1.0),
        )
        self.marker_publisher.publish(message)

    def project_start_to_ground(self, body_position, log=True):
        """Project the visual START marker onto the floor below the body pose."""
        body_position = np.asarray(body_position, dtype=np.float64).copy()
        available = self.index.find_available_layers(body_position[:2])
        if not available:
            self.node.get_logger().warning(
                'Cannot project START marker to a traversable PCT floor'
            )
            return body_position, -1

        expected_ground = body_position[2] - self.path_height_offset
        selected = min(
            available,
            key=lambda item: (
                abs(
                    item['elevation']
                    + self.z_alignment
                    - expected_ground
                ),
                item['spatial_distance'],
                item['traversability_cost'],
            ),
        )
        ground_position = body_position.copy()
        ground_position[2] = selected['elevation'] + self.z_alignment
        if log:
            self.node.get_logger().info(
                'START display projected from body z=%.3fm to floor z=%.3fm, L%d; '
                'planning coordinates are unchanged'
                % (
                    body_position[2],
                    ground_position[2],
                    selected['layer_idx'],
                )
            )
        return ground_position, selected['layer_idx']

    def add_labeled_sphere(self, message, marker_id, position, label, color):
        stamp = self.node.get_clock().now().to_msg()
        sphere = Marker()
        sphere.header.frame_id = self.frame_id
        sphere.header.stamp = stamp
        sphere.ns = 'pct_waypoints'
        sphere.id = marker_id
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = float(position[0])
        sphere.pose.position.y = float(position[1])
        sphere.pose.position.z = float(position[2])
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.25
        sphere.color.r = color[0]
        sphere.color.g = color[1]
        sphere.color.b = color[2]
        sphere.color.a = 1.0
        message.markers.append(sphere)

        text = Marker()
        text.header = sphere.header
        text.ns = 'pct_waypoint_labels'
        text.id = marker_id + 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(position[0])
        text.pose.position.y = float(position[1])
        text.pose.position.z = float(position[2]) + 0.3
        text.pose.orientation.w = 1.0
        text.scale.z = 0.22
        text.color.r = text.color.g = text.color.b = text.color.a = 1.0
        text.text = label
        message.markers.append(text)

    def cancel_callback(self, _feedback):
        self.server.erase(self.MARKER_NAME)
        self.server.applyChanges()
        self.node.get_logger().info('Pending PCT goal cancelled')

    def cancel_navigation_callback(self, _message):
        clear = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        clear.markers.append(marker)
        self.marker_publisher.publish(clear)
        self.server.erase(self.MARKER_NAME)
        self.server.applyChanges()

    def shutdown(self):
        self.server.shutdown()