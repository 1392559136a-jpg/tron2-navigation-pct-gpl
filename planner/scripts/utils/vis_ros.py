import math

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.clock import Clock


def traj2ros(traj_3d, frame_id='map', stamp=None):
    path = Path()
    path.header.stamp = stamp if stamp is not None else Clock().now().to_msg()
    path.header.frame_id = frame_id
    for i in range(traj_3d.shape[0]):
        pose = PoseStamped()
        pose.header = path.header
        pose.pose.position.x = traj_3d[i, 0]
        pose.pose.position.y = traj_3d[i, 1]
        pose.pose.position.z = traj_3d[i, 2]

        if traj_3d.shape[0] > 1:
            if i + 1 < traj_3d.shape[0]:
                dx = traj_3d[i + 1, 0] - traj_3d[i, 0]
                dy = traj_3d[i + 1, 1] - traj_3d[i, 1]
            else:
                dx = traj_3d[i, 0] - traj_3d[i - 1, 0]
                dy = traj_3d[i, 1] - traj_3d[i - 1, 1]
            yaw = math.atan2(dy, dx)
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
        else:
            pose.pose.orientation.w = 1.0
        path.poses.append(pose)
    return path
