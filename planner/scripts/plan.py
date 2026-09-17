import sys
import argparse
import numpy as np
import threading
import pickle
import os
from pathlib import Path as PathLib

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseArray

from utils import traj2ros
from planner_wrapper import TomogramPlanner

sys.path.append('../')
from config import Config

class PCTPlanner(Node):
    def __init__(self, cfg, tomo_file, robot_height, waypoints_type="auto"):
        super().__init__('pct_planner')
        self.cfg = cfg
        self.tomo_file = tomo_file
        self.robot_height = robot_height
        self.waypoints_type = waypoints_type

        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self.path_pub = self.create_publisher(Path, "/pct_path", qos)
        
        # 订阅 waypoints 话题的 QoS 配置
        # 使用兼容的 QoS 设置以接收来自各种发布者（如 RViz2）的消息
        # 尝试多种 QoS 策略以确保兼容性
        waypoints_qos_reliable = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        waypoints_qos_besteffort = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        # 订阅 waypoints 话题
        # 由于 /waypoints 话题可能同时支持两种消息类型，我们同时订阅两种类型以确保能收到消息
        if waypoints_type == "Path" or waypoints_type == "path":
            # 只订阅 Path 类型
            self.waypoints_sub_path = self.create_subscription(
                Path, 
                "/waypoints", 
                self.waypoints_callback_path, 
                waypoints_qos_reliable
            )
            self.get_logger().info("📡 已订阅话题: /waypoints (消息类型: nav_msgs/Path, QoS: RELIABLE)")
        elif waypoints_type == "PoseArray" or waypoints_type == "posearray":
            # 只订阅 PoseArray 类型
            self.waypoints_sub_pose = self.create_subscription(
                PoseArray, 
                "/waypoints", 
                self.waypoints_callback_pose_array, 
                waypoints_qos_reliable
            )
            self.get_logger().info("📡 已订阅话题: /waypoints (消息类型: geometry_msgs/PoseArray, QoS: RELIABLE)")
        else:  # auto 或默认：同时订阅两种类型以确保兼容性
            self.waypoints_sub_pose = self.create_subscription(
                PoseArray, 
                "/waypoints", 
                self.waypoints_callback_pose_array, 
                waypoints_qos_reliable
            )
            self.waypoints_sub_path = self.create_subscription(
                Path, 
                "/waypoints", 
                self.waypoints_callback_path, 
                waypoints_qos_reliable
            )
            self.get_logger().info("📡 已订阅话题: /waypoints (消息类型: geometry_msgs/PoseArray 和 nav_msgs/Path, QoS: RELIABLE)")
            
            # 同时尝试 BEST_EFFORT QoS 以增加兼容性
            self.waypoints_sub_pose_be = self.create_subscription(
                PoseArray, 
                "/waypoints", 
                self.waypoints_callback_pose_array, 
                waypoints_qos_besteffort
            )
            self.waypoints_sub_path_be = self.create_subscription(
                Path, 
                "/waypoints", 
                self.waypoints_callback_path, 
                waypoints_qos_besteffort
            )
            self.get_logger().info("📡 已订阅话题: /waypoints (消息类型: geometry_msgs/PoseArray 和 nav_msgs/Path, QoS: BEST_EFFORT)")

        self.planner = TomogramPlanner(cfg)
        self.planner.loadTomogram(self.tomo_file)

        self.start_pos = None
        self.start_layer = None
        self.end_pos = None
        self.end_layer = None

        # 累积的路径段列表
        self.path_segments = []
        # 每段路径的终点层信息（用于恢复）
        self.path_segment_end_layers = []
        # 累积的路径点列表（按顺序）
        self.waypoints = []  # 存储 (position, layer) 元组
        # 当前使用的点索引（下一个要使用的点）
        self.waypoint_index = 0
        
        # 保存轨迹的文件路径
        script_dir = PathLib(__file__).parent
        self.saved_trajectory_file = script_dir / "saved_trajectory.pkl"

        # 启动命令行输入监听线程
        self.input_thread = threading.Thread(target=self._input_listener, daemon=True)
        self.input_thread.start()

        # 检查是否有保存的轨迹
        self._check_saved_trajectory()

        self.get_logger().info("等待从 /waypoints 话题接收路径点...")
        self.get_logger().info("💡 提示: 系统将根据每个点的 z 坐标自动选择最近的层")
        self.get_logger().info("💡 提示: 第一段需要2个点(point#1起点, point#2终点)，后续每段只需1个新点(point#N终点)")
        self.get_logger().info("💡 提示: 每次规划后自动发布到 /pct_path")
        self.get_logger().info("💡 提示: 输入 '0' 保存轨迹，'1' 清除当前段，'2' 加载轨迹，'3' 反转轨迹返回起点")

    def waypoints_callback_pose_array(self, msg):
        """
        处理从 /waypoints 话题接收到的 PoseArray 消息
        根据每个点的 z 坐标自动选择最近的层
        """
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"📨 收到 /waypoints 消息 (PoseArray)，包含 {len(msg.poses)} 个路径点")
        if len(msg.poses) > 0:
            for i, pose in enumerate(msg.poses[:3]):  # 只打印前3个点
                pos = pose.position
                self.get_logger().info(f"   点 {i}: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
        self.get_logger().info("=" * 50)
        self._process_waypoints(msg.poses)
    
    def waypoints_callback_path(self, msg):
        """
        处理从 /waypoints 话题接收到的 Path 消息
        根据每个点的 z 坐标自动选择最近的层
        """
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"📨 收到 /waypoints 消息 (Path)，包含 {len(msg.poses)} 个路径点")
        if len(msg.poses) > 0:
            for i, pose_stamped in enumerate(msg.poses[:3]):  # 只打印前3个点
                pos = pose_stamped.pose.position
                self.get_logger().info(f"   点 {i}: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
        self.get_logger().info("=" * 50)
        # Path.poses 是 PoseStamped 数组，需要提取 pose
        poses = [pose_stamped.pose for pose_stamped in msg.poses]
        self._process_waypoints(poses)
    
    def _input_listener(self):
        """监听命令行输入：
        - 输入 '0': 保存当前轨迹
        - 输入 '1': 清除当前段
        - 输入 '2': 加载保存的轨迹
        - 输入 '3': 反转轨迹，沿原轨迹返回起点
        """
        while True:
            try:
                user_input = input().strip()
                if user_input == '0':
                    self._save_trajectory()
                elif user_input == '1':
                    self._clear_current_segment()
                elif user_input == '2':
                    self._load_trajectory()
                elif user_input == '3':
                    self._reverse_trajectory()
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                self.get_logger().error(f"输入监听错误: {e}")

    def _clear_current_segment(self):
        """清除最后一段路径，允许重新选择终点"""
        if len(self.path_segments) == 0:
            self.get_logger().warn("⚠️ 没有路径段可以清除")
            return
        
        # 移除最后一段
        removed_segment = self.path_segments.pop()
        # 同时移除对应的层信息
        if len(self.path_segment_end_layers) > 0:
            self.path_segment_end_layers.pop()
        
        # 回退waypoint_index，并移除对应的终点
        if self.waypoint_index > 0:
            # 计算需要移除的终点索引
            # waypoint_index指向下一个要使用的点，所以当前段的终点是waypoint_index-1
            # 需要移除这个终点
            if self.waypoint_index <= len(self.waypoints):
                removed_waypoint = self.waypoints.pop(self.waypoint_index - 1)
                self.get_logger().info(f"🗑️ 已移除终点 point#{self.waypoint_index}: ({removed_waypoint[0]:.3f}, {removed_waypoint[1]:.3f}, {removed_waypoint[2]:.3f})")
            
            # 回退waypoint_index
            self.waypoint_index -= 1
            
            # 如果还有路径段，waypoint_index应该指向上一段的终点
            if len(self.path_segments) > 0:
                # waypoint_index应该等于已完成的段数+1（因为第一段用了2个点）
                # 但我们已经减1了，所以这里不需要再调整
                pass
            else:
                # 如果没有路径段了，重置为0（需要重新选择起点和终点）
                self.waypoint_index = 0
        
        self.get_logger().info(f"🗑️ 已清除最后一段路径 (共 {len(removed_segment)} 个点)")
        self.get_logger().info(f"📌 当前点索引: {self.waypoint_index}，剩余路径点数: {len(self.waypoints)}")
        if self.waypoint_index == 0:
            self.get_logger().info("💡 可以重新选择起点和终点（需要2个点）")
        else:
            self.get_logger().info(f"💡 可以重新选择终点 (point#{self.waypoint_index + 1})")
        
        # 重新发布剩余路径
        self._publish_all_paths()

    def _save_trajectory(self):
        """保存当前轨迹到文件"""
        if len(self.path_segments) == 0:
            self.get_logger().warn("⚠️ 没有轨迹可以保存")
            return
        
        try:
            # 连接所有路径段
            connected_traj = None
            for segment in self.path_segments:
                if connected_traj is None:
                    connected_traj = segment.copy()
                else:
                    if len(segment) > 0 and len(connected_traj) > 0:
                        last_point = connected_traj[-1]
                        first_point = segment[0]
                        dist = np.linalg.norm(last_point[:2] - first_point[:2])
                        if dist < 0.1 and len(segment) > 1:
                            connected_traj = np.vstack([connected_traj, segment[1:]])
                        else:
                            connected_traj = np.vstack([connected_traj, segment])
                    elif len(segment) > 0:
                        connected_traj = np.vstack([connected_traj, segment])
            
            if connected_traj is None or len(connected_traj) == 0:
                self.get_logger().warn("⚠️ 轨迹为空，无法保存")
                return
            
            # 保存轨迹数据
            trajectory_data = {
                'trajectory': connected_traj,
                'path_segments': self.path_segments,
                'path_segment_end_layers': self.path_segment_end_layers,
                'waypoints': self.waypoints,
                'waypoint_index': self.waypoint_index,
                'num_segments': len(self.path_segments),
                'total_points': len(connected_traj)
            }
            
            with open(self.saved_trajectory_file, 'wb') as f:
                pickle.dump(trajectory_data, f)
            
            self.get_logger().info(f"💾 轨迹已保存到: {self.saved_trajectory_file}")
            self.get_logger().info(f"   路径段数: {len(self.path_segments)}")
            self.get_logger().info(f"   总路径点数: {len(connected_traj)}")
            
        except Exception as e:
            self.get_logger().error(f"❌ 保存轨迹失败: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _check_saved_trajectory(self):
        """检查是否有保存的轨迹，并询问是否加载"""
        if not os.path.exists(self.saved_trajectory_file):
            return
        
        try:
            with open(self.saved_trajectory_file, 'rb') as f:
                trajectory_data = pickle.load(f)
            
            num_segments = trajectory_data.get('num_segments', 0)
            total_points = trajectory_data.get('total_points', 0)
            
            self.get_logger().info("=" * 50)
            self.get_logger().info(f"📂 发现保存的轨迹文件:")
            self.get_logger().info(f"   路径段数: {num_segments}")
            self.get_logger().info(f"   总路径点数: {total_points}")
            self.get_logger().info(f"   文件路径: {self.saved_trajectory_file}")
            self.get_logger().info("=" * 50)
            self.get_logger().info("💡 提示: 如果要在启动时自动加载轨迹，请在代码中设置")
            self.get_logger().info("   或者手动调用加载函数（当前版本需要手动加载）")
            
        except Exception as e:
            self.get_logger().warn(f"⚠️ 读取保存的轨迹文件失败: {e}")

    def _load_trajectory(self):
        """加载保存的轨迹"""
        if not os.path.exists(self.saved_trajectory_file):
            self.get_logger().warn("⚠️ 没有找到保存的轨迹文件")
            return False
        
        try:
            with open(self.saved_trajectory_file, 'rb') as f:
                trajectory_data = pickle.load(f)
            
            # 恢复轨迹数据
            self.path_segments = trajectory_data.get('path_segments', [])
            self.path_segment_end_layers = trajectory_data.get('path_segment_end_layers', [])
            self.waypoints = trajectory_data.get('waypoints', [])
            self.waypoint_index = trajectory_data.get('waypoint_index', 0)
            
            # 确保数据是numpy数组
            for i, segment in enumerate(self.path_segments):
                if not isinstance(segment, np.ndarray):
                    self.path_segments[i] = np.array(segment)
            
            for i, waypoint in enumerate(self.waypoints):
                if not isinstance(waypoint, np.ndarray):
                    self.waypoints[i] = np.array(waypoint)
            
            num_segments = len(self.path_segments)
            total_points = sum(len(seg) for seg in self.path_segments)
            
            self.get_logger().info(f"✅ 轨迹已加载:")
            self.get_logger().info(f"   路径段数: {num_segments}")
            self.get_logger().info(f"   总路径点数: {total_points}")
            self.get_logger().info(f"   路径点数: {len(self.waypoints)}")
            
            # 发布加载的轨迹
            self._publish_all_paths()
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"❌ 加载轨迹失败: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return False

    def _reverse_trajectory(self):
        """反转当前轨迹，让机器人沿原轨迹返回起点"""
        if len(self.path_segments) == 0:
            self.get_logger().warn("⚠️ 没有轨迹可以反转")
            return
        
        try:
            # 连接所有路径段（复用_publish_all_paths的逻辑）
            connected_traj = None
            valid_segments = []
            
            for i, segment in enumerate(self.path_segments):
                # 检查segment是否有效
                if segment is None:
                    continue
                if not isinstance(segment, np.ndarray):
                    continue
                if segment.size == 0:
                    continue
                if len(segment.shape) != 2 or segment.shape[1] != 3:
                    continue
                if len(segment) == 0:
                    continue
                valid_segments.append(segment)
            
            if len(valid_segments) == 0:
                self.get_logger().warn("⚠️ 没有有效的路径段，无法反转")
                return
            
            # 连接有效路径段
            for i, segment in enumerate(valid_segments):
                if connected_traj is None:
                    connected_traj = segment.copy()
                else:
                    if len(segment) > 0 and len(connected_traj) > 0:
                        last_point = connected_traj[-1]
                        first_point = segment[0]
                        dist = np.linalg.norm(last_point[:2] - first_point[:2])
                        if dist < 0.1 and len(segment) > 1:
                            connected_traj = np.vstack([connected_traj, segment[1:]])
                        else:
                            connected_traj = np.vstack([connected_traj, segment])
                    elif len(segment) > 0:
                        connected_traj = np.vstack([connected_traj, segment])
            
            if connected_traj is None or len(connected_traj) == 0:
                self.get_logger().warn("⚠️ 连接后的路径为空，无法反转")
                return
            
            # 反转轨迹（从最后一个点到第一个点）
            reversed_traj = np.flipud(connected_traj)
            
            # 发布反转后的轨迹
            self.path_pub.publish(traj2ros(reversed_traj))
            self.get_logger().info(f"🔄 已反转并发布轨迹: {len(reversed_traj)} 个路径点")
            self.get_logger().info(f"   起点: ({reversed_traj[0][0]:.3f}, {reversed_traj[0][1]:.3f}, {reversed_traj[0][2]:.3f})")
            self.get_logger().info(f"   终点: ({reversed_traj[-1][0]:.3f}, {reversed_traj[-1][1]:.3f}, {reversed_traj[-1][2]:.3f})")
            
        except Exception as e:
            self.get_logger().error(f"❌ 反转轨迹异常: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _publish_all_paths(self):
        """连接所有路径段并发布到 /pct_path 话题"""
        try:
            if len(self.path_segments) == 0:
                # 如果没有路径段，发布空路径
                empty_path = Path()
                empty_path.header.stamp = self.get_clock().now().to_msg()
                empty_path.header.frame_id = "map"
                self.path_pub.publish(empty_path)
                self.get_logger().info("📤 已发布空路径到 /pct_path")
                return
            
            # 连接所有路径段
            connected_traj = None
            valid_segments = []
            
            for i, segment in enumerate(self.path_segments):
                # 检查segment是否有效
                if segment is None:
                    self.get_logger().warn(f"⚠️ 路径段 {i} 为 None，跳过")
                    continue
                
                if not isinstance(segment, np.ndarray):
                    self.get_logger().warn(f"⚠️ 路径段 {i} 类型错误 {type(segment)}，跳过")
                    continue
                
                if segment.size == 0:
                    self.get_logger().warn(f"⚠️ 路径段 {i} 为空，跳过")
                    continue
                
                if len(segment.shape) != 2 or segment.shape[1] != 3:
                    self.get_logger().warn(f"⚠️ 路径段 {i} 形状错误 {segment.shape}，跳过")
                    continue
                
                if len(segment) == 0:
                    self.get_logger().warn(f"⚠️ 路径段 {i} 长度为0，跳过")
                    continue
                
                valid_segments.append(segment)
            
            if len(valid_segments) == 0:
                self.get_logger().warn("⚠️ 没有有效的路径段，发布空路径")
                empty_path = Path()
                empty_path.header.stamp = self.get_clock().now().to_msg()
                empty_path.header.frame_id = "map"
                self.path_pub.publish(empty_path)
                return
            
            # 连接有效路径段
            for i, segment in enumerate(valid_segments):
                if connected_traj is None:
                    connected_traj = segment.copy()
                else:
                    # 去除重复点（每段的第一个点通常是上一段的最后一个点）
                    # 如果第一点和上一段的最后一点很接近，则跳过第一点
                    if len(segment) > 0 and len(connected_traj) > 0:
                        last_point = connected_traj[-1]
                        first_point = segment[0]
                        # 计算距离，如果很接近（小于0.1米），则跳过第一点
                        dist = np.linalg.norm(last_point[:2] - first_point[:2])
                        if dist < 0.1 and len(segment) > 1:
                            connected_traj = np.vstack([connected_traj, segment[1:]])
                        else:
                            connected_traj = np.vstack([connected_traj, segment])
                    elif len(segment) > 0:
                        connected_traj = np.vstack([connected_traj, segment])
            
            if connected_traj is not None and len(connected_traj) > 0:
                self.path_pub.publish(traj2ros(connected_traj))
                self.get_logger().info(f"📤 已发布 {len(connected_traj)} 个路径点到 /pct_path (共 {len(valid_segments)} 段)")
            else:
                self.get_logger().warn("⚠️ 连接后的路径为空，无法发布")
                empty_path = Path()
                empty_path.header.stamp = self.get_clock().now().to_msg()
                empty_path.header.frame_id = "map"
                self.path_pub.publish(empty_path)
                
        except Exception as e:
            self.get_logger().error(f"❌ 发布路径异常: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            # 尝试发布空路径
            try:
                empty_path = Path()
                empty_path.header.stamp = self.get_clock().now().to_msg()
                empty_path.header.frame_id = "map"
                self.path_pub.publish(empty_path)
            except:
                pass

    def _process_waypoints(self, poses):
        """
        处理路径点列表（通用的处理函数）
        使用点计数器逻辑：第一段需要2个点(point#1起点, point#2终点)，
        后续每段只需1个新点(point#N终点)，起点自动使用上一段的终点
        
        注意：/waypoints 话题每次发布的是完整的点列表，所以只处理新增的点
        
        Args:
            poses: pose 对象列表（可以是 Pose 或 PoseStamped.pose）
        """
        if len(poses) == 0:
            self.get_logger().warn("⚠️ 收到空的路径点列表")
            return
        
        # 将收到的点转换为numpy数组
        received_points = []
        for pose in poses:
            point_3d = np.array([
                pose.position.x,
                pose.position.y,
                pose.position.z
            ], dtype=np.float32)
            received_points.append(point_3d)
        
        # 只添加新的点（通过坐标比较，避免重复）
        # 由于/waypoints每次发布完整列表，需要检查每个点是否已存在
        added_count = 0
        for new_point in received_points:
            # 检查这个点是否已经在waypoints列表中（通过坐标比较，容差0.01米）
            is_duplicate = False
            for existing_point in self.waypoints:
                dist = np.linalg.norm(new_point[:2] - existing_point[:2])
                if dist < 0.01:  # 如果2D距离小于1cm，认为是同一个点
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                self.waypoints.append(new_point.copy())
                added_count += 1
                self.get_logger().info(f"📌 添加路径点 #{len(self.waypoints)}: ({new_point[0]:.3f}, {new_point[1]:.3f}, {new_point[2]:.3f})")
        
        # 如果没有新点，直接返回
        if added_count == 0:
            self.get_logger().info(f"💡 没有新点，当前已有 {len(self.waypoints)} 个点，已处理到 point#{self.waypoint_index}")
            return
        
        # 检查是否有足够的点进行规划
        if self.waypoint_index == 0:
            # 第一段：需要至少2个点
            if len(self.waypoints) < 2:
                self.get_logger().info(f"💡 等待更多路径点... (当前 {len(self.waypoints)} 个，需要至少 2 个)")
                return
            # 使用 point#1 作为起点，point#2 作为终点
            start_point = self.waypoints[0]
            end_point = self.waypoints[1]
            self.waypoint_index = 2
            self.get_logger().info(f"🚀 第一段路径: point#1 -> point#2")
        else:
            # 后续段：只需要1个新点（起点是上一段的终点）
            if len(self.waypoints) <= self.waypoint_index:
                self.get_logger().info(f"💡 等待更多路径点... (当前 {len(self.waypoints)} 个，需要至少 {self.waypoint_index + 1} 个)")
                return
            # 使用上一段的终点(point#N)作为起点，新点(point#N+1)作为终点
            start_point = self.waypoints[self.waypoint_index - 1]
            end_point = self.waypoints[self.waypoint_index]
            self.waypoint_index += 1
            self.get_logger().info(f"🚀 第 {len(self.path_segments) + 1} 段路径: point#{self.waypoint_index - 1} -> point#{self.waypoint_index}")
        
        # 处理起点
        start_2d = start_point[:2]
        start_z = start_point[2]
        
        # 搜索起点附近的所有可用层
        start_available_layers = self.planner.findAvailableLayers(start_2d, search_radius=1.0)
        
        # 打印可用层信息用于调试（只打印前几个）
        if len(start_available_layers) > 0:
            self.get_logger().info(f"🔍 起点附近找到 {len(start_available_layers)} 个可用层")
            for layer_info in start_available_layers[:3]:  # 只打印前3个
                self.get_logger().info(
                    f"   层 {layer_info['layer_idx']}: 理论高度={layer_info['height']:.3f}m, "
                    f"实际高程={layer_info['elevation']:.3f}m, 可通行性={layer_info['traversability']:.2f}"
                )
        
        # 根据 z 坐标选择最近的层
        start_layer = self._select_layer_by_z(start_z, start_available_layers, start_2d)
        
        # 验证选择的层是否在可行区域
        if len(start_available_layers) > 0:
            start_layer_in_available = any(l['layer_idx'] == start_layer for l in start_available_layers)
            if not start_layer_in_available:
                self.get_logger().warn(f"⚠️ 起点选择的层 {start_layer} 不在可用层中，使用可通行性最高的层")
                best_start_layer = max(start_available_layers, key=lambda x: x['traversability'])
                start_layer = best_start_layer['layer_idx']
                self.get_logger().info(f"   新起点层: {start_layer} (可通行性={best_start_layer['traversability']:.2f})")
        
        # 处理终点
        end_2d = end_point[:2]
        end_z = end_point[2]
        
        # 搜索终点附近的所有可用层
        end_available_layers = self.planner.findAvailableLayers(end_2d, search_radius=1.0)
        
        # 打印可用层信息用于调试（只打印前几个）
        if len(end_available_layers) > 0:
            self.get_logger().info(f"🔍 终点附近找到 {len(end_available_layers)} 个可用层")
            for layer_info in end_available_layers[:3]:  # 只打印前3个
                self.get_logger().info(
                    f"   层 {layer_info['layer_idx']}: 理论高度={layer_info['height']:.3f}m, "
                    f"实际高程={layer_info['elevation']:.3f}m, 可通行性={layer_info['traversability']:.2f}"
                )
        
        # 根据 z 坐标选择最近的层
        end_layer = self._select_layer_by_z(end_z, end_available_layers, end_2d)
        
        # 验证选择的层是否在可行区域
        if len(end_available_layers) > 0:
            end_layer_in_available = any(l['layer_idx'] == end_layer for l in end_available_layers)
            if not end_layer_in_available:
                self.get_logger().warn(f"⚠️ 终点选择的层 {end_layer} 不在可用层中，使用可通行性最高的层")
                best_end_layer = max(end_available_layers, key=lambda x: x['traversability'])
                end_layer = best_end_layer['layer_idx']
                self.get_logger().info(f"   新终点层: {end_layer} (可通行性={best_end_layer['traversability']:.2f})")
        
        # 设置起点和终点
        self.start_pos = start_2d
        self.start_layer = start_layer
        self.end_pos = end_2d
        self.end_layer = end_layer
        
        self.get_logger().info(f"📍 起点: ({start_2d[0]:.3f}, {start_2d[1]:.3f}), z={start_z:.3f}m, 层={start_layer}")
        self.get_logger().info(f"📍 终点: ({end_2d[0]:.3f}, {end_2d[1]:.3f}), z={end_z:.3f}m, 层={end_layer}")
        
        # 如果跨层，给出提示
        if start_layer != end_layer:
            self.get_logger().info(f"🔄 跨层规划: 从层 {start_layer} 到层 {end_layer}，路径应该经过楼梯/gateway")
        
        # 开始规划并立即发布
        self.pct_plan_and_publish()
    
    def _select_layer_by_z(self, z_coord, available_layers, pos_2d):
        """
        根据 z 坐标选择最近的层
        
        Args:
            z_coord: 点的 z 坐标
            available_layers: 可用层列表
            pos_2d: 2D 位置坐标（用于日志）
            
        Returns:
            layer_idx: 选择的层索引
        """
        # 首先使用 z 坐标直接计算层（这是最准确的方法）
        calculated_layer = self.planner.height2layer(z_coord)
        
        if len(available_layers) == 0:
            # 如果没有找到可用层，使用计算出的层
            self.get_logger().warn(f"⚠️ 在位置 {pos_2d} 附近未找到可用层，使用 z 坐标计算层: {calculated_layer}")
            return calculated_layer
        
        # 在可用层中，找到实际高程（elevation）最接近 z_coord 的层
        # 注意：使用 elevation（实际高程）而不是 height（理论高度）
        best_layer_idx = available_layers[0]['layer_idx']
        best_elevation = available_layers[0]['elevation']
        min_diff = abs(best_elevation - z_coord)
        
        for layer_info in available_layers:
            # 使用实际高程（elevation）来匹配
            diff = abs(layer_info['elevation'] - z_coord)
            if diff < min_diff:
                min_diff = diff
                best_layer_idx = layer_info['layer_idx']
                best_elevation = layer_info['elevation']
        
        # 如果计算出的层和选择出的层不一致，给出警告
        if calculated_layer != best_layer_idx:
            self.get_logger().warn(
                f"⚠️ 层选择不一致: z坐标计算层={calculated_layer}, "
                f"可用层中选择层={best_layer_idx} (高程={best_elevation:.3f}m, 差值={min_diff:.3f}m)"
            )
            # 优先使用计算出的层，因为它基于 z 坐标更准确
            best_layer_idx = calculated_layer
            best_elevation = z_coord  # 使用 z 坐标作为高程
        
        self.get_logger().info(
            f"   位置 {pos_2d}: z={z_coord:.3f}m, "
            f"选择层 {best_layer_idx} (高程={best_elevation:.3f}m, 差值={min_diff:.3f}m)"
        )
        
        return best_layer_idx
    
    def pct_plan_and_publish(self):
        """规划路径，累积到路径段列表，并立即发布所有路径"""
        self.get_logger().info(f"🚀 开始规划: 起点层={self.start_layer}, 终点层={self.end_layer}")
        
        try:
            traj_3d = self.planner.plan(self.start_pos, self.end_pos, self.start_layer, self.end_layer, self.robot_height)
            
            # 检查路径是否有效（不是None，不是空数组，形状正确）
            if traj_3d is None:
                self.get_logger().warn("❌ Planning failed: 返回 None")
                return
            
            # 检查是否是numpy数组
            if not isinstance(traj_3d, np.ndarray):
                self.get_logger().warn(f"❌ Planning failed: 返回类型错误 {type(traj_3d)}")
                return
            
            # 检查数组是否为空
            if traj_3d.size == 0:
                self.get_logger().warn("❌ Planning failed: 返回空数组")
                return
            
            # 检查数组形状（应该是 N x 3）
            if len(traj_3d.shape) != 2 or traj_3d.shape[1] != 3:
                self.get_logger().warn(f"❌ Planning failed: 数组形状错误 {traj_3d.shape}, 期望 (N, 3)")
                return
            
            # 检查路径长度
            if len(traj_3d) == 0:
                self.get_logger().warn("❌ Planning failed: 路径长度为0")
                return
            
            # 检查路径是否经过中间层
            z_values = traj_3d[:, 2]
            min_z = np.min(z_values)
            max_z = np.max(z_values)
            z_range = max_z - min_z
            self.get_logger().info(f"📊 路径统计: 点数={len(traj_3d)}, z范围=[{min_z:.3f}, {max_z:.3f}], 高度差={z_range:.3f}m")
            
            # 如果起点和终点在不同层，但路径高度变化很小，可能是直接"飞"上去
            if self.start_layer != self.end_layer and z_range < 0.5:
                self.get_logger().warn("⚠️ 警告: 起点和终点在不同层，但路径高度变化很小，可能没有经过楼梯！")
                self.get_logger().warn("   建议: 检查 gateway 检测参数或数据中的楼梯标记")
            
            # 累积路径段（确保是副本，避免引用问题）
            self.path_segments.append(traj_3d.copy())
            # 保存该段的终点层信息
            self.path_segment_end_layers.append(self.end_layer)
            self.get_logger().info(f"✅ 路径段已累积 (共 {len(self.path_segments)} 段)")
            
            # 立即发布所有路径段
            self._publish_all_paths()
            
        except Exception as e:
            self.get_logger().error(f"❌ Planning 异常: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        finally:
            # 规划完成后，重置起点和终点，准备下一次规划
            # 但保留current_start和current_start_layer作为下一段的起点
            self.start_pos = None
            self.start_layer = None
            self.end_pos = None
            self.end_layer = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, default='Spiral', help='Name of the scene. Available: [\'Spiral\', \'Building\', \'Plaza\', \'Map\']')
    parser.add_argument('--robot_height', type=float, default=0.05, help='Height of the robot')
    parser.add_argument('--waypoints_type', type=str, default='auto', 
                        choices=['PoseArray', 'Path', 'auto'],
                        help='Waypoints message type: Path (default via auto), PoseArray, or auto (detects and subscribes to both types).')
    parser.add_argument('--load_trajectory', action='store_true',
                        help='Load saved trajectory on startup')
    args = parser.parse_args()

    cfg = Config()

    match args.scene:
        case "Spiral":
            tomo_file = "spiral0.3_2"
        case 'Building':
            tomo_file = 'building2_9'
        case 'Plaza':
            tomo_file = 'plaza3_10'
        case "Map":
            tomo_file = "map"
        case _:
            raise ValueError('Invalid scene name')

    # 如果使用 auto，尝试检测话题类型
    waypoints_type = args.waypoints_type
    if waypoints_type == 'auto':
        import subprocess
        try:
            result = subprocess.run(['ros2', 'topic', 'info', '/waypoints'], 
                                  capture_output=True, text=True, timeout=2)
            has_path = 'nav_msgs/msg/Path' in result.stdout
            has_posearray = 'geometry_msgs/msg/PoseArray' in result.stdout
            
            if has_path and has_posearray:
                # 话题同时支持两种类型，保持 auto 模式以同时订阅两种类型
                print("🔍 自动检测: /waypoints 话题同时支持 nav_msgs/Path 和 geometry_msgs/PoseArray，将同时订阅两种类型")
                waypoints_type = 'auto'  # 保持 auto，让 PCTPlanner 同时订阅
            elif has_path:
                waypoints_type = 'Path'
                print("🔍 自动检测: /waypoints 话题类型为 nav_msgs/Path")
            elif has_posearray:
                waypoints_type = 'PoseArray'
                print("🔍 自动检测: /waypoints 话题类型为 geometry_msgs/PoseArray")
            else:
                # 无法检测，使用 auto 模式同时订阅两种类型作为兜底
                print("⚠️ 无法检测话题类型，将同时订阅 nav_msgs/Path 和 geometry_msgs/PoseArray")
                waypoints_type = 'auto'
        except:
            # 检测失败，使用 auto 模式同时订阅两种类型作为兜底
            print("⚠️ 无法检测话题类型，将同时订阅 nav_msgs/Path 和 geometry_msgs/PoseArray")
            waypoints_type = 'auto'

    rclpy.init(args=None)
    node = PCTPlanner(cfg, tomo_file, args.robot_height, waypoints_type)
    
    # 如果指定了加载轨迹，则加载
    if hasattr(args, 'load_trajectory') and args.load_trajectory:
        node._load_trajectory()
    
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

