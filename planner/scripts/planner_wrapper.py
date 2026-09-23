import os
import sys
import pickle
import numpy as np

from utils import *
from stair_centerline import build_stair_centerline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLANNER_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PLANNER_ROOT not in sys.path:
    sys.path.insert(0, PLANNER_ROOT)
from lib import a_star, ele_planner, traj_opt

rsg_root = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))


class TomogramPlanner(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.use_quintic = self.cfg.planner.use_quintic
        self.max_heading_rate = self.cfg.planner.max_heading_rate
        self.astar_cost_threshold = self.cfg.wrapper.astar_cost_threshold
        self.path_height_offset = self.cfg.wrapper.path_height_offset

        self.tomo_dir = rsg_root + self.cfg.wrapper.tomo_dir

        self.resolution = None
        self.center = None
        self.n_slice = None
        self.slice_h0 = None
        self.slice_dh = None
        self.map_dim = []
        self.offset = None

        self.start_idx = np.zeros(3, dtype=np.int32)
        self.end_idx = np.zeros(3, dtype=np.int32)
        self.last_start_layer = -1
        self.last_end_layer = -1
        self.last_centerline_result = None
        
        # 保存tomogram数据以便查询可用层
        self.elev_g = None
        self.trav = None

    def loadTomogram(self, tomo_file):
        if os.path.isabs(tomo_file):
            tomo_path = tomo_file
        else:
            tomo_path = os.path.join(self.tomo_dir, tomo_file)
        if not tomo_path.endswith('.pickle'):
            tomo_path += '.pickle'

        with open(tomo_path, 'rb') as handle:
            data_dict = pickle.load(handle)

            tomogram = np.asarray(data_dict['data'], dtype=np.float32)

            self.resolution = float(data_dict['resolution'])
            self.center = np.asarray(data_dict['center'], dtype=np.double)
            self.n_slice = tomogram.shape[1]
            self.slice_h0 = float(data_dict['slice_h0'])
            self.slice_dh = float(data_dict['slice_dh'])
            self.map_dim = [tomogram.shape[2], tomogram.shape[3]]
            self.offset = np.array([int(self.map_dim[0] / 2), int(self.map_dim[1] / 2)], dtype=np.int32)
            print(f" {self.n_slice=}\n {self.slice_dh=}\n {self.slice_h0=}\n {self.offset=}\n")

        trav = tomogram[0]
        trav_gx = tomogram[1]
        trav_gy = tomogram[2]
        elev_g = tomogram[3]
        elev_g = np.nan_to_num(elev_g, nan=-100)
        elev_c = tomogram[4]
        elev_c = np.nan_to_num(elev_c, nan=1e6)

        # 保存数据以便后续查询
        self.elev_g = elev_g
        self.trav = trav
        self.gateway_map = None  # 将在initPlanner中设置

        self.initPlanner(trav, trav_gx, trav_gy, elev_g, elev_c)

    def initPlanner(self, trav, trav_gx, trav_gy, elev_g, elev_c):
        diff_t = trav[1:] - trav[:-1]
        diff_g = np.abs(elev_g[1:] - elev_g[:-1])

        # 使用配置参数
        trav_threshold = self.cfg.planner.gateway_traversability_threshold
        elev_threshold = self.cfg.planner.gateway_elevation_threshold

        gateway_up = np.zeros_like(trav, dtype=bool)
        mask_t = diff_t < -trav_threshold
        mask_g = (diff_g < elev_threshold) & (~np.isnan(elev_g[1:]))
        gateway_up[:-1] = np.logical_and(mask_t, mask_g)

        gateway_dn = np.zeros_like(trav, dtype=bool)
        mask_t = diff_t > trav_threshold
        mask_g = (diff_g < elev_threshold) & (~np.isnan(elev_g[:-1]))
        gateway_dn[1:] = np.logical_and(mask_t, mask_g)

        gateway = np.zeros_like(trav, dtype=np.int32)
        gateway[gateway_up] = 2
        gateway[gateway_dn] = -2
        
        # 保存gateway地图以便后续查询
        self.gateway_map = gateway
        
        # 打印 gateway 统计信息用于调试
        num_gateway_up = np.sum(gateway_up)
        num_gateway_dn = np.sum(gateway_dn)
        print(f"Gateway 检测统计:")
        print(f"  向上连接点 (gateway_up): {num_gateway_up}")
        print(f"  向下连接点 (gateway_dn): {num_gateway_dn}")
        print(f"  检测阈值: 可通行性差异 > {trav_threshold}, 高程差异 < {elev_threshold}")

        self.planner = ele_planner.OfflineElePlanner(
            max_heading_rate=self.max_heading_rate, use_quintic=self.use_quintic
        )

        self.planner.init_map(
            self.cfg.wrapper.astar_cost_threshold,
            self.cfg.wrapper.safe_cost_margin,
            self.resolution, self.n_slice, self.cfg.wrapper.step_cost_weight,
            trav.reshape(-1, trav.shape[-1]).astype(np.double),
            elev_g.reshape(-1, elev_g.shape[-1]).astype(np.double),
            elev_c.reshape(-1, elev_c.shape[-1]).astype(np.double),
            gateway.reshape(-1, gateway.shape[-1]),
            trav_gy.reshape(-1, trav_gy.shape[-1]).astype(np.double),
            -trav_gx.reshape(-1, trav_gx.shape[-1]).astype(np.double)
        )
        # 机体高度只在输出路径坐标转换时增加，避免和 SCAN 重复抬高。
        self.planner.set_reference_height(0.0)

    def plan(self, start_pos, end_pos, start_layer=None, end_layer=None,
             robot_height=None, keep_goal_on_start_layer=True):
        """Plan between map positions with explicit or localization-based layers."""
        self.last_centerline_result = None
        start_idx = self.pos2idx(start_pos)
        end_idx = self.pos2idx(end_pos)
        start_layer = self.selectLayer(start_pos, start_idx, start_layer)
        if end_layer is None and keep_goal_on_start_layer:
            try:
                end_layer = self.selectLayer(end_pos, end_idx, start_layer)
            except ValueError:
                end_layer = self.selectLayer(end_pos, end_idx, None)
        else:
            end_layer = self.selectLayer(end_pos, end_idx, end_layer)

        self.start_idx[0] = start_layer
        self.start_idx[1:] = start_idx
        self.end_idx[0] = end_layer
        self.end_idx[1:] = end_idx
        self.last_start_layer = int(start_layer)
        self.last_end_layer = int(end_layer)

        # 验证起点和终点是否在可行区域
        start_valid = self._check_position_validity(start_pos, start_layer)
        end_valid = self._check_position_validity(end_pos, end_layer)
        
        if not start_valid:
            print(f"⚠️ 警告: 起点 ({start_pos[0]:.3f}, {start_pos[1]:.3f}) 在层 {start_layer} 可能不在可行区域")
            print(f"   建议: 检查起点位置，确保在可通行区域内")
        if not end_valid:
            print(f"⚠️ 警告: 终点 ({end_pos[0]:.3f}, {end_pos[1]:.3f}) 在层 {end_layer} 可能不在可行区域")
            print(f"   建议: 检查终点位置，确保在可通行区域内")
        
        # 如果跨层，检查是否有gateway可用
        if start_layer != end_layer and self.gateway_map is not None:
            layer_diff = end_layer - start_layer
            if layer_diff > 0:
                # 需要向上，查找向上的gateway
                gateway_count = np.sum(self.gateway_map == 2)
                print(f"🔍 跨层规划: 从层 {start_layer} 到层 {end_layer} (向上 {layer_diff} 层)")
                print(f"   可用向上gateway数量: {gateway_count}")
            else:
                # 需要向下，查找向下的gateway
                gateway_count = np.sum(self.gateway_map == -2)
                print(f"🔍 跨层规划: 从层 {start_layer} 到层 {end_layer} (向下 {abs(layer_diff)} 层)")
                print(f"   可用向下gateway数量: {gateway_count}")
            
            if gateway_count == 0:
                print(f"❌ 错误: 没有可用的gateway进行跨层！路径可能无法正确规划")

        if not self.planner.plan(self.start_idx, self.end_idx, True):
            print("❌ PCT规划或轨迹优化失败")
            return None
        path_finder: a_star.Astar = self.planner.get_path_finder()
        path = path_finder.get_result_matrix()
        if len(path) == 0:
            print(f"❌ A*搜索失败: 无法找到从起点到终点的路径")
            return None
        
        # 检查路径是否经过gateway（如果跨层的话）
        if start_layer != end_layer and len(path) > 0:
            # path的第一列是层索引
            if path.shape[1] > 0:
                path_layers = path[:, 0].astype(int)
                layer_changes = np.diff(path_layers)
                num_layer_changes = np.sum(layer_changes != 0)
                if num_layer_changes > 0:
                    print(f"✅ 路径检测到 {num_layer_changes} 次层切换")
                    # 检查层变化是否平滑（应该是逐步变化的）
                    if np.any(np.abs(layer_changes) > 1):
                        print(f"⚠️ 警告: 路径中有跳跃式层变化（>1层），可能没有经过楼梯")
                else:
                    print(f"❌ 严重警告: 起点在层 {start_layer}，终点在层 {end_layer}，但路径没有层变化！")
                    print(f"   路径直接'飞'上去了，没有经过楼梯！")
                    print(f"   路径层序列: {path_layers[:10]}..." if len(path_layers) > 10 else f"   路径层序列: {path_layers}")

        optimizer: traj_opt.GPMPOptimizer = (
            self.planner.get_trajectory_optimizer()
            if not self.use_quintic
            else self.planner.get_trajectory_optimizer_wnoj()
        )

        opt_init = optimizer.get_opt_init_value()
        init_layer = optimizer.get_opt_init_layer()
        traj_raw = optimizer.get_result_matrix()
        layers = optimizer.get_layers()
        heights = optimizer.get_heights()

        opt_init = np.concatenate([opt_init.transpose(1, 0), init_layer.reshape(-1, 1)], axis=-1)
        traj = np.concatenate([traj_raw, layers.reshape(-1, 1)], axis=-1)
        y_idx = (traj.shape[-1] - 1) // 2
        traj_3d = np.stack([traj[:, 0], traj[:, y_idx], heights / self.resolution], axis=1)
        height_offset = (
            self.path_height_offset
            if robot_height is None
            else float(robot_height)
        )
        traj_3d = transTrajGrid2Map(
            self.map_dim, self.center, self.resolution, traj_3d,
            z_offset=height_offset
        )

        return traj_3d

    def plan_stair_centerline(
        self,
        start_pos,
        end_pos,
        start_layer=None,
        end_layer=None,
        max_cost=None,
        max_support_error=0.25,
        max_support_step=0.30,
        max_layer_step=1,
        smoothing_window=9,
        minimum_xy_length=0.5,
        minimum_vertical_span=0.5,
    ):
        """Build a validated XY-straight path without invoking PCT A*."""
        if self.trav is None or self.elev_g is None:
            raise RuntimeError('Tomogram has not been loaded')

        start_idx = self.pos2idx(start_pos)
        end_idx = self.pos2idx(end_pos)
        selected_start_layer = self.selectLayer(
            start_pos, start_idx, start_layer
        )
        selected_end_layer = None
        if end_layer is not None:
            selected_end_layer = self.selectLayer(
                end_pos, end_idx, end_layer
            )

        result = build_stair_centerline(
            start_pos=start_pos,
            goal_pos=end_pos,
            traversability=self.trav,
            ground_elevation=self.elev_g,
            center=self.center,
            resolution=self.resolution,
            path_height_offset=self.path_height_offset,
            cost_threshold=(
                self.astar_cost_threshold if max_cost is None else max_cost
            ),
            max_support_error=max_support_error,
            max_support_step=max_support_step,
            max_layer_step=max_layer_step,
            smoothing_window=smoothing_window,
            minimum_xy_length=minimum_xy_length,
            minimum_vertical_span=minimum_vertical_span,
            start_layer=selected_start_layer,
            end_layer=selected_end_layer,
        )

        self.start_idx[0] = result.layers[0]
        self.start_idx[1:] = start_idx
        self.end_idx[0] = result.layers[-1]
        self.end_idx[1:] = end_idx
        self.last_start_layer = int(result.layers[0])
        self.last_end_layer = int(result.layers[-1])
        self.last_centerline_result = result
        return result.trajectory

    def pos2idx(self, pos):
        position = np.asarray(pos, dtype=np.float64).reshape(-1)
        if position.size < 2 or not np.isfinite(position[:2]).all():
            raise ValueError('Position must contain finite x and y values')

        idx = (
            np.round((position[:2] - self.center) / self.resolution)
            .astype(np.int32) + self.offset
        )
        row, col = int(idx[0]), int(idx[1])
        if row < 0 or row >= self.map_dim[0] or col < 0 or col >= self.map_dim[1]:
            raise ValueError(
                'Position ({:.3f}, {:.3f}) is outside tomogram bounds'.format(
                    position[0], position[1]
                )
            )
        # PCT internally indexes [layer, column(Y), row(X)].
        return np.array([col, row], dtype=np.int32)

    def selectLayer(self, pos, pos_idx=None, requested_layer=None):
        if self.trav is None or self.elev_g is None:
            raise RuntimeError('Tomogram has not been loaded')
        if pos_idx is None:
            pos_idx = self.pos2idx(pos)

        y_idx, x_idx = int(pos_idx[0]), int(pos_idx[1])
        costs = self.trav[:, x_idx, y_idx].astype(np.float64)
        heights = self.elev_g[:, x_idx, y_idx].astype(np.float64)
        valid = np.isfinite(costs) & np.isfinite(heights)
        valid &= costs <= self.astar_cost_threshold
        valid &= heights > -100.0

        if requested_layer is not None:
            layer = int(requested_layer)
            if layer < 0 or layer >= self.n_slice:
                raise ValueError(
                    'Requested layer {} is outside [0, {})'.format(
                        layer, self.n_slice
                    )
                )
            if not valid[layer]:
                raise ValueError(
                    'Requested layer {} is not traversable at '
                    '({:.3f}, {:.3f}); cost={:.3f}'.format(
                        layer,
                        float(np.asarray(pos)[0]),
                        float(np.asarray(pos)[1]),
                        costs[layer],
                    )
                )
            return layer

        candidates = np.flatnonzero(valid)
        if candidates.size == 0:
            raise ValueError(
                'No traversable tomogram layer at ({:.3f}, {:.3f})'.format(
                    float(np.asarray(pos)[0]), float(np.asarray(pos)[1])
                )
            )

        position = np.asarray(pos, dtype=np.float64).reshape(-1)
        if position.size >= 3 and np.isfinite(position[2]):
            body_z = float(position[2])
            nominal_layer = (body_z - self.slice_h0) / self.slice_dh
            expected_ground = body_z - self.path_height_offset
            scores = np.abs(candidates - nominal_layer)
            scores += (
                0.25
                * np.abs(heights[candidates] - expected_ground)
                / max(self.slice_dh, 1e-6)
            )
            scores += 1e-3 * costs[candidates]
            return int(candidates[np.argmin(scores)])

        return int(candidates[np.argmin(costs[candidates])])

    def height2layer(self, height):
        """
        Convert actual height to layer index
        """
        layer = np.round((height - self.slice_h0) / self.slice_dh).astype(np.int32)
        print(f" {height=}\n {layer=}\n")
        return np.clip(layer, 0, self.n_slice - 1)
    
    def findAvailableLayers(self, pos, search_radius=0.5):
        """
        在点击位置附近搜索所有可用的层
        
        Args:
            pos: 点击位置的2D坐标 [x, y]
            search_radius: 搜索半径（米），默认0.5米
            
        Returns:
            available_layers: 可用层的列表，每个元素为 (layer_idx, height, traversability)
                - layer_idx: 层索引
                - height: 该层的实际高度
                - traversability: 可通行性值（越大越容易通行）
        """
        if self.elev_g is None or self.trav is None:
            return []
        
        # 将位置转换为索引
        # pos2idx返回 [y_idx, x_idx]，需要交换顺序
        pos_idx = self.pos2idx(pos)
        y_idx = int(pos_idx[0])  # pos2idx返回的第一个是y索引
        x_idx = int(pos_idx[1])  # pos2idx返回的第二个是x索引
        
        # 检查索引是否在有效范围内
        # map_dim[0]是x方向，map_dim[1]是y方向
        if x_idx < 0 or x_idx >= self.map_dim[0] or y_idx < 0 or y_idx >= self.map_dim[1]:
            return []
        
        # 计算搜索半径对应的网格数
        search_grids = int(np.ceil(search_radius / self.resolution))
        
        available_layers = []
        
        # 遍历所有层
        for layer_idx in range(self.n_slice):
            # 在搜索半径内查找有效点
            min_x = max(0, x_idx - search_grids)
            max_x = min(self.map_dim[0], x_idx + search_grids + 1)
            min_y = max(0, y_idx - search_grids)
            max_y = min(self.map_dim[1], y_idx + search_grids + 1)
            
            # 获取该层在搜索区域内的数据
            # elev_g和trav的维度是 [n_slice, map_dim[0], map_dim[1]]
            layer_elev = self.elev_g[layer_idx, min_x:max_x, min_y:max_y]
            layer_trav = self.trav[layer_idx, min_x:max_x, min_y:max_y]
            
            # 检查是否有有效的可通行点（elev_g > -100 表示有数据，trav > 0 表示可通行）
            valid_mask = (layer_elev > -100) & (layer_trav > 0) & (~np.isnan(layer_elev))
            
            if np.any(valid_mask):
                # 计算该层的平均高度和最大可通行性
                valid_elev = layer_elev[valid_mask]
                valid_trav = layer_trav[valid_mask]
                
                # elev_g存储的是实际高度值（米），直接使用
                avg_elevation = np.mean(valid_elev)
                max_trav = np.max(valid_trav)
                
                # 计算该层的理论高度
                layer_height = self.slice_h0 + layer_idx * self.slice_dh
                
                available_layers.append({
                    'layer_idx': layer_idx,
                    'height': layer_height,
                    'elevation': avg_elevation,
                    'traversability': max_trav
                })
        
        # 按高度排序
        available_layers.sort(key=lambda x: x['height'])
        
        return available_layers
    
    def _check_position_validity(self, pos, layer):
        """
        检查位置在指定层是否可行
        
        Args:
            pos: 2D位置坐标 [x, y]
            layer: 层索引
            
        Returns:
            bool: 如果位置在可行区域返回True，否则返回False
        """
        if self.trav is None or self.elev_g is None:
            return False
        
        pos_idx = self.pos2idx(pos)
        y_idx = int(pos_idx[0])
        x_idx = int(pos_idx[1])
        
        # 检查索引是否在有效范围内
        if x_idx < 0 or x_idx >= self.map_dim[0] or y_idx < 0 or y_idx >= self.map_dim[1]:
            return False
        
        # 检查该位置在该层的可通行性
        if layer < 0 or layer >= self.n_slice:
            return False
        
        trav_value = self.trav[layer, x_idx, y_idx]
        elev_value = self.elev_g[layer, x_idx, y_idx]
        
        # 可通行性 > 0 且高程有效
        return trav_value > 0 and elev_value > -100 and not np.isnan(elev_value)
