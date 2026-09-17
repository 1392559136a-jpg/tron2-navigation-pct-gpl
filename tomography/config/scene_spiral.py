from .scene import Scene

class SceneSpiral(Scene):
    def __init__(self) -> None:
        super().__init__()

        self.pcd.file_name = 'spiral0.3_2.pcd'

        # 地图基础参数（保持你的显存优化策略）
        self.map.resolution = 0.1  # 分辨率0.4米/像素，平衡精度与显存
        self.map.ground_h = 0.0    # 地面高度基准，保持不变
        self.map.slice_dh = 0.5    # 切片间隔，保持不变

        # 针对楼梯优化的可通行性参数
        self.trav.kernel_size = 9.5  # 从99减小到49：避免大核过度侵蚀楼梯窄通道（改这个又用）
        self.trav.interval_min = 0.20  # 降低空闲点云最小间隔，适配楼梯点云密度
        self.trav.interval_free = 0.50 # 微调空闲区间阈值
        self.trav.slope_max = 1.4  # 从1.40增大到2.00：适配楼梯的大坡度（约63度）
        self.trav.step_max = 0.3   # 从0.10增大到0.18：匹配楼梯台阶高度
        self.trav.standable_ratio = 0.5 # 从0.40降低到0.25：适配楼梯踏步的有效站立比例
        self.trav.cost_barrier = 10.0    # 障碍成本值，保持不变
        self.trav.safe_margin = 0.01     # 从1.2减小到1.0：降低安全裕度，适配楼梯宽度
        self.trav.inflation = 0.05      # 从0.2减小到0.15：减少障碍膨胀，避免楼梯边缘被遮挡
