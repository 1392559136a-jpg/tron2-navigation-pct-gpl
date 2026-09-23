class ConfigPlanner():
    use_quintic = True
    max_heading_rate = 10
    # Gateway 检测参数
    # 可通行性差异阈值：较小的值更容易检测到楼梯，但可能产生误检
    # 建议范围：5.0-10.0，如果楼梯未被检测到，可以尝试降低到 5.0 或 6.0
    gateway_traversability_threshold =2.0# 默认从 8.0 降低到 6.0，更容易检测楼梯
    # 高程差异阈值：楼梯连接点的高程应该很接近
    gateway_elevation_threshold = 0.1  # 高程差异阈值，用于检测连接点


class ConfigWrapper():
    tomo_dir = '/rsc/tomogram/'
    astar_cost_threshold = 20
    safe_cost_margin = 15
    step_cost_weight = 0.2
    path_height_offset = 0.5


class Config():
    planner = ConfigPlanner()
    wrapper = ConfigWrapper()
