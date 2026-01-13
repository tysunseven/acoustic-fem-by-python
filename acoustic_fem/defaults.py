import numpy as np

# 物理常数配置 (与 MATLAB define_acoustic_params.m 保持一致)
PHYSICS_PARAMS = {
    'Lx_design': 0.032,      # 设计区域长度 (m)
    'Lx_half': 10,          # 左右波导长度倍数
    'rhoa': 1.225,          # 空气密度
    'kappaa': 141834.999,   # 空气体积模量
    'p_in': 1,              # 入射声压
    'rhoav': 1.225 * 10**6,       # 夹杂物密度
    'kappaav': 141834.999 * 10**7 # 夹杂物体积模量
}

# 仿真网格配置
GRID_CONFIG = {
    'resolution': 128,      # FEM 网格分辨率 (128x128)
    'input_resolution': 8   # 输入结构分辨率 (8x8)
}