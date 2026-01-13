# acoustic_fem/simulation.py
import numpy as np
from scipy.ndimage import zoom
from .core import AcousticFEMSolver
from .defaults import PHYSICS_PARAMS, GRID_CONFIG

class AcousticSimulator:
    """
    高层封装接口：负责将用户输入的单胞结构转化为物理场并求解。
    """
    def __init__(self, physics_config=None, grid_config=None):
        self.params = PHYSICS_PARAMS.copy()
        if physics_config:
            self.params.update(physics_config)
            
        self.grid_config = GRID_CONFIG.copy()
        if grid_config:
            self.grid_config.update(grid_config)
            
        # 初始化底层求解器 (复用 mesh 缓存)
        # 初始化时使用默认频率占位，后续 predict 会动态更新
        self._init_derived_params(f=2000) 
        self.solver = AcousticFEMSolver(self.params)

    def _init_derived_params(self, f):
        """根据频率计算 omega, k, c 等导出参数"""
        self.params['f'] = f
        self.params['c'] = np.sqrt(self.params['kappaa'] / self.params['rhoa'])
        self.params['omega'] = 2 * np.pi * f
        self.params['k'] = self.params['omega'] / self.params['c']
        # 确保 Ly_design 与 Lx_design 一致 (假设正方形单元)
        self.params['Ly_design'] = self.params['Lx_design']

    def _preprocess_structure(self, unit_cell):
        """
        将输入的单胞 (如 8x8) 映射到仿真网格 (如 128x128)，并填充波导背景。
        """
        # 1. 放大/插值 (Upscaling)
        res_in = self.grid_config.get('input_resolution', unit_cell.shape[0])
        res_out = self.grid_config['resolution']
        
        # 如果尺寸不匹配，进行缩放
        if unit_cell.shape[0] != res_out:
            scale = res_out / unit_cell.shape[0]
            # 使用最近邻插值保持二值特性，或者使用 kron 扩展
            # 这里为了通用性使用 scipy.ndimage.zoom (order=0 为最近邻)
            struct_fem = zoom(unit_cell, scale, order=0)
        else:
            struct_fem = unit_cell

        # 2. 构建全场 (Full Domain Setup)
        Lx_half = self.params['Lx_half']
        resolution = res_out
        
        # 计算总宽度 (左波导 + 单胞 + 右波导)
        full_width_blocks = 2 * Lx_half + 1
        full_width_pixels = full_width_blocks * resolution
        full_height_pixels = resolution
        
        # 初始化全场背景 (0通常代表空气/基体)
        x_total = np.zeros((full_height_pixels, full_width_pixels))
        
        # 将单胞放置在中心
        start_col = Lx_half * resolution
        end_col = start_col + resolution
        x_total[:, start_col:end_col] = struct_fem
        
        return x_total

    def predict(self, unit_cell, frequency):
        """
        标准接口：输入单胞和频率，返回 T 和 R
        
        Args:
            unit_cell (np.ndarray): 二维数组，代表单胞结构 (例如 8x8)
            frequency (float): 频率 Hz
            
        Returns:
            tuple: (T, R) 复数透射与反射系数
        """
        # 1. 更新物理参数
        self._init_derived_params(frequency)
        self.solver.update_params(self.params)
        
        # 2. 预处理结构 (生成全场网格)
        x_total = self._preprocess_structure(unit_cell)
        
        # 3. 调用 FEM 求解
        P_grid = self.solver.solve(x_total)
        
        # 4. 计算指标
        resolution = self.grid_config['resolution']
        T, R = self.solver.calculate_TR(P_grid, resolution)
        
        return T, R