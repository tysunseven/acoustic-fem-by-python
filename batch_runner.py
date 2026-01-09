import argparse
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
from src.solver import AcousticFEMSolver
from src.config import PHYSICS_PARAMS, GRID_CONFIG
from src.utils import load_structures_from_mat, save_results_to_npy
from scipy.ndimage import zoom

def process_single_structure(args):
    """包装函数，用于多进程调用"""
    struct_8x8, freq = args
    
    # 1. 动态更新频率参数
    params = PHYSICS_PARAMS.copy()
    params['f'] = freq
    params['c'] = np.sqrt(params['kappaa'] / params['rhoa'])
    params['omega'] = 2 * np.pi * params['f']
    params['k'] = params['omega'] / params['c']
    params['Ly_design'] = params['Lx_design']
    
    # 2. 插值放大 (使用 kron 替代 zoom 以保证绝对的块状对齐)
    res_in = GRID_CONFIG['input_resolution']
    res_out = GRID_CONFIG['resolution']
    scale = int(res_out / res_in) # 确保是整数，例如 16
    
    # 使用克罗内克积进行精确放大 (每个像素变为 scale x scale 的块)
    # 这一步等价于 MATLAB 的 imresize(..., 'nearest') 且无坐标偏差
    struct_fem = np.kron(struct_8x8, np.ones((scale, scale)))
    
    # 3. 构造全场 (构建包含左右波导的计算域)
    # 获取参数
    Lx_half = params['Lx_half']
    resolution = GRID_CONFIG['resolution'] # 128
    
    # 计算总尺寸
    # 宽度 = 左波导(Lx_half) + 设计区域(1) + 右波导(Lx_half)
    full_width_blocks = 2 * Lx_half + 1
    full_width_pixels = full_width_blocks * resolution
    full_height_pixels = resolution
    
    # 初始化全场为空气 (0)
    x_total = np.zeros((full_height_pixels, full_width_pixels))
    
    # 计算设计区域的插入位置
    start_col = Lx_half * resolution
    end_col = start_col + resolution
    
    # 填入插值后的结构
    # 注意：需确保 struct_fem 尺寸与插入区域严格匹配
    # 如果 zoom 产生细微误差，可以使用切片强制赋值
    if struct_fem.shape != (resolution, resolution):
         # 简单的尺寸保护，防止 crash
         from scipy.ndimage import zoom as scipy_zoom
         scale_fix = resolution / struct_fem.shape[0]
         struct_fem = scipy_zoom(struct_fem, scale_fix, order=0)
         
    x_total[:, start_col:end_col] = struct_fem
    
    # 4. 求解
    try:
        solver = AcousticFEMSolver(params)
        P_grid = solver.solve(x_total)
        
        # 计算透射和反射系数 (传入设计区域分辨率用于定位测量点)
        T, R = solver.calculate_TR(P_grid, resolution)
        return T, R
        
    except Exception as e:
        print(f"Simulation failed for a structure: {e}")
        return np.nan, np.nan

def main():
    parser = argparse.ArgumentParser(description="Acoustic FEM Batch Solver")
    parser.add_argument('--input', type=str, required=True, help='Path to .mat file containing structures')
    parser.add_argument('--freq', type=float, default=3000.0, help='Frequency to compute (Hz)')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel processes')
    args = parser.parse_args()

    print(f"--- Starting Batch Job for {args.freq} Hz ---")
    
    # 1. 加载数据
    structures = load_structures_from_mat(args.input)
    print(f"Loaded {structures.shape[0]} structures.")
    
    # 2. 准备参数
    tasks = [(s, args.freq) for s in structures]
    
    # 3. 并行计算
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # 使用 tqdm 显示进度条 (建议添加 tqdm 到 requirements.txt)
        from tqdm import tqdm
        results = list(tqdm(executor.map(process_single_structure, tasks), total=len(tasks)))
        
    # 4. 保存
    T_vals = np.array([r[0] for r in results])
    R_vals = np.array([r[1] for r in results])
    save_results_to_npy('results', T_vals, R_vals)

if __name__ == '__main__':
    main()