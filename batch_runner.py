import argparse
import numpy as np
import os
import time
from concurrent.futures import ProcessPoolExecutor
from src.solver import AcousticFEMSolver
from src.config import PHYSICS_PARAMS, GRID_CONFIG
from src.utils import load_structures_from_mat

def process_single_structure(args):
    """包装函数，用于多进程调用"""
    struct_8x8, freq = args
    
    # 1. 动态更新频率参数
    params = PHYSICS_PARAMS.copy()
    params['f'] = freq
    # 重新计算依赖频率的衍生参数
    params['c'] = np.sqrt(params['kappaa'] / params['rhoa'])
    params['omega'] = 2 * np.pi * params['f']
    params['k'] = params['omega'] / params['c']
    params['Ly_design'] = params['Lx_design']
    
    # 2. 插值放大 (保留 kron 修正以保证精度)
    res_in = GRID_CONFIG['input_resolution']
    res_out = GRID_CONFIG['resolution']
    scale = int(res_out / res_in) # 确保是整数，例如 16
    
    # 使用克罗内克积进行精确放大
    struct_fem = np.kron(struct_8x8, np.ones((scale, scale)))
    
    # 3. 构造全场 (构建包含左右波导的计算域)
    # 获取参数
    Lx_half = params['Lx_half']
    resolution = GRID_CONFIG['resolution'] # 128
    
    # 计算总尺寸
    full_width_blocks = 2 * Lx_half + 1
    full_width_pixels = full_width_blocks * resolution
    full_height_pixels = resolution
    
    # 初始化全场为空气 (0)
    x_total = np.zeros((full_height_pixels, full_width_pixels))
    
    # 计算设计区域的插入位置
    start_col = Lx_half * resolution
    end_col = start_col + resolution
    
    # 填入结构
    if struct_fem.shape != (resolution, resolution):
         # 简单的尺寸保护
         from scipy.ndimage import zoom as scipy_zoom
         scale_fix = resolution / struct_fem.shape[0]
         struct_fem = scipy_zoom(struct_fem, scale_fix, order=0)
         
    x_total[:, start_col:end_col] = struct_fem
    
    # 4. 求解
    try:
        solver = AcousticFEMSolver(params)
        P_grid = solver.solve(x_total)
        
        # 计算透射和反射系数
        T, R = solver.calculate_TR(P_grid, resolution)
        return T, R
        
    except Exception as e:
        print(f"Simulation failed for a structure at {freq}Hz: {e}")
        return np.nan, np.nan

def main():
    parser = argparse.ArgumentParser(description="Acoustic FEM Batch Solver (Multi-Frequency)")
    parser.add_argument('--input', type=str, required=True, help='Path to .mat file containing structures')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel processes')
    args = parser.parse_args()

    # --- 1. 提取输入文件的基础名称 ---
    # 例如：'data/test_structures.mat' -> 'test_structures'
    input_filename = os.path.basename(args.input)
    file_basename = os.path.splitext(input_filename)[0]
    
    # 定义要计算的 10 个频率点
    target_frequencies = np.linspace(2000, 2900, 10) 
    
    # 2. 加载数据
    print(f"Loading structures from {args.input}...")
    structures = load_structures_from_mat(args.input)
    print(f"Loaded {structures.shape[0]} structures.")
    
    # 准备结果目录
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. 循环计算每个频率
    for freq in target_frequencies:
        print(f"\n--- Starting Batch Job for Frequency: {freq:.1f} Hz ---")
        
        # 准备任务列表
        tasks = [(s, freq) for s in structures]
        
        # 并行计算
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            from tqdm import tqdm
            results = list(tqdm(executor.map(process_single_structure, tasks), total=len(tasks), desc=f"Freq {int(freq)}"))
            
        # 整理结果
        T_vals = np.array([r[0] for r in results])
        R_vals = np.array([r[1] for r in results])
        
        # 4. 动态保存结果 (文件名包含：输入源 + 频率)
        # 例如: T_values_test_structures_2000.npy
        t_filename = f'T_values_{file_basename}_{int(freq)}.npy'
        r_filename = f'R_values_{file_basename}_{int(freq)}.npy'
        
        np.save(os.path.join(output_dir, t_filename), T_vals)
        np.save(os.path.join(output_dir, r_filename), R_vals)
        
        print(f"✅ Saved results to {output_dir}/{t_filename}")

    print("\nAll frequencies computed successfully!")

if __name__ == '__main__':
    main()