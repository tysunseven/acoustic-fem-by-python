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
    
    # 2. 插值放大
    res_in = GRID_CONFIG['input_resolution']
    res_out = GRID_CONFIG['resolution']
    scale = res_out / res_in
    struct_fem = zoom(struct_8x8, scale, order=0) # 最近邻插值
    
    # 3. 构造全场并计算
    # ... (此处复用之前提供的构造 x_total 的逻辑) ...
    # 为简洁省略具体构建代码，参考上一轮的 run_single_simulation
    
    solver = AcousticFEMSolver(params)
    # ... solve ...
    # return T, R
    return 0.0, 0.0 # 占位符

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