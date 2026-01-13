import argparse
import numpy as np
import os
import time
from concurrent.futures import ProcessPoolExecutor
from src.solver import AcousticFEMSolver
from src.config import PHYSICS_PARAMS, GRID_CONFIG
from src.utils import load_structures_from_mat
from scipy.ndimage import zoom as scipy_zoom # 提前导入

# --- 全局变量 (仅在 Worker 进程中有效) ---
_SOLVER_INSTANCE = None

def init_worker(params_template):
    """
    Worker 进程初始化函数。
    每个进程只运行一次，用于创建一个持久的 Solver 实例。
    """
    global _SOLVER_INSTANCE
    # 使用初始参数实例化 Solver，后续只更新频率
    _SOLVER_INSTANCE = AcousticFEMSolver(params_template)

def process_single_structure(args):
    """
    处理单个结构 (运行在 Worker 进程中)
    """
    struct_8x8, freq = args
    
    # 1. 动态更新频率参数
    # 注意：我们直接操作字典，计算出当前频率下的所有物理参数
    params = PHYSICS_PARAMS.copy()
    params['f'] = freq
    params['c'] = np.sqrt(params['kappaa'] / params['rhoa'])
    params['omega'] = 2 * np.pi * params['f']
    params['k'] = params['omega'] / params['c']
    params['Ly_design'] = params['Lx_design']
    
    # 2. 更新全局 Solver 的参数 (避免重新创建对象)
    global _SOLVER_INSTANCE
    _SOLVER_INSTANCE.update_params(params)
    
    # 3. 构造全场 (这一步是 CPU 密集型，仍在 Worker 中做)
    res_in = GRID_CONFIG['input_resolution']
    res_out = GRID_CONFIG['resolution']
    scale = int(res_out / res_in)
    
    struct_fem = np.kron(struct_8x8, np.ones((scale, scale)))
    
    Lx_half = params['Lx_half']
    resolution = GRID_CONFIG['resolution']
    
    full_width_blocks = 2 * Lx_half + 1
    full_width_pixels = full_width_blocks * resolution
    full_height_pixels = resolution
    
    x_total = np.zeros((full_height_pixels, full_width_pixels))
    
    start_col = Lx_half * resolution
    end_col = start_col + resolution
    
    if struct_fem.shape != (resolution, resolution):
         scale_fix = resolution / struct_fem.shape[0]
         struct_fem = scipy_zoom(struct_fem, scale_fix, order=0)
         
    x_total[:, start_col:end_col] = struct_fem
    
    # 4. 求解 (复用 Solver)
    try:
        # init_fem 的结果已被 Solver 内部缓存，这里会非常快
        P_grid = _SOLVER_INSTANCE.solve(x_total)
        T, R = _SOLVER_INSTANCE.calculate_TR(P_grid, resolution)
        return T, R
        
    except Exception as e:
        # print 可能会在多进程中混乱，实际生产建议用 logging
        return np.nan, np.nan

def main():
    parser = argparse.ArgumentParser(description="Acoustic FEM Batch Solver (Optimized)")
    parser.add_argument('--input', type=str, required=True, help='Path to .mat file')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel processes')
    args = parser.parse_args()

    input_filename = os.path.basename(args.input)
    file_basename = os.path.splitext(input_filename)[0]
    
    target_frequencies = np.linspace(2000, 2900, 10) 
    
    print(f"Loading structures from {args.input}...")
    structures = load_structures_from_mat(args.input)
    print(f"Loaded {structures.shape[0]} structures.")
    
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 准备初始参数模板 (用于初始化 Worker)
    initial_params = PHYSICS_PARAMS.copy()
    initial_params['f'] = 2000 # 占位符
    initial_params['c'] = np.sqrt(initial_params['kappaa'] / initial_params['rhoa'])
    initial_params['omega'] = 2 * np.pi * initial_params['f']
    initial_params['k'] = initial_params['omega'] / initial_params['c']
    initial_params['Ly_design'] = initial_params['Lx_design']

    # --- 优化点：ProcessPoolExecutor 在循环外初始化 ---
    print(f"Starting Process Pool with {args.workers} workers...")
    
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(initial_params,)) as executor:
        
        for freq in target_frequencies:
            start_time = time.time()
            print(f"\n--- Starting Batch Job for Frequency: {freq:.1f} Hz ---")
            
            # 准备任务
            tasks = [(s, freq) for s in structures]
            
            # 并行计算
            results = []
            from tqdm import tqdm
            # executor.map 会保持提交顺序
            results = list(tqdm(executor.map(process_single_structure, tasks), total=len(tasks), desc=f"Freq {int(freq)}"))
            
            # 整理和保存
            T_vals = np.array([r[0] for r in results])
            R_vals = np.array([r[1] for r in results])
            
            t_filename = f'T_values_{file_basename}_{int(freq)}.npy'
            r_filename = f'R_values_{file_basename}_{int(freq)}.npy'
            
            np.save(os.path.join(output_dir, t_filename), T_vals)
            np.save(os.path.join(output_dir, r_filename), R_vals)
            
            elapsed = time.time() - start_time
            print(f"✅ Saved results to {output_dir}/{t_filename} (Time: {elapsed:.2f}s)")

    print("\nAll frequencies computed successfully!")

if __name__ == '__main__':
    main()