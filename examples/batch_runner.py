import argparse
import numpy as np
import os
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# 引入新封装的包
from acoustic_fem import AcousticSimulator
# 注意：utils 里的函数可能需要显式引入，或者你也把它加到 __init__.py 里
from acoustic_fem.utils import load_structures_from_mat

# --- 全局变量 (仅在 Worker 进程中有效) ---
_SIMULATOR_INSTANCE = None

def init_worker():
    """
    Worker 进程初始化函数。
    """
    global _SIMULATOR_INSTANCE
    # 初始化仿真器 (内部会自动建立 Mesh 缓存)
    # 可以在这里传入 grid_config={'resolution': 128} 等自定义配置
    _SIMULATOR_INSTANCE = AcousticSimulator()

def process_single_structure(args):
    """
    处理单个结构 (现在的逻辑非常简单，只是调用接口)
    """
    struct_8x8, freq = args
    
    global _SIMULATOR_INSTANCE
    
    try:
        # 直接调用标准接口！
        # 不需要关心网格怎么放大、波导怎么加、参数怎么算
        T, R = _SIMULATOR_INSTANCE.predict(struct_8x8, freq)
        return T, R
    except Exception:
        return np.nan, np.nan

def main():
    parser = argparse.ArgumentParser(description="Acoustic FEM Batch Solver (Refactored)")
    parser.add_argument('--input', type=str, required=True, help='Path to .mat file')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel processes')
    args = parser.parse_args()

    input_filename = os.path.basename(args.input)
    file_basename = os.path.splitext(input_filename)[0]
    
    # 示例频率范围
    target_frequencies = np.linspace(2000, 2900, 10) 
    
    print(f"Loading structures from {args.input}...")
    structures = load_structures_from_mat(args.input)
    print(f"Loaded {structures.shape[0]} structures.")
    
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Starting Process Pool with {args.workers} workers...")
    
    # 这里的 initializer 不需要传参了，因为 Simulator 内部有默认配置
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as executor:
        
        for freq in target_frequencies:
            start_time = time.time()
            print(f"\n--- Starting Batch Job for Frequency: {freq:.1f} Hz ---")
            
            tasks = [(s, freq) for s in structures]
            
            # 提交任务
            results = list(tqdm(executor.map(process_single_structure, tasks), total=len(tasks), desc=f"Freq {int(freq)}"))
            
            # 整理结果
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