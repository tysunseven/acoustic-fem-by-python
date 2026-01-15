import os

# ==============================================================================
# 【关键设置】强制底层数值库使用单线程
# 必须在 import numpy/scipy 之前设置！
# 否则 8 个 worker 会产生 8 x (CPU核数) 个线程，导致严重的上下文切换和性能下降。
# ==============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from acoustic_fem import AcousticSimulator

# --- 全局变量 (用于子进程) ---
_worker_sim = None

def init_worker(grid_conf, phys_conf):
    """
    子进程初始化函数。
    每个 Worker 启动时只运行一次，用于加载 FEM 求解器并构建网格缓存。
    """
    global _worker_sim
    # 在这里实例化，保证每个进程有自己独立的求解器对象
    _worker_sim = AcousticSimulator(grid_config=grid_conf, physics_config=phys_conf)

def solve_task(args):
    """
    子进程执行任务的函数
    """
    struct, freq = args
    global _worker_sim
    # 调用全局的仿真器实例
    return _worker_sim.predict(struct, freq)

def run_parallel_speed_test(num_samples=100, freq=2500, workers=8):
    print(f"--- 并行速度测试 ---")
    print(f"Worker数量: {workers}")
    print(f"底层线程数: 1 (强制单线程模式)")
    
    # --- 1. 配置参数 (根据您的偏好设置) ---
    # 您之前坚持的高精度配置
    GRID_CONFIG = {'resolution': 128} 
    PHYS_CONFIG = {'Lx_half': 10}
    
    print(f"仿真配置: Res={GRID_CONFIG['resolution']}, Lx={PHYS_CONFIG['Lx_half']}")
    print(f"生成 {num_samples} 个随机 8x8 结构...")
    
    # --- 2. 准备数据 ---
    np.random.seed(42) # 固定种子以便复现
    structures = np.random.randint(0, 2, size=(num_samples, 8, 8))
    
    # 打包任务参数 [(struct1, freq), (struct2, freq), ...]
    tasks = [(s, freq) for s in structures]
    
    # --- 3. 正式测试 ---
    print(f"\n开始计算 (频率: {freq} Hz)...")
    start_time = time.time()
    
    # 使用进程池
    # initializer=init_worker 确保每个进程先初始化 FEM 环境
    with ProcessPoolExecutor(max_workers=workers, 
                             initializer=init_worker, 
                             initargs=(GRID_CONFIG, PHYS_CONFIG)) as executor:
        
        # executor.map 会自动分配任务并收集结果
        # 转换为 list 以确保所有任务都执行完毕才停止计时
        results = list(executor.map(solve_task, tasks))
            
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_samples) * 1000
    fps = num_samples / total_time
    
    # --- 4. 输出结果 ---
    print(f"\n--- 测试结果 (Parallel) ---")
    print(f"样本总数: {num_samples}")
    print(f"Worker数: {workers}")
    print(f"总耗时:   {total_time:.4f} 秒")
    print(f"平均耗时: {avg_time_ms:.2f} ms/个")
    print(f"吞吐量:   {fps:.2f} 个/秒")

if __name__ == "__main__":
    # 在 Windows 下必须放在 if __name__ == "__main__": 块中
    run_parallel_speed_test(num_samples=100, workers=8)