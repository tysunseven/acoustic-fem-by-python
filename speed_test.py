import time
import numpy as np
from acoustic_fem import AcousticSimulator

def run_speed_test(num_samples=100, freq=2500):
    """
    生成随机结构并测试求解速度
    :param num_samples: 测试样本数量
    :param freq: 测试频率 (Hz)
    """
    print(f"--- 准备开始速度测试 ---")
    print(f"生成 {num_samples} 个随机 8x8 二值结构...")
    
    # 1. 生成随机数据 (0或1)
    # size: (N, 8, 8)
    structures = np.random.randint(0, 2, size=(num_samples, 8, 8))
    
    # 2. 初始化仿真器
    print("正在初始化 AcousticSimulator (首次加载可能需要构建缓存)...")
    sim = AcousticSimulator()
    
    # 3. 预热 (Warm-up)
    # 第一次运行通常会进行 JIT 编译或缓存初始化，不计入稳定运行时间
    print("正在进行预热 (Warm-up)...")
    sim.predict(structures[0], freq)
    
    # 4. 正式测试
    print(f"\n开始计算 (频率: {freq} Hz)...")
    start_time = time.time()
    
    for i, struct in enumerate(structures):
        sim.predict(struct, freq)
        # 可选：打印进度
        if (i + 1) % 10 == 0:
            print(f"已完成: {i + 1}/{num_samples}", end='\r')
            
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_samples) * 1000
    fps = num_samples / total_time
    
    # 5. 输出结果
    print(f"\n\n--- 测试结果 ---")
    print(f"样本总数: {num_samples}")
    print(f"总耗时:   {total_time:.4f} 秒")
    print(f"平均耗时: {avg_time_ms:.2f} ms/个")
    print(f"吞吐量:   {fps:.2f} 个/秒")

if __name__ == "__main__":
    # 您可以根据需要调整测试样本数
    run_speed_test(num_samples=100)