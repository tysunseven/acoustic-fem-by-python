import numpy as np
import h5py
import os

def check_accuracy():
    print("--- 开始精度对比 ---")
    
    # 1. 加载 Python 计算结果
    try:
        py_T = np.load('results/T_values.npy')
        py_R = np.load('results/R_values.npy')
    except FileNotFoundError:
        print("❌ 找不到 Python 结果，请先运行 batch_runner.py")
        return

    # 2. 加载 MATLAB 真值
    mat_path = 'data/verification_3000Hz.mat'
    if not os.path.exists(mat_path):
        print("❌ 找不到验证数据 file")
        return

    with h5py.File(mat_path, 'r') as f:
        # h5py 读取的数组通常是 (N, 1) 或 (1, N)，需要 flatten
        ref_T_re = np.array(f['Ref_T_Re']).flatten()
        ref_T_im = np.array(f['Ref_T_Im']).flatten()
        ref_R_re = np.array(f['Ref_R_Re']).flatten()
        ref_R_im = np.array(f['Ref_R_Im']).flatten()

    # 组合复数
    ref_T = ref_T_re + 1j * ref_T_im
    ref_R = ref_R_re + 1j * ref_R_im

    # 3. 计算误差
    # 确保长度一致
    n = min(len(py_T), len(ref_T))
    if n == 0: print("数据为空"); return

    print(f"对比样本数: {n}")
    print(f"{'ID':<5} | {'Matlab T (Real)':<15} | {'Python T (Real)':<15} | {'Diff (Abs)':<15}")
    print("-" * 60)

    max_err_T = 0
    max_err_R = 0

    for i in range(n):
        diff_T = abs(py_T[i] - ref_T[i])
        diff_R = abs(py_R[i] - ref_R[i])
        
        max_err_T = max(max_err_T, diff_T)
        max_err_R = max(max_err_R, diff_R)

        if i < 5: # 只打印前5个详情
            print(f"{i:<5} | {ref_T[i].real:<15.6f} | {py_T[i].real:<15.6f} | {diff_T:<15.2e}")

    print("-" * 60)
    print(f"最大透射误差 (Max Error T): {max_err_T:.4e}")
    print(f"最大反射误差 (Max Error R): {max_err_R:.4e}")

    if max_err_T < 1e-9 and max_err_R < 1e-9:
        print("\n✅ 完美匹配！Python 求解器完全可靠。")
    elif max_err_T < 1e-5:
        print("\n⚠️ 误差极小 (可接受)。通常由 float32/64 精度差异或插值引起。")
    else:
        print("\n❌ 误差较大，请检查参数设置 (如 rho/kappa) 是否与 MATLAB 完全一致。")

if __name__ == "__main__":
    check_accuracy()