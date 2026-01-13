import h5py
import numpy as np
import scipy.io as sio

def load_structures_from_mat(file_path, variable_name='Structure_Data'):
    """
    智能读取 .mat 文件 (自动识别 v7.3 或旧版本)
    """
    try:
        # 尝试使用 h5py 读取 (针对 v7.3 大文件)
        with h5py.File(file_path, 'r') as f:
            # 注意: h5py 读取的矩阵通常需要转置 (MATLAB 列优先 vs Python 行优先)
            data = np.array(f[variable_name])
            # data shape usually becomes (N, 8, 8) after transpose handling if needed
            # h5py 读取时，维度通常是 (N, x, y)，且 x,y 可能转置，需根据实际情况调整
            return np.transpose(data, (0, 2, 1)) 
    except OSError:
        # 如果不是 v7.3，回退到 scipy 读取
        mat_data = sio.loadmat(file_path)
        return mat_data[variable_name]

def save_results_to_npy(output_dir, T_array, R_array):
    """保存结果为 .npy"""
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, 'T_values.npy'), T_array)
    np.save(os.path.join(output_dir, 'R_values.npy'), R_array)
    print(f"Results saved to {output_dir}")