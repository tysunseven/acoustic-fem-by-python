import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import math

class AcousticFEMSolver:
    # 类级缓存，用于存储与网格分辨率相关的静态数据
    # Key: (nelx, nely, h)
    # Value: (Ke, Me, iIdx, jIdx, Mle, Mre, iIdxL, jIdxL, iIdxR, jIdxR)
    _MESH_CACHE = {}

    def __init__(self, params):
        self.update_params(params)
        self.q = 1

    def update_params(self, params):
        """
        更新物理参数（用于变频计算），避免重新初始化整个对象
        """
        self.rho_bg = params['rhoa']
        self.kappa_bg = params['kappaa']
        self.rho_incl = params['rhoav']
        self.kappa_incl = params['kappaav']
        self.omega = params['omega']
        self.k = params['k']
        self.p_in = params['p_in']
        
        # 几何参数
        self.Lx_design = params['Lx_design']
        self.Ly_design = params['Ly_design']
        self.Lx_half = params['Lx_half']

    def init_fem(self, nelx, nely, h):
        """
        初始化 FEM 网格。增加了缓存机制。
        """
        cache_key = (nelx, nely, h)
        if cache_key in AcousticFEMSolver._MESH_CACHE:
            return AcousticFEMSolver._MESH_CACHE[cache_key]

        # --- 以下为原有的网格生成逻辑 (保持不变) ---
        nodenum = np.arange( (nelx+1)*(nely+1) ).reshape((nelx+1, nely+1)).T 
        nodenrs = nodenum[:-1, :-1]
        edofVec = nodenrs.flatten(order='F')
        
        stride = nely + 1
        edofMat = np.zeros((nelx*nely, 4), dtype=int)
        edofMat[:, 0] = edofVec
        edofMat[:, 1] = edofVec + 1
        edofMat[:, 2] = edofVec + stride + 1
        edofMat[:, 3] = edofVec + stride
        
        Ke = np.array([
            [4, -1, -2, -1],
            [-1, 4, -1, -2],
            [-2, -1, 4, -1],
            [-1, -2, -1, 4]
        ]) / 6.0
        
        Me = (h**2) * np.array([
            [4, 2, 1, 2],
            [2, 4, 2, 1],
            [1, 2, 4, 2],
            [2, 1, 2, 4]
        ]) / 36.0
        
        iIndex = np.repeat(edofMat, 4, axis=1).flatten()
        jIndex = np.tile(edofMat, 4).flatten()
        
        Mle = h * np.array([
            [2, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 2]
        ]) / 6.0
        
        Mre = h * np.array([
            [0, 0, 0, 0], [0, 2, 1, 0], [0, 1, 2, 0], [0, 0, 0, 0]
        ]) / 6.0
        
        edofMat_L = edofMat[:nely, :]
        edofMat_R = edofMat[-nely:, :] 
        
        iIndexL = np.repeat(edofMat_L, 4, axis=1).flatten()
        jIndexL = np.tile(edofMat_L, 4).flatten()
        
        iIndexR = np.repeat(edofMat_R, 4, axis=1).flatten()
        jIndexR = np.tile(edofMat_R, 4).flatten()
        
        # 存入缓存
        result = (Ke, Me, iIndex, jIndex, Mle, Mre, iIndexL, jIndexL, iIndexR, jIndexR)
        AcousticFEMSolver._MESH_CACHE[cache_key] = result
        return result

    def solve(self, x_structure):
        nely, nelx = x_structure.shape
        h = self.Ly_design / nely
        
        # 1. 获取网格数据 (优先从缓存)
        Ke_base, Me_base, iIdx, jIdx, Mle_base, Mre_base, iIdxL, jIdxL, iIdxR, jIdxR = \
            self.init_fem(nelx, nely, h)
        
        # 材料属性分布计算 (保持原逻辑)
        inv_rho_bg = 1.0 / self.rho_bg
        inv_rho_incl = 1.0 / self.rho_incl
        inv_kappa_bg = 1.0 / self.kappa_bg
        inv_kappa_incl = 1.0 / self.kappa_incl
        
        rhoiv = inv_rho_bg + (inv_rho_incl - inv_rho_bg) * (x_structure**self.q)
        kappaiv = inv_kappa_bg + (inv_kappa_incl - inv_kappa_bg) * (x_structure**self.q)
        
        rhoiv_flat = rhoiv.flatten(order='F')
        kappaiv_flat = kappaiv.flatten(order='F')
        
        # 组装全局矩阵
        # 优化点：虽然这里仍用了 outer，但由于 iIdx 等已缓存，整体速度会有提升
        sKa = np.outer(rhoiv_flat, Ke_base.flatten()).flatten()
        sMa = np.outer(kappaiv_flat, Me_base.flatten()).flatten()
        
        ndof = (nelx + 1) * (nely + 1)
        
        # Sparse Matrix Assembly
        Ka = sparse.coo_matrix((sKa, (iIdx, jIdx)), shape=(ndof, ndof))
        Ka = (Ka + Ka.T) / 2.0
        
        Ma = sparse.coo_matrix((sMa, (iIdx, jIdx)), shape=(ndof, ndof))
        Ma = (Ma + Ma.T) / 2.0
        
        # Boundary Assembly
        rhoiv_left = rhoiv[:, 0]
        rhoiv_right = rhoiv[:, -1]
        
        sMl = np.outer(rhoiv_left, Mle_base.flatten()).flatten()
        sMr = np.outer(rhoiv_right, Mre_base.flatten()).flatten()
        
        Ml = sparse.coo_matrix((sMl, (iIdxL, jIdxL)), shape=(ndof, ndof))
        Ml = (Ml + Ml.T) / 2.0
        
        Mr = sparse.coo_matrix((sMr, (iIdxR, jIdxR)), shape=(ndof, ndof))
        Mr = (Mr + Mr.T) / 2.0
        
        # Load Vector F
        Fa_vals = np.zeros(nely + 1)
        Fa_vals[:-1] += rhoiv_left / 2.0
        Fa_vals[1:] += rhoiv_left / 2.0
        
        Fa = np.zeros(ndof, dtype=complex)
        Fa[:nely+1] = Fa_vals
        
        # System Matrix K_sys
        # 这里的矩阵加法是不可避免的，但由于前面的优化，整体吞吐量增加
        K_sys = Ka - (self.omega**2)*Ma + 1j * self.k * (Ml + Mr)
        
        F_sys = h * 2 * 1j * self.k * self.p_in * Fa
        
        # Solve
        K_sys = K_sys.tocsc()
        P_vec = spsolve(K_sys, F_sys)
        
        P_grid = P_vec.reshape((nelx+1, nely+1)).T
        return P_grid

    def calculate_TR(self, P_grid, nelx_design):
        # 对应 src/+core/calculate_ABCD_midrow.m
        # P_grid shape: (nely_total+1, nelx_total+1)
        
        nely_total, nelx_total = P_grid.shape
        nely_total -= 1
        nelx_total -= 1
        
        mid_row = round((nely_total + 1) / 2) - 1 # Python 0-based index
        
        # 定义采样点 (Block indices)
        x1_block = 1
        x2_block = self.Lx_half - 1
        x3_block = self.Lx_half + 2
        x4_block = 2 * self.Lx_half
        
        # 对应的物理 X 坐标
        X_p1 = x1_block * self.Lx_design
        X_p2 = x2_block * self.Lx_design
        X_p3 = x3_block * self.Lx_design
        X_p4 = x4_block * self.Lx_design
        
        # 对应的节点列索引 (Python 0-based)
        # MATLAB: idx = block * nelx + 1
        # Python: idx = block * nelx
        idx_p1 = x1_block * nelx_design
        idx_p2 = x2_block * nelx_design
        idx_p3 = x3_block * nelx_design
        idx_p4 = x4_block * nelx_design
        
        p1 = P_grid[mid_row, idx_p1]
        p2 = P_grid[mid_row, idx_p2]
        p3 = P_grid[mid_row, idx_p3]
        p4 = P_grid[mid_row, idx_p4]
        
        k = self.k
        
        # Calculate ABCD
        den_AB = 2 * np.sin(k * (X_p1 - X_p2))
        A = (1j * (p1 * np.exp(1j * k * X_p2) - p2 * np.exp(1j * k * X_p1))) / den_AB
        B = (1j * (p2 * np.exp(-1j * k * X_p1) - p1 * np.exp(-1j * k * X_p2))) / den_AB
        
        den_CD = 2 * np.sin(k * (X_p3 - X_p4))
        C = (1j * (p3 * np.exp(1j * k * X_p4) - p4 * np.exp(1j * k * X_p3))) / den_CD
        
        # Calculate T and R
        if abs(A) < 1e-12:
            return np.nan, np.nan
        
        T = C / A
        R = B / A
        
        return T, R