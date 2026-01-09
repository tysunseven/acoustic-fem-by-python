import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import math

class AcousticFEMSolver:
    def __init__(self, params):
        # 物理参数解包
        self.rho_bg = params['rhoa']
        self.kappa_bg = params['kappaa']
        self.rho_incl = params['rhoav']
        self.kappa_incl = params['kappaav']
        self.omega = params['omega']
        self.k = params['k']
        self.p_in = params['p_in']
        
        # 几何参数
        self.Lx_design = params['Lx_design']
        self.Ly_design = params['Ly_design'] # 通常等于 Lx_design
        self.Lx_half = params['Lx_half']
        
        # 预计算常数
        self.q = 1
        
    def init_fem(self, nelx, nely, h):
        # 对应 src/+fem/init_fem.m
        # 注意：Python 是 0-based索引，且 reshape 默认是 C-order (行优先)，
        # 而 MATLAB 是 Fortran-order (列优先)。这里必须小心处理。
        
        # 节点编号 (Node numbering)
        # MATLAB: 1:(nelx+1)*(nely+1) reshaped to (nely+1, nelx+1)
        # Python range: 0 to ...
        nodenum = np.arange( (nelx+1)*(nely+1) ).reshape((nelx+1, nely+1)).T 
        # T (transpose) is needed to match MATLAB's column-major ordering in a row-major array
        
        # 单元自由度映射 (Element DOF map)
        nodenrs = nodenum[:-1, :-1]
        edofVec = nodenrs.flatten(order='F') # Column-major flatten
        
        # edofMat 构建 (每个单元4个节点)
        # MATLAB indices shifted by: [0, nely+1, nely, 1] (but logic is slightly diff due to 0-base)
        # Let's map directly:
        # node 1: (i, j)   -> edofVec
        # node 2: (i+1, j) -> edofVec + 1
        # node 3: (i+1, j+1) -> edofVec + (nely + 1) + 1
        # node 4: (i, j+1) -> edofVec + (nely + 1)
        
        stride = nely + 1
        edofMat = np.zeros((nelx*nely, 4), dtype=int)
        edofMat[:, 0] = edofVec
        edofMat[:, 1] = edofVec + 1
        edofMat[:, 2] = edofVec + stride + 1
        edofMat[:, 3] = edofVec + stride
        
        # 单元矩阵 (Element Matrices)
        # Ke (Stiffness)
        Ke = np.array([
            [4, -1, -2, -1],
            [-1, 4, -1, -2],
            [-2, -1, 4, -1],
            [-1, -2, -1, 4]
        ]) / 6.0
        
        # Me (Mass)
        Me = (h**2) * np.array([
            [4, 2, 1, 2],
            [2, 4, 2, 1],
            [1, 2, 4, 2],
            [2, 1, 2, 4]
        ]) / 36.0
        
        # Construct indices for sparse assembly (COO format)
        # iIndex = kron(edofMat, ones(4,1))
        # jIndex = kron(edofMat, ones(1,4))
        iIndex = np.repeat(edofMat, 4, axis=1).flatten()
        jIndex = np.tile(edofMat, 4).flatten()
        
        # Boundary mass matrices (Left and Right)
        # Mle, Mre from MATLAB code
        Mle = h * np.array([
            [2, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 2]
        ]) / 6.0
        
        Mre = h * np.array([
            [0, 0, 0, 0], [0, 2, 1, 0], [0, 1, 2, 0], [0, 0, 0, 0]
        ]) / 6.0
        
        # Boundary indices
        # Left boundary elements: 1 to nely (in MATLAB) -> 0 to nely-1
        # Right boundary elements: (nelx-1)*nely + 1 to end
        edofMat_L = edofMat[:nely, :]
        edofMat_R = edofMat[-nely:, :] # Last nely elements
        
        iIndexL = np.repeat(edofMat_L, 4, axis=1).flatten()
        jIndexL = np.tile(edofMat_L, 4).flatten()
        
        iIndexR = np.repeat(edofMat_R, 4, axis=1).flatten()
        jIndexR = np.tile(edofMat_R, 4).flatten()
        
        return Ke, Me, iIndex, jIndex, Mle, Mre, iIndexL, jIndexL, iIndexR, jIndexR

    def solve(self, x_structure):
        # x_structure: 2D numpy array (8x8 or 128x128), 0 for air, 1 for solid
        # 对应 src/+core/AcousticTwoDimLeftIncidentRightAbsorb.m
        
        nely, nelx = x_structure.shape
        h = self.Ly_design / nely # 假设均匀网格
        
        # 材料属性分布
        # MATLAB: rhoiv = 1/rhoa + (1/rhoav-1/rhoa)*x.^q;
        inv_rho_bg = 1.0 / self.rho_bg
        inv_rho_incl = 1.0 / self.rho_incl
        inv_kappa_bg = 1.0 / self.kappa_bg
        inv_kappa_incl = 1.0 / self.kappa_incl
        
        rhoiv = inv_rho_bg + (inv_rho_incl - inv_rho_bg) * (x_structure**self.q)
        kappaiv = inv_kappa_bg + (inv_kappa_incl - inv_kappa_bg) * (x_structure**self.q)
        
        # 初始化 FEM 索引和单元矩阵
        Ke_base, Me_base, iIdx, jIdx, Mle_base, Mre_base, iIdxL, jIdxL, iIdxR, jIdxR = \
            self.init_fem(nelx, nely, h)
        
        # 组装全局矩阵
        # Flatten parameters column-wise to match element ordering
        rhoiv_flat = rhoiv.flatten(order='F')
        kappaiv_flat = kappaiv.flatten(order='F')
        
        # sKa = reshape(Ke(:)*(rhoiv(:))', ...)
        # In Python: we repeat rhoiv for each of the 16 entries in element matrix
        sKa = np.outer(rhoiv_flat, Ke_base.flatten()).flatten()
        sMa = np.outer(kappaiv_flat, Me_base.flatten()).flatten()
        
        # Total DOFs
        ndof = (nelx + 1) * (nely + 1)
        
        # Assembly
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
        # Fa(1:nely+1) = [rhoiv(:,1); 0]/2 + [0; rhoiv(:,1)]/2;
        # Python implementation of trapezoidal rule approx for load
        Fa_vals = np.zeros(nely + 1)
        Fa_vals[:-1] += rhoiv_left / 2.0
        Fa_vals[1:] += rhoiv_left / 2.0
        
        Fa = np.zeros(ndof, dtype=complex)
        Fa[:nely+1] = Fa_vals
        
        # System Matrix K_sys = Ka - w^2*Ma + i*k*(Ml + Mr)
        K_sys = Ka - (self.omega**2)*Ma + 1j * self.k * (Ml + Mr)
        
        # Force Vector F_sys
        F_sys = h * 2 * 1j * self.k * self.p_in * Fa
        
        # Solve
        # Use CSC format for efficient arithmetic and solving
        K_sys = K_sys.tocsc()
        P_vec = spsolve(K_sys, F_sys)
        
        # Reshape result back to grid (nely+1, nelx+1)
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