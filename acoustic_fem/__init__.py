import os
if os.environ.get("ACOUSTIC_FEM_FORCE_SINGLE_THREAD", "True") == "True":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

# acoustic_fem/__init__.py
from .simulation import AcousticSimulator

# 也可以直接暴露常用的配置，方便外部修改
from .defaults import PHYSICS_PARAMS, GRID_CONFIG

from .utils import load_structures_from_mat, save_results_to_npy
# 更新 __all__
__all__ = ['AcousticSimulator', 'PHYSICS_PARAMS', 'GRID_CONFIG', 'load_structures_from_mat', 'save_results_to_npy']