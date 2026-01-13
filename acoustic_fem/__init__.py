# acoustic_fem/__init__.py
from .simulation import AcousticSimulator

# 也可以直接暴露常用的配置，方便外部修改
from .defaults import PHYSICS_PARAMS, GRID_CONFIG

__all__ = ['AcousticSimulator', 'PHYSICS_PARAMS', 'GRID_CONFIG']