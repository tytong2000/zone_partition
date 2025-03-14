# solve/base/__init__.py
"""基础模块初始化文件"""

# 按依赖关系顺序导入
from .vrp_instance import VRPInstance
from .vrp_solution import VRPSolution
from .base_solver import BaseSolver

try:
    from .gurobi_solver import GurobiBaseSolver
except ImportError:
    GurobiBaseSolver = None

__all__ = ['VRPInstance', 'VRPSolution', 'BaseSolver', 'GurobiBaseSolver']