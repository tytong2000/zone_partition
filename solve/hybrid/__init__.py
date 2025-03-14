# solvers/hybrid/__init__.py

"""
Hybrid solvers for VRP, combining multiple baseline algorithms:
- Clarke-Wright Savings + Tabu Search (CWTSSolver)
- Clarke-Wright Savings + Variable Neighborhood Search (CWVNSSolver)
- Simulated Annealing + Tabu Search (SATSSolver)
- Simulated Annealing + Variable Neighborhood Search (SAVNSolver)
"""

from .cw_ts_solver import CWTSSolver
from .cw_vns_solver import CWVNSSolver
from .sa_ts_solver import SATSSolver
from .sa_vns_solver import SAVNSSolver

__all__ = [
    "CWTSSolver",
    "CWVNSSolver",
    "SATSSolver",
    "SAVNSSolver"
]
 
