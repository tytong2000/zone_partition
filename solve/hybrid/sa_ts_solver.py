# solvers/hybrid/sa_ts_solver.py

import time
import logging

from ..base.base_solver import BaseSolver
from ..base.vrp_solution import VRPSolution
from ..baseline.sa_solver import SASolver
from ..baseline.ts_solver import TSSolver

class SATSSolver(BaseSolver):
    """
    混合算法：先用 SimulatedAnnealing (SA) 得到初始解，
            再用 TabuSearch (TS) 对该解进行改进。
    """
    def __init__(self, instance, sa_initial_temp=1000.0, sa_cooling_rate=0.95, sa_max_iter=100,
                 ts_tabu_size=10, ts_max_iter=50):
        """
        :param instance: VRPInstance对象
        :param sa_initial_temp: SA的初始温度
        :param sa_cooling_rate: SA的降温系数
        :param sa_max_iter: SA的最大迭代次数
        :param ts_tabu_size: TS的Tabu列表大小
        :param ts_max_iter: TS的最大迭代次数
        """
        super().__init__(instance)
        self.sa_initial_temp = sa_initial_temp
        self.sa_cooling_rate = sa_cooling_rate
        self.sa_max_iter = sa_max_iter
        self.ts_tabu_size = ts_tabu_size
        self.ts_max_iter = ts_max_iter
        self.logger = logging.getLogger(self.__class__.__name__)

        self.final_solution = None

    def solve(self) -> VRPSolution:
        """
        先执行SA得到初始解，如果可行则再用TS改进。
        返回最终解。
        """
        # 第一阶段：SA
        sa_solver = SASolver(
            self.instance,
            initial_temp=self.sa_initial_temp,
            cooling_rate=self.sa_cooling_rate,
            max_iter=self.sa_max_iter
        )
        t0 = time.time()
        initial_solution = sa_solver.solve()
        t1 = time.time()
        
        if (not initial_solution) or (not initial_solution.is_feasible()):
            self.logger.warning("SA第一阶段无可行解，直接返回。")
            self.final_solution = initial_solution
            return initial_solution
        
        # 第二阶段：TS
        ts_solver = TSSolver(
            self.instance,
            tabu_size=self.ts_tabu_size,
            max_iter=self.ts_max_iter
        )
        # 在TabuSearch里，可以把SA的解当作初始解
        ts_solver.current_solution = initial_solution
        
        t2 = time.time()
        improved_solution = ts_solver.solve()
        t3 = time.time()

        improvement = 0.0
        if improved_solution and improved_solution.is_feasible():
            init_obj = initial_solution.objective_value
            final_obj = improved_solution.objective_value
            improvement = (init_obj - final_obj) / (init_obj + 1e-9)
            self.final_solution = improved_solution
        else:
            self.final_solution = initial_solution

        self.logger.info(
            f"[SA+TS] 初始解目标值={initial_solution.objective_value:.2f}, "
            f"最终解目标值={self.final_solution.objective_value:.2f}, "
            f"改进率={improvement*100:.2f}%"
        )

        return self.final_solution

    def get_solution(self) -> VRPSolution:
        if self.final_solution is None:
            return self.solve()
        return self.final_solution
 
