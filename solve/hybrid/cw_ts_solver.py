# solvers/hybrid/cw_ts_solver.py

import time
import logging

from ..base.base_solver import BaseSolver
from ..base.vrp_solution import VRPSolution
from ..baseline.cw_solver import CWSolver
from ..baseline.ts_solver import TSSolver

class CWTSSolver(BaseSolver):
    """
    混合算法：先用 Clarke-Wright Savings (CW) 得到初始解，
            再用 TabuSearch (TS) 对该解进行改进。
    """
    def __init__(self, instance, cw_max_iter=100, ts_tabu_size=10, ts_max_iter=50):
        """
        :param instance: VRPInstance对象
        :param cw_max_iter: Clarke-Wright中的最大迭代次数
        :param ts_tabu_size: 禁忌搜索的Tabu列表大小
        :param ts_max_iter: 禁忌搜索最大迭代次数
        """
        super().__init__(instance)
        self.cw_max_iter = cw_max_iter
        self.ts_tabu_size = ts_tabu_size
        self.ts_max_iter = ts_max_iter
        self.logger = logging.getLogger(self.__class__.__name__)

        self.final_solution = None  # 最终解

    def solve(self) -> VRPSolution:
        """
        先执行Clarke-Wright得到初始解，如果可行则再用TabuSearch改进。
        最后返回改进后的解以及改进率等信息。
        """
        # 第一阶段：CW
        cw_solver = CWSolver(self.instance, max_iter=self.cw_max_iter)
        t0 = time.time()
        initial_solution = cw_solver.solve()
        t1 = time.time()
        
        if (not initial_solution) or (not initial_solution.is_feasible()):
            self.logger.warning("CW第一阶段没有可行解，直接返回空解或不可行解。")
            self.final_solution = initial_solution
            return initial_solution
        
        # 第二阶段：TS
        ts_solver = TSSolver(
            self.instance,
            tabu_size=self.ts_tabu_size,
            max_iter=self.ts_max_iter
        )
        # 在母代码中，通常会把CW的解传给TS作为初始解，这里可以在TSSolver里实现对应接口
        # 或者临时替换ts_solver的初始解逻辑
        # 下面这种方式表示让TS从“current_solution=initial_solution”开始：
        ts_solver.current_solution = initial_solution  # （示例：假定TSSolver内部能识别这个字段）
        
        t2 = time.time()
        improved_solution = ts_solver.solve()
        t3 = time.time()

        # 计算改进率
        improvement = 0.0
        if improved_solution and improved_solution.is_feasible():
            init_obj = initial_solution.objective_value
            final_obj = improved_solution.objective_value
            improvement = (init_obj - final_obj) / (init_obj + 1e-9)
            self.final_solution = improved_solution
        else:
            # 如果TS阶段失败，则保留CW的解
            self.final_solution = initial_solution

        self.logger.info(
            f"[CW+TS] 初始解目标值={initial_solution.objective_value:.2f}, "
            f"最终解目标值={self.final_solution.objective_value:.2f}, "
            f"改进率={improvement*100:.2f}%"
        )

        # 也可把时间、改进率等信息存入 final_solution，以便后续使用
        return self.final_solution

    def get_solution(self) -> VRPSolution:
        if self.final_solution is None:
            return self.solve()
        return self.final_solution
 
