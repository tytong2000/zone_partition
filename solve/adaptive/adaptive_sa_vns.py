# solvers/adaptive/adaptive_sa_vns.py

import time
import logging

from ..base.base_solver import BaseSolver
from ..base.vrp_solution import VRPSolution
from ..baseline.sa_solver import SASolver
from ..adaptive.adaptive_vns import AdaptiveVNSSolver

class AdaptiveSAVNSSolver(BaseSolver):
    """
    自适应混合算法：
      第一阶段：SA求初始解
      第二阶段：Adaptive VNS对初始解做自适应变邻域搜索
    """
    def __init__(self, instance,
                 sa_initial_temp=1000.0,
                 sa_cooling_rate=0.95,
                 sa_max_iter=100,
                 vns_max_iter=30,
                 vns_max_neighborhoods=3,
                 vns_shake_intensity=0.3,
                 vns_min_intensity=0.1,
                 vns_max_intensity=0.5,
                 adapt_frequency=5,
                 **kwargs):
        super().__init__(instance)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # SA参数
        self.sa_initial_temp = sa_initial_temp
        self.sa_cooling_rate = sa_cooling_rate
        self.sa_max_iter = sa_max_iter
        
        # VNS-ADP参数
        self.vns_max_iter = vns_max_iter
        self.vns_max_neighborhoods = vns_max_neighborhoods
        self.vns_shake_intensity = vns_shake_intensity
        self.vns_min_intensity = vns_min_intensity
        self.vns_max_intensity = vns_max_intensity
        self.adapt_frequency = adapt_frequency
        
        # 车辆容量相关参数
        self.vehicle_capacities = kwargs.get('vehicle_capacities', {
            'small': {'weight': 1000, 'volume': 3},
            'medium': {'weight': 3000, 'volume': 10},
            'large': {'weight': 5000, 'volume': 20}
        })
        
        # 商户类型映射
        self.merchant_type_mapping = kwargs.get('merchant_type_mapping', {
            'convenience': {
                'primary': 'small',
                'secondary': 'medium',
                'weight_threshold': 1000,
                'volume_threshold': 3
            },
            'supermarket': {
                'primary': 'medium',
                'secondary': 'large',
                'weight_threshold': 3000,
                'volume_threshold': 10
            },
            'mall': {
                'primary': 'large',
                'secondary': 'medium',
                'weight_threshold': 5000,
                'volume_threshold': 20
            }
        })
        
        self.final_solution = None
    
    def solve(self) -> VRPSolution:
        # 第一阶段：SA
        sa_solver = SASolver(
            self.instance,
            initial_temp=self.sa_initial_temp,
            cooling_rate=self.sa_cooling_rate,
            max_iter=self.sa_max_iter,
            vehicle_capacities=self.vehicle_capacities,  # 传递车辆参数
            merchant_type_mapping=self.merchant_type_mapping  # 传递商户映射
        )
        t0 = time.time()
        initial_sol = sa_solver.solve()
        t1 = time.time()
        
        if not initial_sol or not initial_sol.is_feasible():
            self.logger.warning("SA求解失败或不可行，直接返回。")
            self.final_solution = initial_sol
            return initial_sol
            
        # 第二阶段：Adaptive VNS
        adp_vns_solver = AdaptiveVNSSolver(
            instance=self.instance,
            max_iterations=self.vns_max_iter,
            max_neighborhoods=self.vns_max_neighborhoods,
            shake_intensity=self.vns_shake_intensity,
            min_intensity=self.vns_min_intensity,
            max_intensity=self.vns_max_intensity,
            adapt_frequency=self.adapt_frequency,
            vehicle_capacities=self.vehicle_capacities,  # 传递车辆参数
            merchant_type_mapping=self.merchant_type_mapping  # 传递商户映射
        )
        
        # 让自适应VNS从初始解出发
        adp_vns_solver.initial_solution = initial_sol
        
        t2 = time.time()
        improved_sol = adp_vns_solver.solve()
        t3 = time.time()
        
        improvement = 0.0
        if improved_sol and improved_sol.is_feasible():
            init_obj = initial_sol.objective_value
            final_obj = improved_sol.objective_value
            improvement = (init_obj - final_obj) / (init_obj + 1e-9)
            self.final_solution = improved_sol
        else:
            self.final_solution = initial_sol
            
        self.logger.info(
            f"[SA->VNS-ADP] 初始解目标={initial_sol.objective_value:.2f}, "
            f"改进后目标={self.final_solution.objective_value:.2f}, "
            f"改进率={improvement*100:.2f}%, "
            f"SA时间={t1-t0:.2f}s, VNS时间={t3-t2:.2f}s"
        )
        
        return self.final_solution
        
    def get_solution(self) -> VRPSolution:
        if self.final_solution is None:
            return self.solve()
        return self.final_solution