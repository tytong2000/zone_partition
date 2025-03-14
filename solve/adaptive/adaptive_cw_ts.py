# solvers/adaptive/adaptive_cw_ts.py

import time
import logging
from typing import List
from ..base.base_solver import BaseSolver
from ..base.vrp_solution import VRPSolution
from ..baseline.cw_solver import CWSolver
from ..adaptive.adaptive_ts import AdaptiveTSSolver

# 首先添加 Route 类定义
class Route:
    def __init__(self, customer_ids: List[int], vehicle_type: str):
        self.customer_ids = customer_ids
        self.vehicle_type = vehicle_type

    def copy(self):
        return Route(self.customer_ids.copy(), self.vehicle_type)

class AdaptiveCWTSSolver(BaseSolver):
    """
    自适应混合算法：
      第一阶段：CW求初始解
      第二阶段：Adaptive TS对初始解做自适应禁忌搜索
    """
    def __init__(self, instance,
                 cw_max_iter=100,
                 ts_init_tabu_size=5,
                 ts_max_iter=50,
                 ts_min_tabu_size=5,
                 ts_max_tabu_size=20,
                 adapt_frequency=5,
                 **kwargs):
        """
        :param instance: VRPInstance
        :param cw_max_iter: Clarke-Wright最大迭代
        :param ts_init_tabu_size: 初始禁忌表长度
        :param ts_max_iter: 最大迭代次数
        :param ts_min_tabu_size: 自适应最小tabu大小
        :param ts_max_tabu_size: 自适应最大tabu大小
        :param adapt_frequency: 每多少步进行一次参数自适应
        """
        super().__init__(instance)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # CW和TS的基本参数
        self.cw_max_iter = cw_max_iter
        self.ts_init_tabu_size = ts_init_tabu_size
        self.ts_max_iter = ts_max_iter
        self.ts_min_tabu_size = ts_min_tabu_size
        self.ts_max_tabu_size = ts_max_tabu_size
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
        # 第一阶段：CW
        cw_solver = CWSolver(
            self.instance,
            max_iter=self.cw_max_iter,
            vehicle_capacities=self.vehicle_capacities,  # 传递车辆参数
            merchant_type_mapping=self.merchant_type_mapping  # 传递商户映射
        )
        t0 = time.time()
        initial_sol = cw_solver.solve()
        t1 = time.time()
        
        if not initial_sol or not initial_sol.is_feasible():
            self.logger.warning("CW求解失败或不可行，直接返回。")
            self.final_solution = initial_sol
            return initial_sol
            
        # 第二阶段：Adaptive TS
        adp_ts_solver = AdaptiveTSSolver(
            instance=self.instance,
            tabu_size=self.ts_init_tabu_size,
            max_iterations=self.ts_max_iter,
            min_tabu_size=self.ts_min_tabu_size,
            max_tabu_size=self.ts_max_tabu_size,
            adapt_frequency=self.adapt_frequency,
            vehicle_capacities=self.vehicle_capacities,  # 传递车辆参数
            merchant_type_mapping=self.merchant_type_mapping,  # 传递商户映射
            initial_solution=initial_sol  # 传递初始解
        )
        
        t2 = time.time()
        improved_sol = adp_ts_solver.solve()
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
            f"[CW->TS-ADP] 初始解目标={initial_sol.objective_value:.2f}, "
            f"改进后目标={self.final_solution.objective_value:.2f}, "
            f"改进率={improvement*100:.2f}%, "
            f"CW时间={t1-t0:.2f}s, TS时间={t3-t2:.2f}s"
        )
        
        return self.final_solution
        
    def _ensure_route_objects(self, solution: VRPSolution) -> VRPSolution:
        """确保所有路线都是Route对象"""
        new_solution = solution.copy()
        for i, route in enumerate(new_solution.routes):
            if not isinstance(route, Route):
                # 如果是list, 转换为Route对象
                route_data = route if isinstance(route, list) else route.customer_ids
                v_type = self._get_suitable_vehicle_type(route_data)
                new_solution.routes[i] = Route(
                    customer_ids=route_data,
                    vehicle_type=v_type
                )
        return new_solution

    def _get_suitable_vehicle_type(self, customer_ids: List[int]) -> str:
        """根据客户需求确定合适的车型"""
        total_weight = sum(self.instance.order_demands[i] for i in customer_ids)
        total_volume = sum(self.instance.get_order_volume(i) for i in customer_ids)
        
        if (total_weight <= self.vehicle_capacities['small']['weight'] and 
            total_volume <= self.vehicle_capacities['small']['volume']):
            return 'small'
        elif (total_weight <= self.vehicle_capacities['medium']['weight'] and 
              total_volume <= self.vehicle_capacities['medium']['volume']):
            return 'medium'
        else:
            return 'large'
    
    def solve(self) -> VRPSolution:
        # 第一阶段：CW
        cw_solver = CWSolver(
            self.instance,
            max_iter=self.cw_max_iter,
            vehicle_capacities=self.vehicle_capacities,
            merchant_type_mapping=self.merchant_type_mapping
        )
        t0 = time.time()
        initial_sol = cw_solver.solve()
        t1 = time.time()
        
        if not initial_sol or not initial_sol.is_feasible():
            self.logger.warning("CW求解失败或不可行，直接返回。")
            self.final_solution = initial_sol
            return initial_sol
            
        # 确保初始解的路线是Route对象
        initial_sol = self._ensure_route_objects(initial_sol)
            
        # 第二阶段：Adaptive TS
        adp_ts_solver = AdaptiveTSSolver(
            instance=self.instance,
            tabu_size=self.ts_init_tabu_size,
            max_iterations=self.ts_max_iter,
            min_tabu_size=self.ts_min_tabu_size,
            max_tabu_size=self.ts_max_tabu_size,
            adapt_frequency=self.adapt_frequency,
            vehicle_capacities=self.vehicle_capacities,
            merchant_type_mapping=self.merchant_type_mapping,
            initial_solution=initial_sol
        )
        
        t2 = time.time()
        improved_sol = adp_ts_solver.solve()
        t3 = time.time()
        
        # 确保改进解的路线是Route对象
        if improved_sol and improved_sol.is_feasible():
            improved_sol = self._ensure_route_objects(improved_sol)
            init_obj = initial_sol.objective_value
            final_obj = improved_sol.objective_value
            improvement = (init_obj - final_obj) / (init_obj + 1e-9)
            self.final_solution = improved_sol
        else:
            self.final_solution = initial_sol
            improvement = 0.0
            
        # 添加vehicle_assignments
        if not hasattr(self.final_solution, 'vehicle_assignments'):
            self.final_solution.vehicle_assignments = {}
            for i, route in enumerate(self.final_solution.routes):
                self.final_solution.vehicle_assignments[i] = {
                    'type': route.vehicle_type,
                    'customers': route.customer_ids
                }
            
        self.logger.info(
            f"[CW->TS-ADP] 初始解目标={initial_sol.objective_value:.2f}, "
            f"改进后目标={self.final_solution.objective_value:.2f}, "
            f"改进率={improvement*100:.2f}%, "
            f"CW时间={t1-t0:.2f}s, TS时间={t3-t2:.2f}s"
        )
        
        return self.final_solution
        
    def get_solution(self) -> VRPSolution:
        if self.final_solution is None:
            return self.solve()
        return self.final_solution
 
