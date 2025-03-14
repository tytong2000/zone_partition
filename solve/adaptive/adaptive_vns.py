# solvers/adaptive/adaptive_vns.py

import math
import logging
import random
import numpy as np

from ..base.base_solver import BaseSolver
from ..base.vrp_solution import VRPSolution
from ..baseline.vns_solver import VNSSolver
from typing import List, Dict, Tuple, Optional
from solve.base.vrp_instance import VRPInstance
class Route:
    def __init__(self, customer_ids: List[int], vehicle_type: str):
        self.customer_ids = customer_ids
        self.vehicle_type = vehicle_type

    def copy(self):
        return Route(self.customer_ids.copy(), self.vehicle_type)

class AdaptiveVNSSolver(VNSSolver):
    def __init__(self, instance: "VRPInstance", **kwargs):
        # 1. 先调用父类初始化
        super().__init__(instance, **kwargs)
        
        # 2. 基础参数设置 (保持不变)
        self.initial_solution = kwargs.get('initial_solution', None)
        self.max_iterations = kwargs.get('max_iterations', 100)
        self.max_no_improve = kwargs.get('max_no_improve', 20)
        
        # 3-5. 参数设置 (保持不变)
        self.vehicle_capacities = kwargs.get('vehicle_capacities', {
            'small': {'weight': 1000, 'volume': 3},
            'medium': {'weight': 3000, 'volume': 10},
            'large': {'weight': 5000, 'volume': 20}
        })
        
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
        
        self.adaptive_params = kwargs.get('adaptive_params', {'default': {}})
        if not self.adaptive_params:
            self.adaptive_params = {'default': {}}
            
        self.param_type = list(self.adaptive_params.keys())[0]
        self.min_value = self.adaptive_params.get('min_intensity', 0.1)
        self.max_value = self.adaptive_params.get('max_intensity', 0.5)
        
        # 6. 历史记录和计数器
        self.parameter_history = []
        self.improvement_history = []
        self.stagnation_counter = 0
        
        # 7. 添加vehicle_assignments到metrics
        if not hasattr(self, 'metrics'):
            self.metrics = {}
        self.metrics['vehicle_assignments'] = []

    def solve(self) -> VRPSolution:
        """重写solve方法以添加vehicle_assignments追踪"""
        best_sol = super().solve()
        
        # 确保最终解的routes都是Route对象
        best_sol = self._ensure_route_objects(best_sol)
        
        # 更新vehicle_assignments
        self.metrics['vehicle_assignments'] = [
            {'type': route.vehicle_type, 'customers': route.customer_ids}
            for route in best_sol.routes
        ]
        
        return best_sol
    
    def _get_suitable_vehicle_type(self, customer_ids: List[int]) -> str:
        """根据客户需求确定合适的车型"""
        total_weight = sum(self.instance.order_demands[i] for i in customer_ids)  # 修改此处
        total_volume = sum(self.instance.get_order_volume(i) for i in customer_ids)  # 体积可以使用get_order_volume方法获取
        
        if (total_weight <= self.vehicle_capacities['small']['weight'] and 
            total_volume <= self.vehicle_capacities['small']['volume']):
            return 'small'
        elif (total_weight <= self.vehicle_capacities['medium']['weight'] and 
            total_volume <= self.vehicle_capacities['medium']['volume']):
            return 'medium'
        else:
            return 'large'

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
    
    def _shake(self, solution: VRPSolution, k: int) -> VRPSolution:
        """重写扰动方法以支持多车型"""
        # 先确保所有route都是Route对象
        shaken_sol = self._ensure_route_objects(solution.copy())
        
        intensity = self.shake_intensity * (k + 1)
        
        for route in shaken_sol.routes:
            if len(route.customer_ids) >= 4:
                num_moves = max(1, int(len(route.customer_ids) * intensity))
                for _ in range(num_moves):
                    i = random.randint(1, len(route.customer_ids) - 2)
                    j = random.randint(i + 1, len(route.customer_ids) - 1)
                    
                    temp_route = route.copy()
                    temp_route.customer_ids[i], temp_route.customer_ids[j] = \
                        temp_route.customer_ids[j], temp_route.customer_ids[i]
                    
                    if self._check_route_feasible(temp_route, route.vehicle_type):
                        route.customer_ids[i], route.customer_ids[j] = \
                            route.customer_ids[j], route.customer_ids[i]
        
        return shaken_sol

    def _variable_neighborhood_descent(self, solution: VRPSolution) -> VRPSolution:
        """重写VND以支持多车型"""
        # 先确保所有route都是Route对象
        current = self._ensure_route_objects(solution.copy())
        improved = True
        
        while improved:
            improved = False
            for neighborhood in self.neighborhoods:
                neighbor = self._apply_neighborhood(current, neighborhood)
                
                if (neighbor and neighbor.is_feasible() and 
                    neighbor.objective_value < current.objective_value):
                    current = neighbor
                    improved = True
                    break
                    
        return current

    def _apply_neighborhood(self, solution: VRPSolution, 
                          neighborhood: str) -> Optional[VRPSolution]:
        """重写邻域操作以支持多车型"""
        # 先确保所有route都是Route对象
        new_sol = self._ensure_route_objects(solution.copy())
        
        if neighborhood == '2-opt':
            for route in new_sol.routes:
                if len(route.customer_ids) >= 4:
                    i = random.randint(1, len(route.customer_ids) - 3)
                    j = random.randint(i + 2, len(route.customer_ids) - 1)
                    
                    temp_route = route.copy()
                    temp_route.customer_ids[i:j] = reversed(temp_route.customer_ids[i:j])
                    
                    if self._check_route_feasible(temp_route, route.vehicle_type):
                        route.customer_ids[i:j] = reversed(route.customer_ids[i:j])
                        
        elif neighborhood == 'relocate':
            if len(new_sol.routes) >= 2:
                from_idx = random.randint(0, len(new_sol.routes) - 1)
                to_idx = random.randint(0, len(new_sol.routes) - 1)
                while to_idx == from_idx:
                    to_idx = random.randint(0, len(new_sol.routes) - 1)
                    
                from_route = new_sol.routes[from_idx]
                to_route = new_sol.routes[to_idx]
                
                if len(from_route.customer_ids) >= 3:
                    customer_idx = random.randint(1, len(from_route.customer_ids) - 2)
                    customer = from_route.customer_ids[customer_idx]
                    
                    temp_from_route = from_route.copy()
                    temp_to_route = to_route.copy()
                    
                    temp_from_route.customer_ids.pop(customer_idx)
                    insert_pos = random.randint(1, len(temp_to_route.customer_ids) - 1)
                    temp_to_route.customer_ids.insert(insert_pos, customer)
                    
                    if (self._check_route_feasible(temp_from_route, from_route.vehicle_type) and
                        self._check_route_feasible(temp_to_route, to_route.vehicle_type)):
                        from_route.customer_ids.pop(customer_idx)
                        to_route.customer_ids.insert(insert_pos, customer)
        
        return new_sol if new_sol.is_feasible() else None

    def get_parameter_history(self) -> List[float]:
        """获取参数调整历史"""
        return self.parameter_history