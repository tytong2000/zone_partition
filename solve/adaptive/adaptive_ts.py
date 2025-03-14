# solvers/adaptive/adaptive_ts.py

import math
import logging
import random
import numpy as np

from ..base.base_solver import BaseSolver
from ..base.vrp_solution import VRPSolution
from ..baseline.ts_solver import TSSolver
from solve.base.vrp_instance import VRPInstance
from typing import List, Dict, Tuple, Optional

class Route:
    def __init__(self, customer_ids: List[int], vehicle_type: str):
        self.customer_ids = customer_ids
        self.vehicle_type = vehicle_type

    def copy(self):
        return Route(self.customer_ids.copy(), self.vehicle_type)
    
class AdaptiveTSSolver(TSSolver):
    def __init__(self, instance: "VRPInstance", **kwargs):
        # 1. 先调用父类初始化
        super().__init__(instance, **kwargs)
        
        # 2. 保存初始解
        self.initial_solution = kwargs.get('initial_solution', None)
        
        # 3. 从solver_params获取相应的参数
        self.initial_temp = kwargs.get('initial_temp', 1000.0)
        self.cooling_rate = kwargs.get('cooling_rate', 0.95)
        self.iterations = kwargs.get('iterations', 1000)
        
        # 4. 禁忌表参数
        self.tabu_size = kwargs.get('tabu_size', 10)
        self.max_iterations = kwargs.get('max_iterations', 50)
        self.neighborhood_size = kwargs.get('neighborhood_size', 20)
        
        # 5. 初始化禁忌表
        self.tabu_list = []
        self.aspiration_value = float('inf')
        
        # 6. 车辆容量相关参数
        self.vehicle_capacities = kwargs.get('vehicle_capacities', {
            'small': {'weight': 1000, 'volume': 3},
            'medium': {'weight': 3000, 'volume': 10},
            'large': {'weight': 5000, 'volume': 20}
        })
        
        # 7. 商户类型映射
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
        
        # 8. 自适应参数
        self.adaptive_params = kwargs.get('adaptive_params', {})
        
        # 9. 设置自适应参数范围
        if self.adaptive_params:
            self.param_type = list(self.adaptive_params.keys())[0]
            self.min_value = self.adaptive_params.get('min_tabu_size', 5)
            self.max_value = self.adaptive_params.get('max_tabu_size', 20)
        else:
            # 设置默认值
            self.param_type = 'tabu_size'
            self.min_value = 5
            self.max_value = 20
        
        # 10. 历史记录初始化
        self.parameter_history = []
        self.improvement_history = []
        
    def solve(self) -> VRPSolution:
        self.logger.info("开始AdaptiveTabuSearch求解...")
        
        # 初始化计数器
        self.iterations_count = 0
        self.improvement_count = 0
        self.diversification_count = 0
        self.best_solutions = []
        
        # 使用初始解或构建新解
        if self.initial_solution and self.initial_solution.is_feasible():
            current_sol = self.initial_solution.copy()
        else:
            current_sol = self._build_initial_solution()
            
        if not current_sol.is_feasible():
            return current_sol
            
        # 正确创建最优解
        best_sol = current_sol.copy()
        best_sol = self._ensure_route_objects(best_sol)
        
        self._update_convergence(current_sol.objective_value)
        self.best_solutions.append(best_sol)
        
        # 初始参数
        if self.param_type == 'tabu_size':
            self.tabu_size = self.min_value
        else:
            self.neighborhood_size = self.min_value
            
        self.parameter_history.append(self.min_value)
        
        # 主循环
        no_improve = 0
        last_obj = current_sol.objective_value
        
        while (self.iterations_count < self.max_iterations and 
            no_improve < self.max_iterations // 2):
            self.iterations_count += 1
            
            # 生成邻域解
            best_neighbor = None
            best_move = None
            best_value = float('inf')
            
            for _ in range(self.neighborhood_size):
                neighbor, move = self._generate_neighbor(current_sol)
                if not neighbor.is_feasible():
                    continue
                    
                move_hash = self._hash_move(move)
                
                if (move_hash in self.tabu_list and 
                    neighbor.objective_value >= self.aspiration_value):
                    continue
                    
                if neighbor.objective_value < best_value:
                    best_neighbor = neighbor
                    best_move = move
                    best_value = neighbor.objective_value
                    
            if best_neighbor is None:
                no_improve += 1
                continue
                
            # 更新当前解
            current_sol = best_neighbor
            move_hash = self._hash_move(best_move)
            
            # 更新禁忌表
            self.tabu_list.append(move_hash)
            if len(self.tabu_list) > self.tabu_size:
                self.tabu_list.pop(0)
                
            # 计算改进率
            improvement = (last_obj - current_sol.objective_value) / last_obj
            self.improvement_history.append(improvement)
            last_obj = current_sol.objective_value
            
            # 自适应参数调整
            self._adapt_parameters(improvement)
            
            # 更新最优解
            if current_sol.objective_value < best_sol.objective_value:
                best_sol = current_sol.copy()
                best_sol = self._ensure_route_objects(best_sol)  # 确保是Route对象
                self.aspiration_value = best_sol.objective_value
                self.improvement_count += 1
                no_improve = 0
                self.best_solutions.append(best_sol)
                
            # 收敛历史
            self._update_convergence(current_sol.objective_value)
            self._update_exploration(current_sol)
            
            # 多样化策略
            if no_improve >= 10:
                current_sol = self._diversification(current_sol)
                no_improve = 0
                self.diversification_count += 1
        
        # 在返回之前，确保添加 vehicle_assignments
        best_sol = self._ensure_route_objects(best_sol)
        
        # 添加 vehicle_assignments
        if not hasattr(best_sol, 'vehicle_assignments'):
            best_sol.vehicle_assignments = {}
            for i, route in enumerate(best_sol.routes):
                best_sol.vehicle_assignments[i] = {
                    'type': route.vehicle_type,
                    'customers': route.customer_ids
                }
        
        return best_sol

    def _diversification(self, solution: VRPSolution) -> VRPSolution:
        """重写多样化方法以支持多车型"""
        new_sol = solution.copy()
        # 确保所有路线都是 Route 对象
        new_sol = self._ensure_route_objects(new_sol)
        
        routes = new_sol.routes
        if not routes:
            return new_sol
            
        route_idx = random.randint(0, len(routes) - 1)
        current_route = routes[route_idx]
        
        # 现在 current_route 是 Route 对象，可以安全访问 vehicle_type
        if len(current_route.customer_ids) >= 4:
            # 随机选择扰动操作
            if random.random() < 0.5:
                # 2-opt
                i = random.randint(1, len(current_route.customer_ids) - 3)
                j = random.randint(i + 2, len(current_route.customer_ids) - 1)
                current_route.customer_ids[i:j] = reversed(current_route.customer_ids[i:j])
            else:
                # 随机交换
                i = random.randint(1, len(current_route.customer_ids) - 2)
                j = random.randint(i + 1, len(current_route.customer_ids) - 1)
                current_route.customer_ids[i], current_route.customer_ids[j] = \
                    current_route.customer_ids[j], current_route.customer_ids[i]
        
        return new_sol
        
    def _adapt_parameters(self, improvement: float):
        """自适应参数调整"""
        recent_improvements = self.improvement_history[-5:] if len(self.improvement_history) >= 5 else self.improvement_history
        avg_improvement = np.mean(recent_improvements)
        
        if self.param_type == 'tabu_size':
            if avg_improvement > 0.01:  # 解质量在改善
                self.tabu_size = max(self.min_value, self.tabu_size - 1)
            else:  # 解质量停滞
                self.tabu_size = min(self.max_value, self.tabu_size + 1)
            self.parameter_history.append(self.tabu_size)
            
        else:  # neighborhood_size
            if avg_improvement > 0.01:
                self.neighborhood_size = min(self.max_value, self.neighborhood_size + 1)
            else:
                self.neighborhood_size = max(self.min_value, self.neighborhood_size - 1)
            self.parameter_history.append(self.neighborhood_size)
            
    def get_parameter_history(self) -> List[float]:
        """获取参数调整历史"""
        return self.parameter_history

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