# solve/base/vrp_solution.py

from typing import Any, List, Dict, Optional
import pandas as pd
from collections import defaultdict

class VRPSolution:
    def __init__(self, instance=None):
        # 初始化解的基本信息
        self.routes: List[List[int]] = []  # 路线列表，每个元素是一个订单ID的列表
        self.vehicle_assignments: Dict[int, Dict[str, Any]] = {}  # 存储每条路线的车辆类型信息
        self.instance = instance
        self.vehicle_types = instance.vehicle_types if instance else {}  # 从实例中获取车型信息
        self.objective_value = float('inf')  # 初始目标值
        self.total_distance = 0.0  # 初始总距离
        self.total_time = 0.0  # 初始总时间
        self.total_cost = 0.0  # 初始总成本

    # 在 VRPSolution 类中修改:
    def add_route(self, route: List[int], vehicle_type: str) -> None:
        """
        向解中添加一条新路线，并为其分配车辆类型
        Args:
            route: 路线包含的订单列表
            vehicle_type: 车辆类型
        """
        if not route:
            return
            
        self.routes.append(route)
        self.vehicle_assignments[len(self.routes) - 1] = {
            'type': vehicle_type,
            'fixed_cost': self.vehicle_types.get(vehicle_type, {}).get('cost_per_km', 1.0)
        }
        self._update_metrics()

    def _update_metrics(self) -> None:
        """更新解的各项指标（例如，总距离、总时间、总成本）"""
        dist_sum = 0.0
        fixed_cost = 0.0

        for i, route in enumerate(self.routes):
            if not route:
                continue
            
            # 确保vehicle_assignments中有对应的记录
            if i not in self.vehicle_assignments:
                self.vehicle_assignments[i] = {'type': 'default', 'fixed_cost': 1000.0}
            
            vehicle = self.vehicle_assignments[i]
            fixed_cost += vehicle['fixed_cost']
            
            # 计算路线距离
            wh = self.instance.assignments.get(route[0], 0)
            wh_loc = self.instance.get_depot_location(wh)
            prev_loc = wh_loc
            
            for o_ in route:
                curr_loc = self.instance.get_order_location(o_)
                dist = self.instance.get_road_distance(prev_loc, curr_loc)
                dist_sum += dist
                prev_loc = curr_loc
            
            # 返回仓库的路径
            dist_sum += self.instance.get_road_distance(prev_loc, wh_loc)
        
        # 更新总距离、总时间、总成本
        self.total_distance = dist_sum
        self.total_time = dist_sum / 30000.0  # 假设车速为 30km/h
        self.total_cost = dist_sum + fixed_cost
        self.objective_value = self.total_cost  # 目标值为总成本

    def is_feasible(self) -> bool:
        """检查解的可行性"""
        return (len(self.routes) > 0) and (self.objective_value < float('inf'))

    def copy(self) -> 'VRPSolution':
        """创建解的深拷贝"""
        new_sol = VRPSolution(self.instance)
        new_sol.routes = [r.copy() for r in self.routes]
        new_sol.vehicle_assignments = self.vehicle_assignments.copy()
        new_sol.objective_value = self.objective_value
        new_sol.total_distance = self.total_distance
        new_sol.total_time = self.total_time
        new_sol.total_cost = self.total_cost
        return new_sol
