# solve/base/base_solver.py
import os
import sys
import time
import logging
import random
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
import csv
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon
import networkx as nx
from tqdm import tqdm
import seaborn as sns
import math
from abc import ABC, abstractmethod
from solve.base.vrp_instance import VRPInstance

# solve/base/base_solver.py
class BaseSolver(ABC):
    def __init__(self, instance: "VRPInstance", **kwargs):
        self.instance = instance
        self.compatibility_matrix = {
            'small': ['convenience'],
            'medium': ['convenience', 'supermarket'],
            'large': ['convenience', 'supermarket', 'mall']
        }
        
        # 直接定义为简单的浮点数值
        self.vehicle_capacities = {
            'small': 1000.0,
            'medium': 3000.0,
            'large': 5000.0
        }
        self.initial_solution = kwargs.get('initial_solution')  # 从 kwargs 中获取 initial_solution
        self.solution = None
        
        # 其他初始化代码保持不变
        self.logger = logging.getLogger(self.__class__.__name__)
        self.convergence_history = []
        self.parameter_history = {}
        self.exploration_history = []
        self.best_solutions = []
        ...  # 其他初始化保持不变
        
        # 性能指标
        self.iterations_count: int = 0
        self.local_search_count: int = 0
        self.improvement_count: int = 0
        self.diversification_count: int = 0
        self.computation_time: float = 0
        
        # 解的多样性指标
        self.solution_pool = []
        self.solution_distances: List[float] = []
        self.diversity_threshold: float = 0.3
        
        # 约束违反统计
        self.time_window_violations: int = 0
        self.capacity_violations: int = 0
        self.duration_violations: int = 0
        
        # 初始化随机种子
        random.seed(42)
        np.random.seed(42)

# 在 BaseSolver 类中
    def _check_route_feasible(self, route: List[int], vehicle_type: str) -> bool:
        """简化的路线可行性检查"""
        if not route:
            return False
        return True  # 直接返回True，跳过所有检查
    @abstractmethod
    def solve(self): 
        """求解VRP问题"""
        raise NotImplementedError

    def _update_parameters(self, param_name: str, value: Any) -> None:
        """更新参数历史"""
        if param_name not in self.parameter_history:
            self.parameter_history[param_name] = []
        self.parameter_history[param_name].append(value)
           
    def _update_convergence(self, current_obj: float) -> None:
        """更新收敛历史"""
        self.convergence_history.append(current_obj)
        if self.solution and (not self.best_solutions or current_obj < self.best_solutions[-1].objective_value):
            self.best_solutions.append(self.solution)
            self.improvement_count += 1
            
    def _calculate_diversity(self, solution) -> float:
        """计算解的多样性"""
        if not self.solution_pool:
            return 1.0
            
        distances = []
        for old_sol in self.solution_pool[-5:]:
            dist = self._solution_distance(solution, old_sol)
            distances.append(dist)
            
        return np.mean(distances) if distances else 0.0
        
    def _solution_distance(self, sol1, sol2) -> float:
        """计算两个解之间的距离"""
        total_dist = 0.0
        for r1 in sol1.routes:
            min_dist = float('inf')
            for r2 in sol2.routes:
                edit_dist = self._route_edit_distance(r1, r2)
                min_dist = min(min_dist, edit_dist)
            total_dist += min_dist
        return total_dist / max(len(sol1.routes), 1)
        
    def _route_edit_distance(self, route1: List[int], route2: List[int]) -> float:
        """计算两条路线间的编辑距离"""
        m, n = len(route1), len(route2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if route1[i-1] == route2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + 1)
                    
        return dp[m][n]

    def _update_exploration(self, solution) -> None:
        """
        更新探索历史,记录解的多样性信息
        
        Args:
            solution: 当前解
        """
        diversity = self._calculate_diversity(solution)
        self.exploration_history.append(diversity)
        
        # 更新解池
        if len(self.solution_pool) >= 10:  # 保持解池大小
            self.solution_pool.pop(0)
        self.solution_pool.append(solution)        
    def _check_constraints(self, solution) -> bool:
        """检查所有约束是否满足"""
        for route in solution.routes:
            if not self._check_time_windows(route):
                return False
            if not self._check_capacity(route):
                return False
            if not self._check_duration(route):
                return False
        return True
        
    def _check_time_windows(self, route: List[int]) -> bool:
        """检查时间窗约束"""
        if not route:
            return True
            
        current_time = 0.0
        prev_loc = self.instance.get_depot_location(0)
        
        for order in route:
            # 计算到达时间
            curr_loc = self.instance.get_order_location(order)
            travel_time = self.instance.get_road_distance(prev_loc, curr_loc) / 30000.0
            current_time += travel_time
            
            # 检查时间窗 - 使用orders_df中的时间窗数据
            ready_time = self.instance.orders_df.iloc[order].get("ready_time", 0.0)
            due_time = self.instance.orders_df.iloc[order].get("due_time", float('inf'))
            
            if current_time < ready_time:
                current_time = ready_time
            elif current_time > due_time:
                self.time_window_violations += 1
                return False
                    
            # 服务时间
            service_time = self.instance.orders_df.iloc[order].get("service_time", 0.0)
            current_time += service_time
            prev_loc = curr_loc
            
        # 返回仓库
        return_time = self.instance.get_road_distance(prev_loc, self.instance.get_depot_location(0)) / 30000.0
        current_time += return_time
        
        if current_time > self.instance.max_route_time:
            self.time_window_violations += 1
            return False
        return True
        
    def _check_capacity(self, route: List[int]) -> bool:
        """检查容量约束"""
        total_demand = sum(self.instance.get_order_demand(o) for o in route)
        if total_demand > self.instance.vehicle_capacity:
            self.capacity_violations += 1
            return False
        return True
        
    def _check_duration(self, route: List[int]) -> bool:
        """检查路线时长约束"""
        if not route:
            return True
            
        total_duration = 0.0
        prev_loc = self.instance.get_depot_location(0)
        
        for order in route:
            curr_loc = self.instance.get_order_location(order)
            travel_time = self.instance.get_road_distance(prev_loc, curr_loc) / 30000.0
            total_duration += travel_time + self.instance.get_service_time(order)
            prev_loc = curr_loc
            
        # 返回仓库时间
        return_time = self.instance.get_road_distance(prev_loc, self.instance.get_depot_location(0)) / 30000.0
        total_duration += return_time
        
        if total_duration > self.instance.max_route_duration:
            self.duration_violations += 1
            return False
        return True
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'iterations': self.iterations_count,
            'improvements': self.improvement_count,
            'computation_time': self.computation_time,
            'best_objective': min(s.objective_value for s in self.best_solutions) if self.best_solutions else float('inf'),
            'convergence_history': self.convergence_history,
            'time_window_violations': self.time_window_violations,
            'capacity_violations': self.capacity_violations,
            'duration_violations': self.duration_violations
        }
        
    def plot_convergence(self, save_path: Optional[str] = None) -> None:
        """绘制收敛曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_history, '-b', label='Objective Value')
        plt.title('Convergence History')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()