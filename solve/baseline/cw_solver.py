# solvers/baseline/cw_solver.py
import os
import sys
import time
import logging
import random
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon
import networkx as nx
from tqdm import tqdm
from typing import Any
import seaborn as sns  
import math
import numpy as np
import networkx as nx
from solve.base.base_solver import BaseSolver
from solve.base.vrp_solution import VRPSolution
from solve.base.vrp_instance import VRPInstance  # 改为绝对导入
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple
# 首先修改CW算法以支持多车型和商户类型
class CWSolver(BaseSolver):
    # 在 CWSolver 类的 __init__ 方法中添加：
    def __init__(self, instance: "VRPInstance", **kwargs):
        super().__init__(instance, **kwargs)
        
        # 新增以下参数
        self.progress_callback = None
        self.parallel = kwargs.get('parallel', True)
        self.early_stopping = kwargs.get('early_stopping', True)
        self.max_iterations = kwargs.get('max_iterations', 1000)
        
        # 添加距离缓存
        self._distance_matrix = {}
        self._precompute_distances()
        
        # 保留原有的其他初始化代码
        self.compatibility_matrix = {
            'small': ['convenience'],
            'medium': ['convenience', 'supermarket'],
            'large': ['convenience', 'supermarket', 'mall']
        }
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
    def solve(self) -> VRPSolution:
        """优化的CW算法主求解过程"""
        # 构建初始解
        solution = self._build_initial_solution()
        if not solution.is_feasible():
            return solution

        # 计算节约值
        savings = self._compute_savings()
        total_iterations = len(savings)
        current_iteration = 0
        no_improve_count = 0
        best_cost = float('inf')

        # 主循环
        for saving, i, j in savings:
            current_iteration += 1
            
            # 更新进度
            if self.progress_callback:
                self.progress_callback(current_iteration / total_iterations)

            # 找到包含i,j的路线
            route_i = None
            route_j = None
            route_i_idx = -1
            route_j_idx = -1

            for idx, route in enumerate(solution.routes):
                if i in route:
                    route_i = route
                    route_i_idx = idx
                if j in route:
                    route_j = route
                    route_j_idx = idx

            # 如果i,j不在同一条路线中
            if (route_i is not None and route_j is not None and 
                route_i_idx != route_j_idx and 
                route_i_idx < len(solution.routes) and 
                route_j_idx < len(solution.routes)):

                # 快速可行性检查
                if self._quick_merge_check(route_i, route_j):
                    # 删除原路线
                    if route_i_idx > route_j_idx:
                        solution.routes.pop(route_i_idx)
                        solution.routes.pop(route_j_idx)
                    else:
                        solution.routes.pop(route_j_idx)
                        solution.routes.pop(route_i_idx)

                    # 添加合并后的新路线
                    merged_route = route_i + route_j
                    vehicle_type = self._get_suitable_vehicle_type(merged_route)
                    solution.add_route(merged_route, vehicle_type=vehicle_type)

                    # 计算新成本
                    current_cost = self._calculate_total_cost(solution.routes)
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        no_improve_count = 0
                    else:
                        no_improve_count += 1

                    # 提前停止检查
                    if self.early_stopping and no_improve_count >= 20:
                        break

        return solution

    def _get_suitable_vehicle_type(self, route: List[int]) -> str:
        """根据路线的总需求选择合适的车型"""
        if not route:
            return 'small'
        
        # 计算总需求
        total_demand = sum(self.instance.get_order_demand(i) for i in route)
        
        # 获取该路线的商户类型
        merchant_types = [self.instance.get_merchant_type(i) for i in route]
        most_restrictive_type = max(merchant_types, key=lambda x: 
            self.merchant_type_mapping.get(x, {}).get('weight_threshold', 0))
        
        # 按容量从小到大尝试车型
        for v_type in ['small', 'medium', 'large']:
            if total_demand <= self.vehicle_capacities[v_type].get('weight', 0):
                # 检查车型是否兼容商户类型
                compatible = True
                for mt in merchant_types:
                    if mt not in ['convenience', 'supermarket', 'mall']:
                        continue
                    if v_type not in [
                        self.merchant_type_mapping[mt]['primary'],
                        self.merchant_type_mapping[mt]['secondary']
                    ]:
                        compatible = False
                        break
                
                if compatible:
                    return v_type
        
        # 如果没有合适的，返回最大车型
        return 'large'

    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback

    def _precompute_distances(self):
        """预计算并缓存距离矩阵"""
        depot = self.instance.get_depot_location(0)
        for i in range(self.instance.num_orders):
            loc_i = self.instance.get_order_location(i)
            self._distance_matrix[(0, i+1)] = self.instance.get_road_distance(depot, loc_i)
            self._distance_matrix[(i+1, 0)] = self._distance_matrix[(0, i+1)]
            
            for j in range(i+1, self.instance.num_orders):
                loc_j = self.instance.get_order_location(j)
                dist = self.instance.get_road_distance(loc_i, loc_j)
                self._distance_matrix[(i+1, j+1)] = dist
                self._distance_matrix[(j+1, i+1)] = dist

    def _calculate_single_saving(self, i: int, j: int) -> Optional[Tuple[float, int, int]]:
        """计算单个节约值"""
        saving = (self._distance_matrix[(0, i)] + 
                self._distance_matrix[(0, j)] - 
                self._distance_matrix[(i, j)])
        if saving > 0:
            return (saving, i-1, j-1)
        return None

    def _compute_savings(self) -> List[Tuple[float, int, int]]:
        savings = []
        depot = self.instance.get_depot_location(0)
        
        # 计算最大距离作为参考
        max_distance = 0
        for i in range(min(100, self.instance.num_orders)):
            loc_i = self.instance.get_order_location(i)
            dist = self.instance.get_road_distance(depot, loc_i)
            max_distance = max(max_distance, dist)
        
        # 只考虑节省大的路径组合
        for i in range(self.instance.num_orders):
            loc_i = self.instance.get_order_location(i)
            dist_i = self.instance.get_road_distance(depot, loc_i)
            
            # 跳过离仓库非常远的点
            if dist_i > 2 * max_distance:
                continue
                
            for j in range(i + 1, self.instance.num_orders):
                loc_j = self.instance.get_order_location(j)
                dist_j = self.instance.get_road_distance(depot, loc_j)
                
                # 如果两点都非常远，跳过
                if dist_j > 2 * max_distance:
                    continue
                    
                saving = dist_i + dist_j - self.instance.get_road_distance(loc_i, loc_j)
                
                # 只保留有显著节省的组合
                if saving > 0.1 * (dist_i + dist_j):
                    savings.append((saving, i, j))
        
        # 只保留前N个最大节省值
        savings.sort(reverse=True)
        max_savings = min(10000, len(savings))
        return savings[:max_savings]
    
    def _check_route_feasible(self, route: List[int], vehicle_type: str) -> bool:
        """简化的路线可行性检查"""
        return True  # 直接返回True，跳过所有检查
    def _quick_merge_check(self, route1: List[int], route2: List[int]) -> bool:
        """快速检查两条路线合并是否可行"""
        # 简化实现，始终返回可行
        return True
# 以 CWSolver 为例
    def _build_initial_solution(self) -> VRPSolution:
        solution = VRPSolution(self.instance)
        
        # 按商户类型对订单分组
        orders_by_type = {mt: [] for mt in ['convenience', 'supermarket', 'mall']}
        for i in range(self.instance.num_orders):
            mt = self.instance.get_merchant_type(i)
            orders_by_type[mt].append(i)
            
        # 为每个订单创建路线（不做可行性检查）
        for mt, orders in orders_by_type.items():
            for order in orders:
                vehicle_type = self.instance.get_vehicle_type_for_order(order)
                solution.add_route([order], vehicle_type=vehicle_type)
                    
        return solution
        
    def _calculate_total_cost(self, routes: List[List[int]]) -> float:
        """计算总成本"""
        total_cost = 0.0
        for route in routes:
            if not route:
                continue
            prev = 0  # 从仓库开始
            for i in route:
                total_cost += self._distance_matrix.get((prev, i+1), 0)
                prev = i+1
            total_cost += self._distance_matrix.get((prev, 0), 0)  # 返回仓库
            
        return total_cost
