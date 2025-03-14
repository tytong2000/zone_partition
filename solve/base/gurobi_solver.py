import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd 
from typing import List, Dict, Tuple, Optional
import logging
from .base_solver import BaseSolver
from .vrp_solution import VRPSolution
from .vrp_instance import VRPInstance

class GurobiBaseSolver(BaseSolver):
    """Gurobi求解器基类"""
    
    def __init__(self, instance: "VRPInstance", **kwargs):
        super().__init__(instance, **kwargs)
        
        # Gurobi配置
        self.time_limit = kwargs.get('time_limit', 300)
        self.mip_gap = kwargs.get('mip_gap', 0.01)
        self.threads = kwargs.get('threads', 0)
        
        # 多车型相关
        self.vehicle_types = instance.vehicle_types
        self.merchant_types = self._get_merchant_types()
        
        # 建模相关
        self.model = None
        self.x_vars = {}  # 路径变量
        self.y_vars = {}  # 车辆使用变量
        self.z_vars = {}  # 车型分配变量
        
    def solve(self) -> Optional[VRPSolution]:
        """使用Gurobi求解VRP"""
        try:
            # 1. 创建模型
            self.model = gp.Model("VRP")
            
            # 2. 添加决策变量
            self._add_variables()
            
            # 3. 添加约束
            self._add_constraints()
            
            # 4. 设置目标函数
            self._set_objective()
            
            # 5. 配置求解器参数
            self._config_solver()
            
            # 6. 求解
            self.model.optimize()
            
            # 7. 获取结果
            if self.model.Status == GRB.OPTIMAL or self.model.Status == GRB.TIME_LIMIT:
                solution = self._build_solution()
                return solution
                
        except gp.GurobiError as e:
            self.logger.error(f"Gurobi error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error: {str(e)}")
            
        return None
        
    def _get_merchant_types(self) -> Dict[str, List[int]]:
        """获取商户类型分组"""
        types = {}
        if '类型' in self.instance.orders_df.columns:
            for t in self.instance.orders_df['类型'].unique():
                types[t] = self.instance.orders_df[
                    self.instance.orders_df['类型'] == t
                ].index.tolist()
        return types
        
    def _add_variables(self):
        """添加决策变量"""
        n = self.instance.num_orders
        
        # 1. 路径变量 x[i,j,k,v]: 车型v的第k辆车从i到j
        self.x_vars = {}
        for v_type in self.vehicle_types:
            specs = self.vehicle_types[v_type]
            for k in range(specs['count']):
                for i in range(-1, n):  # -1表示仓库
                    for j in range(-1, n):
                        if i != j:
                            self.x_vars[i,j,k,v_type] = self.model.addVar(
                                vtype=GRB.BINARY,
                                name=f'x_{i}_{j}_{k}_{v_type}'
                            )
                            
        # 2. 车辆使用变量 y[k,v]: 是否使用车型v的第k辆车
        self.y_vars = {}
        for v_type, specs in self.vehicle_types.items():
            for k in range(specs['count']):
                self.y_vars[k,v_type] = self.model.addVar(
                    vtype=GRB.BINARY,
                    name=f'y_{k}_{v_type}'
                )
                
        # 3. 车型分配变量 z[i,v]: 订单i是否由车型v服务
        self.z_vars = {}
        for i in range(n):
            for v_type in self.vehicle_types:
                self.z_vars[i,v_type] = self.model.addVar(
                    vtype=GRB.BINARY,
                    name=f'z_{i}_{v_type}'
                )
                
        # 4. 时间变量 t[i,k,v]: 车型v的第k辆车访问节点i的时间
        self.t_vars = {}
        for v_type in self.vehicle_types:
            for k in range(self.vehicle_types[v_type]['count']):
                for i in range(-1, n):
                    self.t_vars[i,k,v_type] = self.model.addVar(
                        vtype=GRB.CONTINUOUS,
                        name=f't_{i}_{k}_{v_type}'
                    )
                    
        self.model.update()
        
    def _add_constraints(self):
        """添加约束条件"""
        n = self.instance.num_orders
        
        # 1. 每个订单只能被访问一次
        for i in range(n):
            self.model.addConstr(
                gp.quicksum(
                    self.z_vars[i,v_type] 
                    for v_type in self.vehicle_types
                ) == 1,
                f'visit_once_{i}'
            )
            
        # 2. 车辆容量约束
        for v_type, specs in self.vehicle_types.items():
            for k in range(specs['count']):
                self.model.addConstr(
                    gp.quicksum(
                        self.instance.get_order_demand(i) * 
                        gp.quicksum(
                            self.x_vars[i,j,k,v_type]
                            for j in range(-1, n) if j != i
                        )
                        for i in range(n)
                    ) <= specs['capacity'] * self.y_vars[k,v_type],
                    f'capacity_{k}_{v_type}'
                )
                
        # 3. 车辆流平衡约束
        for v_type in self.vehicle_types:
            for k in range(self.vehicle_types[v_type]['count']):
                for h in range(n):
                    self.model.addConstr(
                        gp.quicksum(
                            self.x_vars[i,h,k,v_type]
                            for i in range(-1, n) if i != h
                        ) ==
                        gp.quicksum(
                            self.x_vars[h,j,k,v_type]
                            for j in range(-1, n) if j != h
                        ),
                        f'flow_{h}_{k}_{v_type}'
                    )
                    
        # 4. 时间窗约束(如果有)
        if 'ready_time' in self.instance.orders_df.columns:
            M = 1000000  # Big-M
            for v_type in self.vehicle_types:
                for k in range(self.vehicle_types[v_type]['count']):
                    for i in range(n):
                        # 最早开始时间
                        ready_time = self.instance.orders_df.iloc[i]['ready_time']
                        self.model.addConstr(
                            self.t_vars[i,k,v_type] >= ready_time,
                            f'ready_time_{i}_{k}_{v_type}'
                        )
                        
                        # 最晚完成时间
                        due_time = self.instance.orders_df.iloc[i].get(
                            'due_time',
                            float('inf')
                        )
                        if due_time < float('inf'):
                            self.model.addConstr(
                                self.t_vars[i,k,v_type] <= due_time,
                                f'due_time_{i}_{k}_{v_type}'
                            )
                            
        # 5. 商户类型与车型匹配约束
        for m_type, order_indices in self.merchant_types.items():
            compatible_vehicles = self._get_compatible_vehicles(m_type)
            for i in order_indices:
                self.model.addConstr(
                    gp.quicksum(
                        self.z_vars[i,v_type]
                        for v_type in compatible_vehicles
                    ) == 1,
                    f'merchant_vehicle_match_{i}'
                )
                
    def _set_objective(self):
        """设置目标函数"""
        # 1. 固定成本
        fixed_cost = gp.quicksum(
            self.vehicle_types[v_type]['fixed_cost'] * self.y_vars[k,v_type]
            for v_type in self.vehicle_types
            for k in range(self.vehicle_types[v_type]['count'])
        )
        
        # 2. 运距成本
        n = self.instance.num_orders
        travel_cost = gp.quicksum(
            self.instance.get_road_distance(
                self.instance.get_order_location(i) if i >= 0 
                else self.instance.get_depot_location(0),
                self.instance.get_order_location(j) if j >= 0
                else self.instance.get_depot_location(0)
            ) * self.x_vars[i,j,k,v_type]
            for v_type in self.vehicle_types
            for k in range(self.vehicle_types[v_type]['count'])
            for i in range(-1, n)
            for j in range(-1, n)
            if i != j
        )
        
        # 设置目标
        self.model.setObjective(
            fixed_cost + travel_cost,
            GRB.MINIMIZE
        )
        
    def _config_solver(self):
        """配置求解器参数"""
        self.model.setParam('TimeLimit', self.time_limit)
        self.model.setParam('MIPGap', self.mip_gap)
        self.model.setParam('Threads', self.threads)
        self.model.setParam('OutputFlag', 1)
        
    def _build_solution(self) -> VRPSolution:
        """从Gurobi结果构建VRP解决方案"""
        solution = VRPSolution(self.instance)
        
        n = self.instance.num_orders
        for v_type in self.vehicle_types:
            for k in range(self.vehicle_types[v_type]['count']):
                if self.y_vars[k,v_type].X > 0.5:  # 使用了这辆车
                    route = []
                    current = -1  # 从仓库出发
                    
                    while True:
                        # 找下一个访问点
                        next_point = None
                        for j in range(-1, n):
                            if j != current:
                                if self.x_vars[current,j,k,v_type].X > 0.5:
                                    next_point = j
                                    break
                                    
                        if next_point is None or next_point == -1:
                            break
                            
                        route.append(next_point)
                        current = next_point
                        
                    if route:
                        solution.add_route(route)
                        solution.vehicle_assignments[len(solution.routes)-1] = {
                            'type': v_type,
                            'fixed_cost': self.vehicle_types[v_type]['fixed_cost']
                        }
                        
        return solution
        
    def _get_compatible_vehicles(self, merchant_type: str) -> List[str]:
        """获取与商户类型兼容的车型"""
        # 可以根据业务规则定制
        if merchant_type == '超市':
            return ['medium', 'large']
        elif merchant_type == '便利店':
            return ['small', 'medium']
        elif merchant_type == '仓储中心':
            return ['large']
        else:
            return list(self.vehicle_types.keys()) 
