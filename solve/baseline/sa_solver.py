import random
import math
import time
import numpy as np
from solve.base.base_solver import BaseSolver
from solve.base.vrp_solution import VRPSolution
from solve.base.vrp_instance import VRPInstance
from typing import List, Dict, Tuple, Optional

class SASolver(BaseSolver):
    def __init__(self, instance: "VRPInstance", **kwargs):
        super().__init__(instance, **kwargs)
        # 基本SA参数
        self.progress_callback = None
        self.initial_temp = kwargs.get('initial_temp', 500.0)
        self.cooling_rate = kwargs.get('cooling_rate', 0.95)
        self.iterations = kwargs.get('iterations', 1000)
        self.max_no_improve = kwargs.get('max_no_improve', 100)

        # 车辆类型相关配置
        self.vehicle_types = ['small', 'medium', 'large']
        self.vehicle_capacities = {
            'small': {'weight': 1000, 'volume': 3},
            'medium': {'weight': 3000, 'volume': 10},
            'large': {'weight': 5000, 'volume': 20}
        }
        
        # 商户类型映射
        self.merchant_type_mapping = {
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
        }        
        # 预处理并缓存车型-商户类型兼容性矩阵
        self.compatibility_matrix = self._build_compatibility_matrix()
        
        # 预处理并缓存每个订单的可行车型
        self.order_vehicle_cache = self._preprocess_order_vehicles()
        
        # 性能监控
        self.operation_times = {
            'neighbor_gen': [],
            'feasibility_check': [],
            'objective_calc': []
        }
    def set_progress_callback(self, callback):    # 添加这个方法
        """设置进度回调函数"""
        self.progress_callback = callback
        
    def _build_compatibility_matrix(self) -> Dict[str, List[str]]:
        """预计算车型和商户类型的兼容性矩阵"""
        compatibility = {}
        for v_type in self.instance.vehicle_types.keys():
            compatibility[v_type] = []
            v_cap = self.instance.vehicle_types[v_type].get('capacity', 0)
            
            for m_type in ['convenience', 'supermarket', 'mall']:
                # 简化的兼容性检查规则
                if (m_type == 'convenience' and v_cap >= 1500) or \
                   (m_type == 'supermarket' and v_cap >= 3000) or \
                   (m_type == 'mall' and v_cap >= 5000):
                    compatibility[v_type].append(m_type)
        
        return compatibility

    def _preprocess_order_vehicles(self) -> Dict[int, List[str]]:
        """预处理每个订单的可行车型列表"""
        order_vehicles = {}
        for order_id in range(self.instance.num_orders):
            m_type = self.instance.get_merchant_type(order_id)
            demand = self.instance.get_order_demand(order_id)
            
            # 找出所有满足容量要求的车型
            feasible_vehicles = []
            for v_type, specs in self.instance.vehicle_types.items():
                if specs['capacity'] >= demand and m_type in self.compatibility_matrix[v_type]:
                    feasible_vehicles.append(v_type)
            
            order_vehicles[order_id] = feasible_vehicles
            
        return order_vehicles

    def solve(self) -> VRPSolution:
        """优化版本的SA求解过程"""
        self.logger.info("开始优化版SimulatedAnnealing求解...")
        
        # 快速构建初始解
        current_sol = self._build_initial_solution()
        if not current_sol.is_feasible():
            return current_sol
            
        best_sol = current_sol.copy()
        best_obj = current_sol.objective_value
        
        temperature = self.initial_temp
        no_improve = 0
        
        # 主循环
        while temperature > 1e-6 and self.iterations_count < self.iterations:
            self.iterations_count += 1
            improved = False
            
            # 批量生成多个邻域解
            for _ in range(5):  # 每次迭代尝试5个邻域
                t0 = time.time()
                neighbor_sol = self._generate_neighbor(current_sol)
                self.operation_times['neighbor_gen'].append(time.time() - t0)
                
                if not neighbor_sol.is_feasible():
                    continue
                    
                # 快速评估解的质量
                t0 = time.time()
                delta = neighbor_sol.objective_value - current_sol.objective_value
                self.operation_times['objective_calc'].append(time.time() - t0)
                
                # Metropolis准则
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current_sol = neighbor_sol
                    
                    # 更新最优解
                    if neighbor_sol.objective_value < best_obj:
                        best_sol = neighbor_sol.copy()
                        best_obj = neighbor_sol.objective_value
                        improved = True
                        no_improve = 0
                    break  # 找到更好的解就提前退出内循环
            
            if not improved:
                no_improve += 1
            
            # 提前终止
            if no_improve >= self.max_no_improve:
                break
                
            # 降温
            temperature *= self.cooling_rate
            
            # 每100次迭代输出一次进度
            if self.iterations_count % 100 == 0:
                self.logger.info(f"迭代次数: {self.iterations_count}, "
                               f"温度: {temperature:.2f}, "
                               f"当前最优: {best_obj:.2f}")
        
        # 输出性能统计
        self._log_performance_stats()
        
        return best_sol

    def _build_initial_solution(self) -> VRPSolution:
        """修复后的初始解构建"""
        solution = VRPSolution(self.instance)
        
        # 按需求量排序后分配
        orders = [(i, self.instance.get_order_demand(i)) 
                for i in range(self.instance.num_orders)]
        orders.sort(key=lambda x: x[1], reverse=True)
        
        current_route = []
        current_load = 0
        current_type = None
        
        for order_id, demand in orders:
            # 直接获取该订单的合适车型，不使用缓存
            if not current_type:
                merchant_type = self.instance.get_merchant_type(order_id)
                current_type = self._get_initial_vehicle_type(merchant_type, demand)
            
            # 检查容量约束
            vehicle_capacity = self.instance.vehicle_types[current_type]['capacity']
            if current_load + demand <= vehicle_capacity:
                current_route.append(order_id)
                current_load += demand
            else:
                # 当前路线已满,添加到解中
                if current_route:
                    solution.add_route(current_route, vehicle_type=current_type)
                # 开始新路线
                current_route = [order_id]
                current_load = demand
                merchant_type = self.instance.get_merchant_type(order_id)
                current_type = self._get_initial_vehicle_type(merchant_type, demand)
        
        # 添加最后一条路线
        if current_route:
            solution.add_route(current_route, vehicle_type=current_type)
            
        return solution

    def _get_initial_vehicle_type(self, merchant_type: str, demand: float) -> str:
        """根据商户类型和需求量确定初始车型"""
        if merchant_type not in self.merchant_type_mapping:
            return 'medium'  # 默认使用中型车
            
        mapping = self.merchant_type_mapping[merchant_type]
        primary_type = mapping['primary']
        
        # 检查主要车型是否满足需求
        if demand <= self.vehicle_capacities[primary_type]['weight']:
            return primary_type
        
        # 如果主要车型不满足，使用次要车型
        return mapping['secondary']

    def _generate_neighbor(self, current_sol: VRPSolution) -> VRPSolution:
        """优化的邻域生成"""
        new_sol = current_sol.copy()
        if len(new_sol.routes) < 2:
            return new_sol
            
        # 随机选择操作类型,但给不同操作分配不同权重
        op_weights = {'swap': 0.4, 'relocate': 0.3, 'change_vehicle': 0.3}
        operation = random.choices(list(op_weights.keys()), 
                                 list(op_weights.values()))[0]
        
        if operation == 'swap':
            # 交换操作
            r1, r2 = random.sample(range(len(new_sol.routes)), 2)
            if len(new_sol.routes[r1]) > 0 and len(new_sol.routes[r2]) > 0:
                i = random.randint(0, len(new_sol.routes[r1])-1)
                j = random.randint(0, len(new_sol.routes[r2])-1)
                
                # 快速检查交换可行性
                order1, order2 = new_sol.routes[r1][i], new_sol.routes[r2][j]
                if self._quick_swap_check(order1, order2, 
                                        new_sol.vehicle_assignments[r1]['type'],
                                        new_sol.vehicle_assignments[r2]['type']):
                    new_sol.routes[r1][i], new_sol.routes[r2][j] = order2, order1
        
        elif operation == 'relocate':
            # 重定位操作
            if len(new_sol.routes) >= 2:
                r1 = random.randint(0, len(new_sol.routes)-1)
                if len(new_sol.routes[r1]) > 1:  # 确保不会清空路线
                    r2 = random.randint(0, len(new_sol.routes)-1)
                    while r2 == r1:
                        r2 = random.randint(0, len(new_sol.routes)-1)
                        
                    i = random.randint(0, len(new_sol.routes[r1])-1)
                    order = new_sol.routes[r1][i]
                    
                    # 快速检查重定位可行性
                    if self._quick_relocation_check(order, 
                                                  new_sol.vehicle_assignments[r2]['type']):
                        new_sol.routes[r1].pop(i)
                        insert_pos = random.randint(0, len(new_sol.routes[r2]))
                        new_sol.routes[r2].insert(insert_pos, order)
        
        else:  # change_vehicle
            # 改变车型
            r = random.randint(0, len(new_sol.routes)-1)
            current_type = new_sol.vehicle_assignments[r]['type']
            route = new_sol.routes[r]
            
            # 获取路线上所有订单的共同可行车型
            feasible_types = set(self.order_vehicle_cache[route[0]])
            for order in route[1:]:
                feasible_types &= set(self.order_vehicle_cache[order])
            
            # 随机选择一个不同的可行车型
            feasible_types.discard(current_type)
            if feasible_types:
                new_type = random.choice(list(feasible_types))
                new_sol.vehicle_assignments[r]['type'] = new_type
        
        return new_sol

    def _quick_swap_check(self, order1: int, order2: int, 
                         type1: str, type2: str) -> bool:
        """快速检查交换可行性"""
        return (type1 in self.order_vehicle_cache[order2] and 
                type2 in self.order_vehicle_cache[order1])

    def _quick_relocation_check(self, order: int, target_type: str) -> bool:
        """快速检查重定位可行性"""
        return target_type in self.order_vehicle_cache[order]

    def _log_performance_stats(self):
        """输出性能统计"""
        if self.operation_times['neighbor_gen']:
            avg_neighbor = np.mean(self.operation_times['neighbor_gen'])
            avg_feasibility = np.mean(self.operation_times['feasibility_check'])
            avg_objective = np.mean(self.operation_times['objective_calc'])
            
            self.logger.info(f"性能统计:\n"
                           f"平均邻域生成时间: {avg_neighbor:.4f}s\n"
                           f"平均可行性检查时间: {avg_feasibility:.4f}s\n"
                           f"平均目标值计算时间: {avg_objective:.4f}s")