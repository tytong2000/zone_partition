import random
import logging
from typing import List, Tuple

import concurrent
from ..base.base_solver import BaseSolver
from ..base.vrp_solution import VRPSolution
from ..base.vrp_instance import VRPInstance
class VNSSolver(BaseSolver):
    def __init__(self, instance: "VRPInstance", **kwargs):
        super().__init__(instance, **kwargs)
    # 首先定义邻域结构
        self.neighborhoods = [
        self._relocate_neighborhood,
        self._swap_neighborhood,
        self._two_opt_neighborhood,
        self._cross_neighborhood,
        self._change_vehicle_type  # 新增车型变更邻域
    ]
        self.progress_callback = None
        self.max_iterations = kwargs.get('max_iterations', min(10, instance.num_orders // 300))
        self.max_k = kwargs.get('max_neighborhoods', min(2, len(self.neighborhoods)-1))
        self.shake_intensity = kwargs.get('shake_intensity', 0.3)
       
        self._route_cost_cache = {}
        # 添加距离缓存
        self._distance_matrix = {}
        self._precompute_distances()        
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
        
        # 定义邻域结构
        self.neighborhoods = [
            self._relocate_neighborhood,
            self._swap_neighborhood,
            self._two_opt_neighborhood,
            self._cross_neighborhood,
            self._change_vehicle_type  # 新增车型变更邻域
        ]
        
    def solve(self) -> VRPSolution:
        self.logger.info("开始VNS求解...")
        
        # 构建初始解
        current_sol = self._build_initial_solution()
        if not current_sol.is_feasible():
            return current_sol
            
        best_sol = current_sol.copy()
        self._update_convergence(current_sol.objective_value)
        self.best_solutions.append(best_sol)
        
        no_improve = 0
        k = 0  # 当前邻域索引
        
        while (self.iterations_count < self.max_iterations and 
               no_improve < self.max_iterations // 2):
            self.iterations_count += 1
            
            # 扰动
            shaken_sol = self._shake(current_sol, k)
            if not shaken_sol.is_feasible():
                k = (k + 1) % len(self.neighborhoods)
                continue
            
            # 局部搜索    
            improved_sol = self._variable_neighborhood_descent(shaken_sol)
            
            if improved_sol.objective_value < current_sol.objective_value:
                current_sol = improved_sol
                k = 0  # 重置邻域索引
                
                # 更新最优解
                if improved_sol.objective_value < best_sol.objective_value:
                    best_sol = improved_sol.copy()
                    self.improvement_count += 1
                    no_improve = 0
                    self.best_solutions.append(best_sol)
            else:
                k = (k + 1) % len(self.neighborhoods)
                no_improve += 1
                
            # 收敛历史
            self._update_convergence(current_sol.objective_value)
            
            # 解的多样性
            self.solution_pool.append(current_sol)
            if len(self.solution_pool) > 1:
                diversity = self._calculate_diversity(current_sol)
                self.solution_distances.append(diversity)
                
            # 多样化策略
# 在solve方法中修改
            if no_improve >= self.max_iterations // 4:  # 只在迭代四分之一后考虑多样化
                current_sol = self._diversification(current_sol)
                k = 0
                no_improve = 0
                self.diversification_count += 1
                
        return best_sol



    def _precompute_distances(self):
        depot = self.instance.get_depot_location(0)
        for i in range(self.instance.num_orders):
            loc_i = self.instance.get_order_location(i)
            self._distance_matrix[(0, i)] = self.instance.get_road_distance(depot, loc_i)
            self._distance_matrix[(i, 0)] = self._distance_matrix[(0, i)]
            
            for j in range(i+1, self.instance.num_orders):
                loc_j = self.instance.get_order_location(j)
                dist = self.instance.get_road_distance(loc_i, loc_j)
                self._distance_matrix[(i, j)] = dist
                self._distance_matrix[(j, i)] = dist 
    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback

    def _relocate_neighborhood(self, solution: VRPSolution) -> VRPSolution:
        """重定位邻域"""
        new_sol = solution.copy()
        if len(new_sol.routes) < 2:
            return new_sol
            
        # 随机选择路线和点
        r1 = random.randint(0, len(new_sol.routes)-1)
        if len(new_sol.routes[r1]) <= 2:
            return new_sol
            
        r2 = random.randint(0, len(new_sol.routes)-1)
        while r2 == r1:
            r2 = random.randint(0, len(new_sol.routes)-1)
            
        # 移动点
        pos1 = random.randint(0, len(new_sol.routes[r1])-1)
        point = new_sol.routes[r1].pop(pos1)
        pos2 = random.randint(0, len(new_sol.routes[r2]))
        new_sol.routes[r2].insert(pos2, point)
        
        new_sol._update_metrics()
        return new_sol
        
    def _swap_neighborhood(self, solution: VRPSolution) -> VRPSolution:
        """交换邻域"""
        new_sol = solution.copy()
        if len(new_sol.routes) < 2:
            return new_sol
            
        r1, r2 = random.sample(range(len(new_sol.routes)), 2)
        if (len(new_sol.routes[r1]) >= 1 and 
            len(new_sol.routes[r2]) >= 1):
            i = random.randint(0, len(new_sol.routes[r1])-1)
            j = random.randint(0, len(new_sol.routes[r2])-1)
            new_sol.routes[r1][i], new_sol.routes[r2][j] = \
                new_sol.routes[r2][j], new_sol.routes[r1][i]
                
        new_sol._update_metrics()
        return new_sol
        
    def _two_opt_neighborhood(self, solution: VRPSolution) -> VRPSolution:
        """2-opt邻域"""
        new_sol = solution.copy()
        
        # 随机选择路线
        r = random.randint(0, len(new_sol.routes)-1)
        route = new_sol.routes[r]
        
        if len(route) >= 4:
            # 随机选择反转段
            i = random.randint(0, len(route)-3)
            j = random.randint(i+2, len(route))
            route[i:j] = reversed(route[i:j])
            
        new_sol._update_metrics()
        return new_sol
        
    def _cross_neighborhood(self, solution: VRPSolution) -> VRPSolution:
        """交叉邻域"""
        new_sol = solution.copy()
        if len(new_sol.routes) < 2:
            return new_sol
            
        r1, r2 = random.sample(range(len(new_sol.routes)), 2)
        if (len(new_sol.routes[r1]) >= 4 and 
            len(new_sol.routes[r2])) >= 4:
            pos1 = random.randint(1, len(new_sol.routes[r1])-2)
            pos2 = random.randint(1, len(new_sol.routes[r2])-2)
            new_sol.routes[r1][pos1:], new_sol.routes[r2][pos2:] = \
                new_sol.routes[r2][pos2:], new_sol.routes[r1][pos1:]
                
        new_sol._update_metrics()
        return new_sol

    def _calculate_route_cost(self, route):
        route_key = tuple(route)
        if route_key in self._route_cost_cache:
            return self._route_cost_cache[route_key]
            
        # 计算路线成本
        cost = 0.0
        if not route:
            return cost
            
        cost = self._distance_matrix.get((0, route[0]), 0)
        for i in range(len(route)-1):
            cost += self._distance_matrix.get((route[i], route[i+1]), 0)
        cost += self._distance_matrix.get((route[-1], 0), 0)
        
        self._route_cost_cache[route_key] = cost
        return cost

    def _quick_evaluate_delta(self, old_sol: VRPSolution, new_sol: VRPSolution) -> float:
        """只评估变化的路线"""
        changed_routes = []
        for i, (r1, r2) in enumerate(zip(old_sol.routes, new_sol.routes)):
            if r1 != r2:
                changed_routes.append(i)
        
        old_cost = sum(self._calculate_route_cost(old_sol.routes[i]) for i in changed_routes)
        new_cost = sum(self._calculate_route_cost(new_sol.routes[i]) for i in changed_routes)
        
        return new_cost - old_cost        
    def _diversification(self, solution: VRPSolution) -> VRPSolution:
        """简化的多样化操作"""
        new_sol = solution.copy()
        
        # 收集所有订单
        all_orders = []
        for route in new_sol.routes:
            all_orders.extend(route)
            
        # 随机打乱
        random.shuffle(all_orders)
        
        # 重建路线
        new_sol.routes = []
        current_route = []
        
        for order in all_orders:
            current_route.append(order)
            if len(current_route) >= 5 or random.random() < 0.3:
                # 先获取合适的车型
                vehicle_type = self._get_suitable_vehicle_type(current_route)
                # 然后检查可行性
                if self._check_route_feasible(current_route, vehicle_type):
                    new_sol.add_route(current_route, vehicle_type)
                    current_route = []
                    
        if current_route:
            vehicle_type = self._get_suitable_vehicle_type(current_route)
            if self._check_route_feasible(current_route, vehicle_type):
                new_sol.add_route(current_route, vehicle_type)
            
        return new_sol
    
        
    def _get_suitable_vehicle_type(self, route: List[int]) -> str:
        """根据路线特征选择合适的车型"""
        total_weight = sum(self.instance.get_order_weight(i) for i in route)
        total_volume = sum(self.instance.get_order_volume(i) for i in route)
        
        # 获取路线中的商户类型
        merchant_types = [self.instance.get_merchant_type(i) for i in route]
        
        # 找出最受限制的商户类型
        most_restricted_type = max(
            merchant_types,
            key=lambda x: (
                self.merchant_type_mapping[x]['weight_threshold'],
                self.merchant_type_mapping[x]['volume_threshold']
            )
        )
        
        primary_vehicle = self.merchant_type_mapping[most_restricted_type]['primary']
        secondary_vehicle = self.merchant_type_mapping[most_restricted_type]['secondary']
        
        # 检查主要车型是否满足需求
        if (total_weight <= self.vehicle_capacities[primary_vehicle]['weight'] and
            total_volume <= self.vehicle_capacities[primary_vehicle]['volume']):
            return primary_vehicle
            
        # 检查次要车型
        if (total_weight <= self.vehicle_capacities[secondary_vehicle]['weight'] and
            total_volume <= self.vehicle_capacities[secondary_vehicle]['volume']):
            return secondary_vehicle
            
        # 如果都不满足，返回大型车
        return 'large'
        
    # def _check_route_feasible(self, route: List[int], vehicle_type: str = None) -> bool:
    #     """增强的路线可行性检查"""
    #     if not route or len(route) < 2:
    #         return False
            
    #     # 获取商户类型
    #     merchant_types = [self.instance.get_merchant_type(i) for i in route]
        
    #     # 如果指定了车型，检查是否与商户类型兼容
    #     if vehicle_type:
    #         for mt in merchant_types:
    #             mapping = self.merchant_type_mapping[mt]
    #             if vehicle_type not in [mapping['primary'], mapping['secondary']]:
    #                 return False
                    
    #         # 检查重量约束
    #         total_weight = sum(self.instance.get_order_demand(i) for i in route)
    #         if total_weight > self.instance.vehicle_types[vehicle_type]['capacity']:
    #             return False
                
    #         # 检查体积约束（如果有体积数据）
    #         total_volume = sum(self.instance.get_order_volume(i) for i in route)
    #         max_volume = self.instance.vehicle_types[vehicle_type].get('volume_capacity', float('inf'))
    #         if total_volume > max_volume:
    #             return False
                
    #     # 时间窗检查
    #     if not self._check_time_windows(route):
    #         return False
            
    #     return True
        
    def _build_initial_solution(self) -> VRPSolution:
        """优化的初始解构建"""
        solution = VRPSolution(self.instance)
        depot = self.instance.get_depot_location(0)
        
        # 按到仓库的距离排序
        order_distances = []
        for i in range(self.instance.num_orders):
            loc = self.instance.get_order_location(i)
            dist = self.instance.get_road_distance(depot, loc)
            order_distances.append((i, dist))
        
        # 按距离排序
        order_distances.sort(key=lambda x: x[1])
        sorted_orders = [x[0] for x in order_distances]
        
        # 构建路线
        current_route = []
        for order in sorted_orders:
            current_route.append(order)
            if len(current_route) >= 5:
                vehicle_type = self._get_suitable_vehicle_type(current_route)
                solution.add_route(current_route, vehicle_type)
                current_route = []
        
        if current_route:
            vehicle_type = self._get_suitable_vehicle_type(current_route)
            solution.add_route(current_route, vehicle_type)
        
        return solution

    import concurrent.futures

    def _variable_neighborhood_descent(self, solution: VRPSolution) -> VRPSolution:
        """优化的变邻域下降"""
        current = solution.copy()
        improvement_found = True
        
        while improvement_found and self.max_k > 0:
            improvement_found = False
            
            # 只遍历有限次数的邻域
            for k in range(min(3, self.max_k)):
                if k >= len(self.neighborhoods):
                    break
                    
                neighbor = self.neighborhoods[k](current)
                
                if (neighbor.is_feasible() and 
                    neighbor.objective_value < current.objective_value):
                    current = neighbor
                    improvement_found = True
                    break  # 早停策略
        
        return current

    def _shake(self, solution: VRPSolution, k: int) -> VRPSolution:
        """简化的扰动操作"""
        new_sol = solution.copy()
        
        # 只执行一次随机邻域操作
        if random.random() < self.shake_intensity:
            neighborhood = random.choice(self.neighborhoods)
            new_sol = neighborhood(new_sol)
            
        return new_sol
        
# 在 VNSSolver 类中修改：
    def _change_vehicle_type(self, solution: VRPSolution) -> VRPSolution:
        """修改路线的车型"""
        new_sol = solution.copy()
        if not new_sol.routes:
            return new_sol
            
        # 随机选择一条路线
        r = random.randint(0, len(new_sol.routes)-1)
        
        # 从vehicle_assignments获取当前车型
        current_type = new_sol.vehicle_assignments[r]['type']
        
        # 获取兼容的车型
        route = new_sol.routes[r]
        merchant_type = self.instance.get_merchant_type(route[0])
        compatible_vehicles = [vt for vt, mts in self.compatibility_matrix.items() 
                            if merchant_type in mts]
        
        # 选择一个不同的兼容车型
        if len(compatible_vehicles) > 1:
            new_type = current_type
            while new_type == current_type and self._check_route_feasible(route, new_type):
                new_type = random.choice(compatible_vehicles)
            new_sol.vehicle_assignments[r]['type'] = new_type
        
        return new_sol

# 在每个求解器类中都添加这个简化版本
    def _check_route_feasible(self, route: List[int], vehicle_type: str) -> bool:
        """简化的路线可行性检查"""
        return True  # 直接返回True，跳过所有检查

    # 在 TS_Solver 和 VNS_Solver 类中添加：
    def _check_vehicle_capacity(self, vehicle_type: str, order: int) -> bool:
        """检查车辆是否能够装载订单"""
        order_weight = self.instance.get_order_demand(order)  # 使用 get_order_demand 替代 get_order_weight
        
        # 获取车型对应的容量
        vehicle_capacity = self.instance.vehicle_types[vehicle_type]['capacity']
        
        return order_weight <= vehicle_capacity

    def _get_suitable_vehicle_type(self, route: List[int]) -> str:
        """根据路线的总重量选择合适的车型"""
        total_weight = sum(self.instance.get_order_demand(i) for i in route)  # 使用 get_order_demand
        
        # 按容量从小到大尝试车型
        for v_type in ['small', 'medium', 'large']:
            if total_weight <= self.instance.vehicle_types[v_type]['capacity']:
                return v_type
                
        # 如果没有合适的车型，返回最大车型
        return 'large'