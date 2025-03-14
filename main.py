#!/usr/bin/env python
# coding: utf-8
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
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon
import networkx as nx
from tqdm import tqdm
from typing import Any
import seaborn as sns  
import math
from jinja2 import Template
from solvers.vrp_instance import RoadNetworkBuilder, EnhancedZonePartitioner, split_merchants_by_zone, run_facility_location, generate_experiment_report, SolverConfig, configure_logging, load_and_prepare_merchant_data
from solvers.vrp_instance import VRPInstance, FacilityLocationSolver, VRPStatisticalAnalyzer
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from solvers.vrp_instance import AdaptiveMetrics
import itertools

@dataclass
class AdvancedFLPConfig:
    fixed_cost: float = 1000.0
    operation_cost: float = 2.0
    coverage_radius: float = 50.0
    multi_levels: List[Tuple[int,int]] = None
    use_partial_coverage: bool = False
    coverage_penalty_factor: float = 1.5
    use_multilevel_capacity: bool = False

@dataclass 
class ZoneFLPSolverConfig:
    use_gurobi: bool = True
    gurobi_time_limit: int = 300
    gurobi_threads: int = 0
    random_seed: int = 42
    use_backup_ga: bool = True
    ga_population: int = 50
    ga_generations: int = 50
    ga_crossover: float = 0.7
    ga_mutation: float = 0.2

# 实验配置类
@dataclass 
class ExperimentConfig:
    # 基础配置
    excel_path: str
    city_path: str 
    road_path: str
    output_dir: str
    
    # 实验参数
    vehicle_capacity: float = 5000.0
    max_route_time: float = 12.0
    random_seed: int = 42
    
    # 并行设置
    parallel_evaluation: bool = True
    max_workers: Optional[int] = None
    
    # 分区参数
    min_clusters: int = 30
    max_clusters: int = 100
    road_buffer_distance: float = 1000.0
    
    # 算法参数
    # ClarkeWright
    cw_parallel_savings: bool = True
    cw_time_window_enforce: bool = False
    
    # SimulatedAnnealing  
    sa_initial_temp: float = 1000.0
    sa_cooling_rate: float = 0.95
    sa_iterations: int = 100
    
    # TabuSearch
    ts_tabu_size: int = 10
    ts_max_iterations: int = 50
    ts_neighborhood_size: int = 20
    
    # VNS
    vns_max_iterations: int = 30
    vns_max_neighborhoods: int = 3
    vns_shake_intensity: float = 0.3

class VRPSolution:
    def __init__(self, instance: "VRPInstance"=None):
        self.instance = instance
        self.routes = []
        self.objective_value = float('inf')
        self.total_distance = 0.0
        self.total_time = 0.0
        self.total_cost = 0.0
        # 添加 vehicle_assignments 属性
        self.vehicle_assignments = {}

    def _update_metrics(self):
        if not self.instance:
            return
        dist_sum = 0.0
        fixed_cost = 0.0

        for i, route in enumerate(self.routes):
            if not route:
                continue
                
            # 确保 vehicle_assignments 中有对应的记录
            if i not in self.vehicle_assignments:
                self.vehicle_assignments[i] = {
                    'type': 'default',
                    'fixed_cost': 1000.0
                }
            
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
                
            # 返回仓库
            dist_sum += self.instance.get_road_distance(prev_loc, wh_loc)
            
        self.total_distance = dist_sum
        self.total_time = dist_sum / 30000.0
        self.total_cost = dist_sum + fixed_cost
        self.objective_value = self.total_cost

    def is_feasible(self):
        return (len(self.routes)>0) and (self.objective_value< float('inf'))

    def copy(self):
        new_ = VRPSolution(self.instance)
        new_.routes= [r.copy() for r in self.routes]
        new_.objective_value= self.objective_value
        new_.total_distance= self.total_distance
        new_.total_time= self.total_time
        return new_

    def add_route(self, route:List[int]):
        self.routes.append(route)
        self._update_metrics()

class BaseSolver:
    def __init__(self, instance:"VRPInstance"):
        self.instance= instance
    def solve(self)-> Optional[VRPSolution]:
        raise NotImplementedError    

class VRPExperiment:
    """VRP实验框架类"""
    
    def __init__(self, config: SolverConfig, zone_map: Dict, warehouses: List, 
                 partitioner: EnhancedZonePartitioner, road_graph: nx.Graph):
        # 添加实验控制
        self.experiment_flags = {
            'run_baseline': True,
            'run_hybrid': True, 
            'run_adaptive': True,
            'run_enhanced': True  # 只运行新增实验
        }        
        self.config = config
        self.zone_map = zone_map
        self.warehouses = warehouses
        self.partitioner = partitioner
        self.road_graph = road_graph
        
        # 存储实验结果
        self.baseline_results = {}
        self.hybrid_results = {}
        self.adaptive_results = {}
        
        self.logger = logging.getLogger("VRPExperiment")

        # 新增评估指标
        self.advanced_metrics = {
            'vehicle_utilization': {},  # 车辆利用率
            'path_diversity': {},      # 路径多样性
            'cost_breakdown': {}       # 成本构成
        }
        
    def _calculate_advanced_metrics(self, solution: VRPSolution) -> Dict:
        metrics = {}
        # 计算车辆利用率
        utilization = {}
        for v_type in solution.instance.vehicle_types:
            used = sum(1 for v in solution.vehicle_assignments.values()
                      if v['type'] == v_type)
            total = solution.instance.vehicle_types[v_type]['count']
            utilization[v_type] = used / total if total > 0 else 0
            
        metrics['vehicle_utilization'] = utilization
        
        # 计算路径多样性
        path_lengths = []
        for route in solution.routes:
            if len(route) >= 2:
                length = self._calc_single_route_length(solution, route)
                path_lengths.append(length)
                
        metrics['path_diversity'] = {
            'std_dev': np.std(path_lengths) if path_lengths else 0,
            'max_min_ratio': max(path_lengths) / min(path_lengths) if path_lengths else 1
        }
        
        # 计算成本构成
        total_fixed = sum(v['fixed_cost'] for v in solution.vehicle_assignments.values())
        metrics['cost_breakdown'] = {
            'fixed_cost': total_fixed,
            'variable_cost': solution.total_distance,
            'fixed_cost_ratio': total_fixed / solution.total_cost if solution.total_cost > 0 else 0
        }
        
        return metrics

    def _calc_single_route_length(self, sol: "VRPSolution", route: List[int]) -> float:
        """根据 sol.instance 中的距离函数，计算整条路线 + 返回仓库 的总距离。"""
        if not route:
            return 0.0
        wh_idx = sol.instance.assignments.get(route[0], 0)  # 该路线绑定的仓库
        wh_loc = sol.instance.get_depot_location(wh_idx)
        dist_sum = 0.0
        prev_loc = wh_loc
        for oid in route:
            cur_loc = sol.instance.get_order_location(oid)
            dist_sum += sol.instance.get_road_distance(prev_loc, cur_loc)
            prev_loc = cur_loc
        # 返回仓库
        dist_sum += sol.instance.get_road_distance(prev_loc, wh_loc)
        return dist_sum
    
    def _create_vrp_instance(self, merchants_df: pd.DataFrame) -> VRPInstance:
        """创建VRP实例"""
        return VRPInstance(
            orders_df=merchants_df,
            road_graph=self.road_graph,
            num_warehouses=1,
            vehicle_capacity=self.config.vehicle_capacity,
            max_route_time=self.config.max_route_time,
            max_search_distance=1e5,
            selected_warehouses=self.warehouses,
        )
    
    def run_baseline_experiment(self, zone_id: str, algo_code: str) -> Optional[Dict]:
        """运行基线算法实验"""
        if zone_id not in self.zone_map:
            return None
            
        merchants_df = self.zone_map[zone_id]
        # ==========【粘贴开始】==========
        # 若 merchants_df 并非 DataFrame，则尝试转一下
        if not isinstance(merchants_df, pd.DataFrame):
            try:
                merchants_df = pd.DataFrame(merchants_df)
                self.logger.debug(f"[{zone_id}] 数据被强制转成 DataFrame: shape={merchants_df.shape}")
            except Exception as exc:
                self.logger.error(f"[{zone_id}] 无法把数据转成 DataFrame: {exc}")
                return None
        # ==========【粘贴结束】==========

        if len(merchants_df) == 0:
            return None
            
        try:
            instance = self._create_vrp_instance(merchants_df)
            solver = self._create_solver(algo_code, instance)
            
            t0 = time.time()
            solution = solver.solve()
            t1 = time.time()
            
            if solution and solution.is_feasible():
                return {
                    'objective': solution.objective_value,
                    'distance': solution.total_distance,
                    'time': t1 - t0,
                    'num_routes': len(solution.routes),
                    'routes': [[int(i) for i in r] for r in solution.routes]
                }
        except Exception as e:
            self.logger.error(f"[{zone_id}] {algo_code} 实验失败: {str(e)}")
            
        return None
        
    def run_hybrid_experiment(self, zone_id: str, first_stage: str, second_stage: str) -> Optional[Dict]:
        if zone_id not in self.zone_map:
            return None
                
        merchants_df = self.zone_map[zone_id]
        if not isinstance(merchants_df, pd.DataFrame):
            try:
                merchants_df = pd.DataFrame(merchants_df)
                self.logger.debug(f"[{zone_id}] 数据被强制转成 DataFrame: shape={merchants_df.shape}")
            except Exception as exc:
                self.logger.error(f"[{zone_id}] 无法把数据转成 DataFrame: {exc}")
                return None
        
        if len(merchants_df) == 0:
            return None
                
        try:
            instance = self._create_vrp_instance(merchants_df)
            
            # 第一阶段
            solver1 = self._create_solver(first_stage, instance)
            initial_sol = solver1.solve()
            
            if not initial_sol or not initial_sol.is_feasible():
                self.logger.warning(f"[{zone_id}] {first_stage} 第一阶段无可行解")
                return None
                    
            # 第二阶段
            solver2 = self._create_solver(second_stage, instance, initial_solution=initial_sol)
            t0 = time.time()
            final_sol = solver2.solve()
            t1 = time.time()
            
            if final_sol and final_sol.is_feasible():
                improvement = (initial_sol.objective_value - final_sol.objective_value) / initial_sol.objective_value
                return {
                    'objective': final_sol.objective_value,
                    'distance': final_sol.total_distance,
                    'time': t1 - t0,
                    'num_routes': len(final_sol.routes),
                    'routes': [[int(i) for i in r] for r in final_sol.routes],
                    'improvement': improvement
                }
            else:
                self.logger.warning(f"[{zone_id}] {second_stage} 第二阶段无可行解")
        except Exception as e:
            self.logger.error(f"[{zone_id}] {first_stage}_{second_stage} 实验失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
                
        return None

    def _create_solver(self, solver_type: str, instance: VRPInstance, **kwargs) -> BaseSolver:
        """创建基础求解器"""
        if solver_type == "CW":
            return ClarkeWrightSolver(instance, **kwargs) 
        elif solver_type == "SA":
            return SimulatedAnnealingSolver(instance, **kwargs)
        elif solver_type == "TS":
            return TabuSearchSolver(instance, **kwargs)
        elif solver_type == "VNS":
            return VNSSolver(instance, **kwargs)
        else:
            raise ValueError(f"未知的求解器类型: {solver_type}")

    def _create_adaptive_solver(self, solver_type: str, instance: "VRPInstance", 
                            adaptive_params: Dict, initial_solution=None) -> BaseSolver:
        if solver_type == "TS":
            solver = AdaptiveTabuSearchSolver(
                instance=instance,
                adaptive_params=adaptive_params, 
                initial_solution=initial_solution  # 添加这个参数传递
            )
            return solver
        elif solver_type == "VNS":
            solver = AdaptiveVNSSolver(
                instance=instance,
                adaptive_params=adaptive_params,
                initial_solution=initial_solution  # 添加这个参数传递 
            )
            return solver
        else:
            raise ValueError(f"不支持的自适应求解器类型: {solver_type}")
        
    def run_adaptive_experiment(self, zone_id: str, algo_type: str, adaptive_params: Dict) -> Optional[Dict]:
        """运行自适应参数实验"""
        if zone_id not in self.zone_map:
            return None
            
        merchants_df = self.zone_map[zone_id]
        if len(merchants_df) == 0:
            return None
            
        try:
            instance = self._create_vrp_instance(merchants_df)
            
            # 处理混合算法情况
            if algo_type in ["CW_TS", "SA_VNS"]:
                first_stage = adaptive_params["first_stage"]
                second_stage = adaptive_params["second_stage"]
                
                # 第一阶段
                solver1 = self._create_solver(first_stage, instance)
                initial_sol = solver1.solve()
                
                if not initial_sol or not initial_sol.is_feasible():
                    self.logger.warning(f"[{zone_id}] {first_stage} 第一阶段无可行解")
                    return None
                    
                # 第二阶段使用自适应版本
                t0 = time.time()
                solver2 = self._create_adaptive_solver(
                    second_stage,
                    instance,
                    adaptive_params,
                    initial_solution=initial_sol
                )
                final_sol = solver2.solve()
                t1 = time.time()
                
                if final_sol and final_sol.is_feasible():
                    improvement = (initial_sol.objective_value - final_sol.objective_value) / initial_sol.objective_value
                    return {
                        'objective': final_sol.objective_value,
                        'distance': final_sol.total_distance,
                        'time': t1 - t0,
                        'num_routes': len(final_sol.routes),
                        'routes': [[int(i) for i in r] for r in final_sol.routes],
                        'improvement': improvement,
                        'parameter_history': solver2.get_parameter_history()
                    }
            else:
                # 单一算法的自适应版本
                solver = self._create_adaptive_solver(algo_type, instance, adaptive_params)
                t0 = time.time()
                solution = solver.solve()
                t1 = time.time()
                
                if solution and solution.is_feasible():
                    return {
                        'objective': solution.objective_value,
                        'distance': solution.total_distance,
                        'time': t1 - t0,
                        'num_routes': len(solution.routes),
                        'routes': [[int(i) for i in r] for r in solution.routes],
                        'parameter_history': solver.get_parameter_history()
                    }
        except Exception as e:
            self.logger.error(f"[{zone_id}] Adaptive {algo_type} 实验失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
        return None
        
    # ==========【粘贴开始】==========
    def save_results(self):
        os.makedirs(os.path.join(self.config.output_dir, 'results'), exist_ok=True)
        
        # 把 baseline_results 转成行列表
        baseline_rows = []
        for zone_id, algo_dict in self.baseline_results.items():
            for algo_code, metrics in algo_dict.items():
                row = {
                    "zone_id": zone_id,
                    "algorithm": algo_code,
                    "objective": metrics.get("objective", 0),
                    "distance": metrics.get("distance", 0),
                    "time": metrics.get("time", 0),
                    "num_routes": metrics.get("num_routes", 0)
                }
                baseline_rows.append(row)
        
        # 写 CSV
        baseline_csv_path = os.path.join(self.config.output_dir, 'results', 'baseline_results.csv')
        df_baseline = pd.DataFrame(baseline_rows)
        df_baseline.to_csv(baseline_csv_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"基线结果已保存为CSV: {baseline_csv_path}")
    
    # ===================【补丁：在 VRPExperiment 类内添加】===================

    def run_enhanced_experiment(self, zone_id: str, exp_config: Dict) -> Optional[Dict]:
        """
        处理多车辆或新增功能的三种配置：
        - 单阶段（exp_config['base_algorithm']）      # 比如多车型 TS
        - 两阶段（exp_config['first_stage', 'second_stage']） # 比如多车型 CW->TS
        - 自适应（exp_config['algorithm'] in ['TS-ADP','VNS-ADP']） # 比如多车型 TS-ADP
        与 run_baseline_experiment / run_hybrid_experiment / run_adaptive_experiment 类似，
        最终返回一个 metrics 字典或 None。
        """

        # ========== 0. 先检查路区是否存在 ========== 
        if zone_id not in self.zone_map:
            return None

        merchants_df = self.zone_map[zone_id]
        if not isinstance(merchants_df, pd.DataFrame):
            try:
                merchants_df = pd.DataFrame(merchants_df)
                self.logger.debug(f"[{zone_id}] 数据被强制转成 DataFrame: shape={merchants_df.shape}")
            except Exception as exc:
                self.logger.error(f"[{zone_id}] 无法把数据转成 DataFrame: {exc}")
                return None

        if len(merchants_df) == 0:
            return None

        try:
            # ========== 1. 创建 VRP 实例 ==========
            instance = self._create_vrp_instance(merchants_df)

            # 如果有多车型需求，就把 exp_config['vehicle_types'] 注入到 instance 或其相关字段
            if 'vehicle_types' in exp_config:
                instance.vehicle_types = exp_config['vehicle_types']
            else:
                # 没给就用默认单车型
                instance.vehicle_types = {
                    'default': {'capacity': instance.vehicle_capacity, 'fixed_cost': 1000, 'count': 1}
                }

            # ========== 2. 判断exp_config，分三类场景 ==========
            if 'base_algorithm' in exp_config:
                # --- A: 多车型单阶段算法（如多车型TS） ---
                base_algo = exp_config['base_algorithm']
                solver = self._create_solver(base_algo, instance, **exp_config)

                t0 = time.time()
                solution = solver.solve()
                t1 = time.time()

                if solution and solution.is_feasible():
                    adv_metrics = self._calculate_advanced_metrics(solution)
                    return {
                        'objective': solution.objective_value,
                        'distance': solution.total_distance,
                        'time': t1 - t0,
                        'num_routes': len(solution.routes),
                        'routes': [[int(i) for i in r] for r in solution.routes],
                        'adv_metrics': adv_metrics
                    }
                else:
                    self.logger.warning(f"[{zone_id}] 多车型 {base_algo} 无可行解")
                    return None

            elif 'first_stage' in exp_config and 'second_stage' in exp_config:
                # --- B: 多车型混合算法（如多车型 CW->TS） ---
                fs = exp_config['first_stage']
                ss = exp_config['second_stage']

                # 第一阶段
                solver1 = self._create_solver(fs, instance, **exp_config)
                initial_sol = solver1.solve()
                if (not initial_sol) or (not initial_sol.is_feasible()):
                    self.logger.warning(f"[{zone_id}] {fs} 第一阶段无可行解")
                    return None

                # 第二阶段
                solver2 = self._create_solver(ss, instance, initial_solution=initial_sol, **exp_config)
                t0 = time.time()
                final_sol = solver2.solve()
                t1 = time.time()

                if final_sol and final_sol.is_feasible():
                    improvement = (initial_sol.objective_value - final_sol.objective_value) / initial_sol.objective_value
                    adv_metrics = self._calculate_advanced_metrics(final_sol)
                    return {
                        'objective': final_sol.objective_value,
                        'distance': final_sol.total_distance,
                        'time': t1 - t0,
                        'num_routes': len(final_sol.routes),
                        'routes': [[int(i) for i in r] for r in final_sol.routes],
                        'improvement': improvement,
                        'adv_metrics': adv_metrics
                    }
                else:
                    self.logger.warning(f"[{zone_id}] 多车型 {ss} 第二阶段无可行解")
                    return None

            elif exp_config.get('algorithm', '').upper() in ['TS-ADP', 'VNS-ADP']:
                # --- C: 多车型自适应算法（如多车型 TS-ADP） ---
                algo_type = exp_config['algorithm'].split('-')[0]  # 'TS' or 'VNS'

                # 关键：过滤掉 'algorithm' 这个字段，避免传给 _create_adaptive_solver
                safe_params = {k: v for k, v in exp_config.items() if k != 'algorithm'}

                solver = self._create_adaptive_solver(
                    solver_type="TS",              # 或者从 exp_config["algorithm"] 里提取 'TS'
                    instance=instance,
                    adaptive_params=exp_config["adaptive_params"],
                )

                t0 = time.time()
                solution = solver.solve()
                t1 = time.time()

                if solution and solution.is_feasible():
                    adv_metrics = self._calculate_advanced_metrics(solution)
                    return {
                        'objective': solution.objective_value,
                        'distance': solution.total_distance,
                        'time': t1 - t0,
                        'num_routes': len(solution.routes),
                        'routes': [[int(i) for i in r] for r in solution.routes],
                        'parameter_history': solver.get_parameter_history(),
                        'adv_metrics': adv_metrics
                    }
                else:
                    self.logger.warning(f"[{zone_id}] 多车型 {algo_type}-ADP 无可行解")
                    return None

            else:
                self.logger.error(f"[{zone_id}] 未知的增强实验配置: {exp_config}")
                return None

        except Exception as e:
            self.logger.error(f"[{zone_id}] run_enhanced_experiment 实验失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def run_vehicle_config_experiment(self,
                                      zone_id: str,
                                      solver_type: str,
                                      capacity_candidates: Dict[str, List[Tuple[int,int]]],
                                      fixed_cost_candidates: Dict[str, List[int]],
                                      count_candidates: Dict[str, List[int]]) -> Optional[Dict]:
        """
        【实验16】: 在指定路区 zone_id 上，针对若干“车型容量区间”“固定成本”“可用数量”的组合，
        一次性遍历/测试哪种车辆配置效果最好。
        
        参数:
          - zone_id: 你想测试的路区ID，比如 "Z001"
          - solver_type: 用哪个算法求解，比如 "TS"/"VNS"/"SA"/"CW"
          - capacity_candidates: 每个车型对应一批 (cap_min, cap_max) 候选区间
              形如: {
                'small':  [(1500,2000), (2001,3000)],
                'medium': [(3001,5000)],
                'large':  [(5001,8000),(8001,10000)]
              }
          - fixed_cost_candidates: 每个车型对应若干固定成本候选
              形如: {
                'small':[600,800], 'medium':[900,1000], 'large':[1400,1500]
              }
          - count_candidates: 每个车型对应若干可用车辆数
              形如: {
                'small':[5,10], 'medium':[5], 'large':[2,5]
              }
        
        返回:
          {
            'best_config': { 'small':{...}, 'medium':{...}, 'large':{...} },   # 最优车型配置
            'best_objective': 12345.6,   # 最佳目标值
            'all_results': [ { 'config':..., 'objective':..., 'time':..., 'num_routes':..., ... }, ... ]
          }
          若全都无解，则返回 None
        """

        # 0. 先检查 zone_id 是否存在
        if zone_id not in self.zone_map:
            self.logger.error(f"[Exp16] Zone {zone_id} not found in zone_map.")
            return None

        merchants_df = self.zone_map[zone_id]
        if not isinstance(merchants_df, pd.DataFrame):
            merchants_df = pd.DataFrame(merchants_df)
        if len(merchants_df) == 0:
            self.logger.warning(f"[Exp16] Zone {zone_id} has no merchants.")
            return None

        # 1. 生成所有 possible 组合 (small, medium, large) 的 capacity / fixed_cost / count
        # 这里以3车型为例, 做三重 product.
        # 如果你车型不止3种, 需要自行扩展/改写
        all_configs = []
        vtypes = ['small','medium','large']  # 如果有更多车型可自行修改

        for (cap_min_s, cap_max_s) in capacity_candidates['small']:
            for fc_s in fixed_cost_candidates['small']:
                for cnt_s in count_candidates['small']:
                    for (cap_min_m, cap_max_m) in capacity_candidates['medium']:
                        for fc_m in fixed_cost_candidates['medium']:
                            for cnt_m in count_candidates['medium']:
                                for (cap_min_l, cap_max_l) in capacity_candidates['large']:
                                    for fc_l in fixed_cost_candidates['large']:
                                        for cnt_l in count_candidates['large']:
                                            cfg = {
                                                'small': {
                                                    'cap_min': cap_min_s, 'cap_max': cap_max_s,
                                                    'fixed_cost': fc_s,
                                                    'count': cnt_s
                                                },
                                                'medium': {
                                                    'cap_min': cap_min_m, 'cap_max': cap_max_m,
                                                    'fixed_cost': fc_m,
                                                    'count': cnt_m
                                                },
                                                'large': {
                                                    'cap_min': cap_min_l, 'cap_max': cap_max_l,
                                                    'fixed_cost': fc_l,
                                                    'count': cnt_l
                                                }
                                            }
                                            all_configs.append(cfg)
        
        self.logger.info(f"[Exp16] 共有 {len(all_configs)} 种车辆配置待测试, zone={zone_id}.")

        # 2. 为 zone_id 创建 VRPInstance（不带 vehicle_types, 后面再注入）
        instance = self._create_vrp_instance(merchants_df)

        best_objective = float('inf')
        best_config = None
        all_results = []

        # 3. 依次测试 each config
        for idx, vconf in enumerate(all_configs):
            # 把这个配置写进 instance.vehicle_types
            instance.vehicle_types = vconf

            # 创建 Solver
            solver = self._create_solver(solver_type, instance)
            t0 = time.time()
            solution = solver.solve()
            t1 = time.time()

            if solution and solution.is_feasible():
                obj = solution.objective_value
                used_time = t1 - t0
                if obj < best_objective:
                    best_objective = obj
                    best_config = vconf
                
                row = {
                    'config_index': idx,
                    'vehicle_config': vconf,
                    'objective': obj,
                    'distance': solution.total_distance,
                    'time': used_time,
                    'num_routes': len(solution.routes)
                }
                all_results.append(row)

                self.logger.info(f"[Exp16] idx={idx}, config={vconf} => obj={obj:.2f}, time={used_time:.2f}s")
            else:
                self.logger.warning(f"[Exp16] idx={idx}, config={vconf} => 无可行解")

        if best_config is None:
            self.logger.error(f"[Exp16] {zone_id} 所有车辆配置都无解!")
            return None
        
        self.logger.info(f"[Exp16] {zone_id} 最优车辆配置: {best_config}, objective={best_objective:.2f}")

        # 4. 返回总结
        return {
            'best_config': best_config,
            'best_objective': best_objective,
            'all_results': all_results
        }

    def _compute_vehicle_metrics(self, sol: "VRPSolution") -> Dict[str, Dict]:
        """
        扫描 sol.vehicle_assignments + sol.routes，统计各种车型的使用情况。
        """
        # 如果没有 vehicle_assignments 就返回空
        if not hasattr(sol, 'vehicle_assignments'):
            return {}
        
        # 初始化各车型统计
        stats_map = {}
        for vtype, specs in sol.instance.vehicle_types.items():
            stats_map[vtype] = {
                'used_count': 0,
                'total_fixed_cost': 0.0,
                'total_route_length': 0.0,
                'route_count': 0,
                'available_count': specs.get('count', 1)
            }
        
        # 遍历每条路线
        for r_idx, route in enumerate(sol.routes):
            if r_idx not in sol.vehicle_assignments:
                continue
            vinfo = sol.vehicle_assignments[r_idx]  # 形如 {'type':'small','fixed_cost':800,...}
            vt = vinfo['type']
            
            if vt not in stats_map:
                # 如果出现未在 instance.vehicle_types 中定义的车型，就临时创建
                stats_map[vt] = {
                    'used_count': 0,
                    'total_fixed_cost': 0.0,
                    'total_route_length': 0.0,
                    'route_count': 0,
                    'available_count': 1
                }
            
            stats_map[vt]['used_count'] += 1
            stats_map[vt]['total_fixed_cost'] += vinfo.get('fixed_cost', 0.0)
            
            # 计算该路线长度
            route_len = self._calc_single_route_length(sol, route)
            stats_map[vt]['total_route_length'] += route_len
            stats_map[vt]['route_count'] += 1
        
        # 计算平均路线长度、利用率
        out = {}
        for vtype, data in stats_map.items():
            used_count = data['used_count']
            route_count = data['route_count']
            if route_count > 0:
                avg_len = data['total_route_length'] / route_count
            else:
                avg_len = 0.0
            
            if data['available_count'] > 0:
                utilization = used_count / data['available_count']
            else:
                utilization = 0.0
            
            out[vtype] = {
                'used_count': used_count,
                'utilization': utilization,
                'fixed_cost': data['total_fixed_cost'],
                'avg_route_length': avg_len
            }
        
        return out

    def _compute_path_diversity(self, solution: "VRPSolution") -> Dict[str,Dict]:
        """
        如您只有多车型实验，就写 key='multi_vehicle'；或若要区分算法，可再细分。
        返回 {
        'multi_vehicle':{
            'avg_candidates':...,'length_std':..., 'max_min_ratio':..., 'time_increase':...
        }
        }
        """
        route_lengths = []
        for route in solution.routes:
            route_lengths.append(self._calc_single_route_length(solution, route))
        if len(route_lengths)<2:
            length_std = 0.0
            max_min_ratio = 1.0
        else:
            length_std = float(np.std(route_lengths))
            max_len = max(route_lengths)
            min_len = min(route_lengths)
            max_min_ratio = (max_len/min_len) if min_len>1e-9 else float('inf')

        return {
            "multi_vehicle": {
                "avg_candidates": 1.0,    # 如果您真的有多条候选路径数据可放这里
                "length_std": length_std,
                "max_min_ratio": max_min_ratio,
                "time_increase": 0.0      # 例如计算时间开销增加多少
            }
        }

    def _compute_cost_metrics(self, solution: "VRPSolution") -> Dict[str,Dict]:
        """
        计算固定/变动成本占比。假设 solution.total_distance=dist_sum, plus sum of each vehicle's fixed cost.
        返回: {
        'multi_vehicle':{
            'fixed_ratio':..., 'variable_ratio':..., 'reduction':...
        }
        }
        """
        total_fixed = 0.0
        for vinfo in solution.vehicle_assignments.values():
            total_fixed += vinfo['fixed_cost']
        dist = solution.total_distance
        total_cost = total_fixed+ dist
        if total_cost<=0:
            return {
            "multi_vehicle": {
                "fixed_ratio":0.0, "variable_ratio":0.0, "reduction":0.0
            }
            }
        fixed_ratio = total_fixed/total_cost
        variable_ratio = 1.0 - fixed_ratio
        # reduction => 跟某基线对比,此处仅演示
        reduction = 0.1

        return {
        "multi_vehicle":{
            "fixed_ratio": fixed_ratio,
            "variable_ratio": variable_ratio,
            "reduction": reduction
        }
        }


class BaseSolver:
    def __init__(self, instance:"VRPInstance"):
        self.instance= instance
    def solve(self)-> Optional[VRPSolution]:
        raise NotImplementedError
    
class ExperimentFramework:
    """实验框架类"""
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger("ExperimentFramework")
        
        # 存储实验结果
        self.baseline_results = {}
        self.hybrid_results = {}
        self.adaptive_results = {}
        
        # 存储性能指标
        self.metrics = {
            'baseline': {},
            'hybrid': {},
            'adaptive': {}
        }
        
        # 创建输出目录
        self._setup_directories()
        
    def _setup_directories(self):
        """创建实验相关目录"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 创建子目录
        for subdir in ['logs', 'results', 'plots', 'metrics']:
            path = os.path.join(self.config.output_dir, subdir)
            os.makedirs(path, exist_ok=True)
            
        # 配置日志
        log_file = os.path.join(self.config.output_dir, 'logs', 'experiment.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
            
    def load_data(self) -> Tuple[pd.DataFrame, gpd.GeoDataFrame, nx.Graph]:
        """加载并预处理实验数据"""
        self.logger.info("开始加载数据...")
        
        # 1. 加载订单数据
        df = pd.read_excel(self.config.excel_path)
        if "托运单重量" in df.columns:
            df = df.groupby("商家名称", as_index=False).agg({
                "经度": "first",
                "纬度": "first", 
                "托运单重量": "sum"
            })
        else:
            df = df.drop_duplicates(subset=["商家名称"]).copy()
        
        # 2. 加载地理数据
        city_gdf = gpd.read_file(self.config.city_path)
        if city_gdf.crs is None:
            city_gdf.set_crs("EPSG:4326", inplace=True)
            
        # 3. 构建路网
        rnb = RoadNetworkBuilder(self.config.road_path)
        rnb.build_graph()
        road_graph = rnb.graph
        
        self.logger.info(f"数据加载完成: {len(df)}个商户")
        return df, city_gdf, road_graph
        
    def prepare_zones(self, df: pd.DataFrame, city_gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, Dict]:
        """准备实验路区"""
        self.logger.info("开始准备实验路区...")
        
        # 1. 初始化分区器
        partitioner = EnhancedZonePartitioner(self.config.output_dir)
        partitioner.merchant_gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["经度"], df["纬度"]),
            crs="EPSG:4326"
        )
        partitioner.city_boundary = city_gdf
        
        # 2. 生成路区
        zones_gdf = partitioner.generate_zones(
            min_clusters=self.config.min_clusters,
            max_clusters=self.config.max_clusters,
            road_buffer_distance=self.config.road_buffer_distance
        )
        
        # 3. 拆分商户到路区
        zone_map = split_merchants_by_zone(partitioner.merchant_gdf, zones_gdf)
        
        # 4. 保存路区统计指标
        metrics_df = partitioner.calculate_metrics()
        metrics_path = os.path.join(self.config.output_dir, 'metrics', 'zone_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
        
        self.logger.info(f"路区准备完成: {len(zones_gdf)}个路区")
        return zones_gdf, zone_map
    
    def run_baseline_experiments(self, zone_map: Dict, road_graph: nx.Graph):
        """运行基线算法实验"""
        self.logger.info("\n=== 开始基线算法实验 ===")
        
        baseline_algos = {
            "CW": ClarkeWrightSolver,
            "SA": SimulatedAnnealingSolver,
            "TS": TabuSearchSolver,
            "VNS": VNSSolver
        }
        
        for zone_id, merchants_df in tqdm(zone_map.items(), desc="处理路区"):
            if len(merchants_df) == 0:
                continue
                
            self.logger.info(f"\n处理路区 {zone_id}, 商户数={len(merchants_df)}")
            
            # 为每个路区创建VRP实例
            instance = self._create_vrp_instance(merchants_df, road_graph)
            
            # 运行每个基线算法
            for algo_code, solver_class in baseline_algos.items():
                try:
                    solver = solver_class(
                        instance=instance,
                        config=self.config
                    )
                    t0 = time.time()
                    solution = solver.solve()
                    t1 = time.time()
                    
                    # 记录结果
                    if solution and solution.is_feasible():
                        self.baseline_results.setdefault(zone_id, {})[algo_code] = {
                            'objective': solution.objective_value,
                            'distance': solution.total_distance,
                            'time': t1 - t0,
                            'num_routes': len(solution.routes),
                            'routes': [[int(i) for i in r] for r in solution.routes]
                        }
                        self.logger.info(f"[{zone_id}] {algo_code} 成功: "
                                       f"obj={solution.objective_value:.2f}, "
                                       f"time={t1-t0:.2f}s")
                    else:
                        self.logger.warning(f"[{zone_id}] {algo_code} 无可行解")
                        
                except Exception as e:
                    self.logger.error(f"[{zone_id}] {algo_code} 失败: {str(e)}")
                    
        # 保存基线结果
        self._save_baseline_results()

    def _plot_visualizations(experiment: VRPExperiment, plots_dir: str):
        """生成所有可视化图表"""
        os.makedirs(plots_dir, exist_ok=True)
        
        # 目标值对比图
        plt.figure(figsize=(10, 6))
        df = pd.read_csv('baseline_results.csv')
        
        sns.boxplot(data=df, x='algorithm', y='objective')
        plt.title('各算法目标值对比')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(plots_dir, 'objective_comparison.png'))
        plt.close()
        
        # 求解时间对比图
        plt.figure(figsize=(10, 6)) 
        sns.barplot(data=df, x='algorithm', y='time')
        plt.title('各算法求解时间对比')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(plots_dir, 'time_comparison.png'))
        plt.close()

    def _get_results_path(self, filename: str) -> str:
        """获取结果文件的完整路径"""
        results_dir = os.path.join(self.config.output_dir, 'results')
        # 确保目录存在
        os.makedirs(results_dir, exist_ok=True)
        return os.path.join(results_dir, filename)

    def _plot_objective_comparison(self, plots_dir: str):
        """绘制目标值对比图"""
        try:
            # 获取结果文件路径
            results_path = self._get_results_path('baseline_results.csv')
            
            # 检查文件是否存在
            if not os.path.exists(results_path):
                self.logger.error(f"找不到结果文件: {results_path}")
                return
                
            # 读取数据
            metrics_df = pd.read_csv(results_path)
            
            # 创建图表
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=metrics_df, x='type', y='objective', hue='algorithm')
            
            # 设置标题和标签
            plt.title('各算法目标值对比')
            plt.xlabel('算法类型')
            plt.ylabel('目标值')
            plt.xticks(rotation=45)
            
            # 保存图表
            save_path = os.path.join(plots_dir, 'objective_comparison.png')
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            self.logger.info(f"成功生成目标值对比图: {save_path}")
        except Exception as e:
            self.logger.error(f"生成目标值对比图时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

class EnhancedBaseSolver:
    """增强的VRP算法基类，支持实验指标收集和参数自适应"""
    
    def __init__(self, instance: "VRPInstance", **kwargs):
        self.instance = instance
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 实验数据收集
        self.convergence_history = []  # 收敛历史
        self.parameter_history = {}    # 参数变化历史
        self.exploration_history = []  # 探索历史
        self.best_solutions = []       # 最优解历史
        
        # 性能指标
        self.iterations_count = 0
        self.local_search_count = 0
        self.improvement_count = 0
        self.diversification_count = 0
        
        # 解的多样性指标
        self.solution_pool = []
        self.solution_distances = []
        
        # 时间窗口违反统计
        self.time_window_violations = 0
        self.capacity_violations = 0
        
    def solve(self) -> VRPSolution:
        """求解VRP"""
        raise NotImplementedError
        
    def _update_convergence(self, current_obj: float):
        """更新收敛历史"""
        self.convergence_history.append(current_obj)
        
    def _update_parameters(self, param_name: str, value: Any):
        """更新参数历史"""
        if param_name not in self.parameter_history:
            self.parameter_history[param_name] = []
        self.parameter_history[param_name].append(value)
        
    def _update_exploration(self, solution: VRPSolution):
        """更新探索历史"""
        self.exploration_history.append(self._calculate_diversity(solution))
        
    def _calculate_diversity(self, solution: VRPSolution) -> float:
        """计算解的多样性"""
        if not self.solution_pool:
            return 1.0
        
        # 计算与历史解的平均距离
        distances = []
        for old_sol in self.solution_pool[-5:]:  # 只比较最近5个解
            dist = self._solution_distance(solution, old_sol)
            distances.append(dist)
        
        return np.mean(distances) if distances else 1.0
        
    def _solution_distance(self, sol1: VRPSolution, sol2: VRPSolution) -> float:
        """计算两个解之间的距离"""
        # 使用路线编辑距离
        total_dist = 0
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
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                    
        return dp[m][n]
        
    def _check_time_windows(self, route: List[int]) -> bool:
        """检查时间窗约束"""
        if not route:
            return True
            
        current_time = 0
        prev_loc = self.instance.get_depot_location(0)
        
        for order in route:
            curr_loc = self.instance.get_order_location(order)
            travel_time = self.instance.get_road_distance(prev_loc, curr_loc) / 30000.0
            current_time += travel_time
            
            # 模拟服务时间
            current_time += 0.25  # 15分钟服务时间
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
        
    def get_statistics(self) -> Dict:
        """获取算法运行统计信息"""
        return {
            'iterations': self.iterations_count,
            'local_searches': self.local_search_count,
            'improvements': self.improvement_count,
            'diversifications': self.diversification_count,
            'time_window_violations': self.time_window_violations,
            'capacity_violations': self.capacity_violations,
            'convergence_history': self.convergence_history,
            'parameter_history': self.parameter_history,
            'exploration_history': self.exploration_history,
            'best_solutions': [s.objective_value for s in self.best_solutions],
            'solution_diversity': np.mean(self.solution_distances) if self.solution_distances else 0
        }
    
class ClarkeWrightSolver(EnhancedBaseSolver):
    def __init__(self, instance: "VRPInstance", **kwargs):
        super().__init__(instance, **kwargs)
        self.parallel_savings = kwargs.get('parallel_savings', True)
        self.time_window_enforce = kwargs.get('time_window_enforce', False)
        
    def solve(self) -> VRPSolution:
        self.logger.info("开始Clarke-Wright求解...")
        
        # 初始解
        solution = self._build_initial_solution()
        self._update_convergence(solution.objective_value)
        
        if not solution.is_feasible():
            return solution
            
        # 计算节省值
        savings = self._compute_savings()
        
        # 合并路线
        improved = True
        while improved and savings:
            improved = False
            saving, i, j = savings.pop(0)
            
            route_i = None
            route_j = None
            
            # 查找包含i,j的路线
            for idx, route in enumerate(solution.routes):
                if i in route:
                    route_i = (idx, route)
                if j in route:
                    route_j = (idx, route)
                    
            if not (route_i and route_j) or route_i[0] == route_j[0]:
                continue
                
            # 尝试合并
            merged_route = route_i[1] + route_j[1]
            if self._check_route_feasible(merged_route):
                # 执行合并
                solution.routes[route_i[0]] = merged_route
                solution.routes.pop(route_j[0])
                improved = True
                
                # 更新指标
                solution._update_metrics()
                self.improvement_count += 1
                self._update_convergence(solution.objective_value)
                
        return solution
        
    def _build_initial_solution(self) -> VRPSolution:
        """构建初始解"""
        solution = VRPSolution(self.instance)
        
        # 每个订单独立成一条路线
        for i in range(self.instance.num_orders):
            if self._check_route_feasible([i]):
                solution.add_route([i])
                
        return solution
        
    def _compute_savings(self) -> List[Tuple[float, int, int]]:
        """计算节省值"""
        savings = []
        depot = self.instance.get_depot_location(0)
        
        for i in range(self.instance.num_orders):
            loc_i = self.instance.get_order_location(i)
            
            for j in range(i + 1, self.instance.num_orders):
                loc_j = self.instance.get_order_location(j)
                
                # 计算节省值: c0i + c0j - cij
                saving = (
                    self.instance.get_road_distance(depot, loc_i) +
                    self.instance.get_road_distance(depot, loc_j) -
                    self.instance.get_road_distance(loc_i, loc_j)
                )
                
                if saving > 0:
                    savings.append((saving, i, j))
                    
        # 按节省值降序排序
        savings.sort(reverse=True)
        return savings
        
    def _check_route_feasible(self, route: List[int]) -> bool:
        """检查路线是否可行"""
        # 容量检查
        if not self._check_capacity(route):
            return False
            
        # 时间窗检查
        if self.time_window_enforce and not self._check_time_windows(route):
            return False
            
        return True

class SimulatedAnnealingSolver(EnhancedBaseSolver):
    def __init__(self, instance: "VRPInstance", **kwargs):
        super().__init__(instance, **kwargs)
        self.initial_temp = kwargs.get('initial_temp', 1000.0)
        self.cooling_rate = kwargs.get('cooling_rate', 0.95)
        self.iterations = kwargs.get('iterations', 100)
        
    def solve(self) -> VRPSolution:
        self.logger.info("开始SimulatedAnnealing求解...")
        
        # 构建初始解
        current_sol = self._build_initial_solution()
        if not current_sol.is_feasible():
            return current_sol
            
        best_sol = current_sol.copy()
        self._update_convergence(current_sol.objective_value)
        self.best_solutions.append(best_sol)
        
        # SA主循环
        temperature = self.initial_temp
        
        while temperature > 1e-6 and self.iterations_count < self.iterations:
            self.iterations_count += 1
            
            # 生成邻域解
            neighbor_sol = self._generate_neighbor(current_sol)
            if not neighbor_sol.is_feasible():
                continue
                
            # 计算目标值差异
            delta = neighbor_sol.objective_value - current_sol.objective_value
            
            # Metropolis准则
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_sol = neighbor_sol
                self._update_exploration(current_sol)
                
                # 更新最优解
                if current_sol.objective_value < best_sol.objective_value:
                    best_sol = current_sol.copy()
                    self.improvement_count += 1
                    self.best_solutions.append(best_sol)
                    
            # 收敛历史
            self._update_convergence(current_sol.objective_value)
            
            # 降温
            temperature *= self.cooling_rate
            self._update_parameters('temperature', temperature)
            
            # 解的多样性计算
            self.solution_pool.append(current_sol)
            if len(self.solution_pool) > 1:
                diversity = self._calculate_diversity(current_sol)
                self.solution_distances.append(diversity)
                
            # 如果多样性过低,执行多样化
            if len(self.solution_distances) >= 5:
                recent_diversity = np.mean(self.solution_distances[-5:])
                if recent_diversity < 0.1:  # 多样性阈值
                    current_sol = self._diversification(current_sol)
                    self.diversification_count += 1
                    
        return best_sol
        
    def _build_initial_solution(self) -> VRPSolution:
        """构建初始解"""
        solution = VRPSolution(self.instance)
        orders = list(range(self.instance.num_orders))
        random.shuffle(orders)
        
        current_route = []
        for order in orders:
            current_route.append(order)
            
            # 当路线达到一定长度或随机决定结束时检查可行性
            if (len(current_route) >= 5 or 
                (len(current_route) >= 2 and random.random() < 0.3)):
                if self._check_route_feasible(current_route):
                    solution.add_route(current_route)
                    current_route = []
                else:
                    # 如果当前路线不可行,尝试缩短它
                    while len(current_route) > 2:
                        current_route.pop()
                        if self._check_route_feasible(current_route):
                            solution.add_route(current_route)
                            current_route = []
                            break
                    if current_route:  # 如果还有剩余点,开始新路线
                        last_point = current_route[-1]
                        current_route = [last_point]
                        
        # 处理剩余订单
        if len(current_route) >= 2 and self._check_route_feasible(current_route):
            solution.add_route(current_route)
            
        return solution

    def _generate_neighbor(self, solution: VRPSolution) -> VRPSolution:
        """生成邻域解"""
        new_sol = solution.copy()
        
        if len(new_sol.routes) < 2:
            return new_sol
            
        # 随机选择操作
        operation = random.choice(['swap', 'relocate', '2-opt'])
        
        if operation == 'swap':
            # 交换两个路线中的点
            r1, r2 = random.sample(range(len(new_sol.routes)), 2)
            if len(new_sol.routes[r1]) >= 1 and len(new_sol.routes[r2]) >= 1:
                i = random.randint(0, len(new_sol.routes[r1])-1)
                j = random.randint(0, len(new_sol.routes[r2])-1)
                new_sol.routes[r1][i], new_sol.routes[r2][j] = \
                    new_sol.routes[r2][j], new_sol.routes[r1][i]
        
        elif operation == 'relocate':
            # 重定位操作
            r1 = random.randint(0, len(new_sol.routes)-1)
            if len(new_sol.routes[r1]) > 2:
                r2 = random.randint(0, len(new_sol.routes)-1)
                i = random.randint(0, len(new_sol.routes[r1])-1)
                point = new_sol.routes[r1].pop(i)
                j = random.randint(0, len(new_sol.routes[r2]))
                new_sol.routes[r2].insert(j, point)
        
        else:  # 2-opt
            # 路线内反转
            r = random.randint(0, len(new_sol.routes)-1)
            route = new_sol.routes[r]
            if len(route) >= 4:
                i = random.randint(0, len(route)-3)
                j = random.randint(i+2, len(route))
                route[i:j] = reversed(route[i:j])
                
        new_sol._update_metrics()
        return new_sol
        
    def _diversification(self, solution: VRPSolution) -> VRPSolution:
        """多样化操作"""
        new_sol = solution.copy()
        
        # 路线重组
        all_orders = []
        for route in new_sol.routes:
            all_orders.extend(route)
            
        random.shuffle(all_orders)
        
        # 重新构建路线
        new_sol.routes = []
        current_route = []
        
        for order in all_orders:
            current_route.append(order)
            if len(current_route) >= 5 or random.random() < 0.3:
                if self._check_route_feasible(current_route):
                    new_sol.routes.append(current_route)
                    current_route = []
                    
        if len(current_route) >= 2 and self._check_route_feasible(current_route):
            new_sol.routes.append(current_route)
            
        new_sol._update_metrics()
        return new_sol
        
    def _check_route_feasible(self, route: List[int]) -> bool:
        """检查路线可行性"""
        if not route or len(route) < 2:
            return False
            
        # 容量检查
        if not self._check_capacity(route):
            return False
            
        # 时间窗检查
        if not self._check_time_windows(route):
            return False
            
        return True
    
class TabuSearchSolver(EnhancedBaseSolver):
    def __init__(self, instance: "VRPInstance", **kwargs):
        super().__init__(instance, **kwargs)
        self.tabu_size = kwargs.get('tabu_size', 10)
        self.max_iterations = kwargs.get('max_iterations', 50)
        self.neighborhood_size = kwargs.get('neighborhood_size', 20)
        
        # 禁忌表
        self.tabu_list = []
        self.aspiration_value = float('inf')
        
    def solve(self) -> VRPSolution:
        self.logger.info("开始TabuSearch求解...")
        
        # 构建初始解
        current_sol = self._build_initial_solution()
        if not current_sol.is_feasible():
            return current_sol
            
        best_sol = current_sol.copy()
        self._update_convergence(current_sol.objective_value)
        self.best_solutions.append(best_sol)
        
        # 主循环
        no_improve = 0
        
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
                
                # 判断是否禁忌
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
                
            # 更新最优解
            if current_sol.objective_value < best_sol.objective_value:
                best_sol = current_sol.copy()
                self.aspiration_value = best_sol.objective_value
                self.improvement_count += 1
                no_improve = 0
                self.best_solutions.append(best_sol)
                
            # 收敛历史
            self._update_convergence(current_sol.objective_value)
            
            # 解的多样性
            self.solution_pool.append(current_sol)
            if len(self.solution_pool) > 1:
                diversity = self._calculate_diversity(current_sol)
                self.solution_distances.append(diversity)
                
            # 多样化策略
            if no_improve >= 10:  # 长期停滞
                current_sol = self._diversification(current_sol)
                no_improve = 0
                self.diversification_count += 1
                
        return best_sol
        
    def _build_initial_solution(self) -> VRPSolution:
        """构建初始解"""
        solution = VRPSolution(self.instance)
        unassigned = set(range(self.instance.num_orders))
        depot_loc = self.instance.get_depot_location(0)
        
        while unassigned:
            route = []
            # 选择离仓库最近的点开始
            start = min(unassigned,
                key=lambda x: self.instance.get_road_distance(
                    depot_loc,
                    self.instance.get_order_location(x)
                ))
            route.append(start)
            unassigned.remove(start)
            
            # 构建路线
            while unassigned and len(route) < 5:
                current = route[-1]
                current_loc = self.instance.get_order_location(current)
                
                # 寻找最近的下一个点
                next_point = min(unassigned,
                    key=lambda x: (
                        self.instance.get_road_distance(
                            current_loc,
                            self.instance.get_order_location(x)
                        ) +
                        self.instance.get_road_distance(
                            self.instance.get_order_location(x),
                            depot_loc
                        )
                    ))
                    
                # 检查可行性
                temp_route = route + [next_point]
                if self._check_route_feasible(temp_route):
                    route.append(next_point)
                    unassigned.remove(next_point)
                else:
                    break
                    
            if len(route) >= 2:
                solution.add_route(route)
                
        return solution
        
    def _generate_neighbor(self, solution: VRPSolution) -> Tuple[VRPSolution, Tuple]:
        """生成邻域解,返回(新解,移动)"""
        new_sol = solution.copy()
        
        if len(new_sol.routes) < 2:
            return new_sol, ('none', 0, 0, 0, 0)
        
        # 随机选择操作
        operation = random.choice(['swap', 'relocate', '2-opt', 'cross'])
        move = None
        
        if operation == 'swap':
            try:
                r1, r2 = random.sample(range(len(new_sol.routes)), 2)
                if len(new_sol.routes[r1]) >= 1 and len(new_sol.routes[r2]) >= 1:
                    i = random.randint(0, len(new_sol.routes[r1])-1)
                    j = random.randint(0, len(new_sol.routes[r2])-1)
                    move = ('swap', r1, i, r2, j)
                    new_sol.routes[r1][i], new_sol.routes[r2][j] = \
                        new_sol.routes[r2][j], new_sol.routes[r1][i]
            except ValueError:
                return new_sol, ('none', 0, 0, 0, 0)
                
        elif operation == 'relocate':
            r1 = random.randint(0, len(new_sol.routes)-1)
            if len(new_sol.routes[r1]) > 2:
                r2 = random.randint(0, len(new_sol.routes)-1)
                i = random.randint(0, len(new_sol.routes[r1])-1)
                j = random.randint(0, len(new_sol.routes[r2]))
                point = new_sol.routes[r1].pop(i)
                new_sol.routes[r2].insert(j, point)
                move = ('relocate', r1, i, r2, j)
                    
        elif operation == '2-opt':
            r = random.randint(0, len(new_sol.routes)-1)
            route = new_sol.routes[r]
            if len(route) >= 4:
                i = random.randint(0, len(route)-3)
                j = random.randint(i+2, len(route))
                route[i:j] = reversed(route[i:j])
                move = ('2-opt', r, i, j)
                    
        else:  # cross
            if len(new_sol.routes) >= 2:
                try:
                    r1, r2 = random.sample(range(len(new_sol.routes)), 2)
                    if len(new_sol.routes[r1]) >= 4 and len(new_sol.routes[r2]) >= 4:
                        pos1 = random.randint(1, len(new_sol.routes[r1])-2)
                        pos2 = random.randint(1, len(new_sol.routes[r2])-2)
                        new_sol.routes[r1][pos1:], new_sol.routes[r2][pos2:] = \
                            new_sol.routes[r2][pos2:], new_sol.routes[r1][pos1:]
                        move = ('cross', r1, pos1, r2, pos2)
                except ValueError:
                    return new_sol, ('none', 0, 0, 0, 0)
        
        if move is None:
            move = ('none', 0, 0, 0, 0)
            
        new_sol._update_metrics()
        return new_sol, move
        
    def _hash_move(self, move: Tuple) -> str:
        """计算移动操作的哈希值"""
        return str(move)
        
    def _diversification(self, solution: VRPSolution) -> VRPSolution:
        """多样化操作"""
        new_sol = solution.copy()
        
        # 保存部分好的路线
        good_routes = []
        for route in new_sol.routes:
            if len(route) >= 2:
                route_cost = self._calc_route_cost(route)
                if route_cost < self.aspiration_value / len(new_sol.routes):
                    good_routes.append(route)
                    
        # 清空原有路线
        new_sol.routes = []
        used_orders = set()
        for route in good_routes:
            for order in route:
                used_orders.add(order)
            new_sol.routes.append(route)
            
        # 重新分配未使用的订单
        remaining = set(range(self.instance.num_orders)) - used_orders
        current_route = []
        
        for order in remaining:
            current_route.append(order)
            if len(current_route) >= 5 or random.random() < 0.3:
                if self._check_route_feasible(current_route):
                    new_sol.routes.append(current_route)
                    current_route = []
                    
        if len(current_route) >= 2 and self._check_route_feasible(current_route):
            new_sol.routes.append(current_route)
            
        new_sol._update_metrics()
        return new_sol
        
    def _calc_route_cost(self, route: List[int]) -> float:
        """计算路线成本"""
        if not route:
            return 0.0
            
        total_distance = 0
        depot_loc = self.instance.get_depot_location(0)
        prev_loc = depot_loc
        
        for order in route:
            curr_loc = self.instance.get_order_location(order)
            total_distance += self.instance.get_road_distance(prev_loc, curr_loc)
            prev_loc = curr_loc
            
        total_distance += self.instance.get_road_distance(prev_loc, depot_loc)
        return total_distance
        
    def _check_route_feasible(self, route: List[int]) -> bool:
        """检查路线可行性"""
        if not route or len(route) < 2:
            return False
            
        # 容量检查
        if not self._check_capacity(route):
            return False
            
        # 时间窗检查
        if not self._check_time_windows(route):
            return False
            
        return True
    
class VNSSolver(EnhancedBaseSolver):
    def __init__(self, instance: "VRPInstance", **kwargs):
        super().__init__(instance, **kwargs)
        self.max_iterations = kwargs.get('max_iterations', 30)  
        self.max_k = kwargs.get('max_neighborhoods', 3)
        self.shake_intensity = kwargs.get('shake_intensity', 0.3)
        
        # 定义邻域结构
        self.neighborhoods = [
            self._relocate_neighborhood,
            self._swap_neighborhood,
            self._two_opt_neighborhood,
            self._cross_neighborhood
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
            if no_improve >= 10:
                current_sol = self._diversification(current_sol)
                k = 0
                no_improve = 0
                self.diversification_count += 1
                
        return best_sol
        
    def _build_initial_solution(self) -> VRPSolution:
        """构建初始解"""
        solution = VRPSolution(self.instance)
        unassigned = set(range(self.instance.num_orders))
        depot_loc = self.instance.get_depot_location(0)
        
        while unassigned:
            route = []
            if not route:
                start = min(unassigned,
                    key=lambda x: self.instance.get_road_distance(
                        depot_loc,
                        self.instance.get_order_location(x)
                    ))
                route.append(start)
                unassigned.remove(start)
            
            while unassigned and len(route) < 5:
                last = route[-1]
                last_loc = self.instance.get_order_location(last)
                
                next_point = min(unassigned,
                    key=lambda x: (
                        self.instance.get_road_distance(
                            last_loc,
                            self.instance.get_order_location(x)
                        ) +
                        self.instance.get_road_distance(
                            self.instance.get_order_location(x),
                            depot_loc
                        )
                    ))
                    
                temp_route = route + [next_point]
                if self._check_route_feasible(temp_route):
                    route.append(next_point)
                    unassigned.remove(next_point)
                else:
                    break
                    
            if len(route) >= 2:
                solution.add_route(route)
                
        return solution
        
    def _shake(self, solution: VRPSolution, k: int) -> VRPSolution:
        """扰动操作"""
        new_sol = solution.copy()
        for _ in range(k + 1):  # 扰动强度与k相关
            if random.random() < self.shake_intensity:
                # 应用随机邻域操作
                neighborhood = random.choice(self.neighborhoods)
                new_sol = neighborhood(new_sol)
                
        new_sol._update_metrics()
        return new_sol
        
    def _variable_neighborhood_descent(self, solution: VRPSolution) -> VRPSolution:
        """变邻域下降"""
        current = solution.copy()
        improved = True
        
        while improved:
            improved = False
            self.local_search_count += 1
            
            # 依次尝试每个邻域
            for neighborhood in self.neighborhoods:
                neighbor = neighborhood(current)
                if (neighbor.is_feasible() and 
                    neighbor.objective_value < current.objective_value):
                    current = neighbor
                    improved = True
                    break
                    
        return current
        
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
        
    def _diversification(self, solution: VRPSolution) -> VRPSolution:
        """多样化操作"""
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
                if self._check_route_feasible(current_route):
                    new_sol.routes.append(current_route)
                    current_route = []
                    
        if len(current_route) >= 2 and self._check_route_feasible(current_route):
            new_sol.routes.append(current_route)
            
        new_sol._update_metrics()
        return new_sol
        
    def _check_route_feasible(self, route: List[int]) -> bool:
        """检查路线可行性"""
        if not route or len(route) < 2:
            return False
            
        # 容量检查
        if not self._check_capacity(route):
            return False
            
        # 时间窗检查
        if not self._check_time_windows(route):
            return False
            
        return True
    
class AdaptiveTabuSearchSolver(TabuSearchSolver):
    def __init__(self, instance: "VRPInstance", **kwargs):
        # 1. 先调用父类初始化
        super().__init__(instance, **kwargs)
        
        # 2. 保存初始解
        self.initial_solution = kwargs.get('initial_solution', None)
        
        # 3. 先设置自适应参数
        self.adaptive_params = kwargs.get('adaptive_params', {})
        
        # 4. 然后才能使用 adaptive_params
        if self.adaptive_params:
            self.param_type = list(self.adaptive_params.keys())[0]
            self.min_value = self.adaptive_params.get('min_tabu_size', 5)
            self.max_value = self.adaptive_params.get('max_tabu_size', 20)
        else:
            # 设置默认值
            self.param_type = 'tabu_size'
            self.min_value = 5
            self.max_value = 20
        
        # 5. 其他初始化
        self.parameter_history = []
        self.improvement_history = []
        
    def solve(self) -> VRPSolution:
        self.logger.info("开始AdaptiveTabuSearch求解...")
        
        # 使用初始解或构建新解
        if self.initial_solution and self.initial_solution.is_feasible():
            current_sol = self.initial_solution.copy()
        else:
            current_sol = self._build_initial_solution()
            
        if not current_sol.is_feasible():
            return current_sol
            
        best_sol = current_sol.copy()
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
                
        return best_sol
        
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
    
class AdaptiveVNSSolver(VNSSolver):
    def __init__(self, instance: "VRPInstance", **kwargs):
        super().__init__(instance, **kwargs)
        self.initial_solution = kwargs.get('initial_solution', None)
        # 自适应参数 
        self.adaptive_params = kwargs.get('adaptive_params', {})
        self.param_type = list(self.adaptive_params.keys())[0]
        self.min_value = self.adaptive_params.get('min_intensity', 0.1)
        self.max_value = self.adaptive_params.get('max_intensity', 0.5)
        
        # 参数调整历史
        self.parameter_history = []
        self.improvement_history = []
        self.stagnation_counter = 0
        
    def solve(self) -> VRPSolution:
        self.logger.info("开始AdaptiveVNS求解...")
        
        # 使用初始解或构建新解
        if self.initial_solution and self.initial_solution.is_feasible():
            current_sol = self.initial_solution.copy()
        else:
            current_sol = self._build_initial_solution()
            
        if not current_sol.is_feasible():
            return current_sol
            
        best_sol = current_sol.copy()
        self._update_convergence(current_sol.objective_value)
        self.best_solutions.append(best_sol)
        
        # 初始参数
        if self.param_type == 'perturbation':
            self.shake_intensity = self.min_value
        else:  # neighborhood_change
            self.max_k = int(self.min_value)
            
        self.parameter_history.append(
            self.shake_intensity if self.param_type == 'perturbation' else self.max_k
        )
        
        # 主循环
        no_improve = 0
        k = 0
        last_obj = current_sol.objective_value
        
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
            
            # 计算改进率
            improvement = (last_obj - improved_sol.objective_value) / last_obj
            self.improvement_history.append(improvement)
            
            if improved_sol.objective_value < current_sol.objective_value:
                current_sol = improved_sol
                k = 0
                last_obj = current_sol.objective_value
                self.stagnation_counter = 0
                
                # 更新最优解
                if improved_sol.objective_value < best_sol.objective_value:
                    best_sol = improved_sol.copy()
                    self.improvement_count += 1
                    no_improve = 0
                    self.best_solutions.append(best_sol)
            else:
                k = (k + 1) % len(self.neighborhoods)
                no_improve += 1
                self.stagnation_counter += 1
                
            # 自适应参数调整
            self._adapt_parameters()
            
            # 收敛历史
            self._update_convergence(current_sol.objective_value)
            self._update_exploration(current_sol)
            
            # 多样化策略
            if no_improve >= 10:
                current_sol = self._diversification(current_sol)
                k = 0
                no_improve = 0
                self.diversification_count += 1
                
        return best_sol
        
    def _adapt_parameters(self):
        """自适应参数调整"""
        if self.param_type == 'perturbation':
            if self.stagnation_counter >= 5:  # 连续5次未改进
                # 增加扰动强度
                self.shake_intensity = min(
                    self.max_value,
                    self.shake_intensity * 1.2
                )
            elif self.improvement_count > 0:  # 有改进
                # 减小扰动强度
                self.shake_intensity = max(
                    self.min_value,
                    self.shake_intensity * 0.8
                )
            self.parameter_history.append(self.shake_intensity)
            
        else:  # neighborhood_change
            if self.stagnation_counter >= 5:
                # 增加邻域数量
                self.max_k = min(
                    int(self.max_value),
                    self.max_k + 1
                )
            elif self.improvement_count > 0:
                # 减少邻域数量
                self.max_k = max(
                    int(self.min_value),
                    self.max_k - 1
                )
            self.parameter_history.append(self.max_k)
            
    def get_parameter_history(self) -> List[float]:
        """获取参数调整历史"""
        return self.parameter_history
    
def run_hybrid_experiments(self, zone_map: Dict, road_graph: nx.Graph):
        """运行混合算法实验"""
        self.logger.info("\n=== 开始混合算法实验 ===")
        
        hybrid_combinations = [
            ("CW", "TS", "Clarke-Wright + 禁忌搜索"),
            ("CW", "VNS", "Clarke-Wright + 变邻域搜索"),
            ("SA", "TS", "模拟退火 + 禁忌搜索"),
            ("SA", "VNS", "模拟退火 + 变邻域搜索")
        ]
        
        for zone_id, merchants_df in tqdm(zone_map.items(), desc="处理路区"):
            if len(merchants_df) == 0:
                continue
                
            self.logger.info(f"\n处理路区 {zone_id}, 商户数={len(merchants_df)}")
            instance = self._create_vrp_instance(merchants_df, road_graph)
            
            for first_stage, second_stage, combo_name in hybrid_combinations:
                try:
                    # 第一阶段
                    first_solver = self._create_solver(first_stage, instance)
                    t0 = time.time()
                    initial_sol = first_solver.solve()
                    
                    if not initial_sol or not initial_sol.is_feasible():
                        self.logger.warning(f"[{zone_id}] {combo_name} 第一阶段无可行解")
                        continue
                        
                    # 第二阶段
                    second_solver = self._create_solver(
                        second_stage,
                        instance,
                        initial_solution=initial_sol
                    )
                    improved_sol = second_solver.solve()
                    t1 = time.time()
                    
                    # 记录结果
                    if improved_sol and improved_sol.is_feasible():
                        combo_code = f"{first_stage}_{second_stage}"
                        self.hybrid_results.setdefault(zone_id, {})[combo_code] = {
                            'objective': improved_sol.objective_value,
                            'distance': improved_sol.total_distance,
                            'time': t1 - t0,
                            'num_routes': len(improved_sol.routes),
                            'routes': [[int(i) for i in r] for r in improved_sol.routes],
                            'improvement': (initial_sol.objective_value - 
                                         improved_sol.objective_value) / initial_sol.objective_value
                        }
                        self.logger.info(f"[{zone_id}] {combo_name} 成功: "
                                       f"obj={improved_sol.objective_value:.2f}, "
                                       f"time={t1-t0:.2f}s, "
                                       f"改进={self.hybrid_results[zone_id][combo_code]['improvement']*100:.1f}%")
                    else:
                        self.logger.warning(f"[{zone_id}] {combo_name} 第二阶段失败")
                        
                except Exception as e:
                    self.logger.error(f"[{zone_id}] {combo_name} 失败: {str(e)}")
                    
        # 保存混合算法结果
        self._save_hybrid_results()
        
def run_adaptive_experiments(self, zone_map: Dict, road_graph: nx.Graph):
    """运行自适应参数实验"""
    self.logger.info("\n=== 开始自适应参数实验 ===")
    
    adaptive_configs = [
        ("TS", {
            "tabu_size": "adaptive",
            "min_size": 5,
            "max_size": 20
        }, "自适应禁忌长度"),
        
        ("TS", {
            "neighborhood_size": "adaptive",
            "min_size": 10,
            "max_size": 30
        }, "自适应邻域大小"),
        
        ("VNS", {
            "perturbation": "adaptive",
            "min_intensity": 0.1,
            "max_intensity": 0.5
        }, "自适应扰动强度"),
        
        ("VNS", {
            "neighborhood_change": "adaptive",
            "min_neighborhoods": 2,
            "max_neighborhoods": 5
        }, "自适应邻域切换")
    ]
    
    for zone_id, merchants_df in tqdm(zone_map.items(), desc="处理路区"):
        if len(merchants_df) == 0:
            continue
            
        self.logger.info(f"\n处理路区 {zone_id}, 商户数={len(merchants_df)}")
        instance = self._create_vrp_instance(merchants_df, road_graph)
        
        for algo, params, desc in adaptive_configs:
            try:
                # 创建自适应solver
                solver = self._create_adaptive_solver(
                    algo,
                    instance,
                    params
                )
                
                t0 = time.time()
                solution = solver.solve()
                t1 = time.time()
                
                # 记录结果
                if solution and solution.is_feasible():
                    param_key = list(params.keys())[0]
                    result_key = f"{algo}_ADP_{param_key}"
                    
                            
                    self.adaptive_results.setdefault(zone_id, {})[result_key] = {
                        'objective': solution.objective_value,
                        'distance': solution.total_distance,
                        'time': t1 - t0,
                        'num_routes': len(solution.routes),
                        'routes': [[int(i) for i in r] for r in solution.routes],
                        'parameter_history': solver.parameter_history
                    }
                    self.logger.info(f"[{zone_id}] {desc} 成功: "
                                    f"obj={solution.objective_value:.2f}, "
                                    f"time={t1-t0:.2f}s")
                else:
                    self.logger.warning(f"[{zone_id}] {desc} 无可行解")
                    
            except Exception as e:
                self.logger.error(f"[{zone_id}] {desc} 失败: {str(e)}")
                
    # 保存自适应参数实验结果
    self._save_adaptive_results()
    
def _create_vrp_instance(self, merchants_df: pd.DataFrame, road_graph: nx.Graph) -> "VRPInstance":
    """创建VRP实例"""
    return VRPInstance(
        orders_df=merchants_df,
        road_graph=road_graph,
        num_warehouses=1,
        vehicle_capacity=self.config.vehicle_capacity,
        max_route_time=self.config.max_route_time,
        max_search_distance=1e5,
        selected_warehouses=[0],
        parallel_kdtree=self.config.parallel_evaluation
    )
    
def _create_solver(self, solver_type: str, instance: "VRPInstance", **kwargs) -> BaseSolver:
    """创建基础求解器"""
    solver_map = {
        "CW": ClarkeWrightSolver,
        "SA": SimulatedAnnealingSolver,
        "TS": TabuSearchSolver,
        "VNS": VNSSolver
    }
    
    if solver_type not in solver_map:
        raise ValueError(f"未知的求解器类型: {solver_type}")
        
    solver_class = solver_map[solver_type]
    return solver_class(instance=instance, config=self.config, **kwargs)
    
def _save_baseline_results(self):
    """
    保存基线实验结果（改为CSV）
    """
    # 为安全起见，先确保目录存在
    os.makedirs(os.path.join(self.config.output_dir, 'results'), exist_ok=True)
    
    # 将 baseline_results 从内存转换为“行列表”形式
    rows = []
    for zone_id, algo_dict in self.baseline_results.items():
        for algo_code, metrics in algo_dict.items():
            row = {
                "zone_id": zone_id,
                "algorithm": algo_code,
                "objective": metrics.get("objective", 0),
                "distance": metrics.get("distance", 0),
                "time": metrics.get("time", 0),
                "num_routes": metrics.get("num_routes", 0),
                # 若还想存 "routes" 字段，可序列化为字符串写入CSV
                "routes": str(metrics.get("routes", []))
            }
            rows.append(row)
    
    # 转成 DataFrame 并输出 CSV
    save_path = os.path.join(
        self.config.output_dir,
        'results',
        'baseline_results.csv'
    )
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    self.logger.info(f"基线实验结果已保存 (CSV): {save_path}")
    

def _save_hybrid_results(self):
    """
    保存混合算法实验结果（改为CSV）
    """
    os.makedirs(os.path.join(self.config.output_dir, 'results'), exist_ok=True)
    
    # 同理，将 hybrid_results 转换为行形式
    rows = []
    for zone_id, combo_dict in self.hybrid_results.items():
        for combo, metrics in combo_dict.items():
            row = {
                "zone_id": zone_id,
                "combo": combo,
                "objective": metrics.get("objective", 0),
                "distance": metrics.get("distance", 0),
                "time": metrics.get("time", 0),
                "num_routes": metrics.get("num_routes", 0),
                "improvement": metrics.get("improvement", 0),
                # 序列化路线信息
                "routes": str(metrics.get("routes", []))
            }
            rows.append(row)
    
    save_path = os.path.join(
        self.config.output_dir,
        'results',
        'hybrid_results.csv'
    )
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    self.logger.info(f"混合算法实验结果已保存 (CSV): {save_path}")
    

def _save_adaptive_results(self):
    """
    保存自适应算法实验结果（改为CSV）
    """
    os.makedirs(os.path.join(self.config.output_dir, 'results'), exist_ok=True)
    
    # 将 adaptive_results 转为行形式
    rows = []
    for zone_id, config_dict in self.adaptive_results.items():
        for config_str, metrics in config_dict.items():
            # 注意：自适应算法可能多出“parameter_history”或其他字段
            row = {
                "zone_id": zone_id,
                "config": config_str,
                "objective": metrics.get("objective", 0),
                "distance": metrics.get("distance", 0),
                "time": metrics.get("time", 0),
                "num_routes": metrics.get("num_routes", 0),
                # 把 parameter_history 等复杂结构也转成字符串
                "parameter_history": str(metrics.get("parameter_history", [])),
                # 同理 routes
                "routes": str(metrics.get("routes", []))
            }
            rows.append(row)
    
    save_path = os.path.join(
        self.config.output_dir,
        'results',
        'adaptive_results.csv'
    )
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    self.logger.info(f"自适应参数实验结果已保存 (CSV): {save_path}")
    
def analyze_results(self):
    """分析实验结果"""
    self.logger.info("\n=== 开始分析实验结果 ===")
    
    # 1. 计算并保存性能指标
    self._calculate_performance_metrics()
    
    # 2. 生成可视化
    self._generate_visualizations()
    
    # 3. 生成实验报告
    self._generate_report()
    
def _calculate_performance_metrics(self):
    """计算详细性能指标"""
    metrics_path = os.path.join(
        self.config.output_dir,
        'metrics',
        'experiment_metrics.csv'
    )
    
    all_metrics = []
    
    # 基线算法指标
    for zone_id, zone_results in self.baseline_results.items():
        for algo, result in zone_results.items():
            metrics = {
                'zone_id': zone_id,
                'algorithm': algo,
                'type': 'baseline',
                'objective': result['objective'],
                'distance': result['distance'],
                'time': result['time'],
                'num_routes': result['num_routes'],
                'success_rate': self._calculate_success_rate(result),
                'stability_score': self._calculate_stability(result)
            }
            all_metrics.append(metrics)
            
    # 混合算法指标
    for zone_id, zone_results in self.hybrid_results.items():
        for combo, result in zone_results.items():
            metrics = {
                'zone_id': zone_id,
                'algorithm': combo,
                'type': 'hybrid',
                'objective': result['objective'],
                'distance': result['distance'],
                'time': result['time'],
                'num_routes': result['num_routes'],
                'improvement': result['improvement'],
                'stability_score': self._calculate_stability(result)
            }
            all_metrics.append(metrics)
            
    # 自适应算法指标
    for zone_id, zone_results in self.adaptive_results.items():
        for config, result in zone_results.items():
            adaptive_metrics = AdaptiveMetrics()
            
            # 更新参数历史
            if 'parameter_history' in result:
                for param_value in result['parameter_history']:
                    adaptive_metrics.update_histories(
                        param_value,
                        result['objective']
                    )
            
            # 计算指标
            adaptive_metrics.calculate_metrics()
            
            metrics = {
                'zone_id': zone_id,
                'algorithm': config,
                'type': 'adaptive',
                'objective': result['objective'],
                'distance': result['distance'],
                'time': result['time'],
                'num_routes': result['num_routes'],
                'param_adjustments': adaptive_metrics.param_adjustments,
                'convergence_score': adaptive_metrics.convergence_pattern,
                'stability_score': adaptive_metrics.stability_score,
                'success_rate': adaptive_metrics.success_rate,
                'improvement_rate': adaptive_metrics.improvement_rate,
                'time_increase': result.get('time_increase', 0.0)
            }
            all_metrics.append(metrics)
            
    # 保存到CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    self.logger.info(f"性能指标已保存: {metrics_path}")
    
    return metrics_df

def _calculate_success_rate(self, result):
    """计算成功率"""
    if 'routes' not in result:
        return 0.0
    valid_routes = sum(1 for route in result['routes'] if len(route) >= 2)
    return valid_routes / max(1, len(result['routes']))

def _calculate_stability(self, result):
    """计算稳定性分数"""
    if 'objective_history' not in result:
        return 0.0
    history = result['objective_history']
    if len(history) < 2:
        return 0.0
    variations = [abs(history[i] - history[i-1]) for i in range(1, len(history))]
    return 1.0 / (1.0 + np.std(variations))
    
def _generate_visualizations(self):
    """生成实验结果可视化"""
    plots_dir = os.path.join(self.config.output_dir, 'plots')
    
    # 1. 目标值对比箱线图
    self._plot_objective_comparison(plots_dir)
    
    # 2. 求解时间对比条形图
    self._plot_time_comparison(plots_dir)
    
    # 3. 混合算法改进率
    self._plot_hybrid_improvements(plots_dir)
    
    # 4. 自适应参数变化曲线
    self._plot_parameter_adaptation(plots_dir)
    
def _plot_objective_comparison(self, plots_dir: str):
    """绘制目标值对比箱线图"""
    plt.figure(figsize=(12, 6))
    
    metrics_df = pd.DataFrame(self._calculate_performance_metrics())
    
    # 按算法类型分组绘制箱线图
    sns.boxplot(data=metrics_df, x='type', y='objective', hue='algorithm')
    
    plt.title('不同算法目标值对比')
    plt.xlabel('算法类型')
    plt.ylabel('目标值')
    plt.xticks(rotation=45)
    
    save_path = os.path.join(plots_dir, 'objective_comparison.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
def _plot_time_comparison(self, plots_dir: str):
    """绘制求解时间对比条形图"""
    plt.figure(figsize=(12, 6))
    
    metrics_df = pd.DataFrame(self._calculate_performance_metrics())
    
    # 计算平均求解时间
    time_stats = metrics_df.groupby(['type', 'algorithm'])['time'].mean()
    
    # 绘制条形图
    time_stats.unstack().plot(kind='bar')
    
    plt.title('不同算法平均求解时间对比')
    plt.xlabel('算法类型')
    plt.ylabel('求解时间(秒)')
    plt.xticks(rotation=45)
    
    save_path = os.path.join(plots_dir, 'time_comparison.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
def _plot_hybrid_improvements(self, plots_dir: str):
    """绘制混合算法改进率"""
    plt.figure(figsize=(10, 6))
    
    improvements = []
    labels = []
    
    for zone_results in self.hybrid_results.values():
        for combo, result in zone_results.items():
            improvements.append(result['improvement'] * 100)
            labels.append(combo)
            
    # 绘制箱线图
    plt.boxplot(improvements, labels=labels)
    
    plt.title('混合算法改进率分布')
    plt.xlabel('算法组合')
    plt.ylabel('改进率(%)')
    plt.xticks(rotation=45)
    
    save_path = os.path.join(plots_dir, 'hybrid_improvements.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
def _plot_parameter_adaptation(self, plots_dir: str):
    """绘制参数自适应变化曲线"""
    for zone_id, zone_results in self.adaptive_results.items():
        for config, result in zone_results.items():
            if 'parameter_history' not in result:
                continue
                
            plt.figure(figsize=(10, 6))
            history = result['parameter_history']
            
            plt.plot(range(len(history)), history)
            plt.title(f'参数自适应变化曲线 ({zone_id}-{config})')
            plt.xlabel('迭代次数')
            plt.ylabel('参数值')
            
            save_path = os.path.join(
                plots_dir,
                f'parameter_adaptation_{zone_id}_{config}.png'
            )
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            
def _generate_report(self):
    """生成实验报告"""
    report_path = os.path.join(
        self.config.output_dir,
        'experiment_report.html'
    )
    
    # 使用jinja2模板生成HTML报告
    template = Template("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>VRP算法实验报告</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #2c3e50; }
            h2 { color: #34495e; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f5f6fa; }
            img { max-width: 100%; margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>VRP算法实验报告</h1>
        
        <h2>1. 实验配置</h2>
        <table>
            <tr><th>参数</th><th>值</th></tr>
            {% for param, value in config.items() %}
            <tr><td>{{ param }}</td><td>{{ value }}</td></tr>
            {% endfor %}
        </table>
        
        <h2>2. 数据规模</h2>
        <table>
            <tr><th>统计项</th><th>数值</th></tr>
            {% for stat, value in data_stats.items() %}
            <tr><td>{{ stat }}</td><td>{{ value }}</td></tr>
            {% endfor %}
        </table>
        
        <h2>3. 实验结果汇总</h2>
        
        <h3>3.1 基线算法性能</h3>
        <table>
            <tr>
                <th>算法</th>
                <th>平均目标值</th>
                <th>平均时间(秒)</th>
                <th>平均路线数</th>
            </tr>
            {% for algo, metrics in baseline_metrics.items() %}
            <tr>
                <td>{{ algo }}</td>
                <td>{{ "%.2f"|format(metrics.objective) }}</td>
                <td>{{ "%.2f"|format(metrics.time) }}</td>
                <td>{{ "%.1f"|format(metrics.num_routes) }}</td>
            </tr>
            {% endfor %}
        </table>
        
        <h3>3.2 混合算法性能</h3>
        <table>
            <tr>
                <th>算法组合</th>
                <th>平均目标值</th>
                <th>平均时间(秒)</th>
                <th>平均改进率(%)</th>
            </tr>
            {% for combo, metrics in hybrid_metrics.items() %}
            <tr>
                <td>{{ combo }}</td>
                <td>{{ "%.2f"|format(metrics.objective) }}</td>
                <td>{{ "%.2f"|format(metrics.time) }}</td>
                <td>{{ "%.1f"|format(metrics.improvement*100) }}</td>
            </tr>
            {% endfor %}
        </table>
        
        <h3>3.3 自适应参数性能</h3>
        <table>
            <tr>
                <th>配置</th>
                <th>平均目标值</th>
                <th>平均时间(秒)</th>
                <th>参数收敛性</th>
            </tr>
            {% for config, metrics in adaptive_metrics.items() %}
            <tr>
                <td>{{ config }}</td>
                <td>{{ "%.2f"|format(metrics.objective) }}</td>
                <td>{{ "%.2f"|format(metrics.time) }}</td>
                <td>{{ metrics.convergence }}</td>
            </tr>
            {% endfor %}
        </table>
        
        <h2>4. 结果可视化</h2>
        
        <h3>4.1 目标值对比</h3>
        <img src="plots/objective_comparison.png" alt="目标值对比"/>
        
        <h3>4.2 求解时间对比</h3>
        <img src="plots/time_comparison.png" alt="求解时间对比"/>
        
        <h3>4.3 混合算法改进率</h3>
        <img src="plots/hybrid_improvements.png" alt="混合算法改进率"/>
        
        <h3>4.4 参数自适应曲线</h3>
        {% for param_plot in parameter_plots %}
        <img src="{{ param_plot }}" alt="参数自适应曲线"/>
        {% endfor %}
        
        <h2>5. 结论分析</h2>
        {{ conclusions }}
        
    </body>
    </html>
    """)
    
    self._generate_visualizations()
    # 准备报告数据
    report_data = {
        'config': self.config.__dict__,
        'data_stats': self._get_data_stats(),
        'baseline_metrics': self._get_baseline_metrics(),
        'hybrid_metrics': self._get_hybrid_metrics(),
        'adaptive_metrics': self._get_adaptive_metrics(),
        'parameter_plots': self._get_parameter_plots(),
        'conclusions': self._generate_conclusions()
    }
    
    # 生成HTML报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(template.render(**report_data))
        
    self.logger.info(f"实验报告已生成: {report_path}")
    
def _get_data_stats(self) -> Dict:
    """获取数据统计信息"""
    stats = {
        '总路区数': len(set(r['zone_id'] for r in self.baseline_results)),
        '总商户数': sum(len(r.get('routes', [])) 
                    for results in self.baseline_results.values() 
                    for r in results.values()),
        '平均每区商户数': sum(len(r.get('routes', []))
                        for results in self.baseline_results.values()
                        for r in results.values()) / 
                        max(1, len(self.baseline_results))
    }
    return stats
    
def _get_baseline_metrics(self) -> Dict:
    """获取基线算法指标"""
    metrics = {}
    for algo in ['CW', 'SA', 'TS', 'VNS']:
        algo_results = []
        for zone_results in self.baseline_results.values():
            if algo in zone_results:
                algo_results.append(zone_results[algo])
                
        if algo_results:
            metrics[algo] = {
                'objective': np.mean([r['objective'] for r in algo_results]),
                'time': np.mean([r['time'] for r in algo_results]),
                'num_routes': np.mean([r['num_routes'] for r in algo_results])
            }
    return metrics
    
def _get_hybrid_metrics(self) -> Dict:
    """获取混合算法指标"""
    metrics = {}
    for combo in ['CW_TS', 'CW_VNS', 'SA_TS', 'SA_VNS']:
        combo_results = []
        for zone_results in self.hybrid_results.values():
            if combo in zone_results:
                combo_results.append(zone_results[combo])
                
        if combo_results:
            metrics[combo] = {
                'objective': np.mean([r['objective'] for r in combo_results]),
                'time': np.mean([r['time'] for r in combo_results]),
                'improvement': np.mean([r['improvement'] for r in combo_results])
            }
    return metrics
    
def _get_adaptive_metrics(self):
    """获取自适应算法详细指标"""
    metrics = {}
    
    # TS-ADP (实验9)
    ts_results = self._get_algorithm_results('TS_ADP_tabu_size')
    if ts_results:
        metrics['ts_initial_tabu_size'] = ts_results['parameter_history'][0]
        metrics['ts_final_tabu_size'] = ts_results['parameter_history'][-1]
        metrics['ts_convergence_pattern'] = self._analyze_convergence(ts_results['parameter_history'])
        metrics['ts_base_objective'] = self._get_base_objective('TS')
        metrics['ts_adp_objective'] = ts_results['objective']
        metrics['ts_improvement_rate'] = (metrics['ts_base_objective'] - metrics['ts_adp_objective']) / metrics['ts_base_objective']
        metrics['ts_time_increase'] = (ts_results['time'] / self._get_base_time('TS')) - 1.0
        metrics['ts_stability_change'] = self._calculate_stability_change('TS', ts_results)
        metrics['ts_overall_evaluation'] = self._evaluate_performance(metrics['ts_improvement_rate'], metrics['ts_time_increase'])
    
    # VNS-ADP (实验10)
    vns_results = self._get_algorithm_results('VNS_ADP_neighborhood_change')
    if vns_results:
        metrics['vns_initial_neighborhoods'] = vns_results['parameter_history'][0]
        metrics['vns_final_neighborhoods'] = vns_results['parameter_history'][-1]
        metrics['vns_convergence_pattern'] = self._analyze_convergence(vns_results['parameter_history'])
        metrics['vns_base_objective'] = self._get_base_objective('VNS')
        metrics['vns_adp_objective'] = vns_results['objective']
        metrics['vns_improvement_rate'] = (metrics['vns_base_objective'] - metrics['vns_adp_objective']) / metrics['vns_base_objective']
        metrics['vns_time_increase'] = (vns_results['time'] / self._get_base_time('VNS')) - 1.0
        metrics['vns_stability_change'] = self._calculate_stability_change('VNS', vns_results)
        metrics['vns_overall_evaluation'] = self._evaluate_performance(metrics['vns_improvement_rate'], metrics['vns_time_increase'])
    
    # CW→TS-ADP (实验11)
    cwts_results = self._get_algorithm_results('CW_TS_ADP_tabu_size')
    if cwts_results:
        metrics['cwts_initial_param'] = cwts_results['parameter_history'][0]
        metrics['cwts_final_param'] = cwts_results['parameter_history'][-1]
        metrics['cwts_convergence_pattern'] = self._analyze_convergence(cwts_results['parameter_history'])
        metrics['cwts_base_objective'] = self._get_base_objective('CW_TS')
        metrics['cwts_adp_objective'] = cwts_results['objective']
        metrics['cwts_improvement_rate'] = (metrics['cwts_base_objective'] - metrics['cwts_adp_objective']) / metrics['cwts_base_objective']
        metrics['cwts_time_increase'] = (cwts_results['time'] / self._get_base_time('CW_TS')) - 1.0
        metrics['cwts_stability_change'] = self._calculate_stability_change('CW_TS', cwts_results)
        metrics['cwts_overall_evaluation'] = self._evaluate_performance(metrics['cwts_improvement_rate'], metrics['cwts_time_increase'])
    
    # SA→VNS-ADP (实验12)
    savns_results = self._get_algorithm_results('SA_VNS_ADP_intensity')
    if savns_results:
        metrics['savns_initial_intensity'] = savns_results['parameter_history'][0]
        metrics['savns_final_intensity'] = savns_results['parameter_history'][-1]
        metrics['savns_convergence_pattern'] = self._analyze_convergence(savns_results['parameter_history'])
        metrics['savns_base_objective'] = self._get_base_objective('SA_VNS')
        metrics['savns_adp_objective'] = savns_results['objective']
        metrics['savns_improvement_rate'] = (metrics['savns_base_objective'] - metrics['savns_adp_objective']) / metrics['savns_base_objective']
        metrics['savns_time_increase'] = (savns_results['time'] / self._get_base_time('SA_VNS')) - 1.0
        metrics['savns_stability_change'] = self._calculate_stability_change('SA_VNS', savns_results)
        metrics['savns_overall_evaluation'] = self._evaluate_performance(metrics['savns_improvement_rate'], metrics['savns_time_increase'])
    
    return metrics

def _get_algorithm_results(self, algo_name):
    """获取指定算法的结果"""
    for zone_results in self.adaptive_results.values():
        if algo_name in zone_results:
            return zone_results[algo_name]
    return None
    
def _get_parameter_plots(self) -> List[str]:
    """获取参数自适应曲线图路径"""
    plots_dir = os.path.join(self.config.output_dir, 'plots')
    return [f for f in os.listdir(plots_dir) 
            if f.startswith('parameter_adaptation_')]
            
def _analyze_parameter_convergence(self, results: List[Dict]) -> str:
    """分析参数收敛性"""
    converged = 0
    total = len(results)
    
    for result in results:
        if 'parameter_history' in result:
            history = result['parameter_history']
            if len(history) >= 2:
                # 计算最后5次迭代的标准差
                std = np.std(history[-5:])
                if std < 0.01:  # 收敛阈值
                    converged += 1
                    
    convergence_rate = converged / max(1, total)
    
    if convergence_rate > 0.8:
        return "良好"
    elif convergence_rate > 0.5:
        return "一般"
    else:
        return "欠佳"
        
def _generate_conclusions(self) -> str:
    """生成实验结论"""
    baseline_metrics = self._get_baseline_metrics()
    hybrid_metrics = self._get_hybrid_metrics()
    adaptive_metrics = self._get_adaptive_metrics()
    
    # 找出最佳基线算法
    best_baseline = min(baseline_metrics.items(),
                        key=lambda x: x[1]['objective'])
                        
    # 找出最佳混合算法
    best_hybrid = min(hybrid_metrics.items(),
                        key=lambda x: x[1]['objective'])
                        
    # 计算平均改进率
    avg_improvement = np.mean([m['improvement'] 
                                for m in hybrid_metrics.values()])
                                
    conclusions = f"""
    <p>基于实验结果,我们得出以下主要结论:</p>
    
    <p>1. 算法性能对比</p>
    <ul>
        <li>在基线算法中,{best_baseline[0]}表现最佳,平均目标值为{best_baseline[1]['objective']:.2f}</li>
        <li>混合算法{best_hybrid[0]}取得了最好效果,平均目标值为{best_hybrid[1]['objective']:.2f}</li>
        <li>混合算法相比基线算法平均提升了{avg_improvement*100:.1f}%</li>
    </ul>
    
    <p>2. 自适应参数分析</p>
    <ul>
    """
    
    for config, metrics in adaptive_metrics.items():
        conclusions += f"<li>{config}的参数收敛性{metrics['convergence']}</li>"
        
    conclusions += """
    </ul>
    
    <p>3. 总体建议</p>
    <ul>
        <li>对于一般规模问题,建议使用混合算法方案</li>
        <li>对于大规模问题,自适应参数方案可能更具优势</li>
        <li>算法选择需要根据具体问题特点和计算资源进行权衡</li>
    </ul>
    """
    
    return conclusions
        
def run_experiments():
    """运行完整实验"""
    # 1. 初始化
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    config = ExperimentConfig(
        excel_path = os.path.join(project_root, "data", "littleorders.xlsx"),
        city_path = os.path.join(project_root, "data", "wuhan_city.geojson"),
        road_path = os.path.join(project_root, "data", "best_aligned_road_network.geojson"),
        output_dir = os.path.join(project_root, f"experiments_{timestamp}")
    )
    
    # 2. 创建实验框架
    framework = ExperimentFramework(config)
    
    # 3. 加载数据
    df, city_gdf, road_graph = framework.load_data()
    
    # 4. 准备路区
    zones_gdf, zone_map = framework.prepare_zones(df, city_gdf)
    
    # 5. 运行三组实验
    framework.run_baseline_experiments(zone_map, road_graph)
    framework.run_hybrid_experiments(zone_map, road_graph)
    framework.run_adaptive_experiments(zone_map, road_graph)
    
    # 6. 分析结果
    framework.analyze_results()
    
            
        
def main():
    # 基础配置与初始化
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 实验配置
    config = SolverConfig(
        excel_path = os.path.join(project_root, "data", "orders.xlsx"),
        city_path = os.path.join(project_root, "data", "wuhan_city.geojson"),
        road_path = os.path.join(project_root, "data", "best_aligned_road_network.geojson"),
        output_dir = os.path.join(project_root, f"experiments_{timestamp}"),
        
        # 实验参数配置
        vehicle_capacity = 5000.0,
        max_route_time = 12.0,
        random_seed = 42,
        
        # 并行计算配置
        parallel_evaluation = True,
        max_workers = os.cpu_count(),
        
        # 算法特定参数
        cw_parallel_savings = True,
        cw_time_window_enforce = False
    )
    
    # 创建输出目录结构
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, 'metrics'), exist_ok=True)

    # 设置日志
    configure_logging(config.output_dir)
    logging.info("=== 开始VRP算法对比实验 ===")
    
    # 第一步: 数据准备
    logging.info("第一步: 准备实验数据...")
    deduped_df = load_and_prepare_merchant_data(config.excel_path)
    
    # 第二步: 初始化分区器
    logging.info("第二步: 初始化区域分区...")
    partitioner = EnhancedZonePartitioner(config.output_dir)
    partitioner.merchant_gdf = gpd.GeoDataFrame(
        deduped_df,
        geometry=gpd.points_from_xy(deduped_df["经度"], deduped_df["纬度"]),
        crs="EPSG:4326"
    )
    partitioner.city_boundary = gpd.read_file(config.city_path).to_crs("EPSG:4326")
    partitioner.road_network = gpd.read_file(config.road_path).to_crs("EPSG:4326")
    
    # 第三步: 生成实验路区
    logging.info("第三步: 生成实验路区...")
    zones_gdf = partitioner.generate_zones(
        min_clusters=config.min_clusters,
        max_clusters=config.max_clusters,
        road_buffer_distance=config.road_buffer_distance
    )
    
    # 第四步: 构建路网
    logging.info("第四步: 构建实验路网...")
    rnb = RoadNetworkBuilder(config.road_path)
    rnb.build_graph()
    road_graph = rnb.graph
    
    # 第五步: 准备实验数据
    logging.info("第五步: 准备区域数据...")
    zone_map = split_merchants_by_zone(partitioner.merchant_gdf, zones_gdf)
    # 配置选址参数
    advanced_cfg = AdvancedFLPConfig(
        fixed_cost=1000.0,
        operation_cost=2.0, 
        coverage_radius=50.0
    )

    solver_cfg = ZoneFLPSolverConfig(
        use_gurobi=True,
        gurobi_time_limit=300,
        random_seed=42  
    )

    # 执行选址
    flp_solver = FacilityLocationSolver(
        zone_map=zone_map,
        advanced_cfg=advanced_cfg,
        solver_cfg=solver_cfg
    )    
    # 第六步: 仓库选址
    logging.info("第六步: 执行仓库选址...")
    warehouses = run_facility_location(zone_map, config)
    # ==========【粘贴开始】==========
    # 把 run_facility_location 返回的每个仓库字典，提取成 (longitude, latitude) 元组
    warehouses_coords = []
    for w_ in warehouses:
        # w_ 大概是 {"zone_id":"Zxx","longitude":111.1,"latitude":30.2,"capacity":...}
        warehouses_coords.append((w_["longitude"], w_["latitude"]))

    logging.info(f"已从 {len(warehouses)} 个字典中生成 {len(warehouses_coords)} 个仓库坐标元组.")
    # ==========【粘贴结束】==========
        
    # 创建实验框架实例
    experiment = VRPExperiment(
        config=config,
        zone_map=zone_map,
        warehouses=warehouses_coords,
        partitioner=partitioner,
        road_graph=road_graph
    )
    # 设置实验开关
    experiment.experiment_flags = {
        'run_baseline': True,
        'run_hybrid': True,
        'run_adaptive': True,
        'run_enhanced': True
    }

    # 人为来个 experiment.run_vehicle_config_experiment(...) 
    # 给它一些候选值:
    capacity_candidates = {
        'small':  [(1500,2000), (2001,3000)],
        'medium': [(3001,5000)],
        'large':  [(5001,8000),(8001,10000)]
    }
    fixed_cost_candidates = {
        'small': [600,800],
        'medium':[900,1000],
        'large': [1400,1500]
    }
    count_candidates = {
        'small':[5,10],
        'medium':[5],
        'large':[2,5]
    }

    # 选一个 zone，比如"Z001"，选Solver="TS"
    result_dict = experiment.run_vehicle_config_experiment(
        zone_id="Z001",
        solver_type="TS",
        capacity_candidates=capacity_candidates,
        fixed_cost_candidates=fixed_cost_candidates,
        count_candidates=count_candidates
    )

    if result_dict:
        print("\n=== 实验16 结果 ===")
        print("最优配置:", result_dict['best_config'])
        print("最优目标值:", result_dict['best_objective'])
        # all_results 里存了每次组合的详情
        print("共记录组合数:", len(result_dict['all_results']))

        # 比如:
        ex16_rows = result_dict['all_results']
        # ex16_rows 是很多 dict, 里边还有 'vehicle_config' 这种嵌套dict 
        # 你要序列化一下
        df_ex16 = pd.DataFrame(ex16_rows)
        # JSON序列化
        df_ex16['vehicle_config'] = df_ex16['vehicle_config'].apply(lambda x: json.dumps(x, ensure_ascii=False))
        
        # 存文件
        out_csv = os.path.join(config.output_dir, 'results', 'experiment16_vehicle_config.csv')
        df_ex16.to_csv(out_csv, index=False, encoding='utf-8-sig')
        logging.info(f"实验16车辆配置结果已保存: {out_csv}")


    # ===== 开始12个实验 =====
    logging.info("\n=== 开始执行12组实验 ===")
    
    # 1-4: 基线算法实验
    logging.info("\n[实验1-4] 基线算法实验...")
    baseline_algos = {
        "CW": "Clarke-Wright算法",
        "SA": "模拟退火算法",
        "TS": "禁忌搜索算法",
        "VNS": "变邻域搜索算法"
    }
    
    if experiment.experiment_flags['run_baseline']:
        for algo_code, algo_name in baseline_algos.items():
            logging.info(f"\n开始 {algo_name} 基线实验...")
            for zone_id in zone_map.keys():
                metrics = experiment.run_baseline_experiment(zone_id, algo_code)
                if metrics:
                    if zone_id not in experiment.baseline_results:
                        experiment.baseline_results[zone_id] = {}
                    # 这里是关键修改
                    experiment.baseline_results[zone_id][algo_code] = metrics

    
    # 5-8: 混合算法实验
    logging.info("\n[实验5-8] 混合算法实验...")
    hybrid_combinations = [
        ("CW", "TS", "Clarke-Wright + 禁忌搜索"),
        ("CW", "VNS", "Clarke-Wright + 变邻域搜索"),
        ("SA", "TS", "模拟退火 + 禁忌搜索"),
        ("SA", "VNS", "模拟退火 + 变邻域搜索")
    ]
    
    if experiment.experiment_flags['run_hybrid']:
        for first_stage, second_stage, combo_name in hybrid_combinations:
            logging.info(f"\n开始 {combo_name} 混合实验...")
            for zone_id in zone_map.keys():
                metrics = experiment.run_hybrid_experiment(
                    zone_id,
                    first_stage,
                    second_stage
                )
                if metrics:
                # 修改这里
                    if zone_id not in experiment.hybrid_results:
                        experiment.hybrid_results[zone_id] = {}
                    experiment.hybrid_results[zone_id][f"{first_stage}_{second_stage}"] = metrics
    
    # 9-12: 自适应参数实验
    logging.info("\n[实验9-12] 自适应参数实验...")
    adaptive_configs = [
        # Exp.9: TS vs TS-ADP
        ("TS", {
            "tabu_size": "adaptive",
            "min_size": 5,
            "max_size": 20
        }, "TS vs TS-ADP"),
        
        # Exp.10: VNS vs VNS-ADP 
        ("VNS", {
            "neighborhood_change": "adaptive",
            "min_neighborhoods": 2,
            "max_neighborhoods": 5
        }, "VNS vs VNS-ADP"),
        
        # Exp.11: CW→TS (固定) vs CW→TS-ADP
        ("CW_TS", {
            "first_stage": "CW",
            "second_stage": "TS",
            "tabu_size": "adaptive",
            "min_size": 5,
            "max_size": 20
        }, "CW→TS vs CW→TS-ADP"),
        
        # Exp.12: SA→VNS (固定) vs SA→VNS-ADP
        ("SA_VNS", {
            "first_stage": "SA",
            "second_stage": "VNS", 
            "neighborhood_change": "adaptive",
            "min_neighborhoods": 2,
            "max_neighborhoods": 5
        }, "SA→VNS vs SA→VNS-ADP")
    ]

    # 3. 执行实验
    logging.info("\n=== 开始实验(9-12) ===")
    if experiment.experiment_flags['run_adaptive']:
        for algo, params, desc in adaptive_configs:
            logging.info(f"\n开始 {desc} 实验...")
            for zone_id in zone_map.keys():
                metrics = experiment.run_adaptive_experiment(
                    zone_id,
                    algo,
                    params
                )
                if metrics:
                    experiment.adaptive_results[zone_id] = {
                        f"{algo}_ADP_{list(params.keys())[0]}": metrics
                    }

    enhanced_configs = {
        'multi_vehicle_ts': {  # 实验13: 基于TS的多车型实验
            'base_algorithm': 'TS',
            'tabu_size': 10,
            'max_iterations': 50,
            'vehicle_types': {
                'small': {'capacity': 3000, 'fixed_cost': 800, 'count': 20},
                'medium': {'capacity': 5000, 'fixed_cost': 1000, 'count': 10},
                'large': {'capacity': 8000, 'fixed_cost': 1500, 'count': 5}
            }
        },
        'multi_vehicle_cwts': {  # 实验14: 基于CW_TS的多车型实验
            'first_stage': 'CW',
            'second_stage': 'TS',
            'tabu_size': 10,
            'vehicle_types': {
                'small': {'capacity': 3000, 'fixed_cost': 800, 'count': 20},
                'medium': {'capacity': 5000, 'fixed_cost': 1000, 'count': 10},
                'large': {'capacity': 8000, 'fixed_cost': 1500, 'count': 5}
            }
        },
        'multi_vehicle_tsadp': {  # 实验15: 基于TS-ADP的多车型实验
            'algorithm': 'TS-ADP',
            'vehicle_types': {
                'small': {'capacity': 3000, 'fixed_cost': 800, 'count': 20},
                'medium': {'capacity': 5000, 'fixed_cost': 1000, 'count': 10},
                'large': {'capacity': 8000, 'fixed_cost': 1500, 'count': 5},

            },
            # 这里新建一个 adaptive_params，用来放 min_tabu_size / max_tabu_size
            "adaptive_params": {
                "min_tabu_size": 5,
                "max_tabu_size": 20
            }            
        }
    }

    # 3. 执行增强实验(13-15)
    logging.info("\n=== 开始多车型增强实验 ===")
    if experiment.experiment_flags['run_enhanced']:
        for exp_name, exp_config in enhanced_configs.items():
            logging.info(f"\n开始 {exp_name} 实验...")
            for zone_id in zone_map.keys():
                metrics = experiment.run_enhanced_experiment(
                    zone_id,
                    exp_config
                )
                if metrics:
                    if 'enhanced_results' not in experiment.__dict__:
                        experiment.enhanced_results = {}
                    if zone_id not in experiment.enhanced_results:
                        experiment.enhanced_results[zone_id] = {}
                    experiment.enhanced_results[zone_id][exp_name] = metrics

    # 4. 保存结果和生成报告
    logging.info("\n保存所有实验结果...")
    experiment.save_results()  # 需要修改save_results()以包含enhanced_results
   
# 执行统计分析
    logging.info("\n执行统计分析...")
    results_csv = os.path.join(config.output_dir, 'results', 'baseline_results.csv')
    try:
        if os.path.exists(results_csv) and os.path.getsize(results_csv) > 0:
            results_df = pd.read_csv(results_csv)
            analyzer = VRPStatisticalAnalyzer(config.output_dir)
            statistical_results = analyzer.run_full_analysis()
        else:
            logging.warning("基线结果文件为空，创建默认统计分析结果")
            statistical_results = {
                'pairwise_t_tests': [],
                'pairwise_wilcoxon': [],
                'friedman_test': {
                    'valid': False,
                    'error': '无基线数据',
                    'statistic': 0,
                    'p_value': 1.0,
                    'significant': False
                },
                'average_rankings': {}
            }
    except Exception as e:
        logging.error(f"统计分析出错: {str(e)}")
        statistical_results = {
            'pairwise_t_tests': [],
            'pairwise_wilcoxon': [],
            'friedman_test': {
                'valid': False,
                'error': str(e),
                'statistic': 0,
                'p_value': 1.0,
                'significant': False
            },
            'average_rankings': {}
        }

    # 生成实验报告
    logging.info("\n生成实验分析报告...")
    report_path = os.path.join(config.output_dir, "experiment_report.html")
    try:
        generate_experiment_report(
            experiment=experiment,
            statistical_analysis=statistical_results,
            report_path=report_path
        )
        logging.info(f"实验报告已生成: {report_path}")
    except Exception as e:
        logging.error(f"生成报告失败: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())

    logging.info("=== 实验全部完成 ===")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"实验执行出错: {str(e)}")
        raise e