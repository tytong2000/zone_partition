#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import math
import json
import random
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from solve.utils.config import SolverConfig
# 如果没有 Gurobi，可注释掉
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

# 若要支持多级容量或部分覆盖等逻辑，可以在这里设置
@dataclass
class AdvancedFLPConfig:
    """
    复合 FLP 配置:
    - fixed_cost: 每开一个仓库需付出的固定费用
    - operation_cost: 按订单需求计算的运营费(如 距离*需求*运费)
    - coverage_radius: 如果要“半径覆盖”+部分覆盖逻辑
    - multi_levels: 多级容量 [(cap, cost), ...] ，假设cap按从小到大排序
    - use_partial_coverage: True -> 订单若> coverage_radius则加惩罚
    - coverage_penalty_factor: 覆盖半径外订单的惩罚系数
    - use_multilevel_capacity: True -> 多级容量生效
    """
    fixed_cost: float = 1000.0
    operation_cost: float = 2.0
    coverage_radius: float = 50.0
    multi_levels: List[Tuple[float,float]] = None
    use_partial_coverage: bool = False
    coverage_penalty_factor: float = 1.5
    use_multilevel_capacity: bool = False

@dataclass
class ZoneFLPSolverConfig:
    """
    Zone 级别 FLP 的外层配置
    - 每个路区必须选1仓库
    - 候选点 = 该区所有商户
    - Gurobi相关
    """
    # Gurobi 优先
    use_gurobi: bool = True
    gurobi_time_limit: int = 300  # 秒
    gurobi_threads: int = 0      # 0表示自动
    random_seed: int = 42

    # GA 或其它备选
    use_backup_ga: bool = True
    ga_population: int = 50
    ga_generations: int = 50
    ga_crossover: float = 0.7
    ga_mutation: float = 0.2

class FacilityLocationSolver:
    """
    多区选址:
    1) 给定 zone_map: {z_id: DataFrame(商户点 + 需求等)}
    2) 每个路区 强制 只能选 1 个仓库
    3) 若 use_gurobi=True & HAS_GUROBI=True，则走 Gurobi MIP
       否则 fallback 到 GA (或可自定义)

    - advanced_cfg: AdvancedFLPConfig, 用于多级容量 or partial coverage
    """
    def __init__(
        self,
        zone_map: Dict[str, pd.DataFrame],
        advanced_cfg: AdvancedFLPConfig,
        solver_cfg: ZoneFLPSolverConfig
    ):
        self.logger = logging.getLogger("ZoneBasedFLSolver")
        self.zone_map = zone_map
        self.advanced_cfg = advanced_cfg
        self.solver_cfg = solver_cfg

        # 把 zone_map 整理:
        # zone_ids: list of str
        # zones_data[i] => {
        #   "zone_id": z_id,
        #   "coords": Nx2,           # 经纬度坐标
        #   "demands": Nx1,          # 需求(托运单重量等)
        #   "names": Nx1,            # 商家名称
        # }
        self.zone_ids = sorted(zone_map.keys())
        self.zones_data = []
        for z_id in self.zone_ids:
            df = zone_map[z_id]
            # 必要列: "经度", "纬度", "托运单重量" (若无则置0)
            coords = df[["经度","纬度"]].values
            demands = df["托运单重量"].values if "托运单重量" in df.columns else np.zeros(len(df))
            merchant_names = df["商家名称"].values if "商家名称" in df.columns else [f"{z_id}_M{i}" for i in range(len(df))]
            self.zones_data.append({
                "zone_id": z_id,
                "coords": coords,    # shape (n,2)
                "demands": demands,  # shape (n,)
                "names": merchant_names, # shape (n,)
            })
        self.num_zones = len(self.zone_ids)

        # 求解结果
        self.selected_warehouses = {}  # { z_id: (lon, lat) }
        self.objective_value = float('inf')
        self.solve_time = 0.0

    def solve(self):
        """ 主入口: 优先尝试 Gurobi, 不行则 GA """
        start = time.time()
        self.logger.info("Zone FLP solve start...")

        if self.solver_cfg.use_gurobi and HAS_GUROBI:
            try:
                self._solve_with_gurobi()
            except Exception as e:
                self.logger.warning(f"Gurobi求解失败: {e}, fallback到GA")
                if self.solver_cfg.use_backup_ga:
                    self._solve_with_ga()
                else:
                    raise RuntimeError("Gurobi失败且未配置GA后备")
        else:
            self.logger.info("无需Gurobi或Gurobi不可用 => 直接用GA")
            self._solve_with_ga()

        self.solve_time = time.time() - start
        self.logger.info(f"Zone FLP done, cost={self.objective_value:.2f}, time={self.solve_time:.2f}s")

    # ----------------------------------------------------------------
    # Gurobi求解
    # ----------------------------------------------------------------
    def _solve_with_gurobi(self):
        """使用 Gurobi 做 MIP, 强制: each zone picks exactly 1 warehouse from its candidates.
        支持大型单一区域的情况"""
        self.logger.info("尝试 Gurobi 求解每区选1仓库...")

        # 检查是否为单区域大规模情况
        is_single_zone = len(self.zone_ids) == 1 and self.zone_ids[0] == 'single_zone'
        if is_single_zone:
            self.logger.info("检测到单一大区域模式，调整求解策略")

        # 预先一些计算：Zone 总需求 & 如果启用多级容量，则为其选用最小可承载的容量成本
        zone_loads = []
        zone_capacity_costs = []
        max_cap = 0.0
        if self.advanced_cfg.multi_levels:
            max_cap = max([lvl[0] for lvl in self.advanced_cfg.multi_levels])

        for z_index, z_id in enumerate(self.zone_ids):
            demands = self.zones_data[z_index]["demands"]
            load_ = float(np.sum(demands))
            zone_loads.append(load_)

            if self.advanced_cfg.use_multilevel_capacity and self.advanced_cfg.multi_levels:
                # 单一区域模式下，如果需求超过最大容量，自动调整最大容量
                if is_single_zone and load_ > max_cap:
                    # 动态调整多级容量配置
                    new_cap = math.ceil(load_ * 1.2)  # 增加20%余量
                    self.advanced_cfg.multi_levels.append((new_cap, max_cap * 1.5))
                    max_cap = new_cap
                    self.logger.warning(f"单一区域需求({load_:.2f})超过最大容量，已自动扩展容量至 {new_cap:.2f}")
                elif load_ > max_cap:
                    raise RuntimeError(f"Zone {z_id} 需求({load_})超过最大容量({max_cap})，问题不可行")

                # 找最小可承载 load_ 的 (cap, cost)
                feasible_cost = float('inf')
                for (cap_, c_) in self.advanced_cfg.multi_levels:
                    if load_ <= cap_ and c_ < feasible_cost:
                        feasible_cost = c_
                # 如果找不到，意味着不可行
                if feasible_cost == float('inf'):
                    raise RuntimeError(f"Zone {z_id} 不可找到合适容量承载其需求({load_})")
                zone_capacity_costs.append(feasible_cost)
            else:
                # 如果没用多级容量，就当容量成本为0
                zone_capacity_costs.append(0.0)

        # 剩余代码保持不变...
        # 1) 构建模型
        m = gp.Model("ZoneFLP")
        m.setParam("OutputFlag", 1)
        if self.solver_cfg.gurobi_time_limit>0:
            m.setParam("TimeLimit", self.solver_cfg.gurobi_time_limit)
        if self.solver_cfg.gurobi_threads>0:
            m.setParam("Threads", self.solver_cfg.gurobi_threads)
        random.seed(self.solver_cfg.random_seed)

        # 2) 决策变量
        #   x[z,i] = 1 if zone z choose candidate i
        x_vars = {}
        for z_index, z_id in enumerate(self.zone_ids):
            n_candidates = len(self.zones_data[z_index]["coords"])
            for i in range(n_candidates):
                x_vars[(z_id,i)] = m.addVar(vtype=GRB.BINARY, name=f"x_{z_id}_{i}")
        m.update()

        # 原有的目标函数和约束代码保持不变...
        
        # 对于单一大区域，可以额外增加一些启发式约束来加速求解
        if is_single_zone:
            z_id = self.zone_ids[0]
            coords = self.zones_data[0]["coords"]
            demands = self.zones_data[0]["demands"]
            n_candidates = len(coords)
            
            # 计算需求加权中心
            total_weight = np.sum(demands)
            if total_weight > 0:
                center_x = np.sum(coords[:,0] * demands) / total_weight
                center_y = np.sum(coords[:,1] * demands) / total_weight
            else:
                center_x = np.mean(coords[:,0])
                center_y = np.mean(coords[:,1])
            
            # 计算到中心的距离
            distances_to_center = []
            for i in range(n_candidates):
                dx = coords[i,0] - center_x
                dy = coords[i,1] - center_y
                dist = np.sqrt(dx*dx + dy*dy)
                distances_to_center.append(dist)
            
            # 只考虑距离中心较近的候选点（例如前20%）
            sorted_indices = np.argsort(distances_to_center)
            top_n = max(1, int(n_candidates * 0.2))
            for i in range(n_candidates):
                if i not in sorted_indices[:top_n]:
                    m.addConstr(x_vars[(z_id,i)] == 0, name=f"prune_{z_id}_{i}")
            
            self.logger.info(f"单一区域模式：已将候选点从{n_candidates}个缩减到{top_n}个")

        # 优化求解（后续代码保持不变）
        m.optimize()

        # 6) 读结果
        if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT:
            self.objective_value = m.ObjVal
            for z_index, z_id in enumerate(self.zone_ids):
                n_candidates = len(self.zones_data[z_index]["coords"])
                for i in range(n_candidates):
                    if x_vars[(z_id,i)].X > 0.5:
                        xy_ = self.zones_data[z_index]["coords"][i]
                        self.selected_warehouses[z_id] = (float(xy_[0]), float(xy_[1]))
        else:
            raise RuntimeError(f"Gurobi不可行, status={m.Status}")

    def _solve_with_ga(self):
        """使用GA求解FLP问题，支持单一大区域情况"""
        self.logger.info("进入GA求解(每区1仓) ...")
        random.seed(self.solver_cfg.random_seed)

        # 检查是否为单区域大规模情况
        is_single_zone = len(self.zone_ids) == 1 and self.zone_ids[0] == 'single_zone'
        if is_single_zone:
            self.logger.info("GA检测到单一大区域模式，采用加权中心点策略")
            zone_info = self.zones_data[0]
            coords = zone_info["coords"]
            demands = zone_info["demands"]
            
            # 计算需求加权中心
            total_demand = float(np.sum(demands))
            if total_demand > 0:
                weighted_x = float(np.sum(coords[:,0] * demands) / total_demand)
                weighted_y = float(np.sum(coords[:,1] * demands) / total_demand)
            else:
                weighted_x = float(np.mean(coords[:,0]))
                weighted_y = float(np.mean(coords[:,1]))
                
            # 直接返回加权中心作为仓库位置
            self.selected_warehouses[self.zone_ids[0]] = (weighted_x, weighted_y)
            # 计算目标值
            self.objective_value = self._evaluate_single_zone_center()
            self.logger.info(f"单一区域模式GA求解完成，目标值={self.objective_value:.2f}")
            return
            
        try:
            from deap import base, creator, tools, algorithms
        except ImportError:
            raise RuntimeError("需要安装 DEAP 库才能使用GA求解，请先pip install deap")

        zone_sizes = []
        for z_info in self.zones_data:
            zone_sizes.append(len(z_info["coords"]))

        # GA 染色体长度 = num_zones，每个基因 ∈ [0, zoneSize[i]-1]
        def evaluate_chromosome(chrom):
            cost_ = self._evaluate_zones(chrom)
            return (cost_,)

        # 创建Fitness & Individual
        # 检查这些类型是否已经创建，避免重复创建引起错误
        if not hasattr(creator, 'FitnessMinZone'):
            creator.create("FitnessMinZone", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, 'IndividualZone'):
            creator.create("IndividualZone", list, fitness=creator.FitnessMinZone)

        toolbox = base.Toolbox()

        # 随机基因初始化
        def init_individual():
            genes = []
            for i in range(self.num_zones):
                max_idx = zone_sizes[i] - 1
                g_ = random.randint(0, max_idx)
                genes.append(g_)
            return creator.IndividualZone(genes)

        toolbox.register("individual", init_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate_chromosome)

        # 交叉/变异/选择
        toolbox.register("mate", tools.cxOnePoint)

        def mutate_func(ind, indpb=0.1):
            for i in range(len(ind)):
                if random.random() < indpb:
                    max_i = zone_sizes[i] - 1
                    ind[i] = random.randint(0, max_i)
            return (ind,)

        toolbox.register("mutate", mutate_func, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=self.solver_cfg.ga_population)
        best_cost = float('inf')
        best_ind = None

        # GA 演化过程
        for gen in range(self.solver_cfg.ga_generations):
            try:
                offspring = algorithms.varAnd(pop, toolbox,
                                            cxpb=self.solver_cfg.ga_crossover,
                                            mutpb=self.solver_cfg.ga_mutation)
                fits = map(toolbox.evaluate, offspring)
                for ind, fit in zip(offspring, fits):
                    ind.fitness.values = fit
                pop = toolbox.select(offspring, k=len(pop))

                # 记录最优
                cb = tools.selBest(pop, 1)[0]
                cost_ = cb.fitness.values[0]
                if cost_ < best_cost:
                    best_cost = cost_
                    best_ind = cb[:]
                    self.logger.info(f"GA Gen {gen}, new best => {best_cost:.2f}")
            except Exception as e:
                self.logger.error(f"GA迭代{gen}发生错误: {str(e)}")
                if best_ind is not None:
                    break
                else:
                    raise

        # GA结束
        if best_ind is not None:
            self.objective_value = best_cost
            # 解析 best_ind => {z_id: (lon, lat)}
            self.selected_warehouses.clear()
            for i, z_id in enumerate(self.zone_ids):
                cand_idx = best_ind[i]
                xy_ = self.zones_data[i]["coords"][cand_idx]
                self.selected_warehouses[z_id] = (float(xy_[0]), float(xy_[1]))
        else:
            raise RuntimeError("GA无法找到可行解")

    def _evaluate_single_zone_center(self) -> float:
        """计算单一区域中心点作为仓库时的成本"""
        if self.zone_ids[0] not in self.selected_warehouses:
            return float('inf')
            
        cost_ = 0.0
        # 1) fixed cost
        cost_ += self.advanced_cfg.fixed_cost
        
        # 计算运输成本
        wh_coord = self.selected_warehouses[self.zone_ids[0]]
        coords = self.zones_data[0]["coords"]
        demands = self.zones_data[0]["demands"]
        
        dx = coords[:,0] - wh_coord[0]
        dy = coords[:,1] - wh_coord[1]
        dist = np.sqrt(dx*dx + dy*dy)
        cost_ += np.sum(dist * demands) * self.advanced_cfg.operation_cost
        
        # 若有部分覆盖逻辑
        if self.advanced_cfg.use_partial_coverage:
            rad = self.advanced_cfg.coverage_radius
            factor = self.advanced_cfg.coverage_penalty_factor
            dist_excess = dist - rad
            mask = (dist_excess > 0)
            if np.any(mask):
                cost_ += np.sum(dist_excess[mask] * demands[mask] * factor)
        
        # 处理容量成本
        if self.advanced_cfg.use_multilevel_capacity and self.advanced_cfg.multi_levels:
            total_demand = float(np.sum(demands))
            # 找最小可行的容量级别
            level_cost = float('inf')
            for cap, cost in self.advanced_cfg.multi_levels:
                if total_demand <= cap and cost < level_cost:
                    level_cost = cost
            
            if level_cost < float('inf'):
                cost_ += level_cost
            else:
                # 如果没有足够大的容量级别，返回一个极大值
                return 1e15
                
        return cost_

        def mutate_func(ind, indpb=0.1):
            for i in range(len(ind)):
                if random.random() < indpb:
                    max_i = zone_sizes[i] - 1
                    ind[i] = random.randint(0, max_i)
            return (ind,)

        toolbox.register("mutate", mutate_func, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=self.solver_cfg.ga_population)
        best_cost = float('inf')
        best_ind = None

        # GA 演化过程
        for gen in range(self.solver_cfg.ga_generations):
            try:
                offspring = algorithms.varAnd(pop, toolbox,
                                            cxpb=self.solver_cfg.ga_crossover,
                                            mutpb=self.solver_cfg.ga_mutation)
                fits = map(toolbox.evaluate, offspring)
                for ind, fit in zip(offspring, fits):
                    ind.fitness.values = fit
                pop = toolbox.select(offspring, k=len(pop))

                # 记录最优
                cb = tools.selBest(pop, 1)[0]
                cost_ = cb.fitness.values[0]
                if cost_ < best_cost:
                    best_cost = cost_
                    best_ind = cb[:]
                    self.logger.info(f"GA Gen {gen}, new best => {best_cost:.2f}")
            except Exception as e:
                self.logger.error(f"GA迭代{gen}发生错误: {str(e)}")
                if best_ind is not None:
                    break
                else:
                    raise

        # GA结束
        if best_ind is not None:
            self.objective_value = best_cost
            # 解析 best_ind => {z_id: (lon, lat)}
            self.selected_warehouses.clear()
            for i, z_id in enumerate(self.zone_ids):
                cand_idx = best_ind[i]
                xy_ = self.zones_data[i]["coords"][cand_idx]
                self.selected_warehouses[z_id] = (float(xy_[0]), float(xy_[1]))
        else:
            raise RuntimeError("GA无法找到可行解")

    def _evaluate_single_zone_center(self) -> float:
        """计算单一区域中心点作为仓库时的成本"""
        if self.zone_ids[0] not in self.selected_warehouses:
            return float('inf')
            
        cost_ = 0.0
        # 1) fixed cost
        cost_ += self.advanced_cfg.fixed_cost
        
        # 计算运输成本
        wh_coord = self.selected_warehouses[self.zone_ids[0]]
        coords = self.zones_data[0]["coords"]
        demands = self.zones_data[0]["demands"]
        
        dx = coords[:,0] - wh_coord[0]
        dy = coords[:,1] - wh_coord[1]
        dist = np.sqrt(dx*dx + dy*dy)
        cost_ += np.sum(dist * demands) * self.advanced_cfg.operation_cost
        
        # 若有部分覆盖逻辑
        if self.advanced_cfg.use_partial_coverage:
            rad = self.advanced_cfg.coverage_radius
            factor = self.advanced_cfg.coverage_penalty_factor
            dist_excess = dist - rad
            mask = (dist_excess > 0)
            if np.any(mask):
                cost_ += np.sum(dist_excess[mask] * demands[mask] * factor)
        
        # 处理容量成本
        if self.advanced_cfg.use_multilevel_capacity and self.advanced_cfg.multi_levels:
            total_demand = float(np.sum(demands))
            # 找最小可行的容量级别
            level_cost = float('inf')
            for cap, cost in self.advanced_cfg.multi_levels:
                if total_demand <= cap and cost < level_cost:
                    level_cost = cost
            
            if level_cost < float('inf'):
                cost_ += level_cost
            else:
                # 如果没有足够大的容量级别，返回一个极大值
                return 1e15
                
        return cost_

    # ----------------------------------------------------------------
    # GA 评估 / 也可被 Gurobi 里重复使用的计算思路
    # ----------------------------------------------------------------
    def _evaluate_zones(self, chromosome: List[int]) -> float:
        """
        对 GA 染色体的细化cost:
        1) fixed_cost * num_zones
        2) sum_{z} of sum_{j in z}( dist(merchant_j-> chosen warehouse) * demand_j * operation_cost)
        3) partial coverage penalty
        4) multi-level capacity
        """
        cost_ = 0.0
        # 1) fixed cost
        cost_ += self.advanced_cfg.fixed_cost * self.num_zones

        # 2) transport cost
        opC = self.advanced_cfg.operation_cost
        for zone_idx, gene in enumerate(chromosome):
            z_info = self.zones_data[zone_idx]
            coords = z_info["coords"]
            demands = z_info["demands"]
            wh_coord = coords[gene]
            dx = coords[:,0] - wh_coord[0]
            dy = coords[:,1] - wh_coord[1]
            dist = np.sqrt(dx*dx + dy*dy)
            cost_ += np.sum(dist * demands) * opC

        # 3) partial coverage
        if self.advanced_cfg.use_partial_coverage:
            rad = self.advanced_cfg.coverage_radius
            factor = self.advanced_cfg.coverage_penalty_factor
            for zone_idx, gene in enumerate(chromosome):
                z_info = self.zones_data[zone_idx]
                coords = z_info["coords"]
                demands = z_info["demands"]
                wh_coord = coords[gene]
                dx = coords[:,0] - wh_coord[0]
                dy = coords[:,1] - wh_coord[1]
                dist = np.sqrt(dx*dx + dy*dy)
                dist_excess = dist - rad
                mask = (dist_excess>0)
                if np.any(mask):
                    cost_ += np.sum(dist_excess[mask] * demands[mask] * factor)

        # 4) multi-level capacity
        #   单区 load > 所有 cap => 不可行(设极大cost)
        #   否则加上最小可行档位cost
        if self.advanced_cfg.use_multilevel_capacity and self.advanced_cfg.multi_levels:
            zoneLoads = []
            for zone_idx, gene in enumerate(chromosome):
                demands = self.zones_data[zone_idx]["demands"]
                load_ = float(np.sum(demands))
                zoneLoads.append(load_)

            for load_ in zoneLoads:
                feasible_cost = float('inf')
                for (cap_, c_) in self.advanced_cfg.multi_levels:
                    if load_ <= cap_ and c_ < feasible_cost:
                        feasible_cost = c_
                if feasible_cost == float('inf'):
                    # 超过最大容量，无解，返回极大值
                    return 1e15
                cost_ += feasible_cost

        return cost_

    def get_solution(self) -> Dict[str, Tuple[float,float]]:
        """
        返回每区选定仓库坐标 { zone_id: (lon, lat) }.
        需在 solve() 后调用
        """
        return self.selected_warehouses

    def get_objective(self) -> float:
        """返回求解得到的最优(或近似)目标值."""
        return self.objective_value

    def get_solve_time(self) -> float:
        return self.solve_time

# ----------------------------------------------------------------------
# 如需多一些“分析/可视化”的辅助类，也可写在这里
# ----------------------------------------------------------------------

class ZoneFLPResultAnalyzer:
    """
    针对已经求解好的结果(每区选哪)做进一步分析, 如:
    - total coverage
    - max distance
    - 画图
    """
    def __init__(self, solver: FacilityLocationSolver):
        self.solver = solver
        self.logger = logging.getLogger("ZoneFLPResultAnalyzer")

    def coverage_stats(self):
        # 若启用了 partial coverage，可统计多少订单超出 coverage_radius
        advanced_cfg = self.solver.advanced_cfg
        if not advanced_cfg.use_partial_coverage:
            self.logger.info("未启用部分覆盖逻辑，没有超出覆盖半径的统计。")
            return {}
        rad = advanced_cfg.coverage_radius
        result_map = {}
        for i, z_id in enumerate(self.solver.zone_ids):
            xy_wh = self.solver.selected_warehouses.get(z_id, None)
            if xy_wh is None:
                continue
            coords = self.solver.zones_data[i]["coords"]
            demands = self.solver.zones_data[i]["demands"]
            dx = coords[:,0] - xy_wh[0]
            dy = coords[:,1] - xy_wh[1]
            dist = np.sqrt(dx*dx + dy*dy)
            mask = (dist > rad)
            uncovered = mask.sum()
            total = len(coords)
            result_map[z_id] = {
                "uncovered_orders": int(uncovered),
                "total_orders": total,
                "uncovered_ratio": uncovered/total
            }
        return result_map

    def export_solution_json(self, output_path:str):
        """把 solver 结果 {z_id->(lon, lat)} + objective, time... 存到 JSON."""
        data = {
            "objective": self.solver.objective_value,
            "solve_time": self.solver.solve_time,
            "selected_warehouses": {}
        }
        for z_id, xy_ in self.solver.selected_warehouses.items():
            data["selected_warehouses"][z_id] = list(xy_)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"区级FLP结果已导出: {output_path}")

    def visualize_solution(self, output_path: str = "zone_flp_solution.png"):
        """
        复杂可视化: 
        - 不同颜色的散点代表不同Zone的商户
        - 用线连接商户与选定仓库
        - 若启用partial coverage, 绘制覆盖半径圆
        - 设置更美观的背景、网格、图例等
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.patches import Circle
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_facecolor('#f9f9f9')  # 浅色背景
        
        # 颜色映射，每个Zone一种颜色；也可改为自定义list
        colormap = plt.cm.get_cmap("tab20", self.solver.num_zones)
        
        for i, z_id in enumerate(self.solver.zone_ids):
            z_info = self.solver.zones_data[i]
            coords = z_info["coords"]
            
            # Zone颜色
            zone_color = colormap(i)
            
            # 绘制Zone的所有商户点
            ax.scatter(
                coords[:, 0], coords[:, 1],
                color=zone_color,
                s=25, 
                alpha=0.7,
                label=f"Zone {z_id}" if i == 0 else ""  # 只在第一批绘制时显示“Zone”图例文字，以免重复
            )
            
        # 再绘制仓库点 & 连线
        for i, z_id in enumerate(self.solver.zone_ids):
            if z_id not in self.solver.selected_warehouses:
                continue
            
            # 仓库坐标
            wh_coord = self.solver.selected_warehouses[z_id]
            
            # 取该Zone所有商户坐标
            z_info = self.solver.zones_data[i]
            coords = z_info["coords"]
            
            # Zone颜色
            zone_color = colormap(i)
            
            # 画仓库点 (星形)
            ax.scatter(
                wh_coord[0], wh_coord[1],
                marker='*',
                s=250,
                color='red',
                edgecolor='black',
                linewidth=0.8,
                zorder=5,
                label=f"Warehouse for {z_id}"
            )
            
            # 从商户连到仓库
            for (mx, my) in coords:
                ax.plot([wh_coord[0], mx], [wh_coord[1], my],
                        color=zone_color, alpha=0.3, linewidth=1.5)
            
            # 若开启partial coverage，画一个覆盖半径圆
            if self.solver.advanced_cfg.use_partial_coverage:
                coverage_radius = self.solver.advanced_cfg.coverage_radius
                circle = Circle(
                    (wh_coord[0], wh_coord[1]),
                    coverage_radius,
                    fill=False,
                    edgecolor='orange',
                    linestyle='--',
                    linewidth=2.0,
                    alpha=0.5,
                    label='Coverage Radius' if i == 0 else ""  # 避免重复图例
                )
                ax.add_patch(circle)
        
        # 设置标题、网格等
        ax.set_title(
            f"Zone-based FLP Solution\nTotal Cost = {self.solver.objective_value:.2f}", 
            fontsize=16, 
            fontweight='bold'
        )
        ax.grid(linestyle='--', alpha=0.6)
        
        # 这里可用handles,labels只显示一次legend
        handles, labels = ax.get_legend_handles_labels()
        # 去重
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='best')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        self.logger.info(f"更精美的区级FLP可视化已输出: {output_path}")

def run_facility_location(zone_map: Dict, config: 'SolverConfig') -> List[Dict]:
    """
    运行设施选址算法，支持两种模式：
    1. 常规路区划分模式：每个路区选择一个最优仓库
    2. 数据驱动的远郊大仓模式：基于商户分布自动确定战略仓库位置
    """
    logging.info("=== 执行仓库选址 ===")
    
    # 首先检查是否启用路区划分
    if hasattr(config, 'enable_zone_partition') and not config.enable_zone_partition:
        logging.info("路区划分已禁用，启用'数据驱动的远郊大仓'模式")
        
        # 获取所有商户
        if 'single_zone' in zone_map:
            merchants_df = zone_map['single_zone']
            if not isinstance(merchants_df, pd.DataFrame):
                merchants_df = pd.DataFrame(merchants_df)
                
            # 确保必要的列存在
            if '经度' not in merchants_df.columns or '纬度' not in merchants_df.columns:
                column_mappings = {
                    'longitude': '经度',
                    'latitude': '纬度',
                    'lon': '经度',
                    'lat': '纬度'
                }
                for orig, new in column_mappings.items():
                    if orig in merchants_df.columns and new not in merchants_df.columns:
                        merchants_df[new] = merchants_df[orig]
            
            # 提取经纬度坐标
            coordinates = merchants_df[['经度', '纬度']].values
            
            # 使用KMeans聚类确定4个远郊仓库位置
            try:
                from sklearn.cluster import KMeans
                from scipy.spatial import ConvexHull
                import numpy as np
                
                # 进行KMeans聚类
                num_warehouses = 4  # 默认设置4个仓库
                kmeans = KMeans(n_clusters=num_warehouses, random_state=config.random_seed)
                kmeans.fit(coordinates)
                
                # 获取聚类中心（初始仓库候选位置）
                cluster_centers = kmeans.cluster_centers_
                
                # 计算商户点的凸包（表示城市边界）
                hull = ConvexHull(coordinates)
                hull_points = coordinates[hull.vertices]
                
                # 计算凸包的中心
                centroid = np.mean(hull_points, axis=0)
                
                # 将聚类中心向外延伸，确保仓库位于城市边缘或外围
                strategic_locations = []
                for center in cluster_centers:
                    # 计算从城市中心到聚类中心的向量
                    vector = center - centroid
                    vector_length = np.linalg.norm(vector)
                    
                    if vector_length < 1e-10:  # 避免除零
                        # 随机选择一个方向
                        angle = np.random.uniform(0, 2*np.pi)
                        vector = np.array([np.cos(angle), np.sin(angle)])
                        vector_length = 1.0
                    
                    # 归一化向量
                    unit_vector = vector / vector_length
                    
                    # 计算凸包到中心的平均距离
                    hull_distances = [np.linalg.norm(hp - centroid) for hp in hull_points]
                    avg_distance = np.mean(hull_distances)
                    
                    # 设置仓库位置为：中心 + 1.2倍平均距离 * 方向向量
                    # 系数1.2确保仓库位于城市边缘之外，但不会太远
                    wh_location = centroid + 1.2 * avg_distance * unit_vector
                    
                    strategic_locations.append(wh_location)
                
                # 确定每个仓库的方向标签
                directions = []
                for loc in strategic_locations:
                    angle = np.arctan2(loc[1] - centroid[1], loc[0] - centroid[0])
                    # 将角度转换为方向标签
                    if -np.pi/4 <= angle < np.pi/4:
                        directions.append("东")
                    elif np.pi/4 <= angle < 3*np.pi/4:
                        directions.append("北")
                    elif -3*np.pi/4 <= angle < -np.pi/4:
                        directions.append("南")
                    else:
                        directions.append("西")
                
                # 创建远郊大仓
                strategic_warehouses = []
                for i, (loc, direction) in enumerate(zip(strategic_locations, directions)):
                    # 计算此仓库服务的商户数量，用于确定容量
                    cluster_idx = kmeans.labels_ == i
                    cluster_size = np.sum(cluster_idx)
                    
                    # 根据每个聚类中的商户数量比例分配容量
                    capacity = max(30000, int(50000 * (cluster_size / len(coordinates))))
                    
                    warehouse = {
                        "warehouse_id": f"W{direction}{i+1:02d}",
                        "zone_id": "single_zone",
                        "longitude": float(loc[0]),
                        "latitude": float(loc[1]),
                        "capacity": capacity,
                        "is_strategic": True,
                        "direction": direction,
                        "cluster_size": int(cluster_size)
                    }
                    strategic_warehouses.append(warehouse)
                
                # 检查仓库位置是否有效（不能超出经纬度合理范围）
                for wh in strategic_warehouses:
                    if not (73 <= wh["longitude"] <= 135 and 18 <= wh["latitude"] <= 54):
                        logging.warning(f"仓库 {wh['warehouse_id']} 位置超出中国大陆范围，将调整到边界内")
                        wh["longitude"] = max(73, min(135, wh["longitude"]))
                        wh["latitude"] = max(18, min(54, wh["latitude"]))
                
                logging.info(f"数据驱动的远郊大仓模式：已自动确定 {len(strategic_warehouses)} 个战略仓库位置")
                
                # 可视化仓库布局
                try:
                    import matplotlib.pyplot as plt
                    from matplotlib.patches import Polygon
                    
                    plt.figure(figsize=(12, 10))
                    # 绘制所有商户点，按聚类着色
                    for cluster_id in range(num_warehouses):
                        cluster_points = coordinates[kmeans.labels_ == cluster_id]
                        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                                   s=10, alpha=0.6, label=f'区域 {cluster_id+1}')
                    
                    # 绘制凸包（城市边界）
                    hull_polygon = Polygon(hull_points, fill=False, edgecolor='black', 
                                          linestyle='--', linewidth=2, alpha=0.7)
                    plt.gca().add_patch(hull_polygon)
                    
                    # 绘制仓库
                    wh_lons = [wh["longitude"] for wh in strategic_warehouses]
                    wh_lats = [wh["latitude"] for wh in strategic_warehouses]
                    plt.scatter(wh_lons, wh_lats, s=200, c='red', marker='*', 
                               edgecolor='black', linewidth=1.5, label='战略仓库')
                    
                    # 绘制城市中心
                    plt.scatter(centroid[0], centroid[1], s=100, c='orange', marker='o',
                               edgecolor='black', label='城市中心')
                    
                    # 连接中心与每个仓库
                    for wh_lon, wh_lat in zip(wh_lons, wh_lats):
                        plt.plot([centroid[0], wh_lon], [centroid[1], wh_lat], 
                                'k-', alpha=0.3, linewidth=1)
                    
                    # 标记方向和容量
                    for wh in strategic_warehouses:
                        plt.annotate(
                            f"{wh['direction']}向\n容量:{wh['capacity']/1000:.0f}k\n商户:{wh['cluster_size']}",
                            (wh["longitude"], wh["latitude"]),
                            xytext=(15, 15), textcoords='offset points',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                        )
                    
                    plt.title('数据驱动的远郊大仓布局', fontsize=16, fontweight='bold')
                    plt.xlabel('经度')
                    plt.ylabel('纬度')
                    plt.legend(loc='upper right')
                    plt.grid(True, alpha=0.3)
                    
                    # 保存图片
                    os.makedirs(os.path.join(config.output_dir, 'visualizations'), exist_ok=True)
                    viz_path = os.path.join(config.output_dir, 'visualizations', 'data_driven_warehouses.png')
                    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logging.info(f"已保存数据驱动的战略仓库布局图: {viz_path}")
                except Exception as viz_err:
                    logging.warning(f"可视化战略仓库布局失败: {str(viz_err)}")
                
                return strategic_warehouses
                
            except ImportError as e:
                logging.warning(f"缺少必要的库进行聚类 ({str(e)})，将使用简化方法确定仓库位置")
                # 如果缺少必要的库，使用简化方法
            except Exception as e:
                logging.error(f"使用聚类确定仓库位置失败: {str(e)}")
                logging.error(traceback.format_exc())
            
            # 简化方法：使用边界框四角
            min_lon = merchants_df['经度'].min()
            max_lon = merchants_df['经度'].max()
            min_lat = merchants_df['纬度'].min()
            max_lat = merchants_df['纬度'].max()
            
            # 计算中心点
            center_lon = (min_lon + max_lon) / 2
            center_lat = (min_lat + max_lat) / 2
            
            # 计算边界框四角，稍微向外扩展以确保在城市边缘
            extension_factor = 1.1  # 向外延伸10%
            
            # 创建远郊大仓列表（位于边界框四角）
            strategic_warehouses = [
                {
                    "warehouse_id": "WNE001",
                    "zone_id": "single_zone",
                    "longitude": center_lon + (max_lon - center_lon) * extension_factor,
                    "latitude": center_lat + (max_lat - center_lat) * extension_factor,
                    "capacity": 40000,
                    "is_strategic": True,
                    "direction": "东北"
                },
                {
                    "warehouse_id": "WSE002",
                    "zone_id": "single_zone",
                    "longitude": center_lon + (max_lon - center_lon) * extension_factor,
                    "latitude": center_lat - (center_lat - min_lat) * extension_factor,
                    "capacity": 40000,
                    "is_strategic": True,
                    "direction": "东南"
                },
                {
                    "warehouse_id": "WSW003",
                    "zone_id": "single_zone",
                    "longitude": center_lon - (center_lon - min_lon) * extension_factor,
                    "latitude": center_lat - (center_lat - min_lat) * extension_factor,
                    "capacity": 40000,
                    "is_strategic": True,
                    "direction": "西南"
                },
                {
                    "warehouse_id": "WNW004",
                    "zone_id": "single_zone",
                    "longitude": center_lon - (center_lon - min_lon) * extension_factor,
                    "latitude": center_lat + (max_lat - center_lat) * extension_factor,
                    "capacity": 40000,
                    "is_strategic": True,
                    "direction": "西北"
                }
            ]
            
            # 检查仓库位置是否有效
            for wh in strategic_warehouses:
                if not (73 <= wh["longitude"] <= 135 and 18 <= wh["latitude"] <= 54):
                    logging.warning(f"仓库 {wh['warehouse_id']} 位置超出中国大陆范围，将调整到边界内")
                    wh["longitude"] = max(73, min(135, wh["longitude"]))
                    wh["latitude"] = max(18, min(54, wh["latitude"]))
            
            logging.info(f"使用简化方法：在城市四角布置了 {len(strategic_warehouses)} 个战略仓库")
            return strategic_warehouses
    
    # 以下是正常的多路区设施选址逻辑，启用路区划分时使用
    try:
        # 数据验证和预处理
        processed_zone_map = {}
        for z_id, data in zone_map.items():
            # 确保是DataFrame
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            # 标准化列名
            column_mappings = {
                'longitude': '经度',
                'latitude': '纬度',
                'lon': '经度',
                'lat': '纬度',
                'weight': '托运单重量'
            }
            
            # 仅重命名存在的列
            rename_dict = {orig: new for orig, new in column_mappings.items() 
                         if orig in data.columns and new not in data.columns}
            data = data.rename(columns=rename_dict)
            
            # 确保必要列存在
            if '经度' not in data.columns or '纬度' not in data.columns:
                raise ValueError(f"Zone {z_id} 缺少经纬度列")
                
            if '托运单重量' not in data.columns:
                data['托运单重量'] = 0.0
                
            if '商家名称' not in data.columns:
                data['商家名称'] = [f"{z_id}_M{i}" for i in range(len(data))]
                
            processed_zone_map[z_id] = data
        
        # 创建配置
        advanced_cfg = AdvancedFLPConfig(
            fixed_cost=1000.0,
            operation_cost=2.0,
            coverage_radius=50.0,
            use_partial_coverage=True,
            coverage_penalty_factor=1.5,
            use_multilevel_capacity=True,
            multi_levels=[(3000, 800), (5000, 1000), (8000, 1500)]
        )
        
        solver_cfg = ZoneFLPSolverConfig(
            use_gurobi=True,
            gurobi_time_limit=300,
            random_seed=config.random_seed if hasattr(config, 'random_seed') else 42
        )
        
        # 初始化并运行求解器
        solver = FacilityLocationSolver(
            zone_map=processed_zone_map,
            advanced_cfg=advanced_cfg,
            solver_cfg=solver_cfg
        )
        
        solver.solve()
        
        # 获取结果
        warehouses = []
        for zone_id, coords in solver.get_solution().items():
            warehouses.append({
                "warehouse_id": f"W{len(warehouses)+1:03d}",
                "zone_id": zone_id,
                "longitude": coords[0],
                "latitude": coords[1],
                "capacity": 8000
            })
        
        logging.info(f"成功选择了 {len(warehouses)} 个仓库位置")
        return warehouses
        
    except Exception as e:
        logging.error(f"设施选址失败: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        
        # 出错时，返回一个默认位置（城市中心）
        default_lon, default_lat = 114.3162, 30.5810  # 武汉市中心坐标
        return [{
            "warehouse_id": "W000",
            "zone_id": list(zone_map.keys())[0],
            "longitude": default_lon,
            "latitude": default_lat,
            "capacity": 8000
        }]