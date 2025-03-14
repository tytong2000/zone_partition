import traceback
from sklearn.cluster import KMeans
import os
import time
import logging
import numpy as np
import pandas as pd
import sys
sys.setrecursionlimit(10000)  # 增加递归深度限制
import matplotlib.pyplot as plt
import geopandas as gpd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from gurobipy import Model, GRB
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
from solve.utils.metrics import MetricsCalculator
from solve.baseline.cw_solver import CWSolver
from .config import ConfigManager
from solve.baseline.sa_solver import SASolver
from solve.baseline.ts_solver import TSSolver
from solve.baseline.vns_solver import VNSSolver
from solve.hybrid.cw_ts_solver import CWTSSolver
from solve.hybrid.sa_vns_solver import SAVNSSolver
from solve.hybrid.sa_ts_solver import SATSSolver
from solve.base.vrp_instance import VRPInstance
from solve.adaptive.adaptive_ts import AdaptiveTSSolver
from solve.adaptive.adaptive_vns import AdaptiveVNSSolver
from solve.adaptive.adaptive_sa_vns import AdaptiveSAVNSSolver
from ..base.base_solver import BaseSolver
from solve.utils.metrics import MetricsCalculator
from solve.utils.visualization import EnhancedZonePartitioner,VisualizationTools
from solve.baseline.sa_solver import SASolver
from solve.baseline.ts_solver import TSSolver
from solve.baseline.vns_solver import VNSSolver
from solve.hybrid.cw_ts_solver import CWTSSolver
from solve.hybrid.sa_vns_solver import SAVNSSolver
from solve.hybrid.sa_ts_solver import SATSSolver
from solve.adaptive.adaptive_ts import AdaptiveTSSolver
from solve.adaptive.adaptive_vns import AdaptiveVNSSolver
from solve.adaptive.adaptive_sa_vns import AdaptiveSAVNSSolver
from jinja2 import Template
from typing import TYPE_CHECKING, Dict
from solve.templates.vrp_report_template import VRP_REPORT_TEMPLATE
from jinja2 import Template
from solve.utils.statistical_analyzer import StatisticalAnalyzer
# Check for Gurobi availability
if TYPE_CHECKING:
    from solve.base.vrp_solution import VRPSolution
try:
    import gurobipy as gp
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False
# 1. 首先导入所有需要的类和模块
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
import os
import time
import pandas as pd
import geopandas as gpd
import networkx as nx
from tqdm import tqdm
import numpy as np

class VehicleTypeMatcher:
    def __init__(self, vehicle_types=None):
        if vehicle_types is None:
            # 使用默认配置
            self.vehicle_types = {
                'small': {  
                    'capacity': 1500,  # kg
                    'volume': 4,       # m³
                    'fixed_cost': 500, # 固定成本
                    'suitable_for': ['convenience', 'small_supermarket']
                },
                'medium': {  
                    'capacity': 4000,
                    'volume': 15,
                    'fixed_cost': 800,
                    'suitable_for': ['supermarket', 'small_mall']
                },
                'large': {  
                    'capacity': 8000,
                    'volume': 30,
                    'fixed_cost': 1200,
                    'suitable_for': ['mall', 'large_supermarket']
                }
            }
        else:
            # 使用传入的配置
            self.vehicle_types = vehicle_types

        # 2. 商家类型快速匹配表保持不变
        self.merchant_type_mapping = {
            'convenience': {
                'primary': 'small',
                'secondary': 'medium',
                'weight_threshold': 1000,  # kg
                'volume_threshold': 3      # m³
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
    def match_vehicle_type(self, merchant_type: str, demand_weight: float, demand_volume: float) -> str:
        # 商户类型映射
        merchant_map = {
            '便利店': 'convenience',
            '超市': 'supermarket',
            '购物中心': 'mall'
        }
        
        mapped_type = merchant_map.get(merchant_type)
        if merchant_type not in self.merchant_type_mapping:
            return 'medium'  # 默认返回中型车
            
        mapping = self.merchant_type_mapping[merchant_type]
        
        # 使用预设阈值快速判断（避免复杂计算）
        if demand_weight <= mapping['weight_threshold'] and \
           demand_volume <= mapping['volume_threshold']:
            return mapping['primary']
        else:
            return mapping['secondary']

    def get_vehicle_specs(self, vehicle_type: str) -> dict:
        """获取车型规格（O(1)操作）"""
        return self.vehicle_types.get(vehicle_type, self.vehicle_types['medium'])
# 2. 添加缺失的类定义
@dataclass
class SolverConfig:
    """求解器配置类"""
    excel_path: str
    city_path: str
    road_path: str
    output_dir: str
    vehicle_capacity: float
    max_route_time: float
    random_seed: int
    parallel_evaluation: bool
    max_workers: int
    min_clusters: int
    max_clusters: int
    road_buffer_distance: float

@dataclass
class ExperimentConfig:
    """实验配置类"""
    name: str
    output_dir: str
    random_seed: int
    data: Dict
    solver: Dict
    facility_location: Dict
    parallel: Dict

# 5. 添加缺失的工具类
class RoadNetworkBuilder:
    def __init__(self, road_path: str):
        self.road_path = road_path
        self.graph = None
        
    def build_graph(self):
        """构建路网图"""
        try:
            road_gdf = gpd.read_file(self.road_path)
            G = nx.Graph()
            # 这里添加构建图的逻辑
            self.graph = G
        except Exception as e:
            logging.error(f"构建路网图失败: {str(e)}")
            self.graph = nx.Graph()

def split_merchants_by_zone(merchant_gdf: gpd.GeoDataFrame, zones_gdf: gpd.GeoDataFrame) -> Dict:
    """将商户分配到路区"""
    zone_map = {}
    # 添加分配逻辑
    return zone_map
# Configuration for Advanced Facility Location Problem (FLP)
@dataclass
class VRPExperiment:
    def __init__(self, config: SolverConfig, zone_map: Dict, warehouses: List, 
                partitioner: EnhancedZonePartitioner, road_graph: nx.Graph,
                vehicle_matcher: VehicleTypeMatcher):
        """初始化VRP实验类"""
        self.config = config
        self.zone_map = zone_map
        self.warehouses = warehouses
        self.partitioner = partitioner
        self.road_graph = road_graph
        self.vehicle_matcher = vehicle_matcher
        self.instance = None
        # 直接使用 zone_map 的键作为测试区域
        self.test_areas = list(zone_map.keys())
        self.area_list = self.test_areas 
        self.algorithms = ['CW', 'SA', 'TS', 'VNS']
        self.logger = logging.getLogger("VRPExperiment")

        # 保留其他原有的初始化代码...
        # 保留原有的所有实验控制标志和结果存储
        self.experiment_flags = {
            'run_baseline': True,
            'run_hybrid': True, 
            'run_adaptive': True,
            'run_enhanced': True  
        }
        
        self.baseline_results = {}
        self.hybrid_results = {}
        self.adaptive_results = {}
        self.advanced_metrics = {
            'vehicle_utilization': {},
            'path_diversity': {},
            'cost_breakdown': {}
        }

    def _log_realtime_metrics(self, zone_id: str, algo: str, solution: "VRPSolution"):
        """实时记录每次求解的车型和商户指标"""
        try:
            # 1. 基本指标统计
            basic_info = {
                'experiment_type': 'baseline' if algo in ['CW', 'SA', 'TS', 'VNS'] 
                                else 'hybrid' if '_' in algo 
                                else 'adaptive',
                'zone_id': zone_id,
                'algorithm': algo,
                'objective': solution.objective_value,
                'distance': solution.total_distance,
                'time': solution.solve_time if hasattr(solution, 'solve_time') else 0,
                'num_routes': len(solution.routes)
            }
            
            # 1. 车型统计
            vehicle_stats = {}
            for route_idx, route in enumerate(solution.routes):
                if hasattr(route, 'vehicle_type'):
                    v_type = route.vehicle_type
                else:
                    v_type = self._get_suitable_vehicle_type(route, zone_id)
                    
                if v_type not in vehicle_stats:
                    vehicle_stats[v_type] = {
                        'count': 0,
                        'total_load': 0,
                        'total_distance': 0
                    }
                
                vehicle_stats[v_type]['count'] += 1
                if hasattr(route, 'customer_ids'):
                    customer_ids = route.customer_ids
                else:
                    customer_ids = route
                vehicle_stats[v_type]['total_load'] += sum(self.instance.order_demands[i] for i in customer_ids)
            
            # 3. 商户统计
            merchant_stats = {}
            for route in solution.routes:
                customer_ids = route.customer_ids if hasattr(route, 'customer_ids') else route
                for cid in customer_ids:
                    m_type = self.instance.get_merchant_type(cid)
                    if m_type not in merchant_stats:
                        merchant_stats[m_type] = {
                            'count': 0,
                            'total_demand': 0
                        }
                    merchant_stats[m_type]['count'] += 1
                    merchant_stats[m_type]['total_demand'] += self.instance.get_order_demand(cid)

            # 4. 构造日志消息
            log_message = (
                f"实验结果:\n"
                f"  基本信息:\n"
                f"    实验类型: {basic_info['experiment_type']}\n"
                f"    区域: {basic_info['zone_id']}\n"
                f"    算法: {basic_info['algorithm']}\n"
                f"    目标值: {basic_info['objective']:.2f}\n"
                f"    总距离: {basic_info['distance']:.2f}\n"
                f"    求解时间: {basic_info['time']:.2f}s\n"
                f"    路线数量: {basic_info['num_routes']}\n"
                f"  车型统计:\n"
                + "\n".join(f"    {v_type}: 数量={stats['count']}, 总载重={stats['total_load']:.2f}" 
                        for v_type, stats in vehicle_stats.items())
                + "\n  商户统计:\n"
                + "\n".join(f"    {m_type}: 数量={stats['count']}, 总需求={stats['total_demand']:.2f}"
                        for m_type, stats in merchant_stats.items())
            )
            
            self.logger.info(log_message)
            
            self.logger.info(f"已保存区域 {zone_id} 算法 {algo} 的实时指标")
            
        except Exception as e:
            self.logger.error(f"保存实时指标失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def run_baseline_experiment(self):
        """执行基线算法实验"""
        # 使用正确的区域列表名称
        areas_to_process = self.area_list  
        
        # 创建进度条
        area_pbar = tqdm(areas_to_process, desc="处理路区", unit="区")
        
        for area in area_pbar:
            area_pbar.set_description(f"处理路区 {area}")
            
            for algo in self.algorithms:
                self.logger.info(f"开始处理区域 {area} - 算法 {algo}")
                
                try:
                    # 创建求解器实例
                    instance = self._create_instance(area)
                    solver = self._create_solver(algo, instance)
                    
                    # 记录开始时间
                    start_time = time.time()
                    
                    # 求解
                    self.logger.info(f"开始{algo}求解...")
                    solution = solver.solve()
                    
                    # 记录结束时间和统计信息
                    end_time = time.time()
                    solve_time = end_time - start_time
                    
                    if solution and solution.is_feasible():
                        self.logger.info(f"区域 {area} 算法 {algo} 成功: "
                                       f"obj={solution.objective_value:.2f}, "
                                       f"time={solve_time:.2f}s")
                        self._save_solution(area, algo, solution)
                    else:
                        self.logger.warning(f"区域 {area} 算法 {algo} 无可行解")
                        
                except Exception as e:
                    self.logger.error(f"执行实验失败 (区域={area}, 算法={algo}): {str(e)}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")

    # def check_report_generation(self) -> bool:
    #     """尝试生成报告以确保报告生成过程正常"""
    #     try:
    #         # 这里我们调用一个简单的模板来检测报告是否能成功生成
    #         results_dir = "测试报告生成"
    #         os.makedirs(results_dir, exist_ok=True)

    #         # 使用一个简单的虚拟数据来测试报告生成
    #         template_data = {
    #             'baseline_metrics': {},
    #             'hybrid_metrics': {},
    #             'adaptive_metrics': {},
    #             'plots': []  # 这里可以添加图表路径
    #         }

    #         # 只生成报告，不保存实际结果
    #         template = Template(self.html_template)
    #         html_content = template.render(**template_data)

    #         # 尝试将 HTML 保存到本地
    #         html_path = os.path.join(results_dir, 'test_report.html')
    #         with open(html_path, 'w', encoding='utf-8') as f:
    #             f.write(html_content)

    #         # 如果没有异常发生，则说明报告生成成功
    #         print("报告生成预检查成功！")
    #         return True

    #     except Exception as e:
    #         # 如果生成报告失败，捕获异常并返回 False
    #         print(f"报告生成失败: {str(e)}")
    #         return False
            
    def _create_solver(self, solver_type: str, instance: "VRPInstance", initial_solution=None, **kwargs):
        from solve.base.vrp_solution import VRPSolution
        solver_params = {
            "instance": instance,
            "initial_solution": initial_solution  # 确保传入初始解
        }
        
        solver_map = {
            "CW": CWSolver,
            "SA": SASolver, 
            "TS": TSSolver,
            "VNS": VNSSolver,
            "CW_TS": CWTSSolver,
            "SA_VNS": SAVNSSolver,
            "SA_TS": SATSSolver
        }
        
        solver = solver_map[solver_type](**solver_params)
        if initial_solution:
            solver.solution = initial_solution  # 确保设置初始解
        
        return solver

    def _generate_plots(self):
        """生成所有可视化图表"""
        plots_dir = os.path.join(self.config.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 目标值对比图
        plt.figure(figsize=(12, 6))
        self._plot_objective_comparison()
        plt.savefig(os.path.join(plots_dir, 'objective_comparison.png'))
        plt.close()
        
        # 求解时间对比图
        plt.figure(figsize=(12, 6))
        self._plot_time_comparison()
        plt.savefig(os.path.join(plots_dir, 'time_comparison.png'))
        plt.close()
        
        # 路线数量对比图
        plt.figure(figsize=(12, 6))
        self._plot_routes_comparison()
        plt.savefig(os.path.join(plots_dir, 'routes_comparison.png'))
        plt.close()
    def run_experiments(self):
        """运行所有实验"""
        try:
            self.logger.info("=== 开始执行所有实验 ===")
            
            # 添加数据检查代码
            self.logger.info("=== 检查数据完整性 ===")
            max_index_used = 0
            
            # 检查所有类型的算法结果
            for results_dict in [self.baseline_results, self.hybrid_results, self.adaptive_results]:
                if not results_dict:
                    continue
                    
                for zone_id, algo_dict in results_dict.items():
                    # 检查原始数据长度
                    merchants_df = self.zone_map.get(zone_id)
                    if merchants_df is None:
                        self.logger.error(f"找不到区域 {zone_id} 的商户数据")
                        continue
                        
                    if not isinstance(merchants_df, pd.DataFrame):
                        merchants_df = pd.DataFrame(merchants_df)
                    
                    original_length = len(merchants_df)
                    
                    # 检查路线中使用的最大索引
                    for metrics in algo_dict.values():
                        if 'routes' in metrics:
                            for route in metrics['routes']:
                                if route:
                                    max_index_used = max(max_index_used, max(route))
                    
                    self.logger.info(f"区域 {zone_id} - 原始数据长度: {original_length}, 最大使用索引: {max_index_used}")
                    
                    # 如果发现问题，扩展数据
                    if max_index_used >= original_length:
                        self.logger.warning(
                            f"区域 {zone_id} 需要扩展数据 - "
                            f"当前长度: {original_length}, 需要长度: {max_index_used + 1}"
                        )
            
            # 执行基线算法实验
            if self.experiment_flags['run_baseline']:
                self.logger.info("=== 执行基线算法实验 ===")
                self.run_baseline_experiments()
                
            # 执行混合算法实验
            if self.experiment_flags['run_hybrid']:
                self.logger.info("=== 执行混合算法实验 ===")
                self.run_hybrid_experiments()
                
            # 执行自适应算法实验
            if self.experiment_flags['run_adaptive']:
                self.logger.info("=== 执行自适应算法实验 ===")
                self.run_adaptive_experiments()
                
            # 保存实验结果
            self.save_results()
            
        except Exception as e:
            self.logger.error(f"实验执行出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _verify_experiment_results(self) -> bool:
        """验证实验结果的完整性"""
        # 检查基线算法结果
        if not self.baseline_results:
            self.logger.warning("没有基线算法的实验结果")
            return False
            
        # 检查混合算法结果
        if not self.hybrid_results:
            self.logger.warning("没有混合算法的实验结果")
            return False
            
        # 检查自适应算法结果
        if not self.adaptive_results:
            self.logger.warning("没有自适应算法的实验结果")
            return False
            
        return True
    def _calculate_baseline_metrics(self) -> Dict:
        """计算基线算法指标"""
        metrics = {}
        for algo in ['CW', 'SA', 'TS', 'VNS']:
            results = [m for zone_results in self.baseline_results.values() 
                    for a, m in zone_results.items() if a == algo]
            if results:
                metrics[algo] = {
                    'objective_avg': np.mean([r['objective'] for r in results]),  # 确保这里是一个数值
                    'time_avg': np.mean([r['time'] for r in results]),
                    'routes_avg': np.mean([r['num_routes'] for r in results]),
                    'objective_std': np.std([r['objective'] for r in results]),
                    'success_rate': len(results) / len(self.baseline_results)
                }
        return metrics

    def _calculate_hybrid_metrics(self) -> Dict:
        """计算混合算法指标"""
        metrics = {}
        hybrids = ['CW_TS', 'SA_VNS']
        for algo in hybrids:
            results = [m for zone_results in self.hybrid_results.values() 
                    for a, m in zone_results.items() if a == algo]
            if results:
                metrics[algo] = {
                    'objective_avg': np.mean([r['objective'] for r in results]),  # 确保这里是一个数值
                    'time_avg': np.mean([r['time'] for r in results]),
                    'improvement_avg': np.mean([r.get('improvement', 0) for r in results]),
                    'objective_std': np.std([r['objective'] for r in results]),
                    'stability_score': self._calculate_stability_score(results)
                }
        return metrics

    def _calculate_adaptive_metrics(self) -> Dict:
        """计算自适应算法指标"""
        metrics = {}
        adaptive_algos = ['TS', 'VNS', 'CW_TS', 'SA_VNS']
        for algo in adaptive_algos:
            results = [m for zone_results in self.adaptive_results.values() 
                    for a, m in zone_results.items() if a == algo]
            if results:
                metrics[algo] = {
                    'objective_avg': np.mean([r['objective'] for r in results]),  # 确保这里是一个数值
                    'time_avg': np.mean([r['time'] for r in results]),
                    'convergence_score': self._calculate_convergence_score(results),
                    'param_adjustments': self._count_parameter_adjustments(results),
                    'final_improvement': self._calculate_final_improvement(results)
                }
        return metrics

    def save_experiment_results(self):
        """
        保存实验结果到HTML报告和CSV文件
        """
        try:
            # 确保输出目录存在
            results_dir = os.path.join(self.config.output_dir, 'results')
            os.makedirs(results_dir, exist_ok=True)
            
            # 1. 保存CSV格式的实验结果
            self._save_csv_results(results_dir)
            
            # 2. 生成HTML报告
            self._generate_html_report(results_dir)
            
            logging.info(f"实验结果已保存至目录: {results_dir}")
            
        except Exception as e:
            logging.error(f"保存实验结果失败: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())

    def _save_csv_results(self, results_dir: str):
        """保存实验结果到CSV文件"""
        try:
            # 收集所有结果
            rows = []
            
            # 基线算法结果
            for zone_id, algo_results in self.baseline_results.items():
                for algo, result in algo_results.items():
                    if result:
                        row = {
                            'experiment_type': 'baseline',
                            'zone_id': zone_id,
                            'algorithm': algo,
                            'objective': result['objective'],
                            'distance': result['distance'],
                            'time': result['time'],
                            'num_routes': result['num_routes']
                        }
                        rows.append(row)
            
            # 混合算法结果
            for zone_id, algo_results in self.hybrid_results.items():
                for algo, result in algo_results.items():
                    if result:
                        row = {
                            'experiment_type': 'hybrid',
                            'zone_id': zone_id,
                            'algorithm': algo,
                            'objective': result['objective'],
                            'distance': result['distance'],
                            'time': result['time'],
                            'num_routes': result['num_routes'],
                            'improvement': result.get('improvement', 0)
                        }
                        rows.append(row)
            
            # 自适应算法结果
            for zone_id, algo_results in self.adaptive_results.items():
                for algo, result in algo_results.items():
                    if result:
                        row = {
                            'experiment_type': 'adaptive',
                            'zone_id': zone_id,
                            'algorithm': algo,
                            'objective': result['objective'],
                            'distance': result['distance'],
                            'time': result['time'],
                            'num_routes': result['num_routes']
                        }
                        rows.append(row)
            
            # 创建DataFrame并保存
            if rows:
                df = pd.DataFrame(rows)
                
                # 保存详细结果
                df.to_csv(os.path.join(results_dir, 'detailed_results.csv'), 
                        index=False, encoding='utf-8-sig')
                
                # 保存汇总结果
                summary_df = pd.DataFrame([
                    {
                        'metric': '平均目标值',
                        'baseline': df[df['experiment_type']=='baseline']['objective'].mean(),
                        'hybrid': df[df['experiment_type']=='hybrid']['objective'].mean(),
                        'adaptive': df[df['experiment_type']=='adaptive']['objective'].mean()
                    },
                    {
                        'metric': '平均距离',
                        'baseline': df[df['experiment_type']=='baseline']['distance'].mean(),
                        'hybrid': df[df['experiment_type']=='hybrid']['distance'].mean(),
                        'adaptive': df[df['experiment_type']=='adaptive']['distance'].mean()
                    },
                    {
                        'metric': '平均求解时间',
                        'baseline': df[df['experiment_type']=='baseline']['time'].mean(),
                        'hybrid': df[df['experiment_type']=='hybrid']['time'].mean(),
                        'adaptive': df[df['experiment_type']=='adaptive']['time'].mean()
                    },
                    {
                        'metric': '平均路线数',
                        'baseline': df[df['experiment_type']=='baseline']['num_routes'].mean(),
                        'hybrid': df[df['experiment_type']=='hybrid']['num_routes'].mean(),
                        'adaptive': df[df['experiment_type']=='adaptive']['num_routes'].mean()
                    }
                ])
                
                summary_df.to_csv(os.path.join(results_dir, 'summary_results.csv'),
                                index=False, encoding='utf-8-sig')
                
                logging.info(f"CSV结果已保存至: {results_dir}")
            else:
                logging.warning("没有可用的实验结果可保存")
                
        except Exception as e:
            logging.error(f"保存CSV结果失败: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())

    def _plot_algorithm_comparison(self, plots_dir: str):
        """生成算法性能对比图"""
        try:
            data = []  # 用于存储数据
            labels = []  # 用于存储标签

            # 确保每个数据点都有标签
            for algo_results in [self.baseline_results, self.hybrid_results, self.adaptive_results]:
                for zone_results in algo_results.values():
                    for algo, result in zone_results.items():
                        if result:
                            data.append(result['objective'])
                            labels.append(algo)

            # 检查labels和data的长度是否一致
            if len(labels) != len(data):
                raise ValueError("Labels and data have incompatible dimensions")

            # 生成箱型图
            plt.figure(figsize=(12, 6))
            plt.boxplot(data, labels=labels)
            plt.title('算法目标值对比')
            plt.ylabel('目标值')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'objective_comparison.png'))
            plt.close()

        except Exception as e:
            self.logger.error(f"生成算法对比图失败: {str(e)}")


    def _plot_route_distribution(self, plots_dir: str):
        """生成路线分布图"""
        try:
            plt.figure(figsize=(12, 6))
            routes = []
            labels = []
            
            for algo_results in [self.baseline_results, self.hybrid_results, self.adaptive_results]:
                for zone_results in algo_results.values():
                    for algo, result in zone_results.items():
                        if result:
                            routes.append(result['num_routes'])
                            labels.append(algo)
            
            plt.boxplot([r for r in routes], labels=labels)
            plt.title('路线数量分布')
            plt.ylabel('路线数量')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'route_distribution.png'))
            plt.close()
            
        except Exception as e:
            logging.error(f"生成路线分布图失败: {str(e)}")

    def _plot_cost_analysis(self, plots_dir: str):
        """生成成本分析图"""
        try:
            metrics = self._compute_cost_metrics()
            
            plt.figure(figsize=(12, 6))
            algorithms = ['baseline', 'hybrid', 'adaptive']
            fixed_costs = [metrics[algo]['fixed_ratio'] * 100 for algo in algorithms]
            variable_costs = [metrics[algo]['variable_ratio'] * 100 for algo in algorithms]
            
            x = np.arange(len(algorithms))
            width = 0.35
            
            plt.bar(x - width/2, fixed_costs, width, label='固定成本占比')
            plt.bar(x + width/2, variable_costs, width, label='变动成本占比')
            
            plt.title('成本构成分析')
            plt.xlabel('算法类型')
            plt.ylabel('占比(%)')
            plt.xticks(x, algorithms)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(os.path.join(plots_dir, 'cost_analysis.png'))
            plt.close()
            
        except Exception as e:
            logging.error(f"生成成本分析图失败: {str(e)}")
        
    def get_results(self) -> Dict:
        """获取所有实验结果"""
        return {
            'baseline': self.baseline_results,
            'hybrid': self.hybrid_results,
            'adaptive': self.adaptive_results
        }
    if TYPE_CHECKING:
        from solve.base.vrp_solution import VRPSolution        

    def _calculate_advanced_metrics(self, solution: "VRPSolution") -> Dict:
        metrics = {}
        # ... 其他代码保持不变
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
    
    def _create_vrp_instance(self, merchants_df: pd.DataFrame, zone_id: str = None) -> VRPInstance:
        """创建VRP实例"""
        try:
            # 1. 确保merchants_df格式正确
            if not isinstance(merchants_df, pd.DataFrame):
                merchants_df = pd.DataFrame(merchants_df)
                
            # 2. 直接创建VRP实例
            instance = VRPInstance(
                orders_df=merchants_df,
                road_graph=self.road_graph,
                num_warehouses=1,
                vehicle_capacity=self.config.vehicle_capacity,
                max_route_time=self.config.max_route_time,
                max_search_distance=1e5,
                selected_warehouses=self.warehouses,
            )
            
            return instance
        
        except Exception as e:
            self.logger.error(f"创建VRP实例失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
        
    def run_baseline_experiments(self,zone_id: str):
        """执行所有基线算法实验"""
        self.logger.info("\n=== 执行基线算法实验 ===")
        algorithms = ["CW", "SA", "TS", "VNS"]
        
        total_areas = len(self.test_areas)
        total_algos = len(self.algorithms)
        
        # 创建两层进度条
        area_pbar = tqdm(self.test_areas, desc="处理路区", unit="区")
        
        for area in area_pbar:
            # 更新整体进度信息
            area_pbar.set_description(f"处理路区 {area}")
            self.instance = self._create_vrp_instance(self.zone_map[zone_id], zone_id)
            for algo in self.algorithms:
                self.logger.info(f"开始处理区域 {area} - 算法 {algo}")
                
                try:
                    # 创建求解器实例
                    instance = self._create_vrp_instance(self.zone_map[zone_id], zone_id)
                    solver = self._create_solver(algo, instance)
                    
                    # 记录开始时间
                    start_time = time.time()
                    
                    # 求解
                    self.logger.info(f"开始{algo}求解...")
                    solution = solver.solve()
                    
                    # 记录结束时间和统计信息
                    end_time = time.time()
                    solve_time = end_time - start_time
                    
                    if solution and solution.is_feasible():
                        self.logger.info(f"区域 {area} 算法 {algo} 成功: "
                                    f"obj={solution.objective_value:.2f}, "
                                    f"time={solve_time:.2f}s")
                        self._log_realtime_metrics(area, algo, solution)
                        self._save_solution(area, algo, solution)
                    else:
                        self.logger.warning(f"区域 {area} 算法 {algo} 无可行解")
                        
                except Exception as e:
                    self.logger.error(f"执行实验失败 (区域={area}, 算法={algo}): {str(e)}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
        
    def run_hybrid_experiments(self):
        """执行所有混合算法实验"""
        self.logger.info("\n=== 执行混合算法实验 ===")
        
        # 定义混合算法组合
        hybrid_combinations = [
            ("CW", "TS"),  # Clarke-Wright + 禁忌搜索
            ("SA", "VNS"), # 模拟退火 + 变邻域搜索
            ("SA", "TS")   # 模拟退火 + 禁忌搜索
        ]
        
        # 遍历所有路区和算法组合
        for zone_id in tqdm(self.zone_map.keys(), desc="处理路区"):
            for first_stage, second_stage in hybrid_combinations:
                try:
                    result = self.run_hybrid_experiment(zone_id, first_stage, second_stage)
                    if result:
                        if zone_id not in self.hybrid_results:
                            self.hybrid_results[zone_id] = {}
                        self.hybrid_results[zone_id][f"{first_stage}_{second_stage}"] = result
                        self.logger.info(f"区域 {zone_id} 混合算法 {first_stage}_{second_stage} 执行成功")
                except Exception as e:
                    self.logger.error(f"区域 {zone_id} 混合算法 {first_stage}_{second_stage} 执行失败: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())

    def run_hybrid_experiment(self, zone_id: str, first_stage: str, second_stage: str) -> Optional[Dict]:
        """执行单个混合算法实验"""
        try:
            self.instance = self._create_vrp_instance(self.zone_map[zone_id], zone_id)
            # 数据验证
            if zone_id not in self.zone_map:
                self.logger.warning(f"找不到区域 {zone_id}")
                return None
                
            merchants_df = self.zone_map[zone_id]
            if not isinstance(merchants_df, pd.DataFrame):
                merchants_df = pd.DataFrame(merchants_df)
                
            if merchants_df.empty:
                return None
                
            # 创建VRP实例
            instance = self._create_vrp_instance(merchants_df)
            
            # 第一阶段
            solver1 = self._create_solver(first_stage, instance)
            initial_sol = solver1.solve()
            
            if not initial_sol or not initial_sol.is_feasible():
                self.logger.warning(f"区域 {zone_id} {first_stage} 第一阶段无可行解")
                return None
                
            # 第二阶段
            solver2 = self._create_solver(second_stage, instance, initial_solution=initial_sol)
            t0 = time.time()
            final_sol = solver2.solve()
            t1 = time.time()
            
            if final_sol and final_sol.is_feasible():
                self._log_realtime_metrics(zone_id, f"{first_stage}_{second_stage}", final_sol)
                improvement = (initial_sol.objective_value - final_sol.objective_value) / initial_sol.objective_value
                return {
                    'objective': final_sol.objective_value,
                    'distance': final_sol.total_distance,
                    'time': t1 - t0,
                    'num_routes': len(final_sol.routes),
                    'routes': [[int(i) for i in r] for r in final_sol.routes],
                    'improvement': improvement
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"执行混合算法实验失败 (区域={zone_id}): {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
        
    def run_adaptive_experiments(self):
        """执行所有自适应算法实验"""
        self.logger.info("\n=== 执行自适应算法实验 ===")
        
        # 定义自适应算法配置
        adaptive_configs = [
            {
                'algorithm': 'TS',
                'params': {
                    'tabu_tenure': 'adaptive',
                    'min_tenure': 5,
                    'max_tenure': 20
                }
            },
            {
                'algorithm': 'VNS',
                'params': {
                    'neighborhood_size': 'adaptive',
                    'min_size': 2,
                    'max_size': 5
                }
            }
        ]
        
        # 遍历所有路区和算法配置
        for zone_id in tqdm(self.zone_map.keys(), desc="处理路区"):
            for config in adaptive_configs:
                try:
                    result = self.run_adaptive_experiment(
                        zone_id=zone_id,
                        algo_type=config['algorithm'],
                        adaptive_params=config['params']
                    )
                    
                    if result:
                        if zone_id not in self.adaptive_results:
                            self.adaptive_results[zone_id] = {}
                        self.adaptive_results[zone_id][f"{config['algorithm']}_ADP"] = result
                        self.logger.info(f"区域 {zone_id} 自适应算法 {config['algorithm']} 执行成功")
                except Exception as e:
                    self.logger.error(f"区域 {zone_id} 自适应算法 {config['algorithm']} 执行失败: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())

    def run_adaptive_experiment(self, zone_id: str, algo_type: str, adaptive_params: Dict) -> Optional[Dict]:
        """执行单个自适应算法实验"""
        try:
            instance = self._create_vrp_instance(self.zone_map[zone_id], zone_id)  # 正确
            if zone_id not in self.zone_map:
                self.logger.warning(f"找不到区域 {zone_id}")
                return None
                
            merchants_df = self.zone_map[zone_id]
            if not isinstance(merchants_df, pd.DataFrame):
                merchants_df = pd.DataFrame(merchants_df)
                
            if merchants_df.empty:
                return None
                
            instance = self._create_vrp_instance(merchants_df)
            
            # 创建自适应求解器
            solver = self._create_adaptive_solver(
                solver_type=algo_type,
                instance=instance,
                adaptive_params=adaptive_params
            )
            
            # 求解
            t0 = time.time()
            solution = solver.solve()
            t1 = time.time()
            
            if solution and solution.is_feasible():
                self._log_realtime_metrics(zone_id, f"{algo_type}_ADP", solution)
                # 确保solution有vehicle_assignments
                if not hasattr(solution, 'vehicle_assignments'):
                    solution.vehicle_assignments = {}
                    # 对每条路线，根据其车型创建分配信息
                    for i, route in enumerate(solution.routes):
                        if hasattr(route, 'vehicle_type'):
                            v_type = route.vehicle_type
                        else:
                            # 如果route没有vehicle_type，使用_get_suitable_vehicle_type方法
                            v_type = self._get_suitable_vehicle_type(route)
                        
                        solution.vehicle_assignments[i] = {
                            'type': v_type,
                            'customers': route.customer_ids if hasattr(route, 'customer_ids') else route
                        }
                
                # 构建返回结果，确保包含vehicle_assignments
                result = {
                    'objective': solution.objective_value,
                    'distance': solution.total_distance,
                    'time': t1 - t0,
                    'num_routes': len(solution.routes),
                    'routes': [[int(i) for i in r.customer_ids] for r in solution.routes] if hasattr(solution.routes[0], 'customer_ids') else [[int(i) for i in r] for r in solution.routes],
                    'vehicle_assignments': solution.vehicle_assignments,  # 添加vehicle_assignments
                    'parameter_history': solver.get_parameter_history() if hasattr(solver, 'get_parameter_history') else []
                }
                return result
                    
            return None
                
        except Exception as e:
            self.logger.error(f"执行自适应算法实验失败 (区域={zone_id}, 算法={algo_type}): {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _get_suitable_vehicle_type(self, route, zone_id: str = None) -> str:
        """根据路线需求确定合适的车型"""
        if not self.instance or not hasattr(self.instance, 'order_demands'):
            # 如果instance不存在或没有order_demands属性，先创建它
            if zone_id and zone_id in self.zone_map:
                merchants_df = self.zone_map[zone_id]
                if not isinstance(merchants_df, pd.DataFrame):
                    merchants_df = pd.DataFrame(merchants_df)
                self.instance = self._create_vrp_instance(merchants_df, zone_id)
            else:
                raise ValueError("无法确定车型：instance未初始化且无法通过zone_id创建")

        if hasattr(route, 'customer_ids'):
            customer_ids = route.customer_ids
        else:
            customer_ids = route
                
        total_weight = sum(self.instance.order_demands[i] for i in customer_ids)
        total_volume = sum(self.instance.get_order_volume(i) for i in customer_ids)
            
        # 根据需求量判断合适的车型
        if total_weight <= 1000 and total_volume <= 3:
            return 'small'
        elif total_weight <= 3000 and total_volume <= 10:
            return 'medium'
        else:
            return 'large'
    
    def _create_adaptive_solver(self, solver_type: str, instance: VRPInstance, 
                            adaptive_params: Dict = None, initial_solution=None) -> BaseSolver:
        """创建自适应求解器"""
        solver_params = {
            "instance": instance,
            "initial_solution": initial_solution,
            "adaptive_params": adaptive_params or {}
        }
        
        if solver_type == "TS":
            return AdaptiveTSSolver(**solver_params)
        elif solver_type == "VNS":
            return AdaptiveVNSSolver(**solver_params)
        elif solver_type == "SA_VNS": 
            return AdaptiveSAVNSSolver(**solver_params)
        elif solver_type == "SA_TS": 
            return AdaptiveSAVNSSolver(**solver_params)
        elif solver_type == "CW_TS": 
            return AdaptiveSAVNSSolver(**solver_params)
        elif solver_type == "CW_VNS": 
            return AdaptiveSAVNSSolver(**solver_params)
        else:
            raise ValueError(f"不支持的自适应求解器类型: {solver_type}")
    def _calculate_stability_score(self, results) -> float:
        """
        计算混合算法结果的稳定性评分
        
        Args:
            results: List[Dict] - 包含实验结果的列表，每个结果是一个字典
                必须包含 'objective', 'time', 'num_routes' 等键
                
        Returns:
            float: 稳定性评分，范围[0,1]，越接近1表示越稳定
        """
        if not results:
            return 0.0
            
        # 1. 提取关键指标
        objectives = np.array([r['objective'] for r in results])
        times = np.array([r['time'] for r in results])
        routes = np.array([r['num_routes'] for r in results])
        
        # 2. 计算变异系数(CV = 标准差/平均值)
        def calc_cv(arr):
            mean = np.mean(arr)
            if abs(mean) < 1e-10:  # 避免除零
                return float('inf')
            return np.std(arr) / mean
            
        cv_obj = calc_cv(objectives)
        cv_time = calc_cv(times)
        cv_routes = calc_cv(routes)
        
        # 3. 计算稳定性得分
        # 对三个CV加权平均，并将结果映射到[0,1]区间
        weights = [0.5, 0.3, 0.2]  # 目标值、时间、路线数的权重
        weighted_cv = cv_obj * weights[0] + cv_time * weights[1] + cv_routes * weights[2]
        
        # 使用sigmoid函数将CV映射到[0,1]
        stability = 1 / (1 + np.exp(weighted_cv - 1))
        
        return stability   
    def _get_vehicle_stats(self):
        """获取车辆统计信息"""
        stats = {vt: {'count': 0, 'total_distance': 0, 'total_load': 0} 
                for vt in self.vehicle_types}
        
        for solution in self.solutions:
            for route, vtype in zip(solution.routes, solution.vehicle_types):
                stats[vtype]['count'] += 1
                stats[vtype]['total_distance'] += self._calculate_route_distance(route)
                stats[vtype]['total_load'] += sum(self.instance.get_order_demand(i) 
                                                for i in route)
        
        return stats     
    def _generate_html_report(self, results_dir: str):
        """生成HTML实验报告"""
        try:
            # 1. 首先验证所有必需的数据是否存在
            if not self._verify_experiment_results():
                raise ValueError("无法生成报告：实验结果数据不完整")

            # 2. 计算所有必需的指标，并进行验证
            baseline_metrics = self._calculate_baseline_metrics()
            hybrid_metrics = self._calculate_hybrid_metrics()
            adaptive_metrics = self._calculate_adaptive_metrics()

            # 3. 验证计算结果
            if not baseline_metrics:
                self.logger.error("baseline_metrics 计算失败")
                baseline_metrics = self._get_empty_metrics()  # 提供默认值而不是None
            
            # 4. 准备模板数据，确保所有必需字段都有值
            template_data = {
                'metrics': {
                    'baseline': baseline_metrics,
                    'hybrid': hybrid_metrics or self._get_empty_metrics(),
                    'adaptive': adaptive_metrics or self._get_empty_metrics()
                },
                'vehicle_analysis': {
                    'type_stats': self._get_vehicle_stats()
                },
                'merchant_analysis': {
                    'type_stats': self._get_merchant_stats(),
                    'service_quality': self._calculate_service_metrics()
                },
                'statistical_analysis': {
                    't_tests': self._get_statistical_tests().get('t_tests', {}),
                    'f_tests': self._get_statistical_tests().get('f_tests', {}),
                    'anova': self._get_statistical_tests().get('anova_tests', {})
                },
                'plots': self._generate_plot_paths()
            }

            # 5. 验证模板数据的完整性
            self._validate_template_data(template_data)

            # 6. 渲染模板
            template = Template(self.html_template)
            html_content = template.render(**template_data)

            # 7. 保存HTML报告
            html_path = os.path.join(results_dir, 'experiment_report.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            self.logger.info(f"HTML报告已成功生成: {html_path}")

        except Exception as e:
            self.logger.error(f"生成HTML报告失败: {str(e)}")
            raise

    def _get_empty_metrics(self):
        """提供空的指标结构而不是None"""
        return {
            'objective_avg': 0.0,
            'time_avg': 0.0,
            'routes_avg': 0,
            'objective_std': 0.0,
            'success_rate': 0.0
        }

    def _validate_template_data(self, template_data):
        """验证模板数据的完整性"""
        required_fields = [
            'metrics.baseline',
            'metrics.hybrid',
            'metrics.adaptive',
            'vehicle_analysis.type_stats',
            'merchant_analysis.type_stats',
            'statistical_analysis.t_tests'
        ]

        for field in required_fields:
            parts = field.split('.')
            current = template_data
            for part in parts:
                if part not in current:
                    raise ValueError(f"模板数据缺少必需字段: {field}")
                current = current[part]

    def _save_solution(self, zone_id: str, algo: str, solution: "VRPSolution"):
        """保存求解结果并添加结构化日志"""
        try:
            # 确定实验类型
            if algo in ['CW', 'SA', 'TS', 'VNS']:
                exp_type = 'baseline'
            elif '_' in algo and 'ADP' not in algo:
                exp_type = 'hybrid'
            else:
                exp_type = 'adaptive'
                
            # 统计车型信息
            vehicle_stats = {}
            for route_idx, route in enumerate(solution.routes):
                v_type = self._get_suitable_vehicle_type(route, zone_id)
                if v_type not in vehicle_stats:
                    vehicle_stats[v_type] = {'count': 0, 'total_load': 0}
                vehicle_stats[v_type]['count'] += 1
                vehicle_stats[v_type]['total_load'] += sum(self.instance.order_demands[i] for i in route)
                
            # 统计商户信息
            merchant_stats = {}
            for route in solution.routes:
                for cid in route:
                    m_type = self.instance.get_merchant_type(cid)
                    if m_type not in merchant_stats:
                        merchant_stats[m_type] = {'count': 0, 'total_demand': 0}
                    merchant_stats[m_type]['count'] += 1
                    merchant_stats[m_type]['total_demand'] += self.instance.get_order_demand(cid)

            # 生成结构化日志
            log_str = (
                f"[{exp_type.upper()}] "
                f"Zone={zone_id} "
                f"Algo={algo} "
                f"Obj={solution.objective_value:.2f} "
                f"Dist={solution.total_distance:.2f} "
                f"Time={solution.solve_time if hasattr(solution, 'solve_time') else 0:.2f} "
                f"Routes={len(solution.routes)} | "
                f"Vehicles=["
                + ",".join(f"{k}({v['count']}/{v['total_load']:.0f}kg)" for k,v in vehicle_stats.items())
                + "] | Merchants=["
                + ",".join(f"{k}({v['count']}/{v['total_demand']:.0f}kg)" for k,v in merchant_stats.items())
                + "]"
            )
            
            self.logger.info(log_str)
            
            # 保存结果的其他代码保持不变...
            if zone_id not in self.baseline_results:
                self.baseline_results[zone_id] = {}
                
            self.baseline_results[zone_id][algo] = {
                'objective': solution.objective_value,
                'distance': solution.total_distance,
                'time': solution.solve_time if hasattr(solution, 'solve_time') else 0,
                'num_routes': len(solution.routes),
                'routes': [[int(i) for i in r] for r in solution.routes],
                'vehicle_metrics': self._compute_vehicle_metrics(solution)
            }
                
        except Exception as e:
            self.logger.error(f"保存求解结果失败 (区域={zone_id}, 算法={algo}): {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())


    def _calculate_vehicle_utilization(self) -> Dict:
        """计算各类型车辆的利用率"""
        utilization = {}
        for zone_results in self.baseline_results.values():
            for algo_results in zone_results.values():
                if 'vehicle_metrics' in algo_results:
                    for vtype, metrics in algo_results['vehicle_metrics'].items():
                        if vtype not in utilization:
                            utilization[vtype] = []
                        utilization[vtype].append(metrics['utilization'])
        
        # 计算平均利用率
        return {
            vtype: np.mean(rates) for vtype, rates in utilization.items()
        }

    def _calculate_cost_breakdown(self) -> Dict:
        """计算成本构成分析"""
        cost_data = {
            'fixed_costs': {},
            'variable_costs': {},
            'total_costs': {}
        }
        
        for zone_results in self.baseline_results.values():
            for algo_results in zone_results.values():
                if 'cost_metrics' in algo_results:
                    for vtype, metrics in algo_results['cost_metrics'].items():
                        if vtype not in cost_data['fixed_costs']:
                            cost_data['fixed_costs'][vtype] = []
                            cost_data['variable_costs'][vtype] = []
                            cost_data['total_costs'][vtype] = []
                        
                        cost_data['fixed_costs'][vtype].append(metrics['fixed_cost'])
                        cost_data['variable_costs'][vtype].append(metrics['variable_cost'])
                        cost_data['total_costs'][vtype].append(
                            metrics['fixed_cost'] + metrics['variable_cost']
                        )
        
        # 计算平均成本
        return {
            'avg_fixed_costs': {
                vtype: np.mean(costs) 
                for vtype, costs in cost_data['fixed_costs'].items()
            },
            'avg_variable_costs': {
                vtype: np.mean(costs)
                for vtype, costs in cost_data['variable_costs'].items()
            },
            'avg_total_costs': {
                vtype: np.mean(costs)
                for vtype, costs in cost_data['total_costs'].items()
            }
        }

    def _calculate_service_metrics(self) -> Dict:
        """计算服务质量指标"""
        service_data = {
            'service_times': {},
            'satisfaction_rates': {}
        }
        
        for zone_results in self.baseline_results.values():
            for algo_results in zone_results.values():
                if 'service_metrics' in algo_results:
                    for mtype, metrics in algo_results['service_metrics'].items():
                        if mtype not in service_data['service_times']:
                            service_data['service_times'][mtype] = []
                            service_data['satisfaction_rates'][mtype] = []
                        
                        service_data['service_times'][mtype].append(metrics['service_time'])
                        service_data['satisfaction_rates'][mtype].append(
                            metrics['satisfaction_rate']
                        )
        
        # 计算平均指标
        return {
            'avg_service_times': {
                mtype: np.mean(times)
                for mtype, times in service_data['service_times'].items()
            },
            'avg_satisfaction_rates': {
                mtype: np.mean(rates)
                for mtype, rates in service_data['satisfaction_rates'].items()
            }
        }

    def _generate_plot_paths(self) -> List[str]:
        """生成所有可视化图表的路径"""
        plots_dir = os.path.join(self.config.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 生成各类图表
        self._plot_algorithm_comparison(plots_dir)
        self._plot_vehicle_metrics(plots_dir)
        self._plot_merchant_analysis(plots_dir)
        self._plot_cost_analysis(plots_dir)
        
        # 返回所有图表路径
        return [
            'algorithm_comparison.png',
            'vehicle_metrics.png',
            'merchant_analysis.png',
            'cost_analysis.png'
        ]

    def _plot_vehicle_metrics(self, plots_dir: str):
        """绘制车辆指标分析图"""
        util_data = self._calculate_vehicle_utilization()
        
        plt.figure(figsize=(10, 6))
        vehicle_types = list(util_data.keys())
        utilization_rates = [util_data[vt] for vt in vehicle_types]
        
        plt.bar(vehicle_types, utilization_rates)
        plt.title('各类型车辆利用率')
        plt.xlabel('车型')
        plt.ylabel('利用率')
        plt.savefig(os.path.join(plots_dir, 'vehicle_metrics.png'))
        plt.close()

    def _plot_merchant_analysis(self, plots_dir: str):
        """绘制商户分析图"""
        service_metrics = self._calculate_service_metrics()
        
        plt.figure(figsize=(12, 6))
        merchant_types = list(service_metrics['avg_service_times'].keys())
        service_times = [service_metrics['avg_service_times'][mt] for mt in merchant_types]
        satisfaction = [service_metrics['avg_satisfaction_rates'][mt] for mt in merchant_types]
        
        x = np.arange(len(merchant_types))
        width = 0.35
        
        plt.bar(x - width/2, service_times, width, label='平均服务时间')
        plt.bar(x + width/2, satisfaction, width, label='满意度')
        
        plt.title('商户服务分析')
        plt.xlabel('商户类型')
        plt.xticks(x, merchant_types)
        plt.legend()
        
        plt.savefig(os.path.join(plots_dir, 'merchant_analysis.png'))
        plt.close()

    def save_results(self):
        """保存实验结果"""
        try:
            # 检查是否有任何实验结果
            if not any([self.baseline_results, self.hybrid_results, self.adaptive_results]):
                self.logger.warning("没有任何实验结果可以保存")
                return
    
            # 创建结果目录
            results_dir = os.path.join(self.config.output_dir, 'results')
            os.makedirs(results_dir, exist_ok=True)
            
            # 初始化收集指标的列表
            basic_metrics = []
            vehicle_metrics = []
            merchant_metrics = []
    
            # 处理所有类型的算法结果
            result_mappings = [
                (self.baseline_results, 'baseline'),
                (self.hybrid_results, 'hybrid'),
                (self.adaptive_results, 'adaptive')
            ]
    
            for results_dict, result_type in result_mappings:
                if not results_dict:
                    continue
                    
                self.logger.info(f"正在处理 {result_type} 算法结果...")
                
                for zone_id, algo_dict in results_dict.items():
                    # 初始化或更新实例
                    merchants_df = self.zone_map[zone_id]
                    if not isinstance(merchants_df, pd.DataFrame):
                        merchants_df = pd.DataFrame(merchants_df)
                    self.instance = self._create_vrp_instance(merchants_df, zone_id)
                    
                    try:
                        self.instance = self._create_vrp_instance(merchants_df, zone_id)
                    except Exception as e:
                        self.logger.error(f"创建区域 {zone_id} 的VRP实例失败: {str(e)}")
                        continue
                    
                    # 获取最大有效索引
                    max_valid_idx = len(merchants_df) - 1
                    
                    for algo_code, metrics in algo_dict.items():
                        try:
                            # 1. 处理路线和车辆分配
                            if 'routes' in metrics:
                                # 过滤无效路线
                                valid_routes = []
                                for route in metrics['routes']:
                                    valid_route = [i for i in route if i <= max_valid_idx]
                                    if valid_route:  # 只保留非空路线
                                        valid_routes.append(valid_route)
                                
                                metrics['routes'] = valid_routes
                                metrics['num_routes'] = len(valid_routes)
                                
                                # 更新车辆分配
                                metrics['vehicle_assignments'] = {}
                                for i, route in enumerate(valid_routes):
                                    try:
                                        v_type = self._get_suitable_vehicle_type(route, zone_id)
                                        metrics['vehicle_assignments'][i] = {
                                            'type': v_type,
                                            'customers': route
                                        }
                                    except Exception as e:
                                        self.logger.warning(f"处理路线 {i} 的车型分配失败: {str(e)}")
                                        continue
                            
                            # 2. 添加基本指标
                            basic_row = {
                                'experiment_type': result_type,
                                'zone_id': zone_id,
                                'algorithm': algo_code,
                                'objective': metrics.get('objective', 0),
                                'distance': metrics.get('distance', 0),
                                'time': metrics.get('time', 0),
                                'num_routes': metrics.get('num_routes', 0)
                            }
                            
                            # 3. 添加特定算法类型的指标
                            if result_type == 'hybrid':
                                basic_row.update({
                                    'improvement': metrics.get('improvement', 0),
                                    'first_stage_obj': metrics.get('first_stage_objective', 0),
                                    'second_stage_obj': metrics.get('second_stage_objective', 0)
                                })
                            elif result_type == 'adaptive':
                                basic_row.update({
                                    'convergence_score': metrics.get('convergence_score', 0),
                                    'param_adjustments': metrics.get('param_adjustments', 0),
                                    'adaptation_steps': metrics.get('adaptation_steps', 0)
                                })
                                if 'parameter_history' in metrics:
                                    basic_row['final_parameters'] = str(metrics['parameter_history'][-1]) if metrics['parameter_history'] else "{}"
                            
                            basic_metrics.append(basic_row)
                            
                            # 4. 处理车辆指标
                            if 'routes' in metrics and 'vehicle_assignments' in metrics:
                                for route_idx, route in enumerate(metrics['routes']):
                                    if route_idx in metrics['vehicle_assignments']:
                                        v_assign = metrics['vehicle_assignments'][route_idx]
                                        v_type = v_assign['type']
                                        try:
                                            route_load = sum(self.instance.get_order_demand(i) 
                                                           for i in route if i <= max_valid_idx)
                                            
                                            if route_load > 0:  # 只记录有效负载的路线
                                                vehicle_row = {
                                                    'zone_id': zone_id,
                                                    'algorithm': algo_code,
                                                    'route_id': route_idx,
                                                    'vehicle_type': v_type,
                                                    'route_load': route_load,
                                                    'capacity_utilization': route_load / self.instance.vehicle_types[v_type]['capacity'],
                                                    'route_length': len(route),
                                                    'fixed_cost': v_assign.get('fixed_cost', 0)
                                                }
                                                vehicle_metrics.append(vehicle_row)
                                                
                                                # 5. 处理商户指标
                                                for order_pos, order in enumerate(route):
                                                    if order <= max_valid_idx:
                                                        merchant_row = {
                                                            'zone_id': zone_id,
                                                            'algorithm': algo_code,
                                                            'route_id': route_idx,
                                                            'order_id': order,
                                                            'merchant_type': self.instance.get_merchant_type(order),
                                                            'demand': self.instance.get_order_demand(order),
                                                            'serving_vehicle': v_type,
                                                            'in_route_position': order_pos
                                                        }
                                                        merchant_metrics.append(merchant_row)
                                        except Exception as e:
                                            self.logger.warning(f"处理路线 {route_idx} 的指标失败: {str(e)}")
                                            continue
                                            
                        except Exception as e:
                            self.logger.error(f"处理算法 {algo_code} 的结果失败: {str(e)}")
                            continue
    
            # 保存所有指标到CSV文件
            if basic_metrics:
                # 保存基本指标
                basic_df = pd.DataFrame(basic_metrics)
                basic_df.to_csv(os.path.join(results_dir, 'basic_metrics.csv'), 
                              index=False, encoding='utf-8-sig')
                
                # 保存experiment_results.csv
                basic_df.to_csv(os.path.join(results_dir, 'experiment_results.csv'),
                              index=False, encoding='utf-8-sig')
                
                # 保存车辆分析
                if vehicle_metrics:
                    vehicle_df = pd.DataFrame(vehicle_metrics)
                    vehicle_df.to_csv(os.path.join(results_dir, 'vehicle_analysis.csv'), 
                                    index=False, encoding='utf-8-sig')
                
                # 保存商户分析
                if merchant_metrics:
                    merchant_df = pd.DataFrame(merchant_metrics)
                    merchant_df.to_csv(os.path.join(results_dir, 'merchant_analysis.csv'), 
                                    index=False, encoding='utf-8-sig')
                
                # 生成汇总统计
                if vehicle_metrics and merchant_metrics:
                    self._save_summary_stats(
                        basic_df, 
                        pd.DataFrame(vehicle_metrics), 
                        pd.DataFrame(merchant_metrics), 
                        results_dir
                    )
                
                self.logger.info(f"已成功保存所有实验结果到目录: {results_dir}")
                self.logger.info(f"共处理 {len(basic_metrics)} 条结果记录")
                if vehicle_metrics:
                    self.logger.info(f"包含 {len(vehicle_metrics)} 条车辆记录")
                if merchant_metrics:
                    self.logger.info(f"包含 {len(merchant_metrics)} 条商户记录")
            else:
                self.logger.warning("没有有效的指标可以保存")
                    
        except Exception as e:
            self.logger.error(f"保存结果失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _save_summary_stats(self, basic_df: pd.DataFrame, 
                        vehicle_df: pd.DataFrame, 
                        merchant_df: pd.DataFrame,
                        results_dir: str):
        """生成并保存汇总统计"""
        
        # 1. 算法总体性能汇总
        algo_summary = basic_df.groupby(['experiment_type', 'algorithm']).agg({
            'objective': ['mean', 'std'],
            'distance': ['mean', 'std'],
            'time': 'mean',
            'num_routes': 'mean'
        }).round(2)
        
        # 2. 车型使用情况汇总
        vehicle_summary = vehicle_df.groupby(['algorithm', 'vehicle_type']).agg({
            'route_load': ['mean', 'max'],
            'capacity_utilization': 'mean',
            'route_length': 'mean',
            'fixed_cost': 'sum'
        }).round(2)
        
        # 3. 商户服务情况汇总
        merchant_summary = merchant_df.groupby(['algorithm', 'merchant_type']).agg({
            'demand': ['sum', 'mean'],
            'in_route_position': 'mean'
        }).round(2)
        
        # 保存汇总统计
        algo_summary.to_csv(os.path.join(results_dir, 'algorithm_summary.csv'), 
                        encoding='utf-8-sig')
        vehicle_summary.to_csv(os.path.join(results_dir, 'vehicle_summary.csv'), 
                            encoding='utf-8-sig')
        merchant_summary.to_csv(os.path.join(results_dir, 'merchant_summary.csv'), 
                            encoding='utf-8-sig')    
    def _plot_metrics(self, df: pd.DataFrame, plots_dir: str):
        """生成指标可视化"""
        # 1. 目标值箱线图
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='algorithm', y='objective')
        plt.title('各算法目标值分布')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'objective_boxplot.png'))
        plt.close()
        
        # 2. 求解时间条形图
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='algorithm', y='time')
        plt.title('各算法平均求解时间')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'solve_time_barplot.png'))
        plt.close()
        
        # 3. 路线数量分布图
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df, x='algorithm', y='num_routes')
        plt.title('各算法路线数量分布')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'routes_violin.png'))
        plt.close()
    
    # def run_baseline_experiment(self):
    #     """执行基线算法实验"""
    #     # 使用正确的区域列表名称
    #     areas_to_process = self.area_list  # 使用 area_list 而不是 test_areas
        
    #     # 创建进度条
    #     area_pbar = tqdm(areas_to_process, desc="处理路区", unit="区")
        
    #     for area in area_pbar:
    #         area_pbar.set_description(f"处理路区 {area}")
            
    #         for algo in self.algorithms:
    #             self.logger.info(f"开始处理区域 {area} - 算法 {algo}")
                
    #             try:
    #                 # 创建求解器实例
    #                 instance = self._create_instance(area)
    #                 solver = self._create_solver(algo, instance)
                    
    #                 # 记录开始时间
    #                 start_time = time.time()
                    
    #                 # 求解
    #                 self.logger.info(f"开始{algo}求解...")
    #                 solution = solver.solve()
                    
    #                 # 记录结束时间和统计信息
    #                 end_time = time.time()
    #                 solve_time = end_time - start_time
                    
    #                 if solution and solution.is_feasible():
    #                     self.logger.info(f"区域 {area} 算法 {algo} 成功: "
    #                                    f"obj={solution.objective_value:.2f}, "
    #                                    f"time={solve_time:.2f}s")
    #                     self._save_solution(area, algo, solution)
    #                 else:
    #                     self.logger.warning(f"区域 {area} 算法 {algo} 无可行解")
                        
    #             except Exception as e:
    #                 self.logger.error(f"执行实验失败 (区域={area}, 算法={algo}): {str(e)}")
    #                 self.logger.error(f"Traceback: {traceback.format_exc()}")
                    
    #         # 更新进度条
    #         area_pbar.update(1)
            
    #     area_pbar.close()
        
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
            self.instance = self._create_vrp_instance(self.zone_map[zone_id], zone_id)
            instance = self._create_vrp_instance(merchants_df, zone_id)

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
                    self._log_realtime_metrics(zone_id, base_algo, solution)
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
                    self._log_realtime_metrics(zone_id, f"{fs}_{ss}", final_sol)
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
                    self._log_realtime_metrics(zone_id, f"{algo_type}-ADP", solution)
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
        self.instance = self._create_vrp_instance(self.zone_map[zone_id], zone_id)

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
        instance = self._create_vrp_instance(merchants_df, zone_id)

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
                self._log_realtime_metrics(zone_id, f"{solver_type}_config_{idx}", solution)
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

    def _compute_path_diversity(self, solution: "VRPSolution") -> Dict[str, Dict]:
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

    def _compute_cost_metrics(self, solution: "VRPSolution") -> Dict[str, Dict]:
        """
        扫描 solution.vehicle_assignments + solution.routes，统计各种车型的使用情况。
        """
        if not hasattr(solution, 'vehicle_assignments'):
            return {}

        # 统计各车型使用情况
        stats_map = {}
        for vtype, specs in solution.instance.vehicle_types.items():
            stats_map[vtype] = {
                'used_count': 0,
                'total_fixed_cost': 0.0,
                'total_route_length': 0.0,
                'route_count': 0,
                'available_count': specs.get('count', 1)
            }

        for r_idx, route in enumerate(solution.routes):
            if r_idx not in solution.vehicle_assignments:
                continue
            vinfo = solution.vehicle_assignments[r_idx]  # 形如 {'type':'small','fixed_cost':800,...}
            vt = vinfo['type']

            if vt not in stats_map:
                stats_map[vt] = {
                    'used_count': 0,
                    'total_fixed_cost': 0.0,
                    'total_route_length': 0.0,
                    'route_count': 0,
                    'available_count': 1
                }

            stats_map[vt]['used_count'] += 1
            stats_map[vt]['total_fixed_cost'] += vinfo.get('fixed_cost', 0.0)
            route_len = self._calc_single_route_length(solution, route)
            stats_map[vt]['total_route_length'] += route_len
            stats_map[vt]['route_count'] += 1

        out = {}
        for vtype, data in stats_map.items():
            used_count = data['used_count']
            route_count = data['route_count']
            avg_len = data['total_route_length'] / route_count if route_count > 0 else 0.0
            utilization = used_count / data['available_count'] if data['available_count'] > 0 else 0.0
            out[vtype] = {
                'used_count': used_count,
                'utilization': utilization,
                'fixed_cost': data['total_fixed_cost'],
                'avg_route_length': avg_len
            }

        return out

    
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

    def _get_results_path(self):
        results_dir = os.path.join(self.config.output_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        return os.path.join(results_dir, 'experiment_results.csv')

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