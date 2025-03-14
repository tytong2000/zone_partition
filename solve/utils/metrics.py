import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional

class MetricsCalculator:
    """VRP问题的基础指标计算器"""     
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def calculate_total_distance(solution: List[Tuple[float, float]], 
                            distances: Dict[Tuple[float, float], Dict[Tuple[float, float], float]]) -> float:
        """计算路线总距离"""
        total_distance = 0.0
        for i in range(1, len(solution)):
            total_distance += distances[solution[i - 1]][solution[i]]
        return total_distance    

    def calculate_solution_cost(self, solution: List[Tuple[float, float]], 
                              distances: Dict[Tuple[float, float], Dict[Tuple[float, float], float]], 
                              vehicle_capacity: float, 
                              demand: List[float]) -> float:
        """计算解决方案总成本（包括容量惩罚）"""
        total_cost = 0.0
        load = 0.0

        for i in range(len(solution)):
            current_node = solution[i]
            load += demand[i]
            
            if load > vehicle_capacity:
                total_cost += 10000  # 容量违反惩罚
                
            if i > 0:
                total_cost += distances[solution[i - 1]][current_node]

        return total_cost

    def evaluate_solution_quality(self, solution: List[Tuple[float, float]], 
                                distances: Dict[Tuple[float, float], Dict[Tuple[float, float], float]], 
                                vehicle_capacity: float,
                                demand: List[float]) -> Dict[str, float]:
        """综合解决方案质量评估"""
        metrics = {
            'total_distance': self.calculate_total_distance(solution, distances),
            'total_cost': self.calculate_solution_cost(solution, distances, vehicle_capacity, demand),
            'num_stops': len(solution) - 1,
            'average_distance_between_stops': 0.0,
            'capacity_utilization': sum(demand) / vehicle_capacity if vehicle_capacity > 0 else 0
        }
        
        if len(solution) > 1:
            metrics['average_distance_between_stops'] = metrics['total_distance'] / (len(solution) - 1)
            
        return metrics


class VRPStatisticalAnalyzer:
    """VRP问题的统计分析器"""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics_calculator = MetricsCalculator()
        self.logger = logging.getLogger(__name__)

    def analyze_algorithm_performance(self, results: Dict[str, Dict[str, Dict]]) -> pd.DataFrame:
        """分析不同算法的性能"""
        performance_data = []
        
        for zone_id, zone_results in results.items():
            for algo_name, metrics in zone_results.items():
                performance_data.append({
                    'zone_id': zone_id,
                    'algorithm': algo_name,
                    'total_distance': metrics.get('total_distance', 0),
                    'computation_time': metrics.get('computation_time', 0),
                    'solution_quality': metrics.get('solution_quality', 0),
                    'num_iterations': metrics.get('num_iterations', 0)
                })
                
        return pd.DataFrame(performance_data)

    def generate_statistics_report(self, results: Dict[str, Dict[str, Dict]], vehicle_metrics: Dict, merchant_stats: Dict) -> str:
        """生成统计报告并保存为CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f'statistics_report_{timestamp}.csv')
        
        # 合并算法性能、商户分析和多车型分析
        performance_df = self.analyze_algorithm_performance(results)
        
        # 将结果保存到DataFrame中
        merged_data = performance_df.copy()

        # 添加多车型数据
        for vehicle_type, metrics in vehicle_metrics.items():
            merged_data[f'{vehicle_type}_used_count'] = metrics['used_count']
            merged_data[f'{vehicle_type}_utilization'] = metrics['utilization']
            merged_data[f'{vehicle_type}_turnover_rate'] = metrics['turnover_rate']
        
        # 添加商户类型数据
        for merchant_type, stats in merchant_stats.items():
            merged_data[f'{merchant_type}_avg_volume'] = stats['avg_volume']
            merged_data[f'{merchant_type}_avg_weight'] = stats['avg_weight']
            merged_data[f'{merchant_type}_avg_items'] = stats['avg_items']

        # 保存为CSV
        merged_data.to_csv(report_file, index=False, encoding='utf-8')
        
        logging.info(f"统计报告已生成并保存: {report_file}")
        return report_file

    def run_full_analysis(self, results: Dict[str, Dict[str, Dict]], vehicle_metrics: Dict, merchant_stats: Dict) -> Dict:
        """运行完整分析流程并生成CSV报告"""
        try:
            # 生成统计报告
            report_file = self.generate_statistics_report(results, vehicle_metrics, merchant_stats)
            analysis_results = {'report_file': report_file}
            
            # 返回分析结果
            return analysis_results
        except Exception as e:
            self.logger.error(f"分析过程出错: {str(e)}")
            return {}


def generate_vrp_report_from_csv(csv_file: str, output_dir: str, vehicle_metrics: Dict, merchant_stats: Dict):
    """从 CSV 文件中读取实验结果并生成统计报告"""
    try:
        # 加载实验结果
        results_df = pd.read_csv(csv_file)

        # 将实验结果转换为合适的字典结构
        results = {}
        for _, row in results_df.iterrows():
            zone_id = row['zone_id']
            algorithm = row['algorithm']
            if zone_id not in results:
                results[zone_id] = {}
            results[zone_id][algorithm] = {
                'total_distance': row['total_distance'],
                'computation_time': row['computation_time'],
                'solution_quality': row['solution_quality'],
                'num_iterations': row['num_iterations']
            }

        # 创建分析器并生成报告
        analyzer = VRPStatisticalAnalyzer(output_dir)
        analysis_results = analyzer.run_full_analysis(results, vehicle_metrics, merchant_stats)

        return analysis_results

    except Exception as e:
        logging.error(f"分析失败: {str(e)}")
        return {}
