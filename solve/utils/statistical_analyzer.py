import pandas as pd
import numpy as np
from scipy import stats
import os
from typing import Dict, List, Optional
import logging

class StatisticalAnalyzer:
    def __init__(self, results_dir: str):
        """初始化统计分析器"""
        self.results_dir = results_dir
        # 修改这里：确保包含 'results' 子目录
        self.results_path = os.path.join(results_dir, 'results', 'experiment_results.csv')
        
        # 添加详细的路径检查
        if not os.path.exists(self.results_dir):
            raise FileNotFoundError(f"结果目录不存在: {self.results_dir}")
            
        results_subdir = os.path.join(results_dir, 'results')
        if not os.path.exists(results_subdir):
            raise FileNotFoundError(f"results子目录不存在: {results_subdir}")
            
        if not os.path.exists(self.results_path):
            raise FileNotFoundError(f"结果文件不存在: {self.results_path}")

        self.logger = logging.getLogger("StatisticalAnalyzer")

    def analyze_results(self) -> Dict:
        """
        分析实验结果
        Returns:
            Dict: 包含各类分析结果的字典
        """
        try:
            # 检查结果文件是否存在
            if not os.path.exists(self.results_path):
                self.logger.error(f"结果文件不存在: {self.results_path}")
                return self._get_empty_results()

            # 读取结果数据
            results_df = pd.read_csv(self.results_path)
            
            # 确保数据帧不为空
            if results_df.empty:
                self.logger.warning("结果数据为空")
                return self._get_empty_results()

            # 执行各项分析
            analysis_results = {
                'basic_stats': self._compute_basic_stats(results_df),
                'vehicle_type_analysis': self._analyze_vehicle_types(results_df),
                'merchant_type_analysis': self._analyze_merchant_types(results_df),
                'algorithm_comparison': self._compare_algorithms(results_df),
                'statistical_tests': self._perform_statistical_tests(results_df),
                'by_vehicle_type': self._analyze_by_vehicle_type(),
                'by_merchant_type': self._analyze_by_merchant_type(),
                'vehicle_merchant_compatibility': self._analyze_compatibility(),
                }
            # 返回分析结果
            return analysis_results
        except Exception as e:
            self.logger.error(f"统计分析失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._get_empty_results()

    def _get_empty_results(self) -> Dict:
        """
        返回空的结果结构，确保包含 'statistical_tests' 和 'algorithm_comparison'
        """
        return {
            'basic_stats': {},
            'vehicle_type_analysis': {},
            'merchant_type_analysis': {},
            'algorithm_comparison': {},  # 确保这个键存在
            'statistical_tests': {
                't_tests': {},
                'f_tests': {},
                'anova_tests': {}
            }
        }

    def _compute_basic_stats(self, df: pd.DataFrame) -> Dict:
        """计算基本统计指标"""
        stats_dict = {}
        
        # 检查必要的列是否存在
        for col in ['objective', 'time', 'num_routes']:
            if col in df.columns:
                stats_dict[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median()
                }
        
        return stats_dict

    def _analyze_vehicle_types(self, df: pd.DataFrame) -> Dict:
        """分析不同车型的性能"""
        # 如果没有车型相关列，返回空字典
        if 'vehicle_type' not in df.columns:
            return {}

        vehicle_stats = {}
        for vtype in df['vehicle_type'].unique():
            type_data = df[df['vehicle_type'] == vtype]
            if len(type_data) > 0:
                vehicle_stats[vtype] = {
                    'count': len(type_data),
                    'avg_utilization': type_data['utilization'].mean() if 'utilization' in type_data else 0,
                    'avg_cost': type_data['total_cost'].mean() if 'total_cost' in type_data else 0,
                    'avg_distance': type_data['distance'].mean() if 'distance' in type_data else 0
                }
        
        return vehicle_stats

    def _analyze_merchant_types(self, df: pd.DataFrame) -> Dict:
        """分析不同商户类型的服务情况"""
        # 如果没有商户类型列，返回空字典
        if 'merchant_type' not in df.columns:
            return {}

        merchant_stats = {}
        for mtype in df['merchant_type'].unique():
            type_data = df[df['merchant_type'] == mtype]
            merchant_stats[mtype] = {
                'count': len(type_data)
            }
            # 添加可选统计
            if 'service_time' in type_data:
                merchant_stats[mtype]['avg_service_time'] = type_data['service_time'].mean()
            if 'satisfaction' in type_data:
                merchant_stats[mtype]['avg_satisfaction'] = type_data['satisfaction'].mean()
        
        return merchant_stats

    def _compare_algorithms(self, df: pd.DataFrame) -> Dict:
        """比较不同算法的性能"""
        # 确保algorithm列存在
        if 'algorithm' not in df.columns:
            return {}

        algorithm_stats = {}
        metrics = ['objective', 'time', 'num_routes']
        
        for algo in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algo]
            algorithm_stats[algo] = {}
            
            for metric in metrics:
                if metric in algo_data:
                    algorithm_stats[algo][f'{metric}_mean'] = algo_data[metric].mean()
                    algorithm_stats[algo][f'{metric}_std'] = algo_data[metric].std()
        
        return algorithm_stats

    def _perform_statistical_tests(self, df: pd.DataFrame) -> Dict:
        """执行统计检验"""
        if 'algorithm' not in df.columns or 'objective' not in df.columns:
            return {
                't_tests': {},
                'f_tests': {},
                'anova_tests': {}
            }

        return {
            't_tests': self._run_t_tests(df),
            'f_tests': self._run_f_tests(df),
            'anova_tests': self._run_anova_tests(df)
        }

    def _run_t_tests(self, df: pd.DataFrame) -> Dict:
        """执行t检验"""
        t_test_results = {}
        algorithms = sorted(df['algorithm'].unique())
        
        for i in range(len(algorithms)):
            for j in range(i+1, len(algorithms)):
                algo1, algo2 = algorithms[i], algorithms[j]
                data1 = df[df['algorithm'] == algo1]['objective']
                data2 = df[df['algorithm'] == algo2]['objective']
                
                if len(data1) > 0 and len(data2) > 0:
                    t_stat, p_value = stats.ttest_ind(data1, data2)
                    t_test_results[f'{algo1}_vs_{algo2}'] = {
                        't_statistic': float(t_stat),  # 确保数值可序列化
                        'p_value': float(p_value),
                        'significant': bool(p_value < 0.05)
                    }
        
        return t_test_results

    def _run_f_tests(self, df: pd.DataFrame) -> Dict:
        """执行F检验"""
        f_test_results = {}
        algorithms = sorted(df['algorithm'].unique())
        
        for i in range(len(algorithms)):
            for j in range(i+1, len(algorithms)):
                algo1, algo2 = algorithms[i], algorithms[j]
                data1 = df[df['algorithm'] == algo1]['objective']
                data2 = df[df['algorithm'] == algo2]['objective']
                
                if len(data1) > 0 and len(data2) > 0:
                    try:
                        f_stat, p_value = stats.f_oneway(data1, data2)
                        f_test_results[f'{algo1}_vs_{algo2}'] = {
                            'f_statistic': float(f_stat),
                            'p_value': float(p_value),
                            'significant': bool(p_value < 0.05)
                        }
                    except Exception as e:
                        self.logger.warning(f"F检验失败 ({algo1} vs {algo2}): {str(e)}")
        
        return f_test_results

    def _run_anova_tests(self, df: pd.DataFrame) -> Dict:
        """执行ANOVA分析"""
        try:
            groups = [group['objective'].values for name, group in df.groupby('algorithm') if len(group) > 0]
            if len(groups) > 1:  # 确保至少有两组数据
                f_stat, p_value = stats.f_oneway(*groups)
                return {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05)
                }
        except Exception as e:
            self.logger.warning(f"ANOVA分析失败: {str(e)}")
        
        return {}

    def _save_analysis_results(self, results: Dict):
        """保存分析结果"""
        try:
            # 检查分析结果字典是否包含所需的键
            if 'algorithm_comparison' not in results:
                self.logger.warning("缺少'algorithm_comparison'键，无法保存")
                return

            # 创建分析结果目录
            analysis_dir = os.path.join(self.results_dir, 'analysis')
            os.makedirs(analysis_dir, exist_ok=True)

            # 保存各项分析结果
            for key, data in results.items():
                if data:  # 只保存非空结果
                    file_path = os.path.join(analysis_dir, f'{key}.csv')
                    # 转换为DataFrame并保存
                    if isinstance(data, dict):
                        if key == 'statistical_tests':
                            # 对于统计检验结果，分别保存每种检验
                            for test_type, test_results in data.items():
                                test_path = os.path.join(analysis_dir, f'{test_type}.csv')
                                pd.DataFrame(test_results).to_csv(test_path)
                        else:
                            pd.DataFrame(data).transpose().to_csv(file_path)
                    else:
                        pd.DataFrame(data).to_csv(file_path)

            self.logger.info(f"分析结果已保存至: {analysis_dir}")
        except Exception as e:
            self.logger.error(f"保存分析结果失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
