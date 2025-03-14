#!/usr/bin/env python
# coding: utf-8

import os
import yaml
import json
from typing import Dict, Any, List, Union
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
@dataclass
class SolverConfig:
    """求解器配置类"""
    # 文件路径
    excel_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "littleorders.xlsx")
    city_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "wuhan_city.geojson")
    road_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "best_aligned_road_network.geojson")
    output_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # 算法通用参数
    
    # 算法通用参数
    vehicle_capacity: float = 5000.0
    max_route_time: float = 12.0
    random_seed: int = 42
    
    # 并行计算设置
    parallel_evaluation: bool = True
    max_workers: int = None
    
    # 分区参数
    min_clusters: int = 30
    max_clusters: int = 100
    road_buffer_distance: float = 1000.0
    
    # Clarke-Wright 参数
    cw_parallel_savings: bool = True
    cw_time_window_enforce: bool = False
    
    # Simulated Annealing 参数
    sa_initial_temp: float = 1000.0
    sa_cooling_rate: float = 0.95
    sa_iterations: int = 100
    
    # Tabu Search 参数
    ts_tabu_size: int = 10
    ts_max_iterations: int = 50
    ts_neighborhood_size: int = 20
    
    # VNS 参数
    vns_max_iterations: int = 30
    vns_max_neighborhoods: int = 3
    vns_shake_intensity: float = 0.3
    
    # 车辆配置
    vehicle_types: Dict = field(default_factory=lambda: {
        'small': {'capacity': 3000, 'fixed_cost': 800, 'count': 20},
        'medium': {'capacity': 5000, 'fixed_cost': 1000, 'count': 10},
        'large': {'capacity': 8000, 'fixed_cost': 1500, 'count': 5}
    })

import pandas as pd
import logging
from typing import Dict, Any

def load_and_prepare_merchant_data(excel_path: str) -> pd.DataFrame:
    """
    加载并预处理商户数据，针对不同商户类型进行特殊处理
    
    Args:
        excel_path: Excel文件路径
        
    Returns:
        处理后的DataFrame，包含所有必要的分析字段
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_path)
        
        # 基本数据清洗
        df = df.dropna(subset=['经度', '纬度', '商户类型', '商家名称'])
        
        # 验证商户类型
        valid_types = {'购物中心', '超市', '便利店'}
        invalid_types = set(df['商户类型'].unique()) - valid_types
        if invalid_types:
            raise ValueError(f"发现无效的商户类型: {invalid_types}")
            
        # 添加车型分配比例
        def get_vehicle_ratios(merchant_type):
            if merchant_type == '购物中心':
                return {
                    '6.8米厢式车': 0.6,
                    '12米半挂车': 0.4
                }
            elif merchant_type == '超市':
                return {
                    '金杯车': 0.5,
                    '依维柯车': 0.3,
                    '6.8米厢式车': 0.2
                }
            else:  # 便利店
                return {
                    '金杯车': 0.8,
                    '依维柯车': 0.2
                }
                
        # 添加车次范围
        def get_vehicle_trips_range(merchant_type):
            if merchant_type == '购物中心':
                return (5, 9)
            elif merchant_type == '超市':
                return (15, 25)
            else:  # 便利店
                return (10, 16)
        
        # 扩展数据框，添加分析所需的列
        df['vehicle_ratios'] = df['商户类型'].map(get_vehicle_ratios)
        df['min_trips'], df['max_trips'] = zip(*df['商户类型'].map(get_vehicle_trips_range))
        
        # 计算每个商户的总需求（考虑重量和体积）
        df['总需求'] = df['托运单重量']  # 可以根据需要调整计算方式
        
        # 考虑重量密度进行标准化
        if '重量密度' in df.columns:
            df['标准化需求'] = df['总需求'] * df['重量密度']
        else:
            df['标准化需求'] = df['总需求']
            
        # 计算体积效率（体积/重量比）
        df['体积效率'] = df['托运单体积'] / df['托运单重量']
        
        # 根据商户类型设置容量约束
        capacity_constraints = {
            '购物中心': (2500, 4000),
            '超市': (1500, 2500),
            '便利店': (500, 800)
        }
        df['最小容量'], df['最大容量'] = zip(*df['商户类型'].map(capacity_constraints))
        
        # 数据类型转换和验证
        numeric_columns = [
            '经度', '纬度', '托运单重量', '托运单体积', 
            '标准化需求', '体积效率', '最小容量', '最大容量'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # 删除关键字段缺失的行
        df = df.dropna(subset=['商户类型', '商家名称', '托运单重量', '托运单体积'])
        
        # 添加商户ID（如果没有）
        if '商户ID' not in df.columns:
            df['商户ID'] = df['商家名称'].str.cat(df.index.astype(str), sep='_')
            
        # 记录处理结果
        logging.info(f"数据加载完成:")
        logging.info(f"- 总商户数: {len(df)}")
        for m_type in valid_types:
            count = len(df[df['商户类型'] == m_type])
            logging.info(f"- {m_type}数量: {count}")
            
        return df
        
    except Exception as e:
        logging.error(f"加载商户数据失败: {str(e)}")
        raise

def analyze_merchant_metrics(df: pd.DataFrame) -> Dict:
    """
    计算不同商户类型的关键指标
    
    Args:
        df: 预处理后的数据框
        
    Returns:
        包含各类指标的字典
    """
    metrics = {}
    
    for merchant_type in df['商户类型'].unique():
        type_df = df[df['商户类型'] == merchant_type]
        
        metrics[merchant_type] = {
            'count': len(type_df),
            'avg_volume': type_df['托运单体积'].mean(),
            'avg_weight': type_df['托运单重量'].mean(),
            'avg_density': type_df['重量密度'].mean() if '重量密度' in type_df.columns else None,
            'volume_efficiency': type_df['体积效率'].mean(),
            'total_demand': type_df['总需求'].sum(),
            'avg_trips': (type_df['min_trips'] + type_df['max_trips']).mean() / 2
        }
        
    return metrics

def load_config_files(config_paths: Union[str, List[str]]) -> Dict[str, Any]:
    """
    加载一个或多个配置文件
    
    Args:
        config_paths: 配置文件路径或路径列表
    
    Returns:
        合并后的配置字典
    """
    if isinstance(config_paths, str):
        config_paths = [config_paths]
    
    merged_config = {}
    for path in config_paths:
        try:
            if not os.path.exists(path):
                logging.warning(f"配置文件不存在: {path}")
                continue
                
            with open(path, 'r', encoding='utf-8') as f:
                if path.endswith('.yaml') or path.endswith('.yml'):
                    config = yaml.safe_load(f)
                elif path.endswith('.json'):
                    config = json.load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {path}")
                    
            merged_config = merge_dicts(merged_config, config)
            
        except Exception as e:
            logging.error(f"加载配置文件 {path} 失败: {str(e)}")
            raise
            
    return merged_config

def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    递归合并两个字典，dict2的值会覆盖dict1中的相同键的值
    
    Args:
        dict1: 基础字典
        dict2: 要合并的字典
    
    Returns:
        合并后的新字典
    """
    merged = dict1.copy()
    
    for key, value in dict2.items():
        if (
            key in merged and 
            isinstance(merged[key], dict) and 
            isinstance(value, dict)
        ):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
            
    return merged

class ConfigManager:
    """配置管理类"""
    
    def __init__(self, config_paths: Union[str, List[str]] = None):
        """
        初始化配置管理器
        
        Args:
            config_paths: 配置文件路径或路径列表
        """
        self.config_data = {}
        self.modified_time = None
        
        if config_paths:
            self.load_config(config_paths)
            
    def load_config(self, config_paths: Union[str, List[str]]) -> None:
        """
        加载配置文件
        
        Args:
            config_paths: 配置文件路径或路径列表
        """
        self.config_data = load_config_files(config_paths)
        self.modified_time = datetime.now()
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            default: 默认值
            
        Returns:
            配置值
        """
        try:
            value = self.config_data
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            value: 要设置的值
        """
        keys = key.split('.')
        current = self.config_data
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
            
        current[keys[-1]] = value
        self.modified_time = datetime.now()
        
    def save(self, filepath: str) -> None:
        """
        保存配置到文件
        
        Args:
            filepath: 保存路径
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                    yaml.dump(self.config_data, f, allow_unicode=True, default_flow_style=False)
                elif filepath.endswith('.json'):
                    json.dump(self.config_data, f, ensure_ascii=False, indent=2)
                else:
                    raise ValueError(f"不支持的文件格式: {filepath}")
                    
        except Exception as e:
            logging.error(f"保存配置文件失败: {str(e)}")
            raise
            
    def validate_config(self, required_keys: List[str]) -> bool:
        """
        验证配置是否包含所有必需的键
        
        Args:
            required_keys: 必需的键列表
            
        Returns:
            是否验证通过
        """
        for key in required_keys:
            if self.get(key) is None:
                logging.error(f"缺少必需的配置项: {key}")
                return False
        return True
    
    def create_solver_config(self) -> SolverConfig:
        """
        创建求解器配置实例
        
        Returns:
            SolverConfig实例
        """
        config_dict = {
            'excel_path': self.get('files.excel_path', 'data/orders.xlsx'),
            'city_path': self.get('files.city_path', 'data/city.geojson'),
            'road_path': self.get('files.road_path', 'data/road_network.geojson'),
            'output_dir': self.get('files.output_dir', 'output'),
            'vehicle_capacity': self.get('solver.vehicle_capacity', 5000.0),
            'max_route_time': self.get('solver.max_route_time', 12.0),
            'random_seed': self.get('solver.random_seed', 42),
            'parallel_evaluation': self.get('solver.parallel_evaluation', True),
            'max_workers': self.get('solver.max_workers', None),
            'min_clusters': self.get('solver.min_clusters', 30),
            'max_clusters': self.get('solver.max_clusters', 100),
            'road_buffer_distance': self.get('solver.road_buffer_distance', 1000.0),
            'cw_parallel_savings': self.get('solver.cw.parallel_savings', True),
            'cw_time_window_enforce': self.get('solver.cw.time_window_enforce', False),
            'sa_initial_temp': self.get('solver.sa.initial_temp', 1000.0),
            'sa_cooling_rate': self.get('solver.sa.cooling_rate', 0.95),
            'sa_iterations': self.get('solver.sa.iterations', 100),
            'ts_tabu_size': self.get('solver.ts.tabu_size', 10),
            'ts_max_iterations': self.get('solver.ts.max_iterations', 50),
            'ts_neighborhood_size': self.get('solver.ts.neighborhood_size', 20),
            'vns_max_iterations': self.get('solver.vns.max_iterations', 30),
            'vns_max_neighborhoods': self.get('solver.vns.max_neighborhoods', 3),
            'vns_shake_intensity': self.get('solver.vns.shake_intensity', 0.3),
            'vehicle_types': self.get('solver.vehicle_types', {
                'small': {'capacity': 3000, 'fixed_cost': 800, 'count': 20},
                'medium': {'capacity': 5000, 'fixed_cost': 1000, 'count': 10},
                'large': {'capacity': 8000, 'fixed_cost': 1500, 'count': 5}
            })
        }
        
        return SolverConfig(**config_dict)

# 默认配置管理器实例
default_config = ConfigManager()