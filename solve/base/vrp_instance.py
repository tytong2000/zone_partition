import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd 
from typing import List, Dict, Tuple, Optional
import logging
import networkx as nx
import math

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

class VRPInstance:
    def __init__(
        self,
        orders_df: pd.DataFrame,
        road_graph: Optional[nx.Graph],
        num_warehouses: int = 1,
        vehicle_capacity: float = 2000.0,
        max_route_time: float = 12.0,
        max_search_distance: float = 1e5,
        potential_warehouses: Optional[List[Tuple[float,float]]] = None,
        selected_warehouses: Optional[List[int]] = None,
        parallel_kdtree: bool = True,
        **kwargs 
    ):
        # 基本验证
        if "经度" not in orders_df.columns or "纬度" not in orders_df.columns:
            raise ValueError("orders_df必须包含经度和纬度列")
            
        self.orders_df = orders_df.copy()
        self.num_orders = len(self.orders_df)
        self.logger = logging.getLogger("VRPInstance")

        # 添加默认重量
        if "托运单重量" not in orders_df.columns:
            self.orders_df["托运单重量"] = 0.0
            
        # 基本属性
        self.road_graph = road_graph
        self.road_network_graph = nx.Graph() if road_graph is None else road_graph
        self.road_network_nodes = {}
        self.num_warehouses = num_warehouses
        self.vehicle_capacity = vehicle_capacity
        self.max_route_time = max_route_time
        self.max_search_distance = max_search_distance
        
        # 订单数据
        self.order_locations = self.orders_df[["经度","纬度"]].values
        self.order_demands = self.orders_df["托运单重量"].values

        # 仓库相关
        self.potential_warehouses = potential_warehouses if potential_warehouses else self._generate_potential_warehouses()
        self.selected_warehouses = [0] if selected_warehouses is None else selected_warehouses
        # 添加商户类型属性
        self.merchant_types = kwargs.get('merchant_types', {})  # Dict[int, str] - 订单ID到商户类型的映射
        
        # 修改为字典形式处理车辆类型及其相关属性
        self.vehicle_types = kwargs.get('vehicle_types', {
            "small": {
                "capacity": 1000,
                "compatible_merchants": ["small"],
                "cost_per_km": 1.0
            },
            "medium": {
                "capacity": 3000,
                "compatible_merchants": ["small", "medium"],
                "cost_per_km": 1.5
            },
            "large": {
                "capacity": 5000,
                "compatible_merchants": ["small", "medium", "large"],
                "cost_per_km": 2.0
            }
        })
        # 添加商户类型映射
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
        if selected_warehouses is None:
            self.selected_warehouses = []
        else:
            self.selected_warehouses = selected_warehouses

        # 若 selected_warehouses 里第一个元素是 (lon, lat) 元组，则直接用它们作为仓库位置
        if len(self.selected_warehouses) > 0 and isinstance(self.selected_warehouses[0], tuple):
            self.depot_locations = self.selected_warehouses
        # 否则，假设它是整数下标
        elif len(self.selected_warehouses) > 0:
            self.depot_locations = [self.potential_warehouses[i] for i in self.selected_warehouses]
        else:
            self.depot_locations = []

        # 路网和分配
        self.assignments = {}
        self._precompute_assignments()
        
        # 距离缓存
        self.distance_cache = {}
        self.vehicle_pool = self._initialize_vehicle_pool()
        
        # 路径缓存
        self.path_cache = {}
        self.max_cache_size = 1000
        self.max_path_candidates = 3

    def _initialize_vehicle_pool(self) -> List[Dict]:
        pool = []
        for v_type, specs in self.vehicle_types.items():  # 现在vehicle_types是一个字典
            pool.append({
                'id': f"{v_type}_1",  # 假设所有车辆ID是v_type_1, 可以更改为多辆车
                'type': v_type,
                'capacity': specs['capacity'],
                'cost_per_km': specs['cost_per_km'],
                'used': False
            })
        return pool

    # 增加路径计算方法
    def get_road_distance(self, locA, locB) -> float:
        """获取两个位置之间的路网最短路径距离"""
        if isinstance(locA, (list, tuple)) and isinstance(locB, (list, tuple)):
            if locA == locB:
                return 0.0
        elif isinstance(locA, np.ndarray) and isinstance(locB, np.ndarray):
            if np.array_equal(locA, locB):
                return 0.0

        key_ = (tuple(locA), tuple(locB))
        if key_ in self.distance_cache:
            return self.distance_cache[key_]

        # 添加空图检查
        if self.road_network_graph is None or len(self.road_network_graph.nodes) == 0:
            dist = self._euclidean_distance(locA, locB)
            self.distance_cache[key_] = dist
            return dist

        try:
            node_a = self._get_nearest_node(locA)
            node_b = self._get_nearest_node(locB)
            
            if node_a is None or node_b is None:
                dist = self._euclidean_distance(locA, locB)
            elif node_a == node_b:
                dist = self._euclidean_distance(locA, locB)
            else:
                try:
                    dist_nx = nx.shortest_path_length(
                        self.road_network_graph,
                        node_a,
                        node_b,
                        weight='length'
                    )
                    dist = dist_nx
                    dist += self._euclidean_distance(locA, self.road_network_nodes[node_a])
                    dist += self._euclidean_distance(locB, self.road_network_nodes[node_b])
                except nx.NetworkXNoPath:
                    dist = self._euclidean_distance(locA, locB) * 1.5
        except Exception as e:
            self.logger.debug(f"路网距离计算失败: {str(e)}, 使用欧式距离")
            dist = self._euclidean_distance(locA, locB)

        self.distance_cache[key_] = dist
        return dist
    def init_road_network(self, road_gdf):
        """
        初始化路网图结构，基于road_gdf（如道路线矢量）构造NetworkX图,
        并记录每个节点(坐标)->其位置。
        """
        if road_gdf is None:
            return
        try:
            self.road_network_graph = nx.Graph()
            for idx, row in road_gdf.iterrows():
                if row.geometry.geom_type in ['LineString', 'MultiLineString']:
                    coords = (list(row.geometry.coords)
                              if row.geometry.geom_type == 'LineString'
                              else [c for line in row.geometry.geoms for c in line.coords])
                    for i in range(len(coords)-1):
                        start = coords[i]
                        end = coords[i+1]
                        dist = self._euclidean_distance(start, end)
                        self.road_network_graph.add_edge(start, end, length=dist)
                        self.road_network_nodes[start] = start
                        self.road_network_nodes[end] = end

            self.logger.info(f"路网图构建完成: "
                             f"{len(self.road_network_graph.nodes)} 个节点, "
                             f"{len(self.road_network_graph.edges)} 条边")
        except Exception as e:
            self.logger.error(f"构建路网图失败: {str(e)}")
            self.road_network_graph = nx.Graph()  # 重置为空图

    def _euclidean_distance(self, point1, point2) -> float:
        """
        计算两点间欧氏距离
        """
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]
        return math.sqrt(dx*dx + dy*dy)


    def _get_nearest_node(self, location):
        """
        在 self.road_network_nodes 里找到离 location 最近的节点(坐标键).
        只做简易线性搜索，如需性能可结合 KD-Tree.
        """
        if not self.road_network_nodes:
            return None
        best_node = None
        best_dist = float('inf')
        for nd_coord in self.road_network_nodes.keys():
            d_ = self._euclidean_distance(location, nd_coord)
            if d_ < best_dist:
                best_dist = d_
                best_node = nd_coord
        return best_node


    def _generate_potential_warehouses(self) -> List[Tuple[float,float]]:
        """若未指定潜在仓库位置，则在订单范围内随机生成"""
        min_lon = float(self.orders_df['经度'].min())
        max_lon = float(self.orders_df['经度'].max())
        min_lat = float(self.orders_df['纬度'].min()) 
        max_lat = float(self.orders_df['纬度'].max())
        
        # 生成50个候选点
        candidates = []
        for _ in range(50):
            lon = float(np.random.uniform(min_lon, max_lon))
            lat = float(np.random.uniform(min_lat, max_lat))
            candidates.append((lon, lat))
        
        return candidates


    def _select_warehouses(self, num_warehouses: int, fixed_cost: float, transportation_cost_per_unit: float)-> List[int]:
        """
        若 GUROBI_AVAILABLE，可在此做真正的选址，否则 mock => 选前 num_warehouses 个。
        """
        if not GUROBI_AVAILABLE:
            logging.warning("Gurobi不可用 => mock select first num_warehouses.")
            return list(range(num_warehouses))
        # 否则在这写选址逻辑
        return list(range(min(num_warehouses, 50)))


    def _find_nearest_node_index(self, loc: Tuple[float,float])-> Optional[int]:
        """
        如需使用 KD-Tree 的方式定位 node_coords 的下标，可在这里实现snap。
        """
        if self.kd_tree is None:
            return None
        dist, idx = self.kd_tree.query([loc], k=1)
        if dist[0][0] > self.max_search_distance:
            return None
        return int(idx[0][0])


    def _precompute_nearest_nodes(self):
        """
        针对订单和仓库，预先计算其最近节点(若要用KD-Tree + node_coords)。
        这里只是个空壳，你可以结合 road_network_nodes 进一步完善。
        """
        if not self.road_graph or not self.node_coords or not self.kd_tree:
            return
        # 示例：对每个订单 loc => kd_tree.query => nearest_nodes[i] = nodeIdx
        for i in range(self.num_orders):
            loc= (self.order_locations[i][0], self.order_locations[i][1])
            nd_idx = self._find_nearest_node_index(loc)
            if nd_idx is not None:
                self.nearest_nodes[i] = nd_idx
            else:
                logging.warning(f"订单{i}无法匹配节点(超限?)")

        # 仓库
        for i in range(self.num_warehouses):
            loc= self.depot_locations[i]
            nd_idx= self._find_nearest_node_index(loc)
            if nd_idx is not None:
                self.nearest_nodes[f"depot_{i}"]= nd_idx
            else:
                logging.warning(f"仓库{i}无法匹配节点")


    def _precompute_assignments(self):
        """将订单分配给最近的仓库"""
        for i in range(self.num_orders):
            best_wh = 0  # 默认分配给第一个仓库
            best_dist = float('inf')
            
            loc_i = self.get_order_location(i)
            for wh_idx, wh_loc in enumerate(self.depot_locations):
                dist = self._euclidean_distance(loc_i, wh_loc)
                if dist < best_dist:
                    best_dist = dist
                    best_wh = wh_idx
                    
            self.assignments[i] = best_wh

    def get_depot_location(self, depot_idx: int) -> Tuple[float,float]:
        """获取指定仓库的位置"""
        return self.depot_locations[depot_idx]


    def get_order_location(self, idx: int)-> Tuple[float,float]:
        return (self.order_locations[idx][0], self.order_locations[idx][1])


# 在 VRPInstance 类中修改：
    def get_order_demand(self, idx: int) -> float:
        """使用 pd.to_numeric 确保返回数值类型的需求量"""
        try:
            demand = self.order_demands[idx]
            return pd.to_numeric(demand, errors='coerce')
        except:
            return 0.0

# 在 VRPInstance 类中添加/修改:
    def get_merchant_type(self, order_id: int) -> str:
        """获取订单对应的商户类型"""
        try:
            # 尝试从订单数据中获取商户类型
            merchant_type = self.orders_df.iloc[order_id].get('商户类型', '便利店')
            merchant_map = {
                '便利店': 'convenience',
                '超市': 'supermarket',
                '购物中心': 'mall'
            }
            return merchant_map.get(merchant_type, 'convenience')
        except Exception as e:
            self.logger.warning(f"获取商户类型失败: {str(e)}, 使用默认类型convenience")
            return 'convenience'

    def get_vehicle_type_for_order(self, order_id: int) -> str:
        """
        根据订单的商户类型和需求量确定合适的车型
        返回值为: 'small', 'medium', 或 'large'
        """
        merchant_type = self.get_merchant_type(order_id)
        demand_weight = self.get_order_demand(order_id)
        # 假设体积数据在df中的列名为'volume'或'体积'
        demand_volume = float(self.orders_df.iloc[order_id].get('volume', 0) or 
                            self.orders_df.iloc[order_id].get('体积', 0))
        
        mapping = self.merchant_type_mapping[merchant_type]
        
        # 如果需求超过阈值,使用secondary车型,否则使用primary车型
        if (demand_weight > mapping['weight_threshold'] or 
            demand_volume > mapping['volume_threshold']):
            return mapping['secondary']
        return mapping['primary']

    def get_order_type(self, order_id: int) -> str:
        """
        为了兼容性保留此方法,但实际返回商户类型
        """
        return self.get_merchant_type(order_id)
    def check_route_feasibility(self, route: List[int]) -> bool:
        """
        最后检查路线可行性：
         - 载重 <= vehicle_capacity
         - 第一个订单对应的仓库在 self.assignments 里必须 >=0
         - 如果有时间窗列(ready_time, due_time, service_time)，做简单校验
        """
        if not route:
            return True

        # 检查容量
        total_demand = sum(self.get_order_demand(o_) for o_ in route)
        if total_demand > self.vehicle_capacity:
            logging.debug(f"路线不可行: 总需求({total_demand})超过车辆容量({self.vehicle_capacity})")
            return False

        # 检查仓库分配
        wh_idx = self.assignments.get(route[0], -1)
        if wh_idx < 0:
            logging.debug(f"路线不可行: 订单{route[0]}未分配仓库")
            return False

        # 检查时间窗
        if 'ready_time' in self.orders_df.columns:
            current_time = 0.0
            prev_loc = self.get_depot_location(wh_idx)
            for o_ in route:
                loc = self.get_order_location(o_)
                travel_time = self.get_road_distance(prev_loc, loc) / 30000.0  # 假设速度 30km/h
                current_time += travel_time

                ready_time = self.orders_df.iloc[o_].get("ready_time", 0.0)
                due_time = self.orders_df.iloc[o_].get("due_time", float('inf'))
                service_time = self.orders_df.iloc[o_].get("service_time", 0.0)

                if current_time > due_time:
                    logging.debug(f"路线不可行: 订单{o_}到达时间({current_time:.2f})超过最晚({due_time:.2f})")
                    return False
                if current_time < ready_time:
                    current_time = ready_time

                current_time += service_time
                prev_loc = loc

        return True   
    # 在 VRPInstance 类中添加：
    def get_order_weight(self, order_id: int) -> float:
        """
        获取订单重量（与get_order_demand相同）
        """
        return self.get_order_demand(order_id)
# 在 VRPInstance 类中添加：
    def get_order_volume(self, order_id: int) -> float:
        """获取订单体积"""
        volume = self.orders_df.iloc[order_id].get('体积', 0.0)
        if pd.isna(volume):
            return 0.0
        return float(volume)