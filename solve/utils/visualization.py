# visualization.py
import os
import time
import logging
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from matplotlib import rcParams
from ..base.vrp_solution import VRPSolution
from ..base.vrp_instance import VRPInstance
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi
from matplotlib import rcParams, animation
import networkx as nx
warnings.filterwarnings('ignore')

# 设置中文字体和样式
rcParams['font.sans-serif'] = ['SimHei'] 
rcParams['axes.unicode_minus'] = False

# 设置颜色方案
COLOR_SCHEMES = {
    'standard': plt.cm.Set3(np.linspace(0, 1, 12)),  # 标准视图 - 路线
    'zones': plt.cm.Pastel1(np.linspace(0, 1, 9)),   # 路区颜色
    'density': plt.cm.YlOrRd(np.linspace(0, 1, 9)),  # 密度视图
    'heatmap': plt.cm.viridis(np.linspace(0, 1, 9))  # 热力图视图
}
class EnhancedZonePartitioner:
    def __init__(self, output_dir):
        """初始化分区器，设置输出目录"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置可视化的颜色方案
        self.color_schemes = {
            'standard': plt.cm.Set3(np.linspace(0, 1, 12)),  # 标准视图
            'density': plt.cm.YlOrRd(np.linspace(0, 1, 9)),  # 密度视图
            'heatmap': plt.cm.viridis(np.linspace(0, 1, 9))  # 热力图视图
        }

    def load_data(self, merchant_path, city_path, road_path):
        """加载所需的所有数据"""
        print("正在加载数据...")
        
        # 加载商户数据
        # 注意：这里 merchant_path, city_path, road_path 全是 文件路径
        merchant_df = pd.read_excel(merchant_path)  # 读 Excel
        self.merchant_gdf = gpd.GeoDataFrame(
            merchant_df,
            geometry=gpd.points_from_xy(merchant_df["经度"], merchant_df["纬度"]),
            crs="EPSG:4326"
        )

        self.city_boundary = gpd.read_file(city_path).to_crs("EPSG:4326")
        self.road_network = gpd.read_file(road_path).to_crs("EPSG:4326")

        print("数据加载完成")
        
        # 确保所有数据使用相同的坐标系
        self.city_boundary = self.city_boundary.to_crs("EPSG:4326")
        self.road_network = self.road_network.to_crs("EPSG:4326")
        
        print(f"数据加载完成！共有{len(self.merchant_gdf)}个商户点")
        return self.merchant_gdf

    def split_merchants_by_zone(self, merchant_gdf: gpd.GeoDataFrame, zones_gdf: gpd.GeoDataFrame) -> Dict[str, pd.DataFrame]:
        """将商户点分配到对应的路区中,返回DataFrame而不是索引列表"""
        logging.info("开始将商户分配到路区...")
        
        # 初始化结果字典
        zone_map = {}
        
        # 使用空间索引加速查询
        zones_sindex = zones_gdf.sindex
        
        # 遍历每个商户点
        for idx, merchant in tqdm(merchant_gdf.iterrows(), desc="分配商户到路区", total=len(merchant_gdf)):
            # 使用空间索引找到可能包含该点的路区
            possible_zones = list(zones_sindex.intersection(merchant.geometry.bounds))
            
            if not possible_zones:
                logging.warning(f"商户 {idx} 不在任何路区内")
                continue
                
            # 检查点是否真的在路区内
            for zone_idx in possible_zones:
                zone = zones_gdf.iloc[zone_idx]
                if merchant.geometry.within(zone.geometry):
                    zone_id = zone['路区编号']
                    if zone_id not in zone_map:
                        zone_map[zone_id] = []
                    zone_map[zone_id].append(merchant)
                    break
        
        # 转换列表为DataFrame
        for zone_id in zone_map:
            zone_map[zone_id] = pd.DataFrame(zone_map[zone_id])
        
        return zone_map

    def generate_zones(self, min_clusters=50, max_clusters=70, road_buffer_distance=None):
        """生成优化的路区划分"""
        print("开始生成路区...")
        
        # 预处理边界和路网
        if road_buffer_distance is not None:
            city_boundary_buffered = self.city_boundary.buffer(road_buffer_distance).unary_union
            road_network_buffered = self.road_network.buffer(road_buffer_distance).unary_union
        else:
            city_boundary_buffered = self.city_boundary.unary_union
            road_network_buffered = self.road_network.unary_union
        
        # 优化聚类数量计算
        n_merchants = len(self.merchant_gdf)
        optimal_clusters = min(
            max_clusters,
            max(min_clusters, n_merchants // 50)
        )
        
        print(f"目标路区数量: {optimal_clusters}")
        
        # 使用numpy数组优化坐标处理
        coords = self.merchant_gdf[["经度", "纬度"]].values
        
        # KMeans聚类
        kmeans = KMeans(
            n_clusters=optimal_clusters,
            random_state=42,
            n_init=10
        )
        
        kmeans.fit(coords)
        centers = kmeans.cluster_centers_
        
        # 生成Voronoi图
        vor = Voronoi(centers)
        
        # 预先创建所有多边形列表
        polygons = []
        zone_ids = []
        
        # 批量处理Voronoi区域
        from concurrent.futures import ThreadPoolExecutor
        from functools import partial
        
        def process_region(region, vertices, city_boundary, road_network):
            if not region or -1 in region:
                return None, None
            
            try:
                # 创建多边形
                polygon = Polygon([vertices[v] for v in region])
                
                # 一次性进行所有空间操作
                polygon = polygon.intersection(city_boundary)
                polygon = polygon.intersection(road_network)
                
                # 处理特殊情况
                if isinstance(polygon, MultiLineString):
                    polygon = polygon.buffer(0.001)
                
                if polygon.is_valid and not polygon.is_empty:
                    return polygon, True
                
            except Exception:
                pass
            
            return None, None
        
        # 并行处理区域
        process_func = partial(
            process_region,
            vertices=vor.vertices,
            city_boundary=city_boundary_buffered,
            road_network=road_network_buffered
        )
        
        print("正在处理Voronoi区域...")
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(
                executor.map(process_func, vor.regions),
                total=len(vor.regions),
                desc="处理Voronoi区域"
            ))
        
        # 收集结果
        for i, (polygon, valid) in enumerate(results):
            if valid:
                polygons.append(polygon)
                zone_ids.append(f"Z{str(len(zone_ids)+1).zfill(3)}")
        
        # 创建路区GeoDataFrame
        self.zones_gdf = gpd.GeoDataFrame({
            "路区编号": zone_ids,
            "geometry": polygons
        }, crs="EPSG:4326")
        
        print(f"成功生成{len(self.zones_gdf)}个路区！")
        return self.zones_gdf

    def calculate_metrics(self):
        """计算各个路区的统计指标"""
        print("计算路区统计指标...")
        metrics = []
        
        for idx, zone in tqdm(self.zones_gdf.iterrows(), desc="统计路区指标"):
            # 找出区域内的商户点
            points_in_zone = self.merchant_gdf[self.merchant_gdf.geometry.within(zone.geometry)]
            
            # 计算基础指标
            area = zone.geometry.area * (111000 ** 2)  # 转换为平方米
            perimeter = zone.geometry.length * 111000  # 转换为米
            
            metrics.append({
                "路区编号": zone["路区编号"],
                "面积（平方米）": area,
                "周长（米）": perimeter,
                "商户数量": len(points_in_zone),
                "商户密度": len(points_in_zone) / area if area > 0 else 0
            })
        
        return pd.DataFrame(metrics)

import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams, animation
from matplotlib.colors import Normalize
from typing import Optional, Dict, List
from shapely.geometry import MultiPolygon

class VisualizationTools:
    def __init__(self, output_dir: str):
        """
        初始化可视化工具类
        
        Args:
            output_dir (str): 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置颜色方案
        self.color_schemes = {
            'standard': plt.cm.Set3(np.linspace(0, 1, 12)),  # 标准视图
            'zones': plt.cm.Pastel1(np.linspace(0, 1, 9)),   # 路区颜色
            'density': plt.cm.YlOrRd(np.linspace(0, 1, 9)),  # 密度视图
            'heatmap': plt.cm.viridis(np.linspace(0, 1, 9))  # 热力图视图
        }
        
        # 设置绘图样式
        import seaborn as sns
        sns.set_style("whitegrid")
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

    def visualize_complete_solution(self, solution: VRPSolution, 
                                merchants_df: pd.DataFrame,
                                road_network: nx.Graph,
                                output_path: str):
        """完整解决方案可视化"""
        try:
            # 0. 输出基本信息
            logging.info(f"开始生成可视化, 商户数量: {len(merchants_df)}")
            if solution and hasattr(solution, 'routes'):
                logging.info(f"路线数量: {len(solution.routes)}")
            if road_network:
                logging.info(f"路网边数: {len(road_network.edges())}")

            # 1. 创建图形
            fig, ax = plt.subplots(figsize=(15, 10))

            # 2. 绘制路网（使用实际坐标）
            if road_network and len(road_network.edges()) > 0:
                logging.info("绘制路网...")
                for edge in road_network.edges():
                    node1, node2 = edge
                    if isinstance(node1, (list, tuple)) and isinstance(node2, (list, tuple)):
                        x1, y1 = node1
                        x2, y2 = node2
                        plt.plot([x1, x2], [y1, y2], 
                                color='lightgray', linewidth=0.5, alpha=0.3,
                                zorder=1)  # 确保路网在底层

            # 3. 绘制商户点
            logging.info("绘制商户点...")
            ax.scatter(merchants_df['经度'], merchants_df['纬度'],
                    c='blue', s=30, alpha=0.6, label='商户',
                    zorder=3)  # 商户点在路线之上

            # 4. 绘制路线
            if solution and hasattr(solution, 'routes'):
                logging.info("绘制配送路线...")
                colors = plt.cm.rainbow(np.linspace(0, 1, len(solution.routes)))
                for route_idx, route in enumerate(solution.routes):
                    if not route:  # 跳过空路线
                        continue
                        
                    # 获取路线上的所有点坐标
                    coords = []
                    try:
                        for i in route:
                            if i < len(merchants_df):
                                merchant = merchants_df.iloc[i]
                                coords.append((merchant['经度'], merchant['纬度']))
                    except Exception as e:
                        logging.error(f"处理路线 {route_idx} 时出错: {str(e)}")
                        continue

                    if len(coords) < 2:  # 跳过点数不足的路线
                        continue

                    # 绘制路线连线
                    coords = np.array(coords)
                    ax.plot(coords[:, 0], coords[:, 1],
                        c=colors[route_idx], linewidth=2, alpha=0.7,
                        label=f'路线 {route_idx+1}',
                        zorder=2)  # 路线在商户点之下
                    
                    # 添加箭头指示方向
                    for i in range(len(coords)-1):
                        mid_x = (coords[i][0] + coords[i+1][0]) / 2
                        mid_y = (coords[i][1] + coords[i+1][1]) / 2
                        dx = coords[i+1][0] - coords[i][0]
                        dy = coords[i+1][1] - coords[i][1]
                        plt.arrow(mid_x, mid_y, dx/10, dy/10,
                                head_width=0.0005, head_length=0.001,
                                fc=colors[route_idx], ec=colors[route_idx],
                                alpha=0.7, zorder=2)

            # 5. 设置图表样式
            ax.set_title('配送路线规划结果', fontsize=14, pad=20)
            ax.set_xlabel('经度', fontsize=12)
            ax.set_ylabel('纬度', fontsize=12)
            
            # 添加图例，并调整位置避免遮挡
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                    fontsize=10, frameon=True)
            
            # 添加网格背景
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # 调整布局，确保图例不被截断
            plt.tight_layout()

            # 6. 保存图表
            logging.info(f"保存可视化结果: {output_path}")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logging.info("可视化生成完成")

        except Exception as e:
            logging.error(f"生成可视化失败: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
    
    def plot_zone_distribution(self, zones_gdf: gpd.GeoDataFrame, merchant_gdf: gpd.GeoDataFrame, 
                             city_boundary: gpd.GeoDataFrame, mode: str = 'standard', 
                             save_path: Optional[str] = None) -> None:
        """绘制路区分布图"""
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # 绘制基础图层
        city_boundary.plot(ax=ax, color='white', edgecolor='black', linewidth=1.5)
        
        # 根据模式选择颜色方案
        colors = self.color_schemes[mode]
        
        # 绘制路区
        for idx, zone in zones_gdf.iterrows():
            color = colors[idx % len(colors)]
            if isinstance(zone.geometry, MultiPolygon):
                for poly in zone.geometry.geoms:
                    ax.fill(*poly.exterior.xy, color=color, alpha=0.6)
                    ax.plot(*poly.exterior.xy, color='black', linewidth=0.5)
            else:
                ax.fill(*zone.geometry.exterior.xy, color=color, alpha=0.6)
                ax.plot(*zone.geometry.exterior.xy, color='black', linewidth=0.5)
        
        # 绘制商户点
        merchant_gdf.plot(ax=ax, color='red', markersize=1, alpha=0.5)
        
        # 设置标题和样式
        plt.title(f"武汉市路区划分图 - {mode}模式", fontsize=16, pad=20)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_merchant_density(self, zones_gdf: gpd.GeoDataFrame, merchant_gdf: gpd.GeoDataFrame,
                            save_path: Optional[str] = None) -> None:
        """绘制商户密度热力图"""
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # 计算每个路区的商户密度
        densities = []
        for _, zone in zones_gdf.iterrows():
            points_in_zone = merchant_gdf[merchant_gdf.geometry.within(zone.geometry)]
            density = len(points_in_zone) / zone.geometry.area
            densities.append(density)
        
        # 归一化密度值
        norm = plt.Normalize(min(densities), max(densities))
        
        # 绘制密度图
        zones_gdf.plot(ax=ax, color=[plt.cm.YlOrRd(norm(d)) for d in densities], alpha=0.7)
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=norm)
        plt.colorbar(sm, ax=ax, label='商户密度')
        
        plt.title("商户密度分布图", fontsize=16, pad=20)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_zone_metrics(self, metrics_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """绘制路区统计指标图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        
        # 面积分布
        sns.boxplot(y=metrics_df['面积（平方米）'], ax=ax1)
        ax1.set_title('路区面积分布')
        
        # 周长分布
        sns.boxplot(y=metrics_df['周长（米）'], ax=ax2)
        ax2.set_title('路区周长分布')
        
        # 商户数量分布
        sns.boxplot(y=metrics_df['商户数量'], ax=ax3)
        ax3.set_title('路区商户数量分布')
        
        # 商户密度分布
        sns.boxplot(y=metrics_df['商户密度'], ax=ax4)
        ax4.set_title('路区商户密度分布')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def create_animation(self, zones_gdf: gpd.GeoDataFrame, merchant_gdf: gpd.GeoDataFrame,
                        city_boundary: gpd.GeoDataFrame, save_path: str) -> None:
        """创建路区形成过程的动画"""
        fig, ax = plt.subplots(figsize=(15, 15))
        
        def animate(frame):
            ax.clear()
            
            # 绘制基础图层
            city_boundary.plot(ax=ax, color='white', edgecolor='black', linewidth=1.5)
            merchant_gdf.plot(ax=ax, color='red', markersize=1, alpha=0.5)
            
            # 逐步显示路区
            for idx in range(frame + 1):
                if idx < len(zones_gdf):
                    zone = zones_gdf.iloc[idx]
                    color = self.color_schemes['standard'][idx % len(self.color_schemes['standard'])]
                    if isinstance(zone.geometry, MultiPolygon):
                        for poly in zone.geometry.geoms:
                            ax.fill(*poly.exterior.xy, color=color, alpha=0.6)
                            ax.plot(*poly.exterior.xy, color='black', linewidth=0.5)
                    else:
                        ax.fill(*zone.geometry.exterior.xy, color=color, alpha=0.6)
                        ax.plot(*zone.geometry.exterior.xy, color='black', linewidth=0.5)
            
            plt.title(f"路区形成过程 - 第{frame+1}帧", fontsize=16, pad=20)
            plt.axis('off')
        
        anim = animation.FuncAnimation(fig, animate, frames=len(zones_gdf), 
                                     interval=500, repeat=False)
        anim.save(save_path, writer='pillow')
        plt.close()

    def export_report(self, zones_gdf: gpd.GeoDataFrame, metrics_df: pd.DataFrame,
                     output_path: str) -> None:
        """导出路区分析报告"""
        with pd.ExcelWriter(output_path) as writer:
            # 导出基本信息
            metrics_df.to_excel(writer, sheet_name='路区统计指标', index=False)
            
            # 导出详细信息
            summary = pd.DataFrame({
                '指标': ['总路区数', '平均面积', '平均周长', '平均商户数', '平均商户密度'],
                '值': [
                    len(zones_gdf),
                    metrics_df['面积（平方米）'].mean(),
                    metrics_df['周长（米）'].mean(),
                    metrics_df['商户数量'].mean(),
                    metrics_df['商户密度'].mean()
                ]
            })
            summary.to_excel(writer, sheet_name='总体统计', index=False)