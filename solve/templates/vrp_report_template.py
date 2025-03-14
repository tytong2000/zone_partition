# templates/vrp_report_template.py

VRP_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VRP算法实验报告</title>
    <style>
        body { 
            font-family: "Microsoft YaHei", Arial, sans-serif; 
            margin: 40px;
            line-height: 1.6;
        }
        table { 
            border-collapse: collapse; 
            width: 100%;
            margin: 20px 0;
        }
        th, td { 
            border: 1px solid #ddd; 
            padding: 12px 8px;
            text-align: center;
        }
        th { 
            background-color: #f5f6fa;
            color: #2c3e50;
        }
        .metric { 
            margin: 30px 0;
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1, h2 { 
            color: #2c3e50;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        .visualization {
            text-align: center;
            margin: 20px 0;
        }
        .visualization img {
            max-width: 100%;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .highlight {
            background-color: #f8f9fa;
        }
    </style>
</head>
<body>
    <h1>VRP算法实验报告</h1>
    
    <!-- 1. 基线算法性能 -->
    <div class="metric">
        <h2>1. 基线算法性能</h2>
        <table>
            <tr>
                <th>算法</th>
                <th>平均目标值</th>
                <th>平均时间(秒)</th>
                <th>平均路线数</th>
                <th>标准差</th>
                <th>成功率(%)</th>
            </tr>
            {% for algo, metrics in baseline_metrics.items() %}
            <tr>
                <td>{{ algo }}</td>
                <td>{{ "%.2f"|format(metrics.objective_avg) }}</td>
                <td>{{ "%.2f"|format(metrics.time_avg) }}</td>
                <td>{{ "%.1f"|format(metrics.routes_avg) }}</td>
                <td>{{ "%.2f"|format(metrics.objective_std) }}</td>
                <td>{{ "%.1f"|format(metrics.success_rate * 100) }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <!-- 2. 混合算法性能 -->
    <div class="metric">
        <h2>2. 混合算法性能</h2>
        <table>
            <tr>
                <th>算法组合</th>
                <th>平均目标值</th>
                <th>平均时间(秒)</th>
                <th>平均改进率(%)</th>
                <th>标准差</th>
                <th>稳定性指标</th>
            </tr>
            {% for combo, metrics in hybrid_metrics.items() %}
            <tr>
                <td>{{ combo }}</td>
                <td>{{ "%.2f"|format(metrics.objective_avg) }}</td>
                <td>{{ "%.2f"|format(metrics.time_avg) }}</td>
                <td>{{ "%.1f"|format(metrics.improvement_avg * 100) }}</td>
                <td>{{ "%.2f"|format(metrics.objective_std) }}</td>
                <td>{{ "%.1f"|format(metrics.stability_score) }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <!-- 3. 自适应算法详细分析 -->
    <div class="metric">
        <h2>3. 自适应算法详细分析</h2>
        
        <!-- 3.1 总体性能对比 -->
        <h3>3.1 自适应算法总体性能</h3>
        <table>
            <tr>
                <th>算法配置</th>
                <th>平均目标值</th>
                <th>平均计算时间(秒)</th>
                <th>收敛性评分</th>
                <th>参数调整次数</th>
                <th>最终改进率(%)</th>
            </tr>
            {% for config, metrics in adaptive_metrics.items() %}
            <tr>
                <td>{{ config }}</td>
                <td>{{ "%.2f"|format(metrics.objective_avg) }}</td>
                <td>{{ "%.2f"|format(metrics.time_avg) }}</td>
                <td>{{ metrics.convergence_score }}</td>
                <td>{{ metrics.param_adjustments }}</td>
                <td>{{ "%.1f"|format(metrics.final_improvement * 100) }}</td>
            </tr>
            {% endfor %}
        </table>

        <!-- 3.2 参数自适应过程分析 -->
        <h3>3.2 参数自适应过程分析</h3>
        <table>
            <tr>
                <th>实验编号</th>
                <th>算法类型</th>
                <th>参数类型</th>
                <th>初始值</th>
                <th>最终值</th>
                <th>调整范围</th>
                <th>收敛特征</th>
            </tr>
            <tr class="highlight">
                <td>实验9</td>
                <td>TS-ADP</td>
                <td>禁忌表长度</td>
                <td>{{ "%.1f"|format(metrics.ts_initial_tabu_size) }}</td>
                <td>{{ "%.1f"|format(metrics.ts_final_tabu_size) }}</td>
                <td>[5, 20]</td>
                <td>{{ metrics.ts_convergence_pattern }}</td>
            </tr>
            <tr>
                <td>实验10</td>
                <td>VNS-ADP</td>
                <td>邻域结构数</td>
                <td>{{ "%.1f"|format(metrics.vns_initial_neighborhoods) }}</td>
                <td>{{ "%.1f"|format(metrics.vns_final_neighborhoods) }}</td>
                <td>[2, 5]</td>
                <td>{{ metrics.vns_convergence_pattern }}</td>
            </tr>
            <tr class="highlight">
                <td>实验11</td>
                <td>CW→TS-ADP</td>
                <td>禁忌搜索参数</td>
                <td>{{ "%.1f"|format(metrics.cwts_initial_param) }}</td>
                <td>{{ "%.1f"|format(metrics.cwts_final_param) }}</td>
                <td>[5, 20]</td>
                <td>{{ metrics.cwts_convergence_pattern }}</td>
            </tr>
            <tr>
                <td>实验12</td>
                <td>SA→VNS-ADP</td>
                <td>邻域变化强度</td>
                <td>{{ "%.1f"|format(metrics.savns_initial_intensity) }}</td>
                <td>{{ "%.1f"|format(metrics.savns_final_intensity) }}</td>
                <td>[0.1, 0.5]</td>
                <td>{{ metrics.savns_convergence_pattern }}</td>
            </tr>
        </table>

        <!-- 3.3 性能提升分析 -->
        <h3>3.3 自适应算法性能提升分析</h3>
        <table>
            <tr>
                <th>对比项</th>
                <th>基准算法</th>
                <th>自适应版本</th>
                <th>目标值提升(%)</th>
                <th>时间开销增加(%)</th>
                <th>稳定性变化</th>
                <th>综合评价</th>
            </tr>
            <tr>
                <td>TS vs TS-ADP</td>
                <td>{{ "%.2f"|format(metrics.ts_base_objective) }}</td>
                <td>{{ "%.2f"|format(metrics.ts_adp_objective) }}</td>
                <td>{{ "%.1f"|format(metrics.ts_improvement_rate * 100) }}</td>
                <td>{{ "%.1f"|format(metrics.ts_time_increase * 100) }}</td>
                <td>{{ metrics.ts_stability_change }}</td>
                <td>{{ metrics.ts_overall_evaluation }}</td>
            </tr>
            <tr>
                <td>VNS vs VNS-ADP</td>
                <td>{{ "%.2f"|format(metrics.vns_base_objective) }}</td>
                <td>{{ "%.2f"|format(metrics.vns_adp_objective) }}</td>
                <td>{{ "%.1f"|format(metrics.vns_improvement_rate * 100) }}</td>
                <td>{{ "%.1f"|format(metrics.vns_time_increase * 100) }}</td>
                <td>{{ metrics.vns_stability_change }}</td>
                <td>{{ metrics.vns_overall_evaluation }}</td>
            </tr>
            <tr>
                <td>CW→TS vs CW→TS-ADP</td>
                <td>{{ "%.2f"|format(metrics.cwts_base_objective) }}</td>
                <td>{{ "%.2f"|format(metrics.cwts_adp_objective) }}</td>
                <td>{{ "%.1f"|format(metrics.cwts_improvement_rate * 100) }}</td>
                <td>{{ "%.1f"|format(metrics.cwts_time_increase * 100) }}</td>
                <td>{{ metrics.cwts_stability_change }}</td>
                <td>{{ metrics.cwts_overall_evaluation }}</td>
            </tr>
            <tr>
                <td>SA→VNS vs SA→VNS-ADP</td>
                <td>{{ "%.2f"|format(metrics.savns_base_objective) }}</td>
                <td>{{ "%.2f"|format(metrics.savns_adp_objective) }}</td>
                <td>{{ "%.1f"|format(metrics.savns_improvement_rate * 100) }}</td>
                <td>{{ "%.1f"|format(metrics.savns_time_increase * 100) }}</td>
                <td>{{ metrics.savns_stability_change }}</td>
                <td>{{ metrics.savns_overall_evaluation }}</td>
            </tr>
        </table>
    </div>

    <div class="metric-card">
        <h2>3. 商户类型深度分析</h2>
        
        <h3>3.1 商户基础特征统计</h3>
        <table>
            <tr>
                <th rowspan="2">商户类型</th>
                <th rowspan="2">数量</th>
                <th colspan="3">订单特征</th>
                <th colspan="3">配送需求</th>
                <th colspan="2">经济指标</th>
            </tr>
            <tr>
                <th>平均体积</th>
                <th>平均重量</th>
                <th>平均件数</th>
                <th>高峰期比例</th>
                <th>平均配送频次</th>
                <th>单次配送量</th>
                <th>平均运费</th>
                <th>单位成本</th>
            </tr>
            {% for type, stats in merchant_stats.items() %}
            <tr>
                <td>{{ type }}</td>
                <td>{{ stats.count }}</td>
                <td>{{ "%.2f"|format(stats.avg_volume) }}</td>
                <td>{{ "%.2f"|format(stats.avg_weight) }}</td>
                <td>{{ "%.1f"|format(stats.avg_items) }}</td>
                <td>{{ "%.1f"|format(stats.peak_ratio * 100) }}%</td>
                <td>{{ "%.2f"|format(stats.delivery_frequency) }}</td>
                <td>{{ "%.2f"|format(stats.avg_batch_size) }}</td>
                <td>{{ "%.2f"|format(stats.avg_fee) }}</td>
                <td>{{ "%.2f"|format(stats.unit_cost) }}</td>
            </tr>
            {% endfor %}
        </table>

        <h3>3.2 商户空间分布特征</h3>
        <table>
            <tr>
                <th>商户类型</th>
                <th>平均距离(km)</th>
                <th>密度系数</th>
                <th>聚集度</th>
                <th>邻近商户数</th>
                <th>覆盖半径(km)</th>
            </tr>
            {% for type, stats in merchant_spatial_stats.items() %}
            <tr>
                <td>{{ type }}</td>
                <td>{{ "%.2f"|format(stats.avg_distance) }}</td>
                <td>{{ "%.3f"|format(stats.density_coefficient) }}</td>
                <td>{{ "%.2f"|format(stats.clustering_index) }}</td>
                <td>{{ "%.1f"|format(stats.neighbor_count) }}</td>
                <td>{{ "%.2f"|format(stats.coverage_radius) }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="metric-card">
        <h2>4. 多车型综合评估</h2>
        
        <h3>4.1 车辆使用效率分析</h3>
        <table>
            <tr>
                <th rowspan="2">车型</th>
                <th colspan="3">使用情况</th>
                <th colspan="3">成本结构</th>
                <th colspan="3">路线特征</th>
            </tr>
            <tr>
                <th>实际使用量</th>
                <th>利用率(%)</th>
                <th>周转率</th>
                <th>固定成本</th>
                <th>单位里程成本</th>
                <th>总运营成本</th>
                <th>平均路线长度</th>
                <th>平均订单数</th>
                <th>装载率(%)</th>
            </tr>
            {% for type, metrics in vehicle_metrics.items() %}
            <tr>
                <td>{{ type }}</td>
                <td>{{ metrics.used_count }}</td>
                <td>{{ "%.1f"|format(metrics.utilization * 100) }}</td>
                <td>{{ "%.2f"|format(metrics.turnover_rate) }}</td>
                <td>{{ "%.2f"|format(metrics.fixed_cost) }}</td>
                <td>{{ "%.2f"|format(metrics.unit_distance_cost) }}</td>
                <td>{{ "%.2f"|format(metrics.total_operation_cost) }}</td>
                <td>{{ "%.2f"|format(metrics.avg_route_length) }}</td>
                <td>{{ "%.1f"|format(metrics.avg_orders_per_route) }}</td>
                <td>{{ "%.1f"|format(metrics.load_rate * 100) }}</td>
            </tr>
            {% endfor %}
        </table>

        <h3>4.2 车型匹配度分析</h3>
        <table>
            <tr>
                <th>车型</th>
                <th>商超匹配度</th>
                <th>便利店匹配度</th>
                <th>购物中心匹配度</th>
                <th>容量适应度</th>
                <th>成本效益比</th>
            </tr>
            {% for type, metrics in vehicle_compatibility.items() %}
            <tr>
                <td>{{ type }}</td>
                <td>{{ "%.2f"|format(metrics.supermarket_compatibility) }}</td>
                <td>{{ "%.2f"|format(metrics.convenience_compatibility) }}</td>
                <td>{{ "%.2f"|format(metrics.mall_compatibility) }}</td>
                <td>{{ "%.2f"|format(metrics.capacity_fitness) }}</td>
                <td>{{ "%.2f"|format(metrics.cost_efficiency) }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="metric-card">
        <h2>5. 路径多样性深度分析</h2>
        
        <h3>5.1 路径结构特征</h3>
        <table>
            <tr>
                <th>算法</th>
                <th>平均候选路径数</th>
                <th>路径长度标准差</th>
                <th>最长/最短路径比</th>
                <th>平均分支因子</th>
                <th>路径重叠率(%)</th>
                <th>计算时间增加(%)</th>
            </tr>
            {% for algo, metrics in path_diversity_metrics.items() %}
            <tr>
                <td>{{ algo }}</td>
                <td>{{ "%.1f"|format(metrics.avg_candidates) }}</td>
                <td>{{ "%.2f"|format(metrics.length_std) }}</td>
                <td>{{ "%.2f"|format(metrics.max_min_ratio) }}</td>
                <td>{{ "%.2f"|format(metrics.branching_factor) }}</td>
                <td>{{ "%.1f"|format(metrics.overlap_ratio * 100) }}</td>
                <td>{{ "%.1f"|format(metrics.time_increase * 100) }}</td>
            </tr>
            {% endfor %}
        </table>

        <h3>5.2 路径质量评估</h3>
        <table>
            <tr>
                <th>算法</th>
                <th>区域均衡性</th>
                <th>负载平衡度</th>
                <th>路径平顺度</th>
                <th>收敛稳定性</th>
                <th>车型匹配度</th>
            </tr>
            {% for algo, metrics in path_quality_metrics.items() %}
            <tr>
                <td>{{ algo }}</td>
                <td>{{ "%.2f"|format(metrics.spatial_balance) }}</td>
                <td>{{ "%.2f"|format(metrics.load_balance) }}</td>
                <td>{{ "%.2f"|format(metrics.path_smoothness) }}</td>
                <td>{{ "%.2f"|format(metrics.convergence_stability) }}</td>
                <td>{{ "%.2f"|format(metrics.vehicle_compatibility) }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="metric">
        <h2>6. 综合成本分析</h2>
        <table>
            <tr>
                <th>成本类型</th>
                <th>基础算法</th>
                <th>混合算法</th>
                <th>自适应算法</th>
                <th>多车型方案</th>
            </tr>
            <tr>
                <td>固定成本占比(%)</td>
                <td>{{ "%.1f"|format(cost_metrics.base.fixed_ratio * 100) }}</td>
                <td>{{ "%.1f"|format(cost_metrics.hybrid.fixed_ratio * 100) }}</td>
                <td>{{ "%.1f"|format(cost_metrics.adaptive.fixed_ratio * 100) }}</td>
                <td>{{ "%.1f"|format(cost_metrics.multi_vehicle.fixed_ratio * 100) }}</td>
            </tr>
            <tr>
                <td>变动成本占比(%)</td>
                <td>{{ "%.1f"|format(cost_metrics.base.variable_ratio * 100) }}</td>
                <td>{{ "%.1f"|format(cost_metrics.hybrid.variable_ratio * 100) }}</td>
                <td>{{ "%.1f"|format(cost_metrics.adaptive.variable_ratio * 100) }}</td>
                <td>{{ "%.1f"|format(cost_metrics.multi_vehicle.variable_ratio * 100) }}</td>
            </tr>
            <tr>
                <td>总成本降低率(%)</td>
                <td>-</td>
                <td>{{ "%.1f"|format(cost_metrics.hybrid.reduction * 100) }}</td>
                <td>{{ "%.1f"|format(cost_metrics.adaptive.reduction * 100) }}</td>
                <td>{{ "%.1f"|format(cost_metrics.multi_vehicle.reduction * 100) }}</td>
            </tr>
        </table>
    </div>

<!-- 7. 统计分析结果 -->
    <div class="metric">
        <h2>7. 统计检验分析</h2>
        
        <!-- 7.1 参数检验 -->
        <h3>7.1 成对t检验结果</h3>
        <table>
            <tr>
                <th>算法对比</th>
                <th>效应量(d)</th>
                <th>t统计量</th>
                <th>p值</th>
                <th>显著性</th>
                <th>均值差异</th>
            </tr>
            {% for test in statistical_analysis.t_tests %}
            <tr class="{{ 'highlight' if test.significant }}">
                <td>{{ test.algorithm1 }} vs {{ test.algorithm2 }}</td>
                <td>{{ "%.3f"|format(test.effect_size) }}</td>
                <td>{{ "%.3f"|format(test.statistic) }}</td>
                <td>{{ "%.4f"|format(test.p_value) }}</td>
                <td>{% if test.significant %}✓{% else %}✗{% endif %}</td>
                <td>{{ "%.2f"|format(test.mean_diff) }}</td>
            </tr>
            {% endfor %}
        </table>
        
        <!-- 7.2 非参数检验 -->
        <h3>7.2 Wilcoxon符号秩检验</h3>
        <table>
            <tr>
                <th>算法对比</th>
                <th>效应量(r)</th>
                <th>统计量</th>
                <th>p值</th>
                <th>显著性</th>
            </tr>
            {% for test in statistical_analysis.wilcoxon_tests %}
            <tr class="{{ 'highlight' if test.significant }}">
                <td>{{ test.algorithm1 }} vs {{ test.algorithm2 }}</td>
                <td>{{ "%.3f"|format(test.effect_size) }}</td>
                <td>{{ "%.3f"|format(test.statistic) }}</td>
                <td>{{ "%.4f"|format(test.p_value) }}</td>
                <td>{% if test.significant %}✓{% else %}✗{% endif %}</td>
            </tr>
            {% endfor %}
        </table>
        
        <!-- 7.3 Friedman检验 -->
        <h3>7.3 Friedman检验与后测分析</h3>
        {% if statistical_analysis.friedman_test.valid %}
        <table>
            <tr>
                <th>统计量</th>
                <th>p值</th>
                <th>显著性</th>
                <th>临界差异值(CD)</th>
            </tr>
            <tr>
                <td>{{ "%.3f"|format(statistical_analysis.friedman_test.statistic) }}</td>
                <td>{{ "%.4f"|format(statistical_analysis.friedman_test.p_value) }}</td>
                <td>{% if statistical_analysis.friedman_test.significant %}✓{% else %}✗{% endif %}</td>
                <td>{{ "%.3f"|format(statistical_analysis.friedman_test.cd) }}</td>
            </tr>
        </table>
        
        {% if statistical_analysis.friedman_test.significant %}
        <h4>Nemenyi后测矩阵</h4>
        <div class="nemenyi-matrix">
            {{ statistical_analysis.friedman_test.posthoc.matrix|safe }}
        </div>
        {% endif %}
        {% else %}
        <p>Friedman检验无法执行: {{ statistical_analysis.friedman_test.error }}</p>
        {% endif %}
        
        <!-- 7.4 排名可视化 -->
        <h3>7.4 算法排名与Critical Difference分析</h3>
        <div class="visualization">
            <img src="plots/average_rankings.png" alt="算法平均排名">
            {% if statistical_analysis.friedman_test.significant %}
            <img src="plots/cd_diagram.png" alt="Critical Difference Diagram">
            {% endif %}
        </div>
    </div>
    <!-- 可视化部分也需要更新 -->
    <div class="visualization">
        <h3>多车型分布可视化</h3>
        <img src="plots/vehicle_distribution.png" alt="车型分布">
        
        <h3>路径多样性对比</h3>
        <img src="plots/path_diversity.png" alt="路径多样性">
        
        <h3>成本构成分析</h3>
        <img src="plots/cost_breakdown.png" alt="成本构成">
    </div>
    <!-- 4. 结果分析与讨论 -->
    <div class="metric">
        <h2>4. 结果分析与讨论</h2>
        {{ analysis_text|safe }}
    </div>

    <!-- 5. 可视化展示 -->
    <div class="metric">
        <h2>5. 可视化展示</h2>
        <div class="visualization">
            {% for plot in plots %}
            <figure>
                <img src="{{ plot }}" alt="Performance Plot">
                <figcaption>{{ plot_captions[loop.index0] }}</figcaption>
            </figure>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""