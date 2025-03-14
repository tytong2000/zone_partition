import time
import datetime
from typing import Dict, Optional
import logging
import os

class SmartETATracker:
    def __init__(self):
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        self.start_time = time.time()
        self.current_count = 0
        self.total_count = 1  # Set default to 1 to avoid division by zero
        self.current_phase = ""
        self.current_zone = ""
        self.update_interval = 50
        self.update_count = 0
        self.total_zones = 0
        self.total_experiments = 0
        self.completed_experiments = 0
        # 扩展可用的处理阶段，包含baseline_experiments
        self.phases = [
            "预处理", 
            "特征提取", 
            "模型训练", 
            "评估",
            "baseline_experiments",
            "数据准备",
            "模型优化"
        ]
        self.phase_progress = {}
        self.current_algorithm = ""
        self.last_update_time = time.time()

    def _progress_bar(self, percentage: float) -> str:
        """生成进度条，接收百分比值(0-100)"""
        width = 30  # 进度条宽度
        filled = int(width * percentage / 100)
        bar = '█' * filled + '░' * (width - filled)
        return bar

    def _update_display(self):
        """Update display with proper error handling"""
        try:
            if self.total_count <= 0:
                return
                
            total_progress = (self.current_count / self.total_count) * 100
            elapsed = time.time() - self.start_time
            
            if self.current_count > 0:
                eta = (elapsed / self.current_count) * (self.total_count - self.current_count)
                eta_str = self._format_time(eta)
                finish_time = time.strftime("%H:%M:%S", 
                    time.localtime(time.time() + eta))
            else:
                eta_str = "计算中..."
                finish_time = "计算中..."
                
            print(f"\r- 当前阶段: {self.current_phase}")
            print(f"- 总体进度: [{self._progress_bar(total_progress)}] {total_progress:.1f}%")
            print(f"- 路区进度: {self.current_zone}")
            print(f"- 已用时间: {self._format_time(elapsed)}")
            print(f"- 预计剩余: {eta_str}")
            print(f"- 预计完成时间: {finish_time}", end="\r")
            
        except Exception as e:
            self.logger.warning(f"更新显示时出错: {str(e)}")
            
    def set_total_count(self, total: int):
        """Set total count with validation"""
        if total <= 0:
            self.logger.warning("总数不能小于等于0，设置为默认值1")
            self.total_count = 1
        else:
            self.total_count = total
    
    def set_total_zones(self, total: int):
        """设置总路区数"""
        self.total_zones = total
        self.total_experiments = total * (4 + 4 + 4 + 3)
        self.logger.info(f"总路区数: {total}, 预计总实验数: {self.total_experiments}")
    
    def start_phase(self, phase_name: str, algorithm: str = ""):
        """开始新的处理阶段"""
        if phase_name not in self.phases:
            self.logger.warning(f"未知阶段: {phase_name}")
            return
            
        if self.current_phase != phase_name or algorithm:
            self.current_phase = phase_name
            self.current_algorithm = algorithm
            self.phase_progress[phase_name] = 0
            self.last_update_time = time.time()
            
            # 重置路区计数
            self.current_zone = 0
            
            # 只在阶段改变时显示分隔线，算法改变时不显示
            if not algorithm:
                self.logger.info(f"\n=== 开始 {phase_name} 阶段 ===")
            
            # 立即显示初始进度
            self._log_progress(time.time(), True)
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.0f}分钟"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}小时"
    
    def _get_progress_bar(self, progress: float, width: int = 30) -> str:
        """生成进度条"""
        filled = int(width * progress)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}] {progress:.1%}"
    
    def update(self, n=1):
        """优化的进度更新"""
        self.update_count += n
        self.current_count += n
        self.completed_experiments += n
        if self.update_count >= self.update_interval:
            self._update_display()  # 实际更新显示
            self.update_count = 0

    def _log_progress(self, current_time: float, force_display: bool = False):
        """记录当前进度"""
        if self.total_experiments == 0:
            return
            
        elapsed_time = current_time - self.start_time
        progress = self.completed_experiments / self.total_experiments
        
        if progress > 0 or force_display:
            estimated_total = elapsed_time / max(progress, 0.01)
            remaining_time = max(0, estimated_total - elapsed_time)
            eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining_time)
            
            # 计算处理速度
            experiments_per_second = self.completed_experiments / max(1, elapsed_time)
            
            # 生成进度条
            progress_bar = self._get_progress_bar(progress)
            
            # 显示算法名称（如果有）
            algo_info = f" - {self.current_algorithm}" if self.current_algorithm else ""
            
            self.logger.info(
                f"\n进度更新:"
                f"\n- 当前阶段: {self.current_phase}{algo_info}"
                f"\n- 总体进度: {progress_bar}"
                f"\n- 路区进度: {self.current_zone}/{self.total_zones}"
                f"\n- 实验进度: {self.completed_experiments}/{self.total_experiments}"
                f"\n- 已用时间: {self._format_time(elapsed_time)}"
                f"\n- 预计剩余: {self._format_time(remaining_time)}"
                f"\n- 预计完成时间: {eta.strftime('%H:%M:%S')}"
                f"\n- 处理速度: {experiments_per_second:.2f} 实验/秒"
            )
    
    def update_zone(self, zone_id: str, algorithm: str = ""):
        """更新当前处理的路区"""
        self.current_zone += 1
        if self.current_zone > self.total_zones:
            self.current_zone = 1  # 重置为1，表示新一轮开始
            
        if algorithm:
            self.current_algorithm = algorithm
            
        if self.current_zone % 5 == 0:  # 每5个路区记录一次日志
            self.logger.info(
                f"正在处理路区: {zone_id} ({self.current_zone}/{self.total_zones})"
                f"{' - ' + algorithm if algorithm else ''}"
            )
    
    def complete(self):
        """完成所有处理"""
        end_time = time.time()
        total_time = end_time - self.start_time
        
        self.logger.info(
            f"\n=== 处理完成 ==="
            f"\n总用时: {self._format_time(total_time)}"
            f"\n总实验数: {self.completed_experiments}"
            f"\n处理路区: {self.total_zones}"
            f"\n平均速度: {self.completed_experiments/total_time:.2f} 实验/秒"
        )