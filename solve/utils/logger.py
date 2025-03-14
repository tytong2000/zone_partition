# solve/utils/logger.py

import os
import logging
import time
from typing import Optional
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

class CustomFormatter(logging.Formatter):
    """自定义日志格式器"""
    
    FORMATS = {
        logging.DEBUG: "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
        logging.INFO: "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
        logging.WARNING: "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
        logging.ERROR: "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s",
        logging.CRITICAL: "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(
    name: str = "VRP",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    设置并配置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志文件存储目录，如果为None则仅输出到控制台
        level: 日志级别
        console_output: 是否同时输出到控制台
        max_bytes: 单个日志文件的最大大小（字节）
        backup_count: 保留的日志文件数量
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 创建自定义格式器
    formatter = CustomFormatter()
    
    # 如果指定了日志目录，设置文件处理器
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 按大小轮转的文件处理器
        size_handler = RotatingFileHandler(
            filename=os.path.join(log_dir, f"{name}_{timestamp}.log"),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        size_handler.setFormatter(formatter)
        logger.addHandler(size_handler)
        
        # 按时间轮转的文件处理器（每天）
        time_handler = TimedRotatingFileHandler(
            filename=os.path.join(log_dir, f"{name}_daily_{timestamp}.log"),
            when='midnight',
            interval=1,
            backupCount=backup_count,
            encoding='utf-8'
        )
        time_handler.setFormatter(formatter)
        logger.addHandler(time_handler)
        
        # 错误日志单独存储
        error_handler = RotatingFileHandler(
            filename=os.path.join(log_dir, f"{name}_error_{timestamp}.log"),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
    
    # 如果需要控制台输出，添加流处理器
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

class LoggerManager:
    """日志管理器单例类"""
    _instance = None
    _logger = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_logger(cls, name: str = "VRP", **kwargs) -> logging.Logger:
        """
        获取或创建日志记录器
        
        Args:
            name: 日志记录器名称
            **kwargs: 传递给setup_logger的其他参数
            
        Returns:
            logging.Logger: 日志记录器实例
        """
        if cls._logger is None:
            cls._logger = setup_logger(name, **kwargs)
        return cls._logger
    
    @classmethod
    def reset_logger(cls):
        """重置日志记录器"""
        cls._logger = None

def get_logger(name: str = "VRP", **kwargs) -> logging.Logger:
    """
    获取日志记录器的便捷函数
    
    Args:
        name: 日志记录器名称
        **kwargs: 其他配置参数
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    return LoggerManager.get_logger(name, **kwargs)

# 一些常用的日志装饰器
def log_execution_time(logger):
    """记录函数执行时间的装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"{func.__name__} 执行时间: {end_time - start_time:.2f}秒")
            return result
        return wrapper
    return decorator

def log_exceptions(logger):
    """记录异常的装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{func.__name__} 发生异常: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator