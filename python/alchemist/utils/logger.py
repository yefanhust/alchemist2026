"""
日志模块
配置统一的日志系统
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logger(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
    format_string: Optional[str] = None,
) -> None:
    """
    配置日志系统
    
    Args:
        level: 日志级别
        log_file: 日志文件路径
        rotation: 日志轮转大小
        retention: 日志保留时间
        format_string: 自定义格式
    """
    # 移除默认处理器
    logger.remove()
    
    # 默认格式
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # 控制台输出
    logger.add(
        sys.stderr,
        level=level,
        format=format_string,
        colorize=True,
    )
    
    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=level,
            format=format_string,
            rotation=rotation,
            retention=retention,
            compression="zip",
            encoding="utf-8",
        )
    
    logger.info(f"日志系统初始化完成，级别: {level}")


def get_logger(name: str):
    """
    获取命名日志器
    
    Args:
        name: 日志器名称
        
    Returns:
        日志器实例
    """
    return logger.bind(name=name)
