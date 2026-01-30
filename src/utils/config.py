"""
配置管理模块
统一管理系统配置
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import yaml
from dotenv import load_dotenv


@dataclass
class DatabaseConfig:
    """数据库配置"""
    url: str = "sqlite:///./data/cache/market_data.db"
    pool_size: int = 5
    echo: bool = False


@dataclass
class AlphaVantageConfig:
    """Alpha Vantage 配置"""
    api_key: str = ""
    calls_per_minute: int = 5
    calls_per_day: int = 500


@dataclass
class BrokerConfig:
    """券商配置"""
    commission_rate: float = 0.001
    min_commission: float = 1.0
    slippage_rate: float = 0.0005
    short_selling_enabled: bool = True


@dataclass
class GPUConfig:
    """GPU 配置"""
    enabled: bool = True
    device_ids: list = field(default_factory=lambda: [0, 1])
    memory_fraction: float = 0.8


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    file: str = "./logs/quant_system.log"
    rotation: str = "10 MB"
    retention: str = "1 week"


@dataclass
class Config:
    """
    系统配置
    
    统一管理所有配置项。
    """
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    alphavantage: AlphaVantageConfig = field(default_factory=AlphaVantageConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # 基础配置
    initial_capital: float = 100000.0
    debug: bool = False
    
    def __post_init__(self):
        """从环境变量加载配置"""
        # Alpha Vantage
        self.alphavantage.api_key = os.getenv(
            "ALPHAVANTAGE_API_KEY",
            self.alphavantage.api_key
        )
        
        # 数据库
        self.database.url = os.getenv("DATABASE_URL", self.database.url)
        
        # GPU
        gpu_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if gpu_devices:
            self.gpu.device_ids = [int(d) for d in gpu_devices.split(",")]
        
        # 日志
        self.logging.level = os.getenv("LOG_LEVEL", self.logging.level)
        self.logging.file = os.getenv("LOG_FILE", self.logging.file)
        
        # 调试模式
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        从 YAML 文件加载配置
        
        Args:
            path: 配置文件路径
            
        Returns:
            Config 对象
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """
        从字典加载配置
        
        Args:
            data: 配置字典
            
        Returns:
            Config 对象
        """
        config = cls()
        
        if "database" in data:
            for key, value in data["database"].items():
                setattr(config.database, key, value)
        
        if "alphavantage" in data:
            for key, value in data["alphavantage"].items():
                setattr(config.alphavantage, key, value)
        
        if "broker" in data:
            for key, value in data["broker"].items():
                setattr(config.broker, key, value)
        
        if "gpu" in data:
            for key, value in data["gpu"].items():
                setattr(config.gpu, key, value)
        
        if "logging" in data:
            for key, value in data["logging"].items():
                setattr(config.logging, key, value)
        
        if "initial_capital" in data:
            config.initial_capital = data["initial_capital"]
        
        if "debug" in data:
            config.debug = data["debug"]
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "database": {
                "url": self.database.url,
                "pool_size": self.database.pool_size,
                "echo": self.database.echo,
            },
            "alphavantage": {
                "api_key": "***" if self.alphavantage.api_key else "",
                "calls_per_minute": self.alphavantage.calls_per_minute,
                "calls_per_day": self.alphavantage.calls_per_day,
            },
            "broker": {
                "commission_rate": self.broker.commission_rate,
                "min_commission": self.broker.min_commission,
                "slippage_rate": self.broker.slippage_rate,
                "short_selling_enabled": self.broker.short_selling_enabled,
            },
            "gpu": {
                "enabled": self.gpu.enabled,
                "device_ids": self.gpu.device_ids,
                "memory_fraction": self.gpu.memory_fraction,
            },
            "logging": {
                "level": self.logging.level,
                "file": self.logging.file,
                "rotation": self.logging.rotation,
                "retention": self.logging.retention,
            },
            "initial_capital": self.initial_capital,
            "debug": self.debug,
        }
    
    def save_yaml(self, path: str) -> None:
        """保存配置到 YAML 文件"""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(
    config_path: Optional[str] = None,
    env_file: Optional[str] = None,
) -> Config:
    """
    加载配置
    
    优先级：环境变量 > 配置文件 > 默认值
    
    Args:
        config_path: YAML 配置文件路径
        env_file: .env 文件路径
        
    Returns:
        Config 对象
    """
    # 加载 .env 文件
    if env_file:
        load_dotenv(env_file)
    else:
        # 尝试默认位置
        default_env = Path(".env")
        if default_env.exists():
            load_dotenv(default_env)
    
    # 加载配置文件
    if config_path and Path(config_path).exists():
        config = Config.from_yaml(config_path)
    else:
        config = Config()
    
    return config


# 全局配置实例
_global_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置"""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_config(config: Config) -> None:
    """设置全局配置"""
    global _global_config
    _global_config = config
