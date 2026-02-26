"""
配置管理模块
统一管理系统配置

配置加载优先级:
1. config/config.yaml (用户配置，包含敏感信息，不提交到 Git)
2. config/default.yaml.example (模板配置)
3. 代码内置默认值
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import yaml


@dataclass
class DatabaseConfig:
    """数据库配置"""
    url: str = ""  # 从 config.yaml 加载，或使用 get_data_cache_path() 解析
    pool_size: int = 5
    echo: bool = False


@dataclass
class AlphaVantageConfig:
    """Alpha Vantage 配置"""
    api_key: str = ""
    plan: str = "free"
    calls_per_minute: int = 5
    calls_per_day: int = 500


@dataclass
class RedisConfig:
    """Redis 配置"""
    url: str = "redis://localhost:6379/0"


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
class SSLConfig:
    """SSL 配置"""
    enabled: bool = True
    keyfile: str = ""
    certfile: str = ""


@dataclass
class WebConfig:
    """Web 服务配置"""
    host: str = "0.0.0.0"
    port: int = 8443
    debug: bool = False
    ssl: SSLConfig = field(default_factory=SSLConfig)
    domain: str = ""


@dataclass
class Config:
    """
    系统配置

    统一管理所有配置项。
    """
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    alphavantage: AlphaVantageConfig = field(default_factory=AlphaVantageConfig)
    fred_api_key: str = ""
    redis: RedisConfig = field(default_factory=RedisConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    web: WebConfig = field(default_factory=WebConfig)

    # 基础配置
    initial_capital: float = 100000.0
    debug: bool = False
    testing: bool = False

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
                if hasattr(config.database, key):
                    setattr(config.database, key, value)

        if "alphavantage" in data:
            for key, value in data["alphavantage"].items():
                if hasattr(config.alphavantage, key):
                    setattr(config.alphavantage, key, value)

        if "redis" in data:
            for key, value in data["redis"].items():
                if hasattr(config.redis, key):
                    setattr(config.redis, key, value)

        if "broker" in data:
            for key, value in data["broker"].items():
                if hasattr(config.broker, key):
                    setattr(config.broker, key, value)

        if "gpu" in data:
            for key, value in data["gpu"].items():
                if hasattr(config.gpu, key):
                    setattr(config.gpu, key, value)

        if "logging" in data:
            for key, value in data["logging"].items():
                if hasattr(config.logging, key):
                    setattr(config.logging, key, value)

        if "web" in data:
            web_data = data["web"]
            for key, value in web_data.items():
                if key == "ssl" and isinstance(value, dict):
                    for ssl_key, ssl_value in value.items():
                        if hasattr(config.web.ssl, ssl_key):
                            setattr(config.web.ssl, ssl_key, ssl_value)
                elif hasattr(config.web, key):
                    setattr(config.web, key, value)

        if "fred_api_key" in data:
            config.fred_api_key = data["fred_api_key"]

        if "initial_capital" in data:
            config.initial_capital = data["initial_capital"]

        if "debug" in data:
            config.debug = data["debug"]

        if "testing" in data:
            config.testing = data["testing"]

        return config

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（敏感信息会被隐藏）"""
        return {
            "database": {
                "url": self.database.url,
                "pool_size": self.database.pool_size,
                "echo": self.database.echo,
            },
            "alphavantage": {
                "api_key": "***" if self.alphavantage.api_key else "",
                "plan": self.alphavantage.plan,
                "calls_per_minute": self.alphavantage.calls_per_minute,
                "calls_per_day": self.alphavantage.calls_per_day,
            },
            "redis": {
                "url": self.redis.url,
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
            "web": {
                "host": self.web.host,
                "port": self.web.port,
                "debug": self.web.debug,
                "domain": "***" if self.web.domain else "",
            },
            "initial_capital": self.initial_capital,
            "debug": self.debug,
            "testing": self.testing,
        }

    def save_yaml(self, path: str) -> None:
        """保存配置到 YAML 文件"""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def _find_project_root() -> Path:
    """查找项目根目录"""
    # 从当前文件向上查找包含 config 目录的位置
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "config").is_dir():
            return parent
    # 回退到当前工作目录
    return Path.cwd()


def get_data_cache_path(filename: str = "market_data.db") -> str:
    """
    获取数据缓存文件的绝对路径

    始终解析到 {project_root}/data/cache/{filename}，
    无论脚本从哪个目录运行。

    Args:
        filename: 缓存数据库文件名

    Returns:
        缓存文件的绝对路径字符串
    """
    root = _find_project_root()
    cache_dir = root / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / filename)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    加载配置

    配置文件查找顺序:
    1. 指定的 config_path
    2. config/config.yaml (用户配置)
    3. 默认值

    Args:
        config_path: YAML 配置文件路径

    Returns:
        Config 对象
    """
    # 查找项目根目录
    project_root = _find_project_root()

    # 配置文件搜索路径
    search_paths = []

    if config_path:
        search_paths.append(Path(config_path))

    search_paths.extend([
        project_root / "config" / "config.yaml",
        Path.cwd() / "config" / "config.yaml",
        Path("/workspace/config/config.yaml"),  # 容器内路径
    ])

    # 查找并加载配置文件
    for path in search_paths:
        if path.is_file():
            config = Config.from_yaml(str(path))
            return config

    # 未找到配置文件，使用默认值
    return Config()


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


def reset_config() -> None:
    """重置全局配置（用于测试）"""
    global _global_config
    _global_config = None
