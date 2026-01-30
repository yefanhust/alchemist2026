"""
资产抽象模块
定义不同类型资产的统一接口
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any


class AssetType(Enum):
    """资产类型枚举"""
    STOCK = "stock"           # 股票
    ETF = "etf"               # 交易所交易基金
    CRYPTO = "crypto"         # 加密货币
    FOREX = "forex"           # 外汇
    FUTURES = "futures"       # 期货
    OPTIONS = "options"       # 期权
    INDEX = "index"           # 指数
    BOND = "bond"             # 债券


@dataclass
class Asset:
    """
    资产基类
    
    表示可交易的金融资产，如股票、ETF、加密货币等。
    
    Attributes:
        symbol: 资产代码（如 AAPL, BTC-USD）
        asset_type: 资产类型
        name: 资产名称
        exchange: 交易所
        currency: 计价货币
        tick_size: 最小价格变动
        lot_size: 最小交易单位
        metadata: 额外元数据
    """
    
    symbol: str
    asset_type: AssetType = AssetType.STOCK
    name: Optional[str] = None
    exchange: Optional[str] = None
    currency: str = "USD"
    tick_size: float = 0.01
    lot_size: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """验证和规范化资产数据"""
        self.symbol = self.symbol.upper()
        if self.name is None:
            self.name = self.symbol
    
    def __hash__(self):
        return hash((self.symbol, self.asset_type))
    
    def __eq__(self, other):
        if not isinstance(other, Asset):
            return False
        return self.symbol == other.symbol and self.asset_type == other.asset_type
    
    def __repr__(self):
        return f"Asset({self.symbol}, {self.asset_type.value})"
    
    def round_price(self, price: float) -> float:
        """将价格舍入到最小变动单位"""
        return round(price / self.tick_size) * self.tick_size
    
    def round_quantity(self, quantity: float) -> float:
        """将数量舍入到最小交易单位"""
        return round(quantity / self.lot_size) * self.lot_size
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "symbol": self.symbol,
            "asset_type": self.asset_type.value,
            "name": self.name,
            "exchange": self.exchange,
            "currency": self.currency,
            "tick_size": self.tick_size,
            "lot_size": self.lot_size,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Asset":
        """从字典创建资产"""
        data = data.copy()
        if "asset_type" in data and isinstance(data["asset_type"], str):
            data["asset_type"] = AssetType(data["asset_type"])
        return cls(**data)


@dataclass
class AssetPrice:
    """
    资产价格数据
    
    表示某一时刻的资产价格信息。
    """
    
    asset: Asset
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: Optional[float] = None
    
    @property
    def ohlc(self) -> tuple:
        """返回 OHLC 数据"""
        return (self.open, self.high, self.low, self.close)
    
    @property
    def typical_price(self) -> float:
        """典型价格 (H+L+C)/3"""
        return (self.high + self.low + self.close) / 3
    
    @property
    def vwap(self) -> float:
        """简化的 VWAP 估算"""
        return self.typical_price  # 实际 VWAP 需要更多数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "symbol": self.asset.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "adjusted_close": self.adjusted_close,
        }
