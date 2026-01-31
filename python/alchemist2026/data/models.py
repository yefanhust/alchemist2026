"""
数据模型模块
定义标准化的市场数据结构
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np


@dataclass
class OHLCV:
    """
    OHLCV 数据结构
    
    表示单个时间点的价格和成交量数据。
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: Optional[float] = None
    
    @property
    def typical_price(self) -> float:
        """典型价格"""
        return (self.high + self.low + self.close) / 3
    
    @property
    def range(self) -> float:
        """价格区间"""
        return self.high - self.low
    
    @property
    def body(self) -> float:
        """K线实体"""
        return self.close - self.open
    
    @property
    def is_bullish(self) -> bool:
        """是否阳线"""
        return self.close > self.open
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "adjusted_close": self.adjusted_close,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OHLCV":
        """从字典创建"""
        data = data.copy()
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class MarketData:
    """
    市场数据容器
    
    存储和管理一个资产的历史数据。
    """
    symbol: str
    data: List[OHLCV] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> OHLCV:
        return self.data[index]
    
    @property
    def is_empty(self) -> bool:
        """是否为空"""
        return len(self.data) == 0
    
    @property
    def start_date(self) -> Optional[datetime]:
        """起始日期"""
        if self.is_empty:
            return None
        return self.data[0].timestamp
    
    @property
    def end_date(self) -> Optional[datetime]:
        """结束日期"""
        if self.is_empty:
            return None
        return self.data[-1].timestamp
    
    @property
    def latest(self) -> Optional[OHLCV]:
        """最新数据"""
        if self.is_empty:
            return None
        return self.data[-1]
    
    @property
    def latest_price(self) -> Optional[float]:
        """最新价格"""
        if self.is_empty:
            return None
        return self.data[-1].close
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        转换为 Pandas DataFrame
        
        Returns:
            DataFrame，包含 OHLCV 数据
        """
        if self.is_empty:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "adjusted_close"]
            )
        
        records = [
            {
                "timestamp": d.timestamp,
                "open": d.open,
                "high": d.high,
                "low": d.low,
                "close": d.close,
                "volume": d.volume,
                "adjusted_close": d.adjusted_close,
            }
            for d in self.data
        ]
        
        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df
    
    @classmethod
    def from_dataframe(cls, symbol: str, df: pd.DataFrame) -> "MarketData":
        """
        从 DataFrame 创建
        
        Args:
            symbol: 资产代码
            df: 包含 OHLCV 数据的 DataFrame
            
        Returns:
            MarketData 对象
        """
        data = []
        
        for timestamp, row in df.iterrows():
            ohlcv = OHLCV(
                timestamp=timestamp if isinstance(timestamp, datetime) else datetime.combine(timestamp, datetime.min.time()),
                open=float(row.get("open", row.get("Open", 0))),
                high=float(row.get("high", row.get("High", 0))),
                low=float(row.get("low", row.get("Low", 0))),
                close=float(row.get("close", row.get("Close", 0))),
                volume=float(row.get("volume", row.get("Volume", 0))),
                adjusted_close=float(row.get("adjusted_close", row.get("Adj Close", row.get("close", row.get("Close", 0))))),
            )
            data.append(ohlcv)
        
        return cls(symbol=symbol, data=data)
    
    def to_numpy(self) -> Dict[str, np.ndarray]:
        """
        转换为 NumPy 数组（用于 GPU 加速）
        
        Returns:
            包含各列数据的字典
        """
        if self.is_empty:
            return {
                "open": np.array([]),
                "high": np.array([]),
                "low": np.array([]),
                "close": np.array([]),
                "volume": np.array([]),
            }
        
        return {
            "open": np.array([d.open for d in self.data], dtype=np.float64),
            "high": np.array([d.high for d in self.data], dtype=np.float64),
            "low": np.array([d.low for d in self.data], dtype=np.float64),
            "close": np.array([d.close for d in self.data], dtype=np.float64),
            "volume": np.array([d.volume for d in self.data], dtype=np.float64),
        }
    
    def slice(self, start: datetime, end: datetime) -> "MarketData":
        """
        获取时间范围内的数据
        
        Args:
            start: 开始时间
            end: 结束时间
            
        Returns:
            切片后的 MarketData
        """
        filtered = [d for d in self.data if start <= d.timestamp <= end]
        return MarketData(symbol=self.symbol, data=filtered, metadata=self.metadata)
    
    def append(self, ohlcv: OHLCV) -> None:
        """添加数据点"""
        self.data.append(ohlcv)
    
    def extend(self, other: "MarketData") -> None:
        """合并数据"""
        if other.symbol != self.symbol:
            raise ValueError("资产代码不匹配")
        
        # 去重合并
        existing_timestamps = {d.timestamp for d in self.data}
        for ohlcv in other.data:
            if ohlcv.timestamp not in existing_timestamps:
                self.data.append(ohlcv)
        
        # 排序
        self.data.sort(key=lambda x: x.timestamp)
    
    def resample(self, freq: str = "D") -> "MarketData":
        """
        重采样数据
        
        Args:
            freq: 频率（'D' 日, 'W' 周, 'M' 月）
            
        Returns:
            重采样后的 MarketData
        """
        df = self.to_dataframe()
        
        resampled = df.resample(freq).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "adjusted_close": "last",
        }).dropna()
        
        return MarketData.from_dataframe(self.symbol, resampled)
    
    def returns(self, period: int = 1) -> np.ndarray:
        """
        计算收益率
        
        Args:
            period: 计算周期
            
        Returns:
            收益率数组
        """
        closes = np.array([d.close for d in self.data])
        if len(closes) <= period:
            return np.array([])
        
        return (closes[period:] - closes[:-period]) / closes[:-period]
    
    def log_returns(self, period: int = 1) -> np.ndarray:
        """
        计算对数收益率
        
        Args:
            period: 计算周期
            
        Returns:
            对数收益率数组
        """
        closes = np.array([d.close for d in self.data])
        if len(closes) <= period:
            return np.array([])
        
        return np.log(closes[period:] / closes[:-period])
