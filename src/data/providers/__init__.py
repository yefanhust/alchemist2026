"""
数据提供者基类
定义数据获取的统一接口
"""

from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from enum import Enum

from ..models import MarketData


class DataInterval(Enum):
    """数据时间间隔"""
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1m"


class DataProvider(ABC):
    """
    数据提供者抽象基类
    
    定义获取市场数据的统一接口。
    所有数据源（如 Alpha Vantage, Yahoo Finance 等）都应实现此接口。
    """
    
    def __init__(self, cache_backend=None):
        """
        初始化数据提供者
        
        Args:
            cache_backend: 缓存后端（可选）
        """
        self.cache = cache_backend
    
    @property
    @abstractmethod
    def name(self) -> str:
        """数据源名称"""
        pass
    
    @property
    @abstractmethod
    def supported_intervals(self) -> List[DataInterval]:
        """支持的数据间隔"""
        pass
    
    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: DataInterval = DataInterval.DAILY,
    ) -> MarketData:
        """
        获取历史数据
        
        Args:
            symbol: 资产代码
            start_date: 开始日期
            end_date: 结束日期（默认为今天）
            interval: 数据间隔
            
        Returns:
            MarketData 对象
        """
        pass
    
    @abstractmethod
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        获取最新价格
        
        Args:
            symbol: 资产代码
            
        Returns:
            最新价格
        """
        pass
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取实时报价
        
        Args:
            symbol: 资产代码
            
        Returns:
            报价信息字典
        """
        pass
    
    async def get_multiple_historical_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: DataInterval = DataInterval.DAILY,
    ) -> Dict[str, MarketData]:
        """
        批量获取历史数据
        
        Args:
            symbols: 资产代码列表
            start_date: 开始日期
            end_date: 结束日期
            interval: 数据间隔
            
        Returns:
            {symbol: MarketData} 字典
        """
        result = {}
        for symbol in symbols:
            data = await self.get_historical_data(symbol, start_date, end_date, interval)
            result[symbol] = data
        return result
    
    def _generate_cache_key(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: DataInterval,
    ) -> str:
        """
        生成缓存键
        
        Args:
            symbol: 资产代码
            start_date: 开始日期
            end_date: 结束日期
            interval: 数据间隔
            
        Returns:
            缓存键字符串
        """
        return f"{self.name}:{symbol}:{interval.value}:{start_date.strftime('%Y%m%d')}:{end_date.strftime('%Y%m%d')}"
    
    async def _get_from_cache(self, cache_key: str) -> Optional[MarketData]:
        """从缓存获取数据"""
        if self.cache is None:
            return None
        return await self.cache.get(cache_key)
    
    async def _save_to_cache(self, cache_key: str, data: MarketData) -> None:
        """保存数据到缓存"""
        if self.cache is not None:
            await self.cache.set(cache_key, data)
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        验证资产代码格式
        
        Args:
            symbol: 资产代码
            
        Returns:
            是否有效
        """
        if not symbol or not isinstance(symbol, str):
            return False
        
        # 基本验证：只包含字母、数字、点和横杠
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")
        return all(c in valid_chars for c in symbol.upper())
    
    def __repr__(self):
        return f"DataProvider({self.name})"
