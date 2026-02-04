"""
市场数据 Pydantic 模型
"""

from typing import List, Optional
from pydantic import BaseModel


class OHLCVItem(BaseModel):
    """单条 OHLCV 数据"""
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: Optional[float] = None


class OHLCVResponse(BaseModel):
    """OHLCV 数据响应"""
    symbol: str
    interval: str
    count: int
    data: List[OHLCVItem]


class SymbolInfo(BaseModel):
    """Symbol 信息"""
    symbol: str
    interval: str
    count: int
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    source: Optional[str] = None


class SymbolListResponse(BaseModel):
    """Symbol 列表响应"""
    symbols: List[SymbolInfo]
    total: int
