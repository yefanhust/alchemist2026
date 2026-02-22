"""
黄金回测 API 响应模型
"""

from typing import List, Dict, Optional
from pydantic import BaseModel


class FactorScores(BaseModel):
    """因子得分"""
    technical: float
    cross_market: float
    sentiment: float
    macro: float


class DailySignalPoint(BaseModel):
    """每日信号数据点"""
    date: str
    composite_score: float
    factor_scores: FactorScores
    tactical_action: str  # boost_buy, normal_buy, reduce_buy, skip_buy, partial_sell, hold
    signal_type: Optional[str] = None  # buy, sell, or null
    signal_strength: Optional[float] = None


class PricePoint(BaseModel):
    """价格数据点"""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class ThresholdParams(BaseModel):
    """战术阈值参数"""
    boost_buy: float
    normal_buy: float
    reduce_buy: float
    skip_buy: float
    partial_sell: float


class PositionParams(BaseModel):
    """仓位管理参数"""
    buy_day: int
    boost_multiplier: float
    reduce_multiplier: float
    sell_fraction: float
    force_sell_interval_days: Optional[int] = None
    force_sell_fraction: Optional[float] = None
    force_sell_profit_thresh: Optional[float] = None


class StrategyParams(BaseModel):
    """当前使用的策略参数"""
    source: str  # "optimized" or "default"
    source_file: Optional[str] = None
    weights: FactorScores
    thresholds: ThresholdParams
    position: PositionParams


class GoldBacktestResponse(BaseModel):
    """黄金回测响应"""
    gold_prices: List[PricePoint]
    signals: List[DailySignalPoint]
    available_indicators: List[str]
    indicator_series: Dict[str, List[dict]]
    factor_weights: FactorScores
    strategy_params: Optional[StrategyParams] = None
