"""
策略模块
提供策略基类和内置策略实现
"""

from strategy.base import Strategy, Signal, SignalType
from strategy.indicators.base import Indicator
from strategy.gold import GoldTradingStrategy

__all__ = [
    "Strategy",
    "Signal",
    "SignalType",
    "Indicator",
    "GoldTradingStrategy",
]
