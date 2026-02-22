"""
黄金策略信号计算模块
包含技术面、跨市场、情绪和宏观因子的信号计算
"""

from .technical import TechnicalSignals
from .cross_market import CrossMarketSignals
from .sentiment import SentimentSignals
from .macro import MacroSignals

__all__ = [
    "TechnicalSignals",
    "CrossMarketSignals",
    "SentimentSignals",
    "MacroSignals",
]
