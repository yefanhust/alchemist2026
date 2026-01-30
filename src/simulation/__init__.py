"""
模拟交易模块
提供回测和模拟交易功能
"""

from .engine import SimulationEngine
from .broker import VirtualBroker
from .backtest import Backtester, BacktestResult

__all__ = [
    "SimulationEngine",
    "VirtualBroker",
    "Backtester",
    "BacktestResult",
]
