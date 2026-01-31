"""
模拟交易模块
提供回测和模拟交易功能
"""

from simulation.engine import SimulationEngine
from simulation.broker import VirtualBroker
from simulation.backtest import Backtester, BacktestResult

__all__ = [
    "SimulationEngine",
    "VirtualBroker",
    "Backtester",
    "BacktestResult",
]
