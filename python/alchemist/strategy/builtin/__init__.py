"""
内置策略模块
"""

from strategy.builtin.sma_crossover import SMACrossoverStrategy
from strategy.builtin.mean_reversion import MeanReversionStrategy

__all__ = ["SMACrossoverStrategy", "MeanReversionStrategy"]
