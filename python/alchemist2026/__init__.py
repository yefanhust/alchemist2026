"""
金融指标跟踪及智能交易系统

模块:
- core: 核心抽象（资产、组合、订单、持仓）
- data: 数据获取和缓存
- strategy: 交易策略
- simulation: 模拟交易和回测
- gpu: GPU加速工具
- utils: 通用工具
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from core import Asset, Portfolio, Order, Position
from strategy import Strategy, Signal

__all__ = [
    "Asset",
    "Portfolio",
    "Order",
    "Position",
    "Strategy",
    "Signal",
]
