"""
黄金投资策略模块

择时增强型定投策略，整合技术面、跨市场、情绪和宏观因子。

使用示例:
    from strategy.gold import GoldTradingStrategy

    # 创建策略实例（默认配置）
    strategy = GoldTradingStrategy()

    # 或自定义权重和阈值
    from strategy.gold import FactorWeights, TacticalThresholds, PositionConfig

    strategy = GoldTradingStrategy(
        weights=FactorWeights(
            technical=0.35,
            cross_market=0.25,
            sentiment=0.15,
            macro=0.25,
        ),
        thresholds=TacticalThresholds(
            boost_buy=0.3,
            normal_buy=0.0,
            reduce_buy=-0.3,
            partial_sell=-0.6,
        ),
        position_config=PositionConfig(
            buy_day=2,  # 周三
        ),
    )
"""

from .assets import (
    GoldAssets,
    DEFAULT_GOLD_ASSETS,
    CROSS_MARKET_ETF_SYMBOLS,
    CROSS_MARKET_SPECIAL_DATA,
)
from .strategy import (
    GoldTradingStrategy,
    FactorWeights,
    TacticalThresholds,
    PositionConfig,
)
from .signals import (
    TechnicalSignals,
    CrossMarketSignals,
    SentimentSignals,
    MacroSignals,
)

__all__ = [
    # 策略
    "GoldTradingStrategy",
    "FactorWeights",
    "TacticalThresholds",
    "PositionConfig",
    # 资产
    "GoldAssets",
    "DEFAULT_GOLD_ASSETS",
    "CROSS_MARKET_ETF_SYMBOLS",
    "CROSS_MARKET_SPECIAL_DATA",
    # 信号计算器
    "TechnicalSignals",
    "CrossMarketSignals",
    "SentimentSignals",
    "MacroSignals",
]
