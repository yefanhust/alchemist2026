"""
市场情绪因子信号计算
权重: 0.15

分析矿业股相对强度（领先指标）和 ETF 成交量模式。
"""

from typing import Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class SentimentSignalResult:
    """情绪信号结果"""
    miners_relative: float     # 矿业股相对强度 [-1, 1]
    volume_pattern: float      # 成交量模式信号 [-1, 1]
    composite: float           # 综合情绪信号 [-1, 1]


class SentimentSignals:
    """
    市场情绪信号计算器

    分析以下情绪指标：
    - 矿业股 (GDX) vs 黄金 ETF (GLD) 相对强度（领先指标）
    - GLD 成交量模式（资金流向代理）
    """

    def __init__(
        self,
        relative_strength_period: int = 20,
        volume_period: int = 10,
    ):
        self.relative_strength_period = relative_strength_period
        self.volume_period = volume_period

    def calculate(
        self,
        gold_etf_prices: np.ndarray,
        gold_miners_prices: Optional[np.ndarray] = None,
        gold_etf_volume: Optional[np.ndarray] = None,
        **_kwargs,
    ) -> SentimentSignalResult:
        """
        计算市场情绪信号

        Args:
            gold_etf_prices: 黄金 ETF 价格数组
            gold_miners_prices: 黄金矿业 ETF (GDX) 价格数组
            gold_etf_volume: 黄金 ETF 成交量数组

        Returns:
            SentimentSignalResult 对象
        """
        miners_signal = self._calculate_relative_strength(
            gold_miners_prices, gold_etf_prices
        )
        volume_signal = self._calculate_volume_pattern(gold_etf_volume)

        # 综合情绪信号（按可用数据动态加权）
        period = self.relative_strength_period
        weights = {
            "miners": (miners_signal, 0.70,
                       gold_miners_prices is not None and len(gold_miners_prices) >= period),
            "volume": (volume_signal, 0.30,
                       gold_etf_volume is not None and len(gold_etf_volume) >= self.volume_period),
        }

        total_weight = sum(w for _, w, avail in weights.values() if avail)
        if total_weight > 0:
            composite = sum(sig * w for sig, w, avail in weights.values() if avail) / total_weight
        else:
            composite = 0.0

        return SentimentSignalResult(
            miners_relative=miners_signal,
            volume_pattern=volume_signal,
            composite=np.clip(composite, -1.0, 1.0),
        )

    def _calculate_relative_strength(
        self,
        miners: Optional[np.ndarray],
        gold_etf: np.ndarray,
    ) -> float:
        """
        计算矿业股相对强度

        矿业股通常是黄金价格的领先指标。
        当矿业股表现强于黄金 ETF 时，预示黄金可能上涨。
        """
        if miners is None or len(miners) < self.relative_strength_period:
            return 0.0

        if len(gold_etf) < self.relative_strength_period:
            return 0.0

        period = self.relative_strength_period

        # 中期相对强度
        miners_return = (miners[-1] - miners[-period]) / miners[-period]
        etf_return = (gold_etf[-1] - gold_etf[-period]) / gold_etf[-period]
        relative_return = miners_return - etf_return

        # 短期相对强度（更敏感的信号）
        short_period = min(5, period)
        miners_short_return = (miners[-1] - miners[-short_period]) / miners[-short_period]
        etf_short_return = (gold_etf[-1] - gold_etf[-short_period]) / gold_etf[-short_period]
        short_relative = miners_short_return - etf_short_return

        signal = relative_return * 0.6 + short_relative * 0.4
        return np.clip(signal * 10, -1.0, 1.0)

    def _calculate_volume_pattern(
        self,
        volume: Optional[np.ndarray],
    ) -> float:
        """
        分析成交量模式

        放量通常确认趋势方向，缩量可能预示趋势减弱。
        """
        if volume is None or len(volume) < self.volume_period:
            return 0.0

        period = self.volume_period
        avg_volume = np.mean(volume[-period:-1])  # 排除最新数据

        if avg_volume <= 0:
            return 0.0

        current_volume = volume[-1]
        volume_ratio = current_volume / avg_volume

        if volume_ratio > 1.5:
            return 0.3   # 放量，趋势确认
        elif volume_ratio < 0.5:
            return -0.2  # 缩量，兴趣下降
        else:
            return 0.0
