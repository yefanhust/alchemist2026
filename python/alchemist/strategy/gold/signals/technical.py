"""
技术面因子信号计算
权重: 0.30

使用黄金 ETF 价格数据计算技术分析信号。
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class TechnicalSignalResult:
    """技术信号结果"""
    trend: float          # 趋势强度 [-1, 1]
    momentum: float       # 动量信号 [-1, 1]
    volatility: float     # 波动率突破信号 [-1, 1]
    volume: float         # 成交量信号 [-1, 1]
    composite: float      # 综合技术信号 [-1, 1]


class TechnicalSignals:
    """
    技术分析信号计算器

    使用黄金 ETF 价格数据计算以下技术指标：
    - 趋势强度（多周期均线）
    - 动量指标（RSI、ROC）
    - 波动率突破
    - 成交量分析
    """

    def __init__(
        self,
        trend_periods: List[int] = None,
        momentum_periods: List[int] = None,
        volatility_period: int = 20,
        volume_period: int = 20,
    ):
        """
        初始化技术信号计算器

        Args:
            trend_periods: 趋势计算周期列表
            momentum_periods: 动量计算周期列表
            volatility_period: 波动率计算周期
            volume_period: 成交量分析周期
        """
        self.trend_periods = trend_periods or [20, 50, 200]
        self.momentum_periods = momentum_periods or [14, 21]
        self.volatility_period = volatility_period
        self.volume_period = volume_period

    def calculate(
        self,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None,
    ) -> TechnicalSignalResult:
        """
        计算技术分析信号

        Args:
            prices: 价格数组（收盘价）
            volumes: 成交量数组（可选）

        Returns:
            TechnicalSignalResult 对象
        """
        trend = self._calculate_trend_strength(prices, self.trend_periods)
        momentum = self._calculate_momentum(prices, self.momentum_periods)
        volatility = self._calculate_volatility_breakout(prices)
        volume = self._analyze_volume_pattern(prices, volumes) if volumes is not None else 0.0

        # 综合技术信号（内部权重分配）
        composite = (
            trend * 0.35 +
            momentum * 0.30 +
            volatility * 0.20 +
            volume * 0.15
        )

        return TechnicalSignalResult(
            trend=trend,
            momentum=momentum,
            volatility=volatility,
            volume=volume,
            composite=np.clip(composite, -1.0, 1.0),
        )

    def _calculate_trend_strength(
        self,
        prices: np.ndarray,
        periods: List[int],
    ) -> float:
        """
        计算趋势强度

        使用多周期移动平均线判断趋势方向和强度。
        当短期均线在长期均线上方时为上涨趋势。

        Args:
            prices: 价格数组
            periods: 周期列表 [短期, 中期, 长期]

        Returns:
            趋势强度 [-1, 1]
        """
        if len(prices) < max(periods):
            return 0.0

        smas = []
        for period in periods:
            if len(prices) >= period:
                sma = np.mean(prices[-period:])
                smas.append(sma)

        if len(smas) < 2:
            return 0.0

        current_price = prices[-1]
        signals = []

        # 价格与各均线的关系
        for sma in smas:
            if sma > 0:
                deviation = (current_price - sma) / sma
                signals.append(np.clip(deviation * 10, -1.0, 1.0))

        # 均线排列分析
        if len(smas) >= 3:
            # 多头排列: 短期 > 中期 > 长期
            if smas[0] > smas[1] > smas[2]:
                signals.append(1.0)
            # 空头排列: 短期 < 中期 < 长期
            elif smas[0] < smas[1] < smas[2]:
                signals.append(-1.0)
            else:
                signals.append(0.0)

        return np.clip(np.mean(signals), -1.0, 1.0) if signals else 0.0

    def _calculate_momentum(
        self,
        prices: np.ndarray,
        periods: List[int],
    ) -> float:
        """
        计算动量信号

        综合 RSI 和 ROC 指标。

        Args:
            prices: 价格数组
            periods: 周期列表

        Returns:
            动量信号 [-1, 1]
        """
        signals = []

        for period in periods:
            if len(prices) < period + 1:
                continue

            # RSI 计算
            rsi = self._calculate_rsi(prices, period)
            if rsi is not None:
                # RSI > 70 超买, RSI < 30 超卖
                # 但在趋势策略中，高 RSI 可能表示强势
                rsi_signal = (rsi - 50) / 50  # 归一化到 [-1, 1]
                signals.append(rsi_signal)

            # ROC (Rate of Change) 计算
            roc = (prices[-1] - prices[-period - 1]) / prices[-period - 1]
            roc_signal = np.clip(roc * 10, -1.0, 1.0)
            signals.append(roc_signal)

        return np.clip(np.mean(signals), -1.0, 1.0) if signals else 0.0

    def _calculate_rsi(self, prices: np.ndarray, period: int) -> Optional[float]:
        """
        计算 RSI 指标

        Args:
            prices: 价格数组
            period: 计算周期

        Returns:
            RSI 值 [0, 100]
        """
        if len(prices) < period + 1:
            return None

        deltas = np.diff(prices[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_volatility_breakout(self, prices: np.ndarray) -> float:
        """
        计算波动率突破信号

        使用布林带突破和 ATR 分析。

        Args:
            prices: 价格数组

        Returns:
            波动率突破信号 [-1, 1]
        """
        period = self.volatility_period
        if len(prices) < period:
            return 0.0

        recent_prices = prices[-period:]
        current_price = prices[-1]

        # 布林带计算
        sma = np.mean(recent_prices)
        std = np.std(recent_prices)

        if std == 0:
            return 0.0

        upper_band = sma + 2 * std
        lower_band = sma - 2 * std

        # 突破信号
        if current_price > upper_band:
            # 上轨突破 - 强势信号
            breakout_strength = (current_price - upper_band) / std
            return np.clip(breakout_strength, 0.0, 1.0)
        elif current_price < lower_band:
            # 下轨突破 - 弱势信号
            breakout_strength = (lower_band - current_price) / std
            return np.clip(-breakout_strength, -1.0, 0.0)
        else:
            # 带内运行 - 相对位置
            position = (current_price - lower_band) / (upper_band - lower_band)
            return (position - 0.5) * 0.5  # 缩小范围

    def _analyze_volume_pattern(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
    ) -> float:
        """
        分析成交量模式

        识别放量上涨/下跌、缩量调整等模式。

        Args:
            prices: 价格数组
            volumes: 成交量数组

        Returns:
            成交量信号 [-1, 1]
        """
        period = self.volume_period
        if len(prices) < period or len(volumes) < period:
            return 0.0

        recent_volumes = volumes[-period:]
        avg_volume = np.mean(recent_volumes[:-1])  # 排除最新成交量

        if avg_volume == 0:
            return 0.0

        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume

        # 价格变动方向
        price_change = prices[-1] - prices[-2]
        price_direction = 1 if price_change > 0 else (-1 if price_change < 0 else 0)

        # 放量上涨/下跌的信号强度
        if volume_ratio > 1.5:
            # 放量
            return np.clip(price_direction * (volume_ratio - 1) * 0.5, -1.0, 1.0)
        elif volume_ratio < 0.5:
            # 缩量 - 趋势可能减弱
            return np.clip(-price_direction * (1 - volume_ratio) * 0.3, -1.0, 1.0)
        else:
            # 正常成交量
            return price_direction * 0.1
