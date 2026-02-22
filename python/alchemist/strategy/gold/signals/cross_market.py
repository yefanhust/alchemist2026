"""
跨市场联动因子信号计算
权重: 0.25

分析黄金与其他市场（美元、股市）的关联性。
使用 ETF 代理和外汇数据：UUP（美元）、SPY（股市）、EUR/USD、USD/JPY。
"""

from typing import Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class CrossMarketSignalResult:
    """跨市场信号结果"""
    usd_correlation: float      # 美元负相关信号 [-1, 1]
    equity_safe_haven: float    # 股市避险信号 [-1, 1]
    composite: float            # 综合跨市场信号 [-1, 1]


class CrossMarketSignals:
    """
    跨市场关联分析

    分析黄金与以下市场的关联性：
    - 美元（UUP ETF + EUR/USD + USD/JPY 综合）
    - 股票市场（SPY — 避险需求）
    """

    def __init__(
        self,
        correlation_window: int = 30,
        safe_haven_window: int = 20,
    ):
        self.correlation_window = correlation_window
        self.safe_haven_window = safe_haven_window

    def calculate(
        self,
        gold_prices: np.ndarray,
        usd_index: Optional[np.ndarray] = None,
        sp500: Optional[np.ndarray] = None,
        eur_usd: Optional[np.ndarray] = None,
        usd_jpy: Optional[np.ndarray] = None,
        **_kwargs,
    ) -> CrossMarketSignalResult:
        """
        计算跨市场关联信号

        Args:
            gold_prices: 黄金价格数组
            usd_index: 美元 ETF (UUP) 价格数组
            sp500: 标普500 ETF (SPY) 价格数组
            eur_usd: EUR/USD 汇率数组
            usd_jpy: USD/JPY 汇率数组

        Returns:
            CrossMarketSignalResult 对象
        """
        usd_signal = self._calculate_usd_signal(
            gold_prices, usd_index, eur_usd, usd_jpy
        )
        safe_haven_signal = self._calculate_safe_haven_ratio(sp500, gold_prices)

        # 综合跨市场信号（按可用数据动态加权）
        weights = {
            "usd": (usd_signal, 0.55, self._has_usd_data(usd_index, eur_usd, usd_jpy)),
            "safe_haven": (safe_haven_signal, 0.45,
                           sp500 is not None and len(sp500) >= self.safe_haven_window),
        }

        total_weight = sum(w for _, w, avail in weights.values() if avail)
        if total_weight > 0:
            composite = sum(sig * w for sig, w, avail in weights.values() if avail) / total_weight
        else:
            composite = 0.0

        return CrossMarketSignalResult(
            usd_correlation=usd_signal,
            equity_safe_haven=safe_haven_signal,
            composite=np.clip(composite, -1.0, 1.0),
        )

    def _has_usd_data(
        self,
        usd_index: Optional[np.ndarray],
        eur_usd: Optional[np.ndarray],
        usd_jpy: Optional[np.ndarray],
    ) -> bool:
        """检查是否有任何美元相关数据"""
        window = self.correlation_window
        return any(
            arr is not None and len(arr) >= window
            for arr in [usd_index, eur_usd, usd_jpy]
        )

    def _calculate_correlation(
        self,
        series1: np.ndarray,
        series2: np.ndarray,
        window: int,
    ) -> float:
        """计算滚动相关系数"""
        if len(series1) < window or len(series2) < window:
            return 0.0

        s1 = series1[-window:]
        s2 = series2[-window:]

        returns1 = np.diff(s1) / s1[:-1]
        returns2 = np.diff(s2) / s2[:-1]

        if len(returns1) < 2 or len(returns2) < 2:
            return 0.0

        std1 = np.std(returns1)
        std2 = np.std(returns2)

        if std1 == 0 or std2 == 0:
            return 0.0

        correlation = np.corrcoef(returns1, returns2)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    def _calculate_usd_signal(
        self,
        gold_prices: np.ndarray,
        usd_index: Optional[np.ndarray],
        eur_usd: Optional[np.ndarray],
        usd_jpy: Optional[np.ndarray],
    ) -> float:
        """
        计算美元综合信号

        综合 UUP ETF、EUR/USD、USD/JPY 数据。
        美元走弱利好黄金 → 信号为正。

        Args:
            gold_prices: 黄金价格数组
            usd_index: UUP ETF 价格数组
            eur_usd: EUR/USD 汇率数组
            usd_jpy: USD/JPY 汇率数组

        Returns:
            美元信号 [-1, 1]，正值表示利好黄金
        """
        signals = []
        window = self.correlation_window

        # UUP (美元 ETF) — 美元涨则利空黄金
        if usd_index is not None and len(usd_index) >= window:
            correlation = self._calculate_correlation(
                gold_prices, usd_index, window
            )
            corr_signal = -correlation

            usd_return = (usd_index[-1] - usd_index[-window]) / usd_index[-window]
            trend_signal = -np.clip(usd_return * 8, -0.5, 0.5)

            signals.append((corr_signal + trend_signal) / 1.5)

        # EUR/USD — 欧元走强 = 美元走弱 = 利好黄金
        if eur_usd is not None and len(eur_usd) >= window:
            eur_return = (eur_usd[-1] - eur_usd[-window]) / eur_usd[-window]
            signals.append(np.clip(eur_return * 10, -1.0, 1.0))

        # USD/JPY — 日元走强(USD/JPY下跌) = 避险 = 利好黄金
        if usd_jpy is not None and len(usd_jpy) >= window:
            jpy_return = (usd_jpy[-1] - usd_jpy[-window]) / usd_jpy[-window]
            signals.append(-np.clip(jpy_return * 8, -1.0, 1.0))

        if not signals:
            return 0.0

        return np.clip(np.mean(signals), -1.0, 1.0)

    def _calculate_safe_haven_ratio(
        self,
        sp500: Optional[np.ndarray],
        gold_prices: np.ndarray,
    ) -> float:
        """
        计算避险需求信号

        当股市下跌时，黄金作为避险资产通常会上涨。

        Args:
            sp500: 标普500价格数组
            gold_prices: 黄金价格数组

        Returns:
            避险信号 [-1, 1]，正值表示避险需求增加
        """
        if sp500 is None or len(sp500) < self.safe_haven_window:
            return 0.0

        window = self.safe_haven_window

        sp500_return = (sp500[-1] - sp500[-window]) / sp500[-window]
        gold_return = (gold_prices[-1] - gold_prices[-window]) / gold_prices[-window]

        if sp500_return < -0.02:
            if gold_return > 0:
                return np.clip(0.5 + abs(gold_return) * 10, 0.5, 1.0)
            else:
                return np.clip(-0.2 - sp500_return * 2, -0.5, 0.0)
        elif sp500_return > 0.02:
            if gold_return < 0:
                return np.clip(-0.3 - abs(gold_return) * 5, -1.0, -0.3)
            else:
                return 0.0
        else:
            return 0.0
