"""
宏观因子信号计算
权重: 0.20

分析宏观经济环境对黄金的影响：实际收益率、货币强度、风险偏好。
使用数据：TREASURY_YIELD（真实收益率）、TIP（通胀预期）、
         TLT（国债价格）、VIXY（VIX 代理）、SPY、UUP、EUR/USD、USD/JPY。
"""

from typing import Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class MacroSignalResult:
    """宏观信号结果"""
    real_yield: float           # 实际收益率信号 [-1, 1]
    currency_strength: float    # 货币强度信号 [-1, 1]
    risk_on_off: float          # 风险偏好信号 [-1, 1]
    composite: float            # 综合宏观信号 [-1, 1]


class MacroSignals:
    """
    宏观环境分析

    分析以下宏观因子：
    - 实际收益率（TREASURY_YIELD - TIP 通胀预期）
    - 货币强度（UUP、EUR/USD、USD/JPY）
    - 风险偏好（VIXY、SPY vs TLT）
    """

    def __init__(
        self,
        lookback_period: int = 20,
        yield_sensitivity: float = 1.0,
    ):
        self.lookback_period = lookback_period
        self.yield_sensitivity = yield_sensitivity

    def calculate(
        self,
        treasury_yield: Optional[np.ndarray] = None,
        inflation_expectations: Optional[np.ndarray] = None,
        usd_index: Optional[np.ndarray] = None,
        eur_usd: Optional[np.ndarray] = None,
        usd_jpy: Optional[np.ndarray] = None,
        vix: Optional[np.ndarray] = None,
        sp500: Optional[np.ndarray] = None,
        treasury_prices: Optional[np.ndarray] = None,
        **_kwargs,
    ) -> MacroSignalResult:
        """
        计算宏观环境信号

        Args:
            treasury_yield: 国债收益率数组（真实 yield 百分比，如 4.25）
            inflation_expectations: TIP ETF 价格数组（通胀预期代理）
            usd_index: UUP ETF 价格数组
            eur_usd: EUR/USD 汇率数组
            usd_jpy: USD/JPY 汇率数组
            vix: VIXY ETF 价格数组（VIX 代理）
            sp500: SPY ETF 价格数组
            treasury_prices: TLT ETF 价格数组

        Returns:
            MacroSignalResult 对象
        """
        real_yield_signal = self._calculate_real_yield_signal(
            treasury_yield, inflation_expectations
        )
        currency_signal = self._analyze_currency_basket(
            usd_index, eur_usd, usd_jpy
        )
        risk_signal = self._calculate_risk_appetite(
            vix, sp500, treasury_prices
        )

        # 综合宏观信号（按可用数据动态加权）
        period = self.lookback_period
        weights = {
            "real_yield": (real_yield_signal, 0.40,
                           (treasury_yield is not None and len(treasury_yield) >= period)
                           or (inflation_expectations is not None and len(inflation_expectations) >= period)),
            "currency": (currency_signal, 0.30,
                         any(arr is not None and len(arr) >= period
                             for arr in [usd_index, eur_usd, usd_jpy])),
            "risk": (risk_signal, 0.30,
                     any(arr is not None and len(arr) >= period
                         for arr in [vix, sp500, treasury_prices])),
        }

        total_weight = sum(w for _, w, avail in weights.values() if avail)
        if total_weight > 0:
            composite = sum(sig * w for sig, w, avail in weights.values() if avail) / total_weight
        else:
            composite = 0.0

        return MacroSignalResult(
            real_yield=real_yield_signal,
            currency_strength=currency_signal,
            risk_on_off=risk_signal,
            composite=np.clip(composite, -1.0, 1.0),
        )

    def _calculate_real_yield_signal(
        self,
        treasury_yield: Optional[np.ndarray],
        inflation_expectations: Optional[np.ndarray],
    ) -> float:
        """
        计算实际收益率信号

        黄金无收益，当实际利率低（甚至为负）时，
        持有黄金的机会成本降低，利好黄金。

        - treasury_yield: 真实国债收益率百分比（如 4.25 = 4.25%）
        - inflation_expectations: TIP ETF 价格（上涨 = 通胀预期上升 = 利好黄金）

        Returns:
            实际收益率信号 [-1, 1]，负实际利率时信号为正
        """
        period = self.lookback_period
        signals = []

        # 方案 A: 使用真实国债收益率
        if treasury_yield is not None and len(treasury_yield) >= period:
            current_yield = treasury_yield[-1]

            # 收益率水平信号：高收益率利空黄金
            # 以 3% 为中性水平
            level_signal = -np.clip((current_yield - 3.0) * 0.2, -0.5, 0.5)

            # 收益率变化信号：收益率下降利好黄金
            yield_change = current_yield - treasury_yield[-period]
            change_signal = -np.clip(yield_change * self.yield_sensitivity * 2, -0.5, 0.5)

            signals.append(level_signal + change_signal)

        # 方案 B: 使用 TIP ETF 作为通胀预期代理
        if inflation_expectations is not None and len(inflation_expectations) >= period:
            # TIP 上涨 = 通胀预期上升 = 实际利率下降 = 利好黄金
            tip_return = (inflation_expectations[-1] - inflation_expectations[-period]) / inflation_expectations[-period]
            signals.append(np.clip(tip_return * 10, -0.5, 0.5))

        if not signals:
            return 0.0

        return np.clip(np.mean(signals), -1.0, 1.0)

    def _analyze_currency_basket(
        self,
        usd_index: Optional[np.ndarray],
        eur_usd: Optional[np.ndarray],
        usd_jpy: Optional[np.ndarray],
    ) -> float:
        """
        分析货币篮子强度

        美元走弱通常利好黄金。

        Args:
            usd_index: UUP ETF 价格数组
            eur_usd: EUR/USD 汇率数组
            usd_jpy: USD/JPY 汇率数组

        Returns:
            货币强度信号 [-1, 1]，美元走弱时信号为正
        """
        signals = []
        period = self.lookback_period

        # UUP (美元 ETF) 分析
        if usd_index is not None and len(usd_index) >= period:
            usd_return = (usd_index[-1] - usd_index[-period]) / usd_index[-period]
            signals.append(-np.clip(usd_return * 10, -1.0, 1.0))

            short_period = min(5, period)
            usd_short_return = (usd_index[-1] - usd_index[-short_period]) / usd_index[-short_period]
            signals.append(-np.clip(usd_short_return * 15, -0.5, 0.5))

        # EUR/USD 分析
        if eur_usd is not None and len(eur_usd) >= period:
            eur_return = (eur_usd[-1] - eur_usd[-period]) / eur_usd[-period]
            signals.append(np.clip(eur_return * 10, -0.5, 0.5))

        # USD/JPY 分析
        if usd_jpy is not None and len(usd_jpy) >= period:
            jpy_return = (usd_jpy[-1] - usd_jpy[-period]) / usd_jpy[-period]
            signals.append(-np.clip(jpy_return * 8, -0.5, 0.5))

        return np.mean(signals) if signals else 0.0

    def _calculate_risk_appetite(
        self,
        vix: Optional[np.ndarray],
        sp500: Optional[np.ndarray],
        treasury_prices: Optional[np.ndarray],
    ) -> float:
        """
        计算风险偏好指标

        Risk-off 环境通常利好黄金。
        使用 VIXY ETF 替代 VIX 指数（VIXY 上涨 = 恐慌增加）。

        Args:
            vix: VIXY ETF 价格数组
            sp500: SPY ETF 价格数组
            treasury_prices: TLT ETF 价格数组

        Returns:
            风险偏好信号 [-1, 1]，Risk-off 时信号为正
        """
        signals = []
        period = self.lookback_period

        # VIXY (VIX 代理 ETF) 分析
        if vix is not None and len(vix) >= period:
            # VIXY 上涨 = 恐慌增加 = risk-off = 利好黄金
            vix_return = (vix[-1] - vix[-min(5, period)]) / vix[-min(5, period)]
            signals.append(np.clip(vix_return * 3, -0.5, 0.5))

            # VIXY 相对均值的位置
            avg_vix = np.mean(vix[-period:])
            if avg_vix > 0:
                vix_ratio = vix[-1] / avg_vix
                if vix_ratio > 1.3:
                    signals.append(0.5)   # 恐慌高于均值 30%
                elif vix_ratio < 0.7:
                    signals.append(-0.3)  # 恐慌低于均值 30%
                else:
                    signals.append(0.0)

        # 股债相对表现
        if sp500 is not None and treasury_prices is not None:
            if len(sp500) >= period and len(treasury_prices) >= period:
                sp500_return = (sp500[-1] - sp500[-period]) / sp500[-period]
                tlt_return = (treasury_prices[-1] - treasury_prices[-period]) / treasury_prices[-period]

                # 股票跑输债券 → risk-off → 利好黄金
                relative_return = sp500_return - tlt_return
                signals.append(-np.clip(relative_return * 5, -0.5, 0.5))

        # 仅有股市数据
        elif sp500 is not None and len(sp500) >= period:
            sp500_return = (sp500[-1] - sp500[-period]) / sp500[-period]

            if sp500_return < -0.05:
                signals.append(np.clip(-sp500_return * 5, 0.0, 0.8))
            elif sp500_return > 0.05:
                signals.append(-0.2)
            else:
                signals.append(0.0)

        return np.mean(signals) if signals else 0.0
