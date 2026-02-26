"""
宏观因子（第三步：验证判断 — 宏观与跨资产部分）

Fed模型、20法则、巴菲特指标、利率环境、VIX
宏观因子为全局统一分数，所有股票共享。
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger


class MacroFactors:
    """
    宏观环境因子计算器

    基于宏观经济数据评估整体市场的估值环境。
    输出全局统一的 macro_score，作用于所有股票。
    """

    FACTOR_WEIGHTS = {
        "fed_model": 0.25,
        "rule_of_20": 0.20,
        "buffett_indicator": 0.25,
        "yield_curve": 0.15,
        "vix": 0.15,
    }

    def calculate(
        self,
        macro_data: Dict[str, Any],
        market_pe: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        计算宏观因子

        Args:
            macro_data: 宏观数据快照 {
                "DGS10": float,       # 10年期国债收益率
                "DGS2": float,        # 2年期国债收益率
                "FEDFUNDS": float,    # 联邦基金利率
                "VIXCLS": float,      # VIX
                "BAA10Y": float,      # 信用利差
                "T10YIE": float,      # 盈亏平衡通胀率
                "yield_curve_spread": float,
            }
            market_pe: 标普500 市盈率（用于 Fed 模型和 20 法则）

        Returns:
            {"macro_score": float, "factors": {...}, "details": {...}}
        """
        factors = {}
        details = {}

        # Fed 模型
        fed_score = self._fed_model(
            market_pe=market_pe,
            dgs10=macro_data.get("DGS10"),
        )
        if fed_score is not None:
            factors["fed_model"] = fed_score
            details["earnings_yield"] = round(1 / market_pe * 100, 2) if market_pe and market_pe > 0 else None
            details["treasury_10y"] = macro_data.get("DGS10")

        # 20 法则
        rule20_score = self._rule_of_20(
            market_pe=market_pe,
            inflation=macro_data.get("T10YIE"),
        )
        if rule20_score is not None:
            factors["rule_of_20"] = rule20_score
            details["rule_of_20_sum"] = round(
                (market_pe or 0) + (macro_data.get("T10YIE") or 0), 2
            )

        # 巴菲特指标（需要 Wilshire 5000 / GDP 数据）
        buffett_score = self._buffett_indicator(macro_data)
        if buffett_score is not None:
            factors["buffett_indicator"] = buffett_score
            details["buffett_ratio"] = macro_data.get("buffett_ratio")

        # 收益率曲线
        yc_score = self._yield_curve_score(macro_data)
        if yc_score is not None:
            factors["yield_curve"] = yc_score
            details["yield_curve_spread"] = macro_data.get("yield_curve_spread")

        # VIX
        vix_score = self._vix_score(macro_data.get("VIXCLS"))
        if vix_score is not None:
            factors["vix"] = vix_score
            details["vix"] = macro_data.get("VIXCLS")

        # 加权综合
        macro_score = 0.0
        total_weight = 0.0
        for key, weight in self.FACTOR_WEIGHTS.items():
            if key in factors and factors[key] is not None:
                macro_score += factors[key] * weight
                total_weight += weight

        if total_weight > 0:
            macro_score /= total_weight

        return {
            "macro_score": float(np.clip(macro_score, -1, 1)),
            "factors": factors,
            "details": details,
        }

    @staticmethod
    def _fed_model(
        market_pe: Optional[float],
        dgs10: Optional[float],
    ) -> Optional[float]:
        """
        Fed 模型

        盈利收益率 (1/PE) vs 10年期国债收益率
        盈利收益率 >> 国债 → 股票便宜（低估）→ 负分
        盈利收益率 << 国债 → 股票贵（高估）→ 正分
        """
        if market_pe is None or dgs10 is None or market_pe <= 0:
            return None

        earnings_yield = (1 / market_pe) * 100  # 百分比
        spread = earnings_yield - dgs10  # 正值 = 股票便宜

        # spread > 2% → 股票有吸引力 (低估)
        # spread < -1% → 债券更有吸引力 (高估)
        return float(np.clip(-np.tanh(spread / 2), -1, 1))

    @staticmethod
    def _rule_of_20(
        market_pe: Optional[float],
        inflation: Optional[float],
    ) -> Optional[float]:
        """
        20 法则

        PE + 通胀率 > 20 → 市场过热 (高估)
        PE + 通胀率 < 20 → 市场低估
        """
        if market_pe is None or inflation is None:
            return None

        rule_sum = market_pe + inflation
        deviation = rule_sum - 20

        # deviation > 5 → 明显高估, < -5 → 明显低估
        return float(np.clip(np.tanh(deviation / 5), -1, 1))

    @staticmethod
    def _buffett_indicator(macro_data: Dict[str, Any]) -> Optional[float]:
        """
        巴菲特指标

        总市值 / GDP
        > 150% → 强烈高估
        100-150% → 偏高估
        75-100% → 合理
        < 75% → 低估
        """
        ratio = macro_data.get("buffett_ratio")
        if ratio is None:
            return None

        if ratio > 200:
            return 0.9
        elif ratio > 150:
            return 0.5
        elif ratio > 100:
            return 0.2
        elif ratio > 75:
            return -0.2
        elif ratio > 50:
            return -0.5
        else:
            return -0.8

    @staticmethod
    def _yield_curve_score(macro_data: Dict[str, Any]) -> Optional[float]:
        """
        收益率曲线因子

        正常正斜率 → 经济扩张 → 中性
        倒挂（负斜率）→ 衰退信号 → 偏高估（风险高）
        """
        spread = macro_data.get("yield_curve_spread")
        if spread is None:
            return None

        # 倒挂 (spread < 0) → 经济风险 → 偏高估
        if spread < -0.5:
            return 0.5
        elif spread < 0:
            return 0.2
        elif spread > 1.0:
            return -0.2  # 正常斜率 → 偏有利
        else:
            return 0.0

    @staticmethod
    def _vix_score(vix: Optional[float]) -> Optional[float]:
        """
        VIX 恐慌指数因子

        VIX < 15 → 极度乐观（可能高估）
        VIX 15-20 → 正常
        VIX 20-30 → 紧张
        VIX > 30 → 恐慌（可能低估）
        VIX > 40 → 极度恐慌（低估信号）
        """
        if vix is None:
            return None

        if vix > 40:
            return -0.7  # 极度恐慌 → 低估
        elif vix > 30:
            return -0.4
        elif vix > 25:
            return -0.1
        elif vix > 18:
            return 0.0
        elif vix > 12:
            return 0.2
        else:
            return 0.5   # 过度自满 → 可能高估
