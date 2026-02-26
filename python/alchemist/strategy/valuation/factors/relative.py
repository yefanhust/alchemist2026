"""
相对估值因子（第一步：快速筛选）

将个股的估值比率与同行业中位数及自身历史百分位对比，
识别估值处于极端位置的股票。

每个指标输出 -1.0（极度低估）到 +1.0（极度高估）
"""

from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


def _zscore_to_score(value: float, median: float, std: float) -> float:
    """
    将 Z-Score 映射到 [-1, 1]

    Z-Score > 0 → 高于中位数（偏高估）
    Z-Score < 0 → 低于中位数（偏低估）
    用 tanh 压缩到 [-1, 1]
    """
    if std == 0 or std is None:
        return 0.0
    z = (value - median) / std
    return float(np.tanh(z / 2))  # /2 使得 z=±2 大致映射到 ±0.76


def _percentile_score(value: float, low: float, high: float) -> float:
    """
    基于百分位映射到 [-1, 1]

    value 在 [low, high] 范围内：
    - 接近 low → -1 (低估)
    - 接近 high → +1 (高估)
    """
    if high <= low:
        return 0.0
    pct = (value - low) / (high - low)
    return float(np.clip(pct * 2 - 1, -1, 1))


class RelativeValuationFactors:
    """
    相对估值因子计算器

    使用行业内 Z-Score 标准化来判断个股的相对估值水平。
    """

    # 各因子在相对估值中的权重
    FACTOR_WEIGHTS = {
        "pe": 0.20,
        "pb": 0.12,
        "ps": 0.10,
        "peg": 0.15,
        "ev_ebitda": 0.13,
        "p_fcf": 0.10,
        "shareholder_yield": 0.10,
        "week52_position": 0.10,
    }

    def calculate(
        self,
        stock_info: Dict[str, Any],
        industry_peers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        计算相对估值因子

        Args:
            stock_info: 目标股票信息（含 pe_ratio, pb_ratio 等）
            industry_peers: 同行业股票信息列表

        Returns:
            {"relative_score": float, "factors": {...}, "details": {...}}
        """
        factors = {}
        details = {}

        # PE 偏离度
        factors["pe"] = self._calc_ratio_factor(
            stock_info.get("pe_ratio"),
            [p.get("pe_ratio") for p in industry_peers],
            "PE Ratio"
        )
        details["pe_ratio"] = stock_info.get("pe_ratio")
        details["pe_industry_median"] = self._median([p.get("pe_ratio") for p in industry_peers])

        # PB 偏离度
        factors["pb"] = self._calc_ratio_factor(
            stock_info.get("pb_ratio"),
            [p.get("pb_ratio") for p in industry_peers],
            "PB Ratio"
        )
        details["pb_ratio"] = stock_info.get("pb_ratio")

        # PS 偏离度
        factors["ps"] = self._calc_ratio_factor(
            stock_info.get("ps_ratio"),
            [p.get("ps_ratio") for p in industry_peers],
            "PS Ratio"
        )
        details["ps_ratio"] = stock_info.get("ps_ratio")

        # PEG
        factors["peg"] = self._calc_peg_factor(stock_info.get("peg_ratio"))
        details["peg_ratio"] = stock_info.get("peg_ratio")

        # EV/EBITDA
        factors["ev_ebitda"] = self._calc_ratio_factor(
            stock_info.get("ev_to_ebitda"),
            [p.get("ev_to_ebitda") for p in industry_peers],
            "EV/EBITDA"
        )
        details["ev_to_ebitda"] = stock_info.get("ev_to_ebitda")

        # P/FCF
        factors["p_fcf"] = self._calc_ratio_factor(
            stock_info.get("price_to_fcf"),
            [p.get("price_to_fcf") for p in industry_peers],
            "P/FCF"
        )

        # 股东收益率（股息率 + 回购）
        factors["shareholder_yield"] = self._calc_shareholder_yield(stock_info)
        details["dividend_yield"] = stock_info.get("dividend_yield")

        # 52周位置
        factors["week52_position"] = self._calc_52week_position(stock_info)
        details["high_52week"] = stock_info.get("high_52week")
        details["low_52week"] = stock_info.get("low_52week")

        # 综合相对估值分数
        relative_score = 0.0
        total_weight = 0.0
        for key, weight in self.FACTOR_WEIGHTS.items():
            if factors.get(key) is not None:
                relative_score += factors[key] * weight
                total_weight += weight

        if total_weight > 0:
            relative_score /= total_weight

        return {
            "relative_score": float(np.clip(relative_score, -1, 1)),
            "factors": factors,
            "details": details,
        }

    def _calc_ratio_factor(
        self,
        value: Optional[float],
        peer_values: List[Optional[float]],
        name: str,
    ) -> Optional[float]:
        """计算估值比率的行业 Z-Score 因子"""
        if value is None or value <= 0:
            return None

        # 过滤有效值
        valid_peers = [v for v in peer_values if v is not None and v > 0]
        if len(valid_peers) < 3:
            return None

        arr = np.array(valid_peers)
        median = float(np.median(arr))
        std = float(np.std(arr))

        if std < 1e-6:
            return 0.0

        # 估值比率越高 → 越高估 → score越大
        return _zscore_to_score(value, median, std)

    def _calc_peg_factor(self, peg: Optional[float]) -> Optional[float]:
        """
        PEG 因子

        PEG < 0.5 → 强烈低估 (-0.8)
        PEG < 1.0 → 低估 (-0.4)
        PEG = 1.0 → 合理 (0)
        PEG > 2.0 → 高估 (+0.4)
        PEG > 3.0 → 强烈高估 (+0.8)
        """
        if peg is None or peg <= 0:
            return None

        if peg < 0.5:
            return -0.8
        elif peg < 1.0:
            return -0.4 + (peg - 0.5) * 0.8  # [-0.4, 0]
        elif peg < 2.0:
            return (peg - 1.0) * 0.4  # [0, 0.4]
        elif peg < 3.0:
            return 0.4 + (peg - 2.0) * 0.4  # [0.4, 0.8]
        else:
            return min(1.0, 0.8 + (peg - 3.0) * 0.1)

    def _calc_shareholder_yield(self, stock_info: Dict[str, Any]) -> Optional[float]:
        """
        股东收益率因子

        股东收益率 = 股息率 + 回购收益率
        高收益率 → 低估信号（负分），低收益率 → 高估信号（正分）
        """
        div_yield = stock_info.get("dividend_yield")
        if div_yield is None:
            return None

        # 简化版：仅用股息率，回购需要历史shares数据
        # 股息率 > 5% → 强烈低估, < 0.5% → 可能高估
        if div_yield > 0.05:
            return -0.6
        elif div_yield > 0.03:
            return -0.3
        elif div_yield > 0.015:
            return 0.0
        elif div_yield > 0.005:
            return 0.2
        else:
            return 0.4

    def _calc_52week_position(self, stock_info: Dict[str, Any]) -> Optional[float]:
        """52周高低位置因子"""
        high = stock_info.get("high_52week")
        low = stock_info.get("low_52week")
        # 用 EPS 近似当前价 (市盈率 × EPS ≈ 股价)
        pe = stock_info.get("pe_ratio")
        eps = stock_info.get("eps")

        if high is None or low is None or high <= low:
            return None

        # 估算当前价
        if pe is not None and eps is not None and pe > 0 and eps > 0:
            current_price = pe * eps
        else:
            return None

        return _percentile_score(current_price, low, high)

    @staticmethod
    def _median(values: List[Optional[float]]) -> Optional[float]:
        valid = [v for v in values if v is not None and v > 0]
        if not valid:
            return None
        return float(np.median(valid))
