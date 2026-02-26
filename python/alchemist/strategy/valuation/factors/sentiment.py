"""
情绪因子（第三步：验证判断 — 市场情绪部分）

RSI、成交量异常、波动率偏离、空头持仓、内部人交易、价格动量
"""

from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


class SentimentFactors:
    """
    市场情绪因子计算器

    结合技术指标和市场微观结构数据评估市场情绪。
    """

    FACTOR_WEIGHTS = {
        "rsi": 0.20,
        "volume_anomaly": 0.15,
        "volatility": 0.10,
        "short_interest": 0.20,
        "insider_trading": 0.20,
        "momentum": 0.15,
    }

    def calculate(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        horizon_days: int = 63,
        short_data: Optional[Dict[str, Any]] = None,
        insider_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        计算情绪因子

        Args:
            prices: 收盘价数组（按时间正序）
            volumes: 成交量数组
            horizon_days: 投资时间窗口对应的交易日数
            short_data: yfinance 空头持仓数据
            insider_data: yfinance 内部人交易数据

        Returns:
            {"sentiment_score": float, "factors": {...}, "details": {...}}
        """
        if len(prices) < 30:
            return {"sentiment_score": 0.0, "factors": {}, "details": {}}

        factors = {}
        details = {}

        # RSI
        rsi_val = self._calculate_rsi(prices)
        if rsi_val is not None:
            factors["rsi"] = self._rsi_to_score(rsi_val)
            details["rsi"] = round(rsi_val, 2)

        # 成交量异常
        vol_score, vol_detail = self._volume_anomaly(volumes)
        if vol_score is not None:
            factors["volume_anomaly"] = vol_score
            details["volume_ratio"] = vol_detail

        # 波动率偏离
        vol_dev = self._volatility_deviation(prices, horizon_days)
        if vol_dev is not None:
            factors["volatility"] = vol_dev
            details["volatility_ratio"] = round(vol_dev, 2)

        # 空头持仓
        if short_data:
            factors["short_interest"] = self._short_interest_score(short_data)
            details["short_percent"] = short_data.get("short_percent_of_float")
            details["short_ratio"] = short_data.get("short_ratio")

        # 内部人交易
        if insider_data:
            factors["insider_trading"] = self._insider_score(insider_data)
            details["insider_net_direction"] = insider_data.get("net_direction")

        # 价格动量
        if len(prices) > horizon_days:
            momentum = (prices[-1] - prices[-horizon_days]) / prices[-horizon_days]
            factors["momentum"] = self._momentum_to_score(momentum)
            details["momentum_return"] = round(momentum, 4)

        # 加权综合
        sentiment_score = 0.0
        total_weight = 0.0
        for key, weight in self.FACTOR_WEIGHTS.items():
            if key in factors and factors[key] is not None:
                sentiment_score += factors[key] * weight
                total_weight += weight

        if total_weight > 0:
            sentiment_score /= total_weight

        return {
            "sentiment_score": float(np.clip(sentiment_score, -1, 1)),
            "factors": factors,
            "details": details,
        }

    @staticmethod
    def _calculate_rsi(prices: np.ndarray, period: int = 14) -> Optional[float]:
        """计算 RSI"""
        if len(prices) < period + 1:
            return None

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    @staticmethod
    def _rsi_to_score(rsi: float) -> float:
        """
        RSI → 情绪分数

        RSI < 20 → -0.8 (极度超卖/低估)
        RSI < 30 → -0.5
        RSI 30-70 → 线性映射 [-0.2, 0.2]
        RSI > 70 → +0.5
        RSI > 80 → +0.8 (极度超买/高估)
        """
        if rsi < 20:
            return -0.8
        elif rsi < 30:
            return -0.5 + (rsi - 20) * 0.03  # -0.5 to -0.2
        elif rsi <= 70:
            return (rsi - 50) / 50 * 0.2  # -0.2 to 0.2
        elif rsi < 80:
            return 0.2 + (rsi - 70) * 0.03  # 0.2 to 0.5
        else:
            return 0.8

    @staticmethod
    def _volume_anomaly(volumes: np.ndarray, period: int = 20) -> tuple:
        """
        成交量异常检测

        Returns:
            (score, volume_ratio)
        """
        if len(volumes) < period + 1:
            return None, None

        recent_vol = volumes[-1]
        avg_vol = np.mean(volumes[-period - 1:-1])

        if avg_vol == 0:
            return None, None

        ratio = recent_vol / avg_vol

        # 成交量放大本身是中性的，需要结合价格方向
        # 这里简单处理：异常高量 → 情绪极端
        if ratio > 3.0:
            score = 0.4  # 异常放量，可能过热
        elif ratio > 2.0:
            score = 0.2
        elif ratio < 0.3:
            score = -0.2  # 缩量，缺乏关注（可能低估）
        else:
            score = 0.0

        return score, round(ratio, 2)

    @staticmethod
    def _volatility_deviation(
        prices: np.ndarray,
        horizon_days: int,
    ) -> Optional[float]:
        """波动率偏离（近期 vs 历史）"""
        if len(prices) < horizon_days + 20:
            return None

        returns = np.diff(np.log(prices))

        recent_vol = np.std(returns[-20:]) * np.sqrt(252)
        hist_vol = np.std(returns[-horizon_days:]) * np.sqrt(252)

        if hist_vol == 0:
            return None

        ratio = recent_vol / hist_vol

        # 波动率大幅上升 → 恐慌（可能低估），但也可能是泡沫破裂
        # 简化：高波动率视为情绪极端信号
        if ratio > 2.0:
            return 0.3  # 高波动（不确定方向）
        elif ratio < 0.5:
            return -0.1  # 低波动（温和）
        else:
            return 0.0

    @staticmethod
    def _short_interest_score(short_data: Dict[str, Any]) -> float:
        """
        空头持仓因子

        极高空头 → 过度悲观 → 低估信号 (负分)
        极低空头 → 市场共识太乐观 → 高估信号 (正分)
        """
        short_pct = short_data.get("short_percent_of_float")
        if short_pct is None:
            return 0.0

        # short_percent_of_float 通常为小数，如 0.05 = 5%
        if isinstance(short_pct, (int, float)):
            pct = float(short_pct)
            # 如果值 > 1，可能是百分比形式 (5 = 5%)
            if pct > 1:
                pct = pct / 100

            if pct > 0.30:
                return -0.8  # 极高空头 → 可能轧空（低估）
            elif pct > 0.15:
                return -0.4
            elif pct > 0.05:
                return -0.1
            elif pct < 0.01:
                return 0.3   # 几乎无空头 → 共识太乐观
            else:
                return 0.0

        return 0.0

    @staticmethod
    def _insider_score(insider_data: Dict[str, Any]) -> float:
        """
        内部人交易因子

        集体增持 → 看多 → 低估 (负分)
        集体减持 → 看空 → 高估 (正分)
        """
        net_direction = insider_data.get("net_direction", 0.0)
        total = insider_data.get("total_transactions", 0)

        if total == 0:
            return 0.0

        # net_direction: +1全部买入, -1全部卖出
        # 反转映射：买入多 = 低估信号
        return float(np.clip(-net_direction * 0.6, -0.6, 0.6))

    @staticmethod
    def _momentum_to_score(momentum: float) -> float:
        """
        价格动量因子

        大幅上涨 → 可能过热 (高估)
        大幅下跌 → 可能超跌 (低估)
        """
        # 动量越高，越可能高估
        return float(np.clip(np.tanh(momentum * 3), -1, 1))
