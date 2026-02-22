"""
每周定投基准策略

在每周固定一天买入等金额的标的，作为基准策略与主策略进行比较。
将初始资金平均分配到整个投资期限的每一周，保证资金不会提前耗尽。
"""

import math
from datetime import datetime
from typing import Dict, List, Any, Optional

from strategy.base import Strategy, Signal, SignalType
from core.asset import Asset
from core.portfolio import Portfolio
from data.models import MarketData


# 星期几名称映射
WEEKDAY_NAMES = {
    0: "周一",
    1: "周二",
    2: "周三",
    3: "周四",
    4: "周五",
}

WEEKDAY_NAMES_EN = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
}


class WeeklyDCAStrategy(Strategy):
    """
    每周定投策略 (Dollar-Cost Averaging)

    在每周固定一天买入固定金额的标的。
    仅做多，不做空，不卖出。

    资金分配：将剩余现金平均分配到剩余的投资周数中，
    保证整个投资期限内每周都能买入，不会提前耗尽资金。

    Args:
        target_day: 目标星期几 (0=周一, 4=周五)
        end_date: 投资截止日期，用于计算剩余周数以平摊资金
    """

    def __init__(
        self,
        target_day: int = 0,
        end_date: Optional[datetime] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        if target_day < 0 or target_day > 4:
            raise ValueError(f"target_day 必须在 0-4 之间，当前为 {target_day}")

        day_name = WEEKDAY_NAMES.get(target_day, str(target_day))
        day_name_en = WEEKDAY_NAMES_EN.get(target_day, str(target_day))
        super().__init__(f"DCA-{day_name_en}", params)

        self.target_day = target_day
        self.end_date = end_date
        self._day_name = day_name
        self._last_buy_week: Optional[int] = None

    @property
    def required_history(self) -> int:
        return 1

    def _compute_strength(
        self, ts: datetime, portfolio: Portfolio,
    ) -> float:
        """
        动态计算信号强度，使剩余现金平均分配到剩余周数。

        engine position_sizer 公式:
            position_value = min(net_value, initial_capital) * 0.1 * strength
        我们需要:
            position_value = portfolio.cash / remaining_weeks
        所以:
            strength = portfolio.cash / remaining_weeks
                       / (min(net_value, initial_capital) * 0.1)

        简化：base_value ≈ initial_capital（买入持有策略净值通常 >= 初始资金）
            strength ≈ portfolio.cash / remaining_weeks / (initial_capital * 0.1)
        """
        if self.end_date is None:
            # 未指定截止日期，回退到固定比例
            return 0.1

        remaining_days = (self.end_date - ts).days
        remaining_weeks = max(1, remaining_days / 7)

        per_week_amount = portfolio.cash / remaining_weeks
        base_value = portfolio.initial_capital * 0.1

        if base_value <= 0:
            return 0.1

        return per_week_amount / base_value

    def generate_signals(
        self,
        asset: Asset,
        data: MarketData,
        portfolio: Portfolio,
    ) -> List[Signal]:
        """
        生成交易信号

        仅在目标星期几生成买入信号，每周最多买入一次。
        """
        if data.is_empty:
            return []

        latest = data.latest
        ts = latest.timestamp
        weekday = ts.weekday()

        if weekday != self.target_day:
            return []

        # 防止同一周重复买入（ISO 周号 + 年份）
        iso_year, iso_week, _ = ts.isocalendar()
        week_key = iso_year * 100 + iso_week
        if self._last_buy_week == week_key:
            return []

        # 无剩余现金则跳过
        if portfolio.cash <= 0:
            return []

        self._last_buy_week = week_key
        strength = self._compute_strength(ts, portfolio)

        return [
            Signal(
                signal_type=SignalType.BUY,
                asset=asset,
                timestamp=ts,
                strength=strength,
                metadata={
                    "strategy_id": self.name,
                    "weekday": self._day_name,
                },
            )
        ]

    def reset(self) -> None:
        super().reset()
        self._last_buy_week = None
