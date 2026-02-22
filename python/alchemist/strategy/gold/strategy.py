"""
黄金择时增强型定投策略

核心思路：以每周定投为基础，用多因子综合得分调节买入量，
在极端看空时才卖出部分仓位。

与纯择时策略的区别：
- 不会长期空仓，在趋势市场中保持参与
- 因子得分影响买入量而非买/不买的二元决策
- 仅在极端信号下部分减仓，避免频繁交易
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np
import yaml

from strategy.base import Strategy, Signal, SignalType
from core.asset import Asset
from core.portfolio import Portfolio
from data.models import MarketData

from .signals import (
    TechnicalSignals,
    CrossMarketSignals,
    SentimentSignals,
    MacroSignals,
)


@dataclass
class TacticalThresholds:
    """战术调整阈值"""
    boost_buy: float = 0.3       # 综合得分 > 0.3 → 加量买入
    normal_buy: float = 0.0      # 得分 > 0.0 → 正常买入
    reduce_buy: float = -0.15    # 得分 > -0.15 → 减量买入
    skip_buy: float = -0.3       # 得分 > -0.3 → 完全不买入；≤ -0.3 → 跳过+可能卖出
    partial_sell: float = -0.6   # 持仓时得分 < -0.6 → 卖出部分仓位


@dataclass
class FactorWeights:
    """因子权重配置（基于实际可用数据）"""
    technical: float = 0.40      # 技术面（GLD 价量数据最完整）
    cross_market: float = 0.25   # 跨市场（UUP, SPY, EUR/USD, USD/JPY）
    sentiment: float = 0.15      # 情绪（GDX vs GLD 相对强度）
    macro: float = 0.20          # 宏观（TIP, TLT, TREASURY_YIELD, VIXY）

    def __post_init__(self):
        total = self.technical + self.cross_market + self.sentiment + self.macro
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"权重之和必须为 1.0，当前为 {total}")


@dataclass
class PositionConfig:
    """仓位管理配置"""
    buy_day: int = 2                 # 每周买入日 (0=周一, 2=周三, 4=周五)
    boost_multiplier: float = 2.0    # 加量买入倍率（相对于基准）
    normal_multiplier: float = 1.0   # 正常买入倍率
    reduce_multiplier: float = 0.5   # 减量买入倍率
    sell_fraction: float = 0.50      # 部分卖出比例
    min_hold_days: int = 10          # 最小持有交易日
    cooldown_days: int = 3           # 交易后冷却交易日
    # 强制止盈配置
    force_sell_interval_days: int = 180    # 强制卖出间隔天数（约半年一次）
    force_sell_fraction: float = 0.30      # 强制卖出比例
    force_sell_profit_threshold: float = 0.05  # 仅在持仓盈利超过此比例时强制卖出


class GoldTradingStrategy(Strategy):
    """
    黄金择时增强型定投策略

    每周固定日进行定投，多因子得分调节买入金额：
    - 综合得分 > 0.3   → 加量买入 (基准 × 2.0)
    - 综合得分 > 0.0   → 正常买入 (基准 × 1.0)
    - 综合得分 > -0.15  → 减量买入 (基准 × 0.5)
    - 综合得分 > -0.3   → 完全不买入
    - 综合得分 ≤ -0.3   → 跳过本次买入
    - 综合得分 < -0.6 且持仓 → 卖出 50% 仓位

    强制止盈：每隔一定天数（默认180天），若持仓盈利超过阈值，
    强制卖出部分仓位以落袋为安（每年约1-2次）。

    基准买入金额 = 剩余现金 / 剩余周数，保证资金平摊到投资期限。

    因子权重（基于实际可用数据）：
    - 技术面 40%: GLD 价量数据
    - 跨市场 25%: UUP, SPY, EUR/USD, USD/JPY
    - 情绪 15%: GDX vs GLD 相对强度
    - 宏观 20%: TREASURY_YIELD, TIP, TLT, VIXY
    """

    def __init__(
        self,
        weights: Optional[FactorWeights] = None,
        thresholds: Optional[TacticalThresholds] = None,
        position_config: Optional[PositionConfig] = None,
        params: Optional[Dict[str, Any]] = None,
        cross_market_symbols: Optional[Dict[str, str]] = None,
        end_date: Optional[datetime] = None,
    ):
        default_params = {
            "trend_periods": [20, 50, 200],
            "momentum_periods": [14, 21],
            "correlation_window": 30,
            "lookback_period": 20,
        }
        if params:
            default_params.update(params)

        super().__init__("GoldTacticalDCA", default_params)

        self.weights = weights or FactorWeights()
        self.thresholds = thresholds or TacticalThresholds()
        self.position_config = position_config or PositionConfig()
        self.end_date = end_date

        # 初始化信号计算器
        self.technical_signals = TechnicalSignals(
            trend_periods=default_params["trend_periods"],
            momentum_periods=default_params["momentum_periods"],
        )
        self.cross_market_signals = CrossMarketSignals(
            correlation_window=default_params["correlation_window"],
        )
        self.sentiment_signals = SentimentSignals()
        self.macro_signals = MacroSignals(
            lookback_period=default_params["lookback_period"],
        )

        # 跨市场 symbol 映射
        self.cross_market_symbols = cross_market_symbols or {}

        # 标记是否从优化参数加载
        self._loaded_from: Optional[str] = None

        # 状态追踪
        self._has_position: Dict[str, bool] = {}
        self._last_trade_date: Optional[datetime] = None
        self._last_buy_week: Optional[int] = None
        self._trade_count: int = 0
        # 强制止盈追踪
        self._last_sell_date: Optional[datetime] = None
        self._position_cost_basis: float = 0.0  # 加权平均买入价
        self._position_quantity: float = 0.0     # 当前持仓数量

    @classmethod
    def from_optimized_params(
        cls,
        yaml_path: str,
        cross_market_symbols: Optional[Dict[str, str]] = None,
        end_date: Optional[datetime] = None,
    ) -> "GoldTradingStrategy":
        """
        从优化结果 YAML 文件加载策略参数

        Args:
            yaml_path: gold_optimized_params.yaml 路径
            cross_market_symbols: 跨市场 symbol 映射
            end_date: 回测截止日期

        Returns:
            使用优化参数的策略实例
        """
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        p = data["params"]
        strategy = cls(
            weights=FactorWeights(
                technical=p["w_technical"],
                cross_market=p["w_cross_market"],
                sentiment=p["w_sentiment"],
                macro=p["w_macro"],
            ),
            thresholds=TacticalThresholds(
                boost_buy=p["thresh_boost"],
                normal_buy=p["thresh_normal"],
                reduce_buy=p["thresh_reduce"],
                skip_buy=p["thresh_skip"],
                partial_sell=p["thresh_sell"],
            ),
            position_config=PositionConfig(
                buy_day=int(p["buy_day"]),
                boost_multiplier=p["boost_multiplier"],
                reduce_multiplier=p["reduce_multiplier"],
                sell_fraction=p["sell_fraction"],
                force_sell_interval_days=int(p["force_sell_interval"]),
                force_sell_fraction=p["force_sell_fraction"],
                force_sell_profit_threshold=p["force_sell_profit_thresh"],
            ),
            cross_market_symbols=cross_market_symbols,
            end_date=end_date,
        )
        strategy._loaded_from = yaml_path
        return strategy

    @property
    def required_history(self) -> int:
        return max(
            max(self.params.get("trend_periods", [200])),
            self.params.get("correlation_window", 30),
            self.params.get("lookback_period", 20),
        ) + 10

    # ------------------------------------------------------------------
    # 主入口：on_data
    # ------------------------------------------------------------------

    def on_data(
        self,
        timestamp: datetime,
        data: Dict[str, 'MarketData'],
        portfolio: 'Portfolio',
    ) -> List[Signal]:
        """处理新数据，生成交易信号"""
        if not self.cross_market_symbols:
            return super().on_data(timestamp, data, portfolio)

        primary = "GLD"
        if primary not in data or len(data[primary]) < self.required_history:
            return []

        # 使用 MarketData 的增量缓存数组，避免每次 O(N) 重建
        cross_data = {}
        for data_key, ticker in self.cross_market_symbols.items():
            if ticker in data and len(data[ticker]) > 0:
                cross_data[data_key] = data[ticker].close_array

        asset = Asset(symbol=primary)
        signals = self._generate_tactical_signals(
            asset, data[primary], cross_data, portfolio,
        )

        self.signals.extend(signals)
        return signals

    # ------------------------------------------------------------------
    # 核心：择时增强型定投信号生成
    # ------------------------------------------------------------------

    def _generate_tactical_signals(
        self,
        asset: Asset,
        data: MarketData,
        cross_market_data: Dict[str, np.ndarray],
        portfolio: Portfolio,
    ) -> List[Signal]:
        """
        生成择时增强型定投信号

        每周固定日买入，因子得分调节买入量；
        极端看空时部分减仓。
        """
        if len(data) < self.required_history:
            return []

        timestamp = data.latest.timestamp
        weekday = timestamp.weekday()

        # 使用 MarketData 的增量缓存数组
        closes = data.close_array
        volumes = data.volume_array if hasattr(data.data[0], 'volume') else None

        market_data = {
            "gold_etf": closes,
            "gold_etf_volume": volumes,
            **cross_market_data,
        }
        factors = self.calculate_factors(market_data)
        composite_score = self._compute_composite_score(factors)

        # 检查冷却期
        if self._in_cooldown(timestamp):
            return []

        has_position = self._has_position.get(asset.symbol, False)
        signals = []

        current_price = closes[-1]

        # ---- 买入逻辑：每周固定日 ----
        if weekday == self.position_config.buy_day:
            # 防止同一周重复买入
            iso_year, iso_week, _ = timestamp.isocalendar()
            week_key = iso_year * 100 + iso_week
            if self._last_buy_week != week_key:
                base_strength = self._compute_base_strength(timestamp, portfolio)
                strength = self._get_buy_strength(composite_score, base_strength)
                if strength > 0 and portfolio.cash > 0:
                    signals.append(Signal(
                        signal_type=SignalType.BUY,
                        asset=asset,
                        timestamp=timestamp,
                        strength=strength,
                        metadata={
                            "strategy_id": self.name,
                            "composite_score": composite_score,
                            "factor_scores": factors,
                            "tactical_action": self._describe_action(composite_score),
                        },
                    ))
                    # 更新持仓成本基础（加权平均）
                    buy_value = strength * portfolio.initial_capital * 0.1
                    buy_qty = buy_value / current_price if current_price > 0 else 0
                    total_cost = self._position_cost_basis * self._position_quantity + buy_value
                    self._position_quantity += buy_qty
                    if self._position_quantity > 0:
                        self._position_cost_basis = total_cost / self._position_quantity
                    self._has_position[asset.symbol] = True
                    self._last_trade_date = timestamp
                    self._last_buy_week = week_key
                    self._trade_count = 0

        # ---- 卖出逻辑：极端看空时部分减仓 ----
        if (has_position
                and composite_score < self.thresholds.partial_sell
                and self._past_min_hold(timestamp)):
            signals.append(Signal(
                signal_type=SignalType.SELL,
                asset=asset,
                timestamp=timestamp,
                strength=self.position_config.sell_fraction,
                metadata={
                    "strategy_id": self.name,
                    "composite_score": composite_score,
                    "factor_scores": factors,
                    "tactical_action": "partial_sell",
                },
            ))
            # 卖出后更新持仓数量（成本基础不变）
            self._position_quantity *= (1 - self.position_config.sell_fraction)
            self._last_trade_date = timestamp
            self._last_sell_date = timestamp
            self._trade_count = 0

        # ---- 强制止盈逻辑：定期落袋为安 ----
        elif (has_position
                and self._should_force_sell(timestamp, current_price)):
            cfg = self.position_config
            signals.append(Signal(
                signal_type=SignalType.SELL,
                asset=asset,
                timestamp=timestamp,
                strength=cfg.force_sell_fraction,
                metadata={
                    "strategy_id": self.name,
                    "composite_score": composite_score,
                    "factor_scores": factors,
                    "tactical_action": "force_take_profit",
                },
            ))
            self._position_quantity *= (1 - cfg.force_sell_fraction)
            self._last_trade_date = timestamp
            self._last_sell_date = timestamp
            self._trade_count = 0

        return signals

    # ------------------------------------------------------------------
    # 因子计算
    # ------------------------------------------------------------------

    def calculate_factors(
        self,
        data: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """计算所有因子得分"""
        factors = {}

        # 技术面因子
        if "gold_etf" in data:
            tech_result = self.technical_signals.calculate(
                prices=data["gold_etf"],
                volumes=data.get("gold_etf_volume"),
            )
            factors["technical"] = tech_result.composite
        else:
            factors["technical"] = 0.0

        # 跨市场联动因子
        cross_result = self.cross_market_signals.calculate(
            gold_prices=data.get("gold_etf", np.array([])),
            usd_index=data.get("usd_index"),
            sp500=data.get("sp500"),
            eur_usd=data.get("eur_usd"),
            usd_jpy=data.get("usd_jpy"),
        )
        factors["cross_market"] = cross_result.composite

        # 市场情绪因子
        sentiment_result = self.sentiment_signals.calculate(
            gold_etf_prices=data.get("gold_etf", np.array([])),
            gold_miners_prices=data.get("gold_miners"),
            gold_etf_volume=data.get("gold_etf_volume"),
        )
        factors["sentiment"] = sentiment_result.composite

        # 宏观因子
        macro_result = self.macro_signals.calculate(
            treasury_yield=data.get("treasury_yield"),
            inflation_expectations=data.get("inflation_expectations"),
            usd_index=data.get("usd_index"),
            eur_usd=data.get("eur_usd"),
            usd_jpy=data.get("usd_jpy"),
            vix=data.get("vix"),
            sp500=data.get("sp500"),
            treasury_prices=data.get("treasury"),
        )
        factors["macro"] = macro_result.composite

        return factors

    def _compute_composite_score(self, factors: Dict[str, float]) -> float:
        """计算综合得分"""
        return (
            factors.get("technical", 0) * self.weights.technical +
            factors.get("cross_market", 0) * self.weights.cross_market +
            factors.get("sentiment", 0) * self.weights.sentiment +
            factors.get("macro", 0) * self.weights.macro
        )

    # ------------------------------------------------------------------
    # 仓位管理辅助
    # ------------------------------------------------------------------

    def _compute_base_strength(
        self, ts: datetime, portfolio: Portfolio,
    ) -> float:
        """
        动态计算基准买入强度，使剩余现金平均分配到剩余周数。

        与 WeeklyDCAStrategy._compute_strength 相同的逻辑：
            base_strength = portfolio.cash / remaining_weeks
                            / (initial_capital * 0.1)
        """
        if self.end_date is None:
            return 0.1  # 未指定截止日期，回退到固定比例

        remaining_days = (self.end_date - ts).days
        remaining_weeks = max(1, remaining_days / 7)

        per_week_amount = portfolio.cash / remaining_weeks
        base_value = portfolio.initial_capital * 0.1

        if base_value <= 0:
            return 0.1

        return per_week_amount / base_value

    def _get_buy_strength(
        self, score: float, base_strength: float,
    ) -> float:
        """
        根据综合得分和动态基准强度决定买入强度（0 表示跳过买入）

        在 base_strength 基础上应用战术倍率（5 档）：
        - 加量买入: boost_multiplier (默认 2x)
        - 正常买入: normal_multiplier (默认 1x)
        - 减量买入: reduce_multiplier (默认 0.5x)
        - 完全不买: 0x（新增，允许在弱看空时完全跳过）
        - 强看空跳过: 0x
        """
        cfg = self.position_config
        thresholds = self.thresholds

        if score > thresholds.boost_buy:
            return base_strength * cfg.boost_multiplier
        elif score > thresholds.normal_buy:
            return base_strength * cfg.normal_multiplier
        elif score > thresholds.reduce_buy:
            return base_strength * cfg.reduce_multiplier
        else:
            return 0.0  # skip_buy 或更低：完全不买入

    def _describe_action(self, score: float) -> str:
        """描述当前战术动作"""
        if score > self.thresholds.boost_buy:
            return "boost_buy"
        elif score > self.thresholds.normal_buy:
            return "normal_buy"
        elif score > self.thresholds.reduce_buy:
            return "reduce_buy"
        elif score > self.thresholds.skip_buy:
            return "skip_buy"
        else:
            return "strong_skip"

    def _in_cooldown(self, current: datetime) -> bool:
        """检查是否在冷却期内"""
        if self._last_trade_date is None:
            return False
        delta = (current - self._last_trade_date).days
        return delta < self.position_config.cooldown_days

    def _past_min_hold(self, current: datetime) -> bool:
        """检查是否已过最小持有期"""
        if self._last_trade_date is None:
            return True
        delta = (current - self._last_trade_date).days
        return delta >= self.position_config.min_hold_days

    def _should_force_sell(self, current: datetime, current_price: float) -> bool:
        """
        检查是否应该强制止盈卖出

        条件：
        1. 距离上次卖出超过 force_sell_interval_days
        2. 当前持仓有正收益，超过 force_sell_profit_threshold
        3. 持仓数量 > 0
        """
        cfg = self.position_config
        if self._position_quantity <= 0 or self._position_cost_basis <= 0:
            return False

        # 检查时间间隔
        if self._last_sell_date is not None:
            days_since_sell = (current - self._last_sell_date).days
            if days_since_sell < cfg.force_sell_interval_days:
                return False
        elif self._last_trade_date is not None:
            # 从未卖出过，用首次交易日期计算
            days_since_first = (current - self._last_trade_date).days
            if days_since_first < cfg.force_sell_interval_days:
                return False

        # 检查盈利
        profit_pct = (current_price - self._position_cost_basis) / self._position_cost_basis
        return profit_pct >= cfg.force_sell_profit_threshold

    # ------------------------------------------------------------------
    # 兼容基类 generate_signals（非跨市场模式回退）
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        asset: Asset,
        data: MarketData,
        portfolio: Portfolio,
    ) -> List[Signal]:
        """基类回退：仅使用 GLD 自身数据"""
        if len(data) < self.required_history:
            return []

        closes = data.close_array
        volumes = data.volume_array if hasattr(data.data[0], 'volume') else None

        market_data = {
            "gold_etf": closes,
            "gold_etf_volume": volumes,
        }

        return self._generate_tactical_signals(
            asset, data, market_data, portfolio,
        )

    # ------------------------------------------------------------------
    # 重置和摘要
    # ------------------------------------------------------------------

    def reset(self) -> None:
        super().reset()
        self._has_position.clear()
        self._last_trade_date = None
        self._last_buy_week = None
        self._trade_count = 0
        self._last_sell_date = None
        self._position_cost_basis = 0.0
        self._position_quantity = 0.0

    def summary(self) -> Dict[str, Any]:
        base_summary = super().summary()
        base_summary.update({
            "weights": {
                "technical": self.weights.technical,
                "cross_market": self.weights.cross_market,
                "sentiment": self.weights.sentiment,
                "macro": self.weights.macro,
            },
            "thresholds": {
                "boost_buy": self.thresholds.boost_buy,
                "normal_buy": self.thresholds.normal_buy,
                "reduce_buy": self.thresholds.reduce_buy,
                "skip_buy": self.thresholds.skip_buy,
                "partial_sell": self.thresholds.partial_sell,
            },
            "position_config": {
                "buy_day": self.position_config.buy_day,
                "min_hold_days": self.position_config.min_hold_days,
                "cooldown_days": self.position_config.cooldown_days,
                "force_sell_interval_days": self.position_config.force_sell_interval_days,
                "force_sell_fraction": self.position_config.force_sell_fraction,
                "force_sell_profit_threshold": self.position_config.force_sell_profit_threshold,
            },
        })
        return base_summary

    def __repr__(self):
        return (
            f"GoldTradingStrategy("
            f"tech={self.weights.technical:.0%}, "
            f"cross={self.weights.cross_market:.0%}, "
            f"sent={self.weights.sentiment:.0%}, "
            f"macro={self.weights.macro:.0%}, "
            f"buy_day={self.position_config.buy_day})"
        )
