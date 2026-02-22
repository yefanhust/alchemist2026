"""
黄金投资策略信号测试
测试各信号模块的数据源获取与计算，以及信号合成逻辑
"""

import sys
import os
from datetime import datetime
import numpy as np
import pytest

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "alchemist"))

from strategy.gold.signals.technical import TechnicalSignals, TechnicalSignalResult
from strategy.gold.signals.cross_market import CrossMarketSignals, CrossMarketSignalResult
from strategy.gold.signals.sentiment import SentimentSignals, SentimentSignalResult
from strategy.gold.signals.macro import MacroSignals, MacroSignalResult
from strategy.gold.strategy import (
    GoldTradingStrategy,
    FactorWeights,
    TacticalThresholds,
    PositionConfig,
)


# =============================================================================
# 测试数据生成工具
# =============================================================================

def generate_uptrend_prices(start=180.0, length=250, daily_return=0.001, noise=0.5):
    """生成上涨趋势价格序列"""
    prices = [start]
    for _ in range(length - 1):
        prices.append(prices[-1] * (1 + daily_return) + np.random.uniform(-noise, noise))
    return np.array(prices)


def generate_downtrend_prices(start=200.0, length=250, daily_return=-0.001, noise=0.5):
    """生成下跌趋势价格序列"""
    prices = [start]
    for _ in range(length - 1):
        prices.append(prices[-1] * (1 + daily_return) + np.random.uniform(-noise, noise))
    return np.array(prices)


def generate_flat_prices(center=190.0, length=250, noise=0.3):
    """生成震荡价格序列"""
    return center + np.random.uniform(-noise, noise, size=length)


def generate_volume_data(length=250, base_volume=1e6, spike_ratio=2.0, spike_at=-1):
    """生成成交量数据，可选在指定位置生成放量"""
    volumes = np.random.uniform(base_volume * 0.8, base_volume * 1.2, size=length)
    if spike_at is not None:
        volumes[spike_at] = base_volume * spike_ratio
    return volumes


# =============================================================================
# 技术面信号测试
# =============================================================================

class TestTechnicalSignalsDataSource:
    """测试技术面信号的数据源获取和处理"""

    def test_calculate_with_price_data(self):
        """验证使用价格数据计算技术信号"""
        ts = TechnicalSignals()
        prices = generate_uptrend_prices(length=250)
        result = ts.calculate(prices)

        assert isinstance(result, TechnicalSignalResult)
        assert -1.0 <= result.trend <= 1.0
        assert -1.0 <= result.momentum <= 1.0
        assert -1.0 <= result.volatility <= 1.0
        assert result.volume == 0.0  # 未提供成交量
        assert -1.0 <= result.composite <= 1.0

    def test_calculate_with_price_and_volume(self):
        """验证同时使用价格和成交量数据"""
        ts = TechnicalSignals()
        prices = generate_uptrend_prices(length=250)
        volumes = generate_volume_data(length=250)
        result = ts.calculate(prices, volumes)

        assert isinstance(result, TechnicalSignalResult)
        assert -1.0 <= result.volume <= 1.0

    def test_insufficient_data_returns_neutral(self):
        """验证数据不足时返回中性信号"""
        ts = TechnicalSignals(trend_periods=[20, 50, 200])
        prices = np.array([180.0, 181.0, 182.0])
        result = ts.calculate(prices)

        assert result.trend == 0.0
        assert result.momentum == 0.0
        assert result.volatility == 0.0

    def test_uptrend_produces_positive_signal(self):
        """验证上涨趋势产生正向信号"""
        np.random.seed(42)
        ts = TechnicalSignals()
        prices = np.linspace(150.0, 200.0, 250)
        result = ts.calculate(prices)

        assert result.trend > 0
        assert result.momentum > 0

    def test_downtrend_produces_negative_signal(self):
        """验证下跌趋势产生负向信号"""
        ts = TechnicalSignals()
        prices = np.linspace(200.0, 150.0, 250)
        result = ts.calculate(prices)

        assert result.trend < 0
        assert result.momentum < 0

    def test_composite_weight_distribution(self):
        """验证技术综合信号的权重分配"""
        ts = TechnicalSignals()
        prices = np.linspace(150.0, 200.0, 250)
        volumes = generate_volume_data(length=250)
        result = ts.calculate(prices, volumes)

        expected_composite = (
            result.trend * 0.35 +
            result.momentum * 0.30 +
            result.volatility * 0.20 +
            result.volume * 0.15
        )
        expected_composite = np.clip(expected_composite, -1.0, 1.0)

        assert abs(result.composite - expected_composite) < 1e-10

    def test_rsi_calculation(self):
        """验证 RSI 指标计算"""
        ts = TechnicalSignals()
        all_up = np.linspace(100, 200, 30)
        rsi = ts._calculate_rsi(all_up, 14)
        assert rsi is not None
        assert rsi > 80

        all_down = np.linspace(200, 100, 30)
        rsi = ts._calculate_rsi(all_down, 14)
        assert rsi is not None
        assert rsi < 20

    def test_volatility_breakout_upper(self):
        """验证上轨突破信号"""
        ts = TechnicalSignals(volatility_period=20)
        stable = np.full(19, 100.0)
        prices = np.append(stable, [120.0])
        result = ts._calculate_volatility_breakout(prices)
        assert result > 0

    def test_volatility_breakout_lower(self):
        """验证下轨突破信号"""
        ts = TechnicalSignals(volatility_period=20)
        stable = np.full(19, 100.0)
        prices = np.append(stable, [80.0])
        result = ts._calculate_volatility_breakout(prices)
        assert result < 0

    def test_volume_spike_with_price_up(self):
        """验证放量上涨信号"""
        ts = TechnicalSignals(volume_period=20)
        prices = np.linspace(100, 105, 25)
        volumes = generate_volume_data(length=25, base_volume=1e6, spike_ratio=3.0, spike_at=-1)
        result = ts._analyze_volume_pattern(prices, volumes)
        assert result > 0

    def test_volume_spike_with_price_down(self):
        """验证放量下跌信号"""
        ts = TechnicalSignals(volume_period=20)
        prices = np.linspace(105, 100, 25)
        volumes = generate_volume_data(length=25, base_volume=1e6, spike_ratio=3.0, spike_at=-1)
        result = ts._analyze_volume_pattern(prices, volumes)
        assert result < 0


# =============================================================================
# 跨市场联动信号测试
# =============================================================================

class TestCrossMarketSignalsDataSource:
    """测试跨市场联动信号的数据源获取和处理"""

    def test_calculate_with_usd_and_sp500(self):
        """验证提供 USD 和 SP500 数据时的计算"""
        cms = CrossMarketSignals(correlation_window=30)
        np.random.seed(42)

        gold = generate_uptrend_prices(start=1800, length=50, noise=5)
        usd = generate_downtrend_prices(start=105, length=50, noise=0.3)
        sp500 = generate_downtrend_prices(start=4500, length=50, noise=10)

        result = cms.calculate(gold, usd, sp500)

        assert isinstance(result, CrossMarketSignalResult)
        assert -1.0 <= result.usd_correlation <= 1.0
        assert -1.0 <= result.equity_safe_haven <= 1.0
        assert -1.0 <= result.composite <= 1.0

    def test_calculate_with_only_gold(self):
        """验证仅提供黄金数据时返回中性信号"""
        cms = CrossMarketSignals()
        gold = generate_uptrend_prices(start=1800, length=50)

        result = cms.calculate(gold)

        assert result.usd_correlation == 0.0
        assert result.equity_safe_haven == 0.0
        assert result.composite == 0.0

    def test_usd_weakness_benefits_gold(self):
        """验证美元走弱利好黄金信号"""
        cms = CrossMarketSignals(correlation_window=30, safe_haven_window=20)
        np.random.seed(42)

        shocks = np.random.randn(49) * 0.01
        gold = 1800 * np.cumprod(np.concatenate([[1.0], 1 + shocks + 0.002]))
        usd = 105 * np.cumprod(np.concatenate([[1.0], 1 - shocks - 0.003]))

        result = cms.calculate(gold, usd_index=usd)

        assert result.usd_correlation > 0, f"美元走弱应利好黄金, 实际={result.usd_correlation}"

    def test_safe_haven_demand_on_stock_crash(self):
        """验证股市暴跌时避险需求信号"""
        cms = CrossMarketSignals(safe_haven_window=20)

        sp500 = np.concatenate([np.full(30, 4500), np.linspace(4500, 4275, 20)])
        gold = np.concatenate([np.full(30, 1800), np.linspace(1800, 1850, 20)])

        result = cms.calculate(gold, sp500=sp500)

        assert result.equity_safe_haven > 0

    def test_risk_on_stock_rally(self):
        """验证股市大涨、金价下跌时的风险偏好信号"""
        cms = CrossMarketSignals(safe_haven_window=20)

        sp500 = np.concatenate([np.full(30, 4000), np.linspace(4000, 4200, 20)])
        gold = np.concatenate([np.full(30, 1900), np.linspace(1900, 1850, 20)])

        result = cms.calculate(gold, sp500=sp500)

        assert result.equity_safe_haven < 0

    def test_eur_usd_strength_signal(self):
        """验证 EUR/USD 走强（美元弱）信号"""
        cms = CrossMarketSignals(correlation_window=30)

        gold = generate_flat_prices(center=1800, length=50)
        eur_usd = np.linspace(1.05, 1.15, 50)

        result = cms.calculate(gold, eur_usd=eur_usd)

        assert result.usd_correlation > 0, "EUR/USD 走强应利好黄金"

    def test_usd_jpy_risk_off_signal(self):
        """验证 USD/JPY 下跌（日元走强/避险）信号"""
        cms = CrossMarketSignals(correlation_window=30)

        gold = generate_flat_prices(center=1800, length=50)
        usd_jpy = np.linspace(155, 140, 50)

        result = cms.calculate(gold, usd_jpy=usd_jpy)

        assert result.usd_correlation > 0, "日元走强应利好黄金"

    def test_insufficient_data_returns_zero(self):
        """验证数据不足时跨市场信号为零"""
        cms = CrossMarketSignals(correlation_window=30, safe_haven_window=20)

        gold = np.array([1800.0, 1805.0])
        usd = np.array([105.0, 104.5])

        result = cms.calculate(gold, usd_index=usd)

        assert result.usd_correlation == 0.0
        assert result.composite == 0.0

    def test_correlation_calculation(self):
        """验证相关性计算方法"""
        cms = CrossMarketSignals(correlation_window=30)
        np.random.seed(42)

        shocks = np.random.randn(39) * 0.01
        x = 100 * np.cumprod(np.concatenate([[1.0], 1 + shocks]))
        y = 50 * np.cumprod(np.concatenate([[1.0], 1 + shocks * 0.8]))
        corr = cms._calculate_correlation(x, y, 30)
        assert corr > 0.9

        y_neg = 100 * np.cumprod(np.concatenate([[1.0], 1 - shocks * 0.8]))
        corr_neg = cms._calculate_correlation(x, y_neg, 30)
        assert corr_neg < -0.9


# =============================================================================
# 市场情绪信号测试
# =============================================================================

class TestSentimentSignalsDataSource:
    """测试市场情绪信号的数据源获取和处理"""

    def test_calculate_with_miners_and_volume(self):
        """验证提供矿业股和成交量数据时的计算"""
        ss = SentimentSignals()
        np.random.seed(42)

        gold_etf = generate_uptrend_prices(start=180, length=30)
        miners = generate_uptrend_prices(start=35, length=30, daily_return=0.002)
        volume = generate_volume_data(length=30)

        result = ss.calculate(gold_etf, miners, gold_etf_volume=volume)

        assert isinstance(result, SentimentSignalResult)
        assert -1.0 <= result.miners_relative <= 1.0
        assert -1.0 <= result.volume_pattern <= 1.0
        assert -1.0 <= result.composite <= 1.0

    def test_calculate_with_only_etf(self):
        """验证仅提供 ETF 价格时返回中性信号"""
        ss = SentimentSignals()
        gold_etf = generate_uptrend_prices(start=180, length=30)

        result = ss.calculate(gold_etf)

        assert result.miners_relative == 0.0
        assert result.volume_pattern == 0.0
        assert result.composite == 0.0

    def test_miners_outperform_bullish_signal(self):
        """验证矿业股跑赢黄金 ETF 时产生看涨信号"""
        ss = SentimentSignals(relative_strength_period=20)

        gold_etf = np.linspace(180, 185, 25)
        miners = np.linspace(35, 38, 25)

        result = ss.calculate(gold_etf, gold_miners_prices=miners)

        assert result.miners_relative > 0

    def test_miners_underperform_bearish_signal(self):
        """验证矿业股跑输黄金 ETF 时产生看跌信号"""
        ss = SentimentSignals(relative_strength_period=20)

        gold_etf = np.linspace(180, 190, 25)
        miners = np.linspace(38, 36, 25)

        result = ss.calculate(gold_etf, gold_miners_prices=miners)

        assert result.miners_relative < 0

    def test_volume_spike_signal(self):
        """验证放量信号"""
        ss = SentimentSignals(volume_period=10)

        gold_etf = generate_flat_prices(center=180, length=20)
        volume = generate_volume_data(length=20, base_volume=1e6, spike_ratio=3.0, spike_at=-1)

        result = ss.calculate(gold_etf, gold_etf_volume=volume)

        assert result.volume_pattern > 0

    def test_insufficient_data_returns_neutral(self):
        """验证数据不足时返回中性信号"""
        ss = SentimentSignals(relative_strength_period=20)

        gold_etf = np.array([180.0, 181.0])
        miners = np.array([35.0, 35.5])

        result = ss.calculate(gold_etf, gold_miners_prices=miners)

        assert result.miners_relative == 0.0


# =============================================================================
# 宏观因子信号测试
# =============================================================================

class TestMacroSignalsDataSource:
    """测试宏观因子信号的数据源获取和处理"""

    def test_calculate_with_all_data(self):
        """验证提供全部宏观数据时的计算"""
        ms = MacroSignals(lookback_period=20)
        np.random.seed(42)

        treasury = np.linspace(3.5, 3.0, 30)
        inflation = np.linspace(100, 105, 30)  # TIP ETF 价格上涨
        usd = generate_downtrend_prices(start=105, length=30, noise=0.2)
        eur_usd = np.linspace(1.08, 1.12, 30)
        usd_jpy = np.linspace(150, 145, 30)
        vix = np.linspace(18, 28, 30)  # VIXY ETF 上涨
        sp500 = generate_downtrend_prices(start=4500, length=30, noise=10)
        treasury_prices = np.linspace(95, 100, 30)

        result = ms.calculate(
            treasury, inflation, usd, eur_usd, usd_jpy, vix, sp500, treasury_prices
        )

        assert isinstance(result, MacroSignalResult)
        assert -1.0 <= result.real_yield <= 1.0
        assert -1.0 <= result.currency_strength <= 1.0
        assert -1.0 <= result.risk_on_off <= 1.0
        assert -1.0 <= result.composite <= 1.0

    def test_calculate_with_no_data(self):
        """验证无宏观数据时返回中性信号"""
        ms = MacroSignals()
        result = ms.calculate()

        assert result.real_yield == 0.0
        assert result.currency_strength == 0.0
        assert result.risk_on_off == 0.0
        assert result.composite == 0.0

    def test_high_yield_bearish(self):
        """验证高国债收益率利空黄金"""
        ms = MacroSignals(lookback_period=20)

        treasury = np.full(30, 5.5)  # 高收益率

        result = ms.calculate(treasury_yield=treasury)

        assert result.real_yield < 0, "高收益率应利空黄金"

    def test_low_yield_bullish(self):
        """验证低国债收益率利好黄金"""
        ms = MacroSignals(lookback_period=20)

        treasury = np.full(30, 1.5)  # 低收益率

        result = ms.calculate(treasury_yield=treasury)

        assert result.real_yield > 0, "低收益率应利好黄金"

    def test_tip_rise_bullish(self):
        """验证 TIP 上涨（通胀预期上升）利好黄金"""
        ms = MacroSignals(lookback_period=20)

        tip_prices = np.linspace(100, 110, 30)  # TIP 价格上涨

        result = ms.calculate(inflation_expectations=tip_prices)

        assert result.real_yield > 0, "TIP 上涨（通胀预期上升）应利好黄金"

    def test_usd_weakness_positive(self):
        """验证美元走弱利好黄金"""
        ms = MacroSignals(lookback_period=20)

        usd = np.linspace(105, 95, 30)

        result = ms.calculate(usd_index=usd)

        assert result.currency_strength > 0

    def test_usd_strength_negative(self):
        """验证美元走强利空黄金"""
        ms = MacroSignals(lookback_period=20)

        usd = np.linspace(95, 105, 30)

        result = ms.calculate(usd_index=usd)

        assert result.currency_strength < 0

    def test_eur_usd_signal(self):
        """验证 EUR/USD 信号"""
        ms = MacroSignals(lookback_period=20)

        eur_usd = np.linspace(1.05, 1.15, 30)

        result = ms.calculate(eur_usd=eur_usd)

        assert result.currency_strength > 0

    def test_vixy_rise_risk_off(self):
        """验证 VIXY 上涨产生 risk-off 信号"""
        ms = MacroSignals(lookback_period=20)

        # VIXY 短期大涨
        vixy = np.concatenate([np.full(25, 15), np.full(5, 25)])

        result = ms.calculate(vix=vixy)

        assert result.risk_on_off > 0, "VIXY 上涨应产生 risk-off 信号"

    def test_stock_bond_divergence(self):
        """验证股债分化时的风险信号"""
        ms = MacroSignals(lookback_period=20)

        sp500 = np.linspace(4500, 4200, 30)
        treasury = np.linspace(95, 100, 30)

        result = ms.calculate(sp500=sp500, treasury_prices=treasury)

        assert result.risk_on_off > 0


# =============================================================================
# 信号合成测试（新的择时增强型定投策略）
# =============================================================================

class TestSignalSynthesis:
    """测试黄金择时增强型定投信号合成逻辑"""

    def test_factor_weights_validation(self):
        """验证因子权重之和必须为 1"""
        weights = FactorWeights(technical=0.25, cross_market=0.25, sentiment=0.25, macro=0.25)
        assert abs(weights.technical + weights.cross_market + weights.sentiment + weights.macro - 1.0) < 0.001

        with pytest.raises(ValueError):
            FactorWeights(technical=0.5, cross_market=0.5, sentiment=0.5, macro=0.5)

    def test_default_weights(self):
        """验证默认权重配置"""
        weights = FactorWeights()
        assert weights.technical == 0.40
        assert weights.cross_market == 0.25
        assert weights.sentiment == 0.15
        assert weights.macro == 0.20

    def test_composite_score_calculation(self):
        """验证综合得分计算"""
        strategy = GoldTradingStrategy()
        factors = {
            "technical": 0.5,
            "cross_market": 0.3,
            "sentiment": 0.4,
            "macro": 0.2,
        }

        score = strategy._compute_composite_score(factors)

        expected_score = (
            0.5 * 0.40 +   # technical
            0.3 * 0.25 +   # cross_market
            0.4 * 0.15 +   # sentiment
            0.2 * 0.20     # macro
        )

        assert abs(score - expected_score) < 1e-10

    def test_buy_strength_boost(self):
        """验证加量买入信号强度"""
        strategy = GoldTradingStrategy()
        strength = strategy._get_buy_strength(base_strength=0.1, score=0.5)  # score > 0.3
        assert strength == 0.1 * strategy.position_config.boost_multiplier  # 0.1 * 2.0 = 0.20

    def test_buy_strength_normal(self):
        """验证正常买入信号强度"""
        strategy = GoldTradingStrategy()
        strength = strategy._get_buy_strength(base_strength=0.1, score=0.15)  # 0.0 < score <= 0.3
        assert strength == 0.1 * strategy.position_config.normal_multiplier  # 0.1 * 1.0 = 0.10

    def test_buy_strength_reduce(self):
        """验证减量买入信号强度"""
        strategy = GoldTradingStrategy()
        strength = strategy._get_buy_strength(base_strength=0.1, score=-0.10)  # reduce_buy < score <= normal_buy
        assert strength == 0.1 * strategy.position_config.reduce_multiplier  # 0.1 * 0.5 = 0.05

    def test_buy_strength_skip(self):
        """验证跳过买入（skip_buy 区间）"""
        strategy = GoldTradingStrategy()
        strength = strategy._get_buy_strength(base_strength=0.1, score=-0.20)  # skip_buy < score <= reduce_buy
        assert strength == 0.0

    def test_buy_strength_strong_skip(self):
        """验证强看空跳过买入"""
        strategy = GoldTradingStrategy()
        strength = strategy._get_buy_strength(base_strength=0.1, score=-0.5)  # score <= skip_buy
        assert strength == 0.0

    def test_tactical_action_description(self):
        """验证战术动作描述"""
        strategy = GoldTradingStrategy()

        assert strategy._describe_action(0.5) == "boost_buy"
        assert strategy._describe_action(0.15) == "normal_buy"
        assert strategy._describe_action(-0.10) == "reduce_buy"
        assert strategy._describe_action(-0.20) == "skip_buy"
        assert strategy._describe_action(-0.5) == "strong_skip"

    def test_custom_thresholds(self):
        """验证自定义阈值"""
        thresholds = TacticalThresholds(boost_buy=0.2, normal_buy=-0.1, reduce_buy=-0.4, partial_sell=-0.7)
        strategy = GoldTradingStrategy(thresholds=thresholds)

        # score=0.25 应触发 boost_buy（阈值 0.2）
        assert strategy._get_buy_strength(base_strength=0.1, score=0.25) == 0.1 * strategy.position_config.boost_multiplier
        # score=-0.05 应触发 normal_buy（阈值 -0.1）
        assert strategy._get_buy_strength(base_strength=0.1, score=-0.05) == 0.1 * strategy.position_config.normal_multiplier

    def test_custom_weights(self):
        """验证自定义权重"""
        weights = FactorWeights(technical=0.50, cross_market=0.20, sentiment=0.10, macro=0.20)
        strategy = GoldTradingStrategy(weights=weights)

        factors = {
            "technical": 1.0,
            "cross_market": 0.0,
            "sentiment": 0.0,
            "macro": 0.0,
        }

        score = strategy._compute_composite_score(factors)
        assert abs(score - 0.50) < 1e-10

    def test_missing_factor_treated_as_zero(self):
        """验证缺失因子按零处理"""
        strategy = GoldTradingStrategy()
        factors = {"technical": 0.5}

        score = strategy._compute_composite_score(factors)
        expected = 0.5 * 0.40
        assert abs(score - expected) < 1e-10

    def test_calculate_factors_integration(self):
        """验证 calculate_factors 整合所有因子"""
        strategy = GoldTradingStrategy()

        data = {
            "gold_etf": np.linspace(170, 200, 250),
            "gold_etf_volume": generate_volume_data(length=250),
            "usd_index": np.linspace(105, 95, 250),
            "sp500": np.linspace(4500, 4300, 250),
            "treasury_yield": np.linspace(3.5, 3.0, 250),
            "gold_miners": np.linspace(30, 35, 250),
            "inflation_expectations": np.linspace(100, 105, 250),
            "eur_usd": np.linspace(1.08, 1.12, 250),
            "usd_jpy": np.linspace(150, 145, 250),
            "vix": np.full(250, 22),
        }

        factors = strategy.calculate_factors(data)

        assert "technical" in factors
        assert "cross_market" in factors
        assert "sentiment" in factors
        assert "macro" in factors

        for name, score in factors.items():
            assert -1.0 <= score <= 1.0, f"因子 {name} 超出范围: {score}"

    def test_calculate_factors_without_cross_market_data(self):
        """验证无跨市场数据时因子计算"""
        strategy = GoldTradingStrategy()

        data = {
            "gold_etf": np.linspace(170, 200, 250),
        }

        factors = strategy.calculate_factors(data)

        assert factors["cross_market"] == 0.0
        assert factors["sentiment"] == 0.0
        assert factors["macro"] == 0.0

    def test_end_to_end_bullish_scenario(self):
        """端到端测试：看涨场景"""
        strategy = GoldTradingStrategy()

        data = {
            "gold_etf": np.linspace(170, 210, 250),
            "gold_etf_volume": generate_volume_data(length=250),
            "usd_index": np.linspace(108, 92, 250),
            "sp500": np.linspace(4500, 4000, 250),
            "treasury_yield": np.linspace(4.0, 2.0, 250),
            "gold_miners": np.linspace(28, 40, 250),
            "inflation_expectations": np.linspace(95, 110, 250),
            "vix": np.linspace(15, 32, 250),
        }

        factors = strategy.calculate_factors(data)
        score = strategy._compute_composite_score(factors)

        assert score > 0, f"看涨环境得分应为正, 实际={score}"

    def test_end_to_end_bearish_scenario(self):
        """端到端测试：看跌场景"""
        strategy = GoldTradingStrategy()

        data = {
            "gold_etf": np.linspace(210, 170, 250),
            "gold_etf_volume": generate_volume_data(length=250),
            "usd_index": np.linspace(92, 108, 250),
            "sp500": np.linspace(4000, 4800, 250),
            "treasury_yield": np.linspace(2.0, 5.0, 250),
            "gold_miners": np.linspace(40, 28, 250),
            "inflation_expectations": np.linspace(110, 95, 250),
            "vix": np.linspace(30, 12, 250),
        }

        factors = strategy.calculate_factors(data)
        score = strategy._compute_composite_score(factors)

        assert score < 0, f"看跌环境得分应为负, 实际={score}"

    def test_strategy_reset(self):
        """验证策略重置"""
        strategy = GoldTradingStrategy()

        strategy._has_position["GLD"] = True
        strategy._last_buy_week = 202601
        strategy._last_trade_date = datetime.now()

        strategy.reset()

        assert len(strategy._has_position) == 0
        assert strategy._last_buy_week is None
        assert strategy._last_trade_date is None

    def test_strategy_summary(self):
        """验证策略摘要信息"""
        strategy = GoldTradingStrategy()
        summary = strategy.summary()

        assert "weights" in summary
        assert "thresholds" in summary
        assert "position_config" in summary
        assert summary["weights"]["technical"] == 0.40

    def test_required_history(self):
        """验证所需最小历史数据量"""
        strategy = GoldTradingStrategy()

        # max(200, 30, 20) + 10 = 210
        assert strategy.required_history == 210

    def test_cooldown_period(self):
        """验证冷却期逻辑"""
        strategy = GoldTradingStrategy()

        # 无交易记录 → 不在冷却期
        assert not strategy._in_cooldown(datetime(2024, 1, 10))

        # 设置上次交易日期
        strategy._last_trade_date = datetime(2024, 1, 8)

        # 1 天后 → 在冷却期（默认 3 天）
        assert strategy._in_cooldown(datetime(2024, 1, 9))

        # 3 天后 → 不在冷却期
        assert not strategy._in_cooldown(datetime(2024, 1, 11))

    def test_min_hold_period(self):
        """验证最小持有期逻辑"""
        strategy = GoldTradingStrategy()

        # 无交易记录 → 已过持有期
        assert strategy._past_min_hold(datetime(2024, 1, 20))

        # 设置上次交易日期
        strategy._last_trade_date = datetime(2024, 1, 1)

        # 5 天后 → 未过最小持有期（默认 10 天）
        assert not strategy._past_min_hold(datetime(2024, 1, 6))

        # 15 天后 → 已过最小持有期
        assert strategy._past_min_hold(datetime(2024, 1, 16))

    def test_repr(self):
        """验证策略的字符串表示"""
        strategy = GoldTradingStrategy()
        repr_str = repr(strategy)

        assert "GoldTradingStrategy" in repr_str
        assert "40%" in repr_str
        assert "25%" in repr_str
        assert "15%" in repr_str
        assert "20%" in repr_str
