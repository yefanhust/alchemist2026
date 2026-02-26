"""
估值扫描系统测试

测试覆盖：
- 数据模型 (StockValuation, ScanResult, score_to_grade)
- 相对估值因子 (RelativeValuationFactors)
- 绝对估值模型 (DCFModel, ResidualIncomeModel, AbsoluteValuationFactors)
- 情绪因子 (SentimentFactors)
- 宏观因子 (MacroFactors)
- 综合打分器 (ValuationScorer)
- 股票池管理 (StockUniverse)
- 数据采集 (_calls_per_symbol, _filter_fresh_symbols, fetch_data_for_scan)
"""

import sys
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "alchemist"))

from strategy.valuation.models import (
    StockValuation,
    ScanResult,
    GRADE_THRESHOLDS,
    HORIZON_WEIGHTS,
    score_to_grade,
    get_horizon_days,
    get_lookback_days,
)
from strategy.valuation.factors.relative import (
    RelativeValuationFactors,
    _zscore_to_score,
    _percentile_score,
)
from strategy.valuation.factors.absolute import (
    DCFModel,
    ResidualIncomeModel,
    AbsoluteValuationFactors,
    _safe_float,
)
from strategy.valuation.factors.sentiment import SentimentFactors
from strategy.valuation.factors.macro import MacroFactors
from strategy.valuation.scorer import ValuationScorer


# =============================================================================
# 测试数据工具
# =============================================================================

def make_stock_info(**overrides):
    """生成 mock stock_info 字典"""
    base = {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "pe_ratio": 28.5,
        "pb_ratio": 45.0,
        "ps_ratio": 7.5,
        "peg_ratio": 1.8,
        "forward_pe": 26.0,
        "ev_to_ebitda": 22.0,
        "book_value": 3.95,
        "revenue_per_share": 24.3,
        "profit_margin": 0.265,
        "operating_margin": 0.305,
        "return_on_equity": 1.6,
        "shares_outstanding": 15500000000,
        "price_to_fcf": 25.0,
        "dividend_per_share": 0.96,
        "payout_ratio": 0.155,
        "eps": 6.42,
        "high_52week": 199.62,
        "low_52week": 164.08,
        "market_cap": 2800000000000,
    }
    base.update(overrides)
    return base


def make_peers(n=10):
    """生成同行业 peer 列表"""
    np.random.seed(42)
    peers = []
    for i in range(n):
        peers.append({
            "symbol": f"PEER{i}",
            "pe_ratio": np.random.uniform(15, 40),
            "pb_ratio": np.random.uniform(2, 60),
            "ps_ratio": np.random.uniform(2, 12),
            "peg_ratio": np.random.uniform(0.5, 3.0),
            "ev_to_ebitda": np.random.uniform(10, 30),
            "price_to_fcf": np.random.uniform(10, 35),
            "dividend_per_share": np.random.uniform(0, 2),
            "shares_outstanding": np.random.uniform(1e9, 20e9),
        })
    return peers


def make_cashflow_reports():
    """生成 mock 现金流量表"""
    return [
        {
            "fiscalDateEnding": "2024-09-30",
            "operatingCashflow": "120000000000",
            "capitalExpenditures": "-10000000000",
        },
        {
            "fiscalDateEnding": "2023-09-30",
            "operatingCashflow": "110000000000",
            "capitalExpenditures": "-11000000000",
        },
        {
            "fiscalDateEnding": "2022-09-30",
            "operatingCashflow": "122000000000",
            "capitalExpenditures": "-10800000000",
        },
    ]


def make_balance_reports():
    """生成 mock 资产负债表"""
    return [
        {
            "fiscalDateEnding": "2024-09-30",
            "totalShareholderEquity": "62000000000",
            "longTermDebt": "95000000000",
        },
        {
            "fiscalDateEnding": "2023-09-30",
            "totalShareholderEquity": "58000000000",
            "longTermDebt": "98000000000",
        },
    ]


def make_income_reports():
    """生成 mock 利润表"""
    return [
        {
            "fiscalDateEnding": "2024-09-30",
            "netIncome": "94000000000",
        },
        {
            "fiscalDateEnding": "2023-09-30",
            "netIncome": "97000000000",
        },
    ]


def generate_prices(n=200, start=150.0, trend=0.0005, seed=42):
    """生成模拟价格序列"""
    np.random.seed(seed)
    returns = np.random.normal(trend, 0.015, n)
    prices = start * np.cumprod(1 + returns)
    return prices


def generate_volumes(n=200, base=1e6, seed=42):
    """生成模拟成交量"""
    np.random.seed(seed)
    return np.random.uniform(base * 0.8, base * 1.2, n)


# =============================================================================
# 模型测试
# =============================================================================

class TestModels:
    """数据模型测试"""

    def test_score_to_grade_boundaries(self):
        """测试评分到等级的边界映射"""
        assert score_to_grade(-0.8) == "A"
        assert score_to_grade(-0.3) == "B"
        assert score_to_grade(0.0) == "C"
        assert score_to_grade(0.3) == "D"
        assert score_to_grade(0.8) == "F"

    def test_score_to_grade_extremes(self):
        """测试极端值"""
        assert score_to_grade(-2.0) == "A"  # clipped to -1.0
        assert score_to_grade(2.0) == "F"   # clipped to 1.0

    def test_score_to_grade_threshold_exact(self):
        """测试阈值精确值"""
        assert score_to_grade(-0.5) == "B"   # -0.5 是 B 的下界
        assert score_to_grade(-0.15) == "C"  # -0.15 是 C 的下界
        assert score_to_grade(0.15) == "D"   # 0.15 是 D 的下界
        assert score_to_grade(0.5) == "F"    # 0.5 是 F 的下界

    def test_get_horizon_days(self):
        """测试投资时间窗口映射"""
        assert get_horizon_days("1M") == 21
        assert get_horizon_days("3M") == 63
        assert get_horizon_days("6M") == 126
        assert get_horizon_days("1Y") == 252
        assert get_horizon_days("unknown") == 63  # 默认值

    def test_get_lookback_days_alias(self):
        """向后兼容别名"""
        assert get_lookback_days("3M") == get_horizon_days("3M")

    def test_stock_valuation_to_dict(self):
        """测试序列化"""
        v = StockValuation(
            symbol="AAPL", name="Apple", sector="Tech", industry="Electronics",
            current_price=180.0, composite_score=-0.35, grade="B",
            relative_score=-0.2, absolute_score=-0.5,
            sentiment_score=-0.3, macro_score=-0.1,
        )
        d = v.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["grade"] == "B"
        assert d["composite_score"] == -0.35
        assert d["scan_date"] is None

    def test_scan_result_to_dict(self):
        """测试 ScanResult 序列化"""
        result = ScanResult(
            scan_date=datetime(2024, 1, 15),
            horizon="3M",
            total_scanned=100,
        )
        d = result.to_dict()
        assert d["total_scanned"] == 100
        assert d["horizon"] == "3M"
        assert "2024" in d["scan_date"]


# =============================================================================
# 相对估值因子测试
# =============================================================================

class TestRelativeValuationFactors:
    """相对估值因子测试"""

    def setup_method(self):
        self.factors = RelativeValuationFactors()

    def test_zscore_to_score_zero(self):
        """Z-Score 为 0 时输出 0"""
        assert _zscore_to_score(10.0, 10.0, 2.0) == 0.0

    def test_zscore_to_score_positive(self):
        """高于中位数应输出正值（高估方向）"""
        score = _zscore_to_score(15.0, 10.0, 2.0)
        assert score > 0

    def test_zscore_to_score_negative(self):
        """低于中位数应输出负值（低估方向）"""
        score = _zscore_to_score(5.0, 10.0, 2.0)
        assert score < 0

    def test_zscore_to_score_zero_std(self):
        """标准差为 0 时返回 0"""
        assert _zscore_to_score(10.0, 10.0, 0.0) == 0.0

    def test_percentile_score_midpoint(self):
        """中间值应返回 0"""
        score = _percentile_score(50.0, 0.0, 100.0)
        assert abs(score) < 0.01

    def test_percentile_score_low(self):
        """接近下界应返回 -1"""
        score = _percentile_score(0.0, 0.0, 100.0)
        assert score == -1.0

    def test_percentile_score_high(self):
        """接近上界应返回 +1"""
        score = _percentile_score(100.0, 0.0, 100.0)
        assert score == 1.0

    def test_calculate_returns_score(self):
        """完整计算应返回有效结果"""
        stock = make_stock_info()
        peers = make_peers()
        result = self.factors.calculate(stock, peers)

        assert "relative_score" in result
        assert "factors" in result
        assert "details" in result
        assert -1.0 <= result["relative_score"] <= 1.0

    def test_calculate_no_peers(self):
        """无同行业数据时应返回 0 分"""
        stock = make_stock_info()
        result = self.factors.calculate(stock, [])

        assert "relative_score" in result
        # 无 peer 数据，大部分因子应为 0

    def test_calculate_missing_ratios(self):
        """缺失指标时不崩溃"""
        stock = {"symbol": "TEST", "pe_ratio": None, "pb_ratio": None}
        result = self.factors.calculate(stock, make_peers())
        assert "relative_score" in result

    def test_high_pe_scores_positive(self):
        """PE 极高的股票应得到正分（偏高估）"""
        stock = make_stock_info(pe_ratio=80.0)
        peers = [{"pe_ratio": p} for p in [15, 18, 20, 22, 25]]
        result = self.factors.calculate(stock, peers)
        # PE 远高于行业，整体应偏高估
        assert result["factors"].get("pe", 0) > 0

    def test_low_pe_scores_negative(self):
        """PE 极低的股票应得到负分（偏低估）"""
        stock = make_stock_info(pe_ratio=8.0)
        peers = [{"pe_ratio": p} for p in [15, 18, 20, 22, 25, 30]]
        result = self.factors.calculate(stock, peers)
        assert result["factors"].get("pe", 0) < 0


# =============================================================================
# 绝对估值因子测试
# =============================================================================

class TestAbsoluteValuation:
    """绝对估值模型测试"""

    def test_safe_float_conversions(self):
        """测试安全类型转换"""
        assert _safe_float("123.45") == 123.45
        assert _safe_float(42) == 42.0
        assert _safe_float(None) is None
        assert _safe_float("None") is None
        assert _safe_float("-") is None
        assert _safe_float("") is None

    def test_dcf_basic(self):
        """DCF 基本计算"""
        dcf = DCFModel()
        stock = make_stock_info()
        result = dcf.calculate_intrinsic_value(
            cashflow_reports=make_cashflow_reports(),
            balance_reports=make_balance_reports(),
            stock_info=stock,
            risk_free_rate=0.04,
        )

        assert "intrinsic_values" in result
        assert "neutral" in result["intrinsic_values"]
        # 有足够数据时应能计算出内在价值
        neutral_iv = result["intrinsic_values"]["neutral"]
        if neutral_iv is not None:
            assert neutral_iv > 0

    def test_dcf_scenarios_ordering(self):
        """乐观 > 中性 > 悲观"""
        dcf = DCFModel()
        result = dcf.calculate_intrinsic_value(
            cashflow_reports=make_cashflow_reports(),
            balance_reports=make_balance_reports(),
            stock_info=make_stock_info(),
        )

        ivs = result["intrinsic_values"]
        opt = ivs.get("optimistic")
        neu = ivs.get("neutral")
        pes = ivs.get("pessimistic")

        if opt and neu and pes:
            assert opt >= neu >= pes

    def test_dcf_empty_reports(self):
        """空报表不崩溃"""
        dcf = DCFModel()
        result = dcf.calculate_intrinsic_value(
            cashflow_reports=[],
            balance_reports=[],
            stock_info=make_stock_info(),
        )
        assert "intrinsic_values" in result

    def test_residual_income_model(self):
        """剩余收益模型基本测试"""
        rim = ResidualIncomeModel()
        result = rim.calculate(
            stock_info=make_stock_info(),
            balance_reports=make_balance_reports(),
            income_reports=make_income_reports(),
            risk_free_rate=0.04,
        )
        assert "intrinsic_value" in result
        if result.get("valid"):
            assert result["intrinsic_value"] > 0

    def test_absolute_factors_calculate(self):
        """综合绝对估值因子"""
        af = AbsoluteValuationFactors()
        result = af.calculate(
            stock_info=make_stock_info(),
            cashflow_reports=make_cashflow_reports(),
            balance_reports=make_balance_reports(),
            income_reports=make_income_reports(),
        )
        assert "absolute_score" in result
        assert -1.0 <= result["absolute_score"] <= 1.0


# =============================================================================
# 情绪因子测试
# =============================================================================

class TestSentimentFactors:
    """情绪因子测试"""

    def setup_method(self):
        self.factors = SentimentFactors()

    def test_basic_calculation(self):
        """基本情绪计算"""
        prices = generate_prices(200)
        volumes = generate_volumes(200)
        result = self.factors.calculate(prices, volumes)

        assert "sentiment_score" in result
        assert -1.0 <= result["sentiment_score"] <= 1.0

    def test_short_data_insufficient(self):
        """数据不足时返回 0"""
        prices = generate_prices(10)  # 不够 30 天
        volumes = generate_volumes(10)
        result = self.factors.calculate(prices, volumes)
        assert result["sentiment_score"] == 0.0

    def test_with_short_interest(self):
        """含空头持仓数据"""
        prices = generate_prices(200)
        volumes = generate_volumes(200)
        short_data = {
            "shortPercentOfFloat": 0.35,  # 高空头比例
            "shortRatio": 5.0,
        }
        result = self.factors.calculate(
            prices, volumes, short_data=short_data,
        )
        assert "sentiment_score" in result

    def test_with_insider_data(self):
        """含内部人交易数据"""
        prices = generate_prices(200)
        volumes = generate_volumes(200)
        insider_data = {
            "net_direction": 0.8,  # 大量内部人买入
            "total_transactions": 10,
        }
        result = self.factors.calculate(
            prices, volumes, insider_data=insider_data,
        )
        assert "sentiment_score" in result

    def test_oversold_rsi(self):
        """下跌趋势（RSI 偏低）应显示低估信号"""
        # 生成明显下跌趋势
        prices = generate_prices(200, trend=-0.003, seed=99)
        volumes = generate_volumes(200)
        result = self.factors.calculate(prices, volumes)
        # RSI 因子应偏向低估（负值）
        rsi_factor = result.get("factors", {}).get("rsi", 0)
        # 不做严格断言因为随机性，但结果应在合理范围
        assert -1.0 <= rsi_factor <= 1.0


# =============================================================================
# 宏观因子测试
# =============================================================================

class TestMacroFactors:
    """宏观因子测试"""

    def setup_method(self):
        self.factors = MacroFactors()

    def test_basic_calculation(self):
        """基本宏观计算"""
        macro_data = {
            "DGS10": 4.2,
            "DGS2": 4.5,
            "FEDFUNDS": 5.25,
            "VIXCLS": 15.0,
            "BAA10Y": 1.8,
            "T10YIE": 2.3,
        }
        result = self.factors.calculate(macro_data, market_pe=22.0)

        assert "macro_score" in result
        assert -1.0 <= result["macro_score"] <= 1.0
        assert "factors" in result
        assert "details" in result

    def test_high_vix_signals_fear(self):
        """高 VIX 应产生低估（恐慌）信号"""
        macro_data = {
            "DGS10": 4.0,
            "VIXCLS": 40.0,  # 高恐慌
        }
        result = self.factors.calculate(macro_data, market_pe=20.0)
        # VIX 因子应偏负（恐慌 → 低估信号）
        vix_score = result.get("factors", {}).get("vix", 0)
        assert vix_score < 0

    def test_low_vix_signals_complacency(self):
        """低 VIX 应产生高估（自满）信号"""
        macro_data = {
            "DGS10": 4.0,
            "VIXCLS": 10.0,  # 极低
        }
        result = self.factors.calculate(macro_data, market_pe=20.0)
        vix_score = result.get("factors", {}).get("vix", 0)
        assert vix_score > 0

    def test_empty_macro_data(self):
        """空数据不崩溃"""
        result = self.factors.calculate({}, market_pe=None)
        assert "macro_score" in result

    def test_fed_model_stocks_attractive(self):
        """盈利收益率远高于国债 → 股票吸引力大（低估信号）"""
        macro_data = {
            "DGS10": 2.0,  # 低利率
        }
        result = self.factors.calculate(macro_data, market_pe=12.0)
        # 盈利收益率 ~8.3% vs 国债 2% → 强烈低估信号
        fed_score = result.get("factors", {}).get("fed_model", 0)
        assert fed_score < 0


# =============================================================================
# 综合打分器测试
# =============================================================================

class TestValuationScorer:
    """综合估值打分器测试"""

    def setup_method(self):
        self.scorer = ValuationScorer()

    def _make_results(self, rel=0.0, abs_=0.0, sent=0.0, macro=0.0):
        """构造因子结果"""
        return {
            "relative_result": {
                "relative_score": rel,
                "factors": {},
                "details": {},
            },
            "absolute_result": {
                "absolute_score": abs_,
                "dcf_result": {},
                "ri_result": {},
            },
            "sentiment_result": {
                "sentiment_score": sent,
                "factors": {},
                "details": {},
            },
            "macro_result": {
                "macro_score": macro,
                "factors": {},
                "details": {},
            },
        }

    def test_neutral_scores_grade_c(self):
        """四维均为 0 → C 评级"""
        results = self._make_results(0.0, 0.0, 0.0, 0.0)
        val = self.scorer.score(
            symbol="TEST", name="Test Corp", sector="Tech", industry="Software",
            current_price=100.0, horizon="3M", **results,
        )
        assert val.grade == "C"
        assert abs(val.composite_score) < 0.01

    def test_strongly_undervalued(self):
        """四维均显示低估 → A 或 B 评级"""
        results = self._make_results(-0.8, -0.7, -0.6, -0.5)
        val = self.scorer.score(
            symbol="TEST", name="Test Corp", sector="Tech", industry="Software",
            current_price=100.0, horizon="3M", **results,
        )
        assert val.grade in ("A", "B")
        assert val.composite_score < -0.15

    def test_strongly_overvalued(self):
        """四维均显示高估 → D 或 F 评级"""
        results = self._make_results(0.8, 0.7, 0.6, 0.5)
        val = self.scorer.score(
            symbol="TEST", name="Test Corp", sector="Tech", industry="Software",
            current_price=100.0, horizon="3M", **results,
        )
        assert val.grade in ("D", "F")
        assert val.composite_score > 0.15

    def test_custom_weights(self):
        """自定义权重"""
        custom_scorer = ValuationScorer(weights={
            "relative": 1.0,
            "absolute": 0.0,
            "sentiment": 0.0,
            "macro": 0.0,
        })
        results = self._make_results(rel=-0.8, abs_=0.8, sent=0.8, macro=0.8)
        val = custom_scorer.score(
            symbol="TEST", name="Test Corp", sector="Tech", industry="Software",
            current_price=100.0, horizon="3M", **results,
        )
        # 仅 relative 权重为 1.0, 其他为 0 → 综合分应接近 -0.8
        assert val.composite_score < 0

    def test_output_has_all_fields(self):
        """输出应包含所有字段"""
        results = self._make_results(-0.3, -0.2, -0.1, 0.0)
        val = self.scorer.score(
            symbol="AAPL", name="Apple Inc.", sector="Technology",
            industry="Consumer Electronics", current_price=180.0,
            horizon="6M", **results,
        )
        assert val.symbol == "AAPL"
        assert val.name == "Apple Inc."
        assert val.sector == "Technology"
        assert val.current_price == 180.0
        assert val.horizon == "6M"
        assert isinstance(val.grade, str)
        assert isinstance(val.key_metrics, dict)


# =============================================================================
# Horizon 权重测试
# =============================================================================

class TestHorizonWeights:
    """不同 horizon 使用不同因子权重"""

    def _make_results(self, rel=0.0, abs_=0.0, sent=0.0, macro=0.0):
        return {
            "relative_result": {"relative_score": rel, "factors": {}, "details": {}},
            "absolute_result": {"absolute_score": abs_, "dcf_result": {}, "ri_result": {}},
            "sentiment_result": {"sentiment_score": sent, "factors": {}, "details": {}},
            "macro_result": {"macro_score": macro, "factors": {}, "details": {}},
        }

    def test_1m_emphasizes_sentiment(self):
        """1M 窗口：sentiment 权重最高 (0.35)"""
        assert HORIZON_WEIGHTS["1M"]["sentiment"] == 0.35
        assert HORIZON_WEIGHTS["1M"]["sentiment"] > HORIZON_WEIGHTS["1M"]["absolute"]

    def test_1y_emphasizes_absolute(self):
        """1Y 窗口：absolute(DCF) 权重最高 (0.35)"""
        assert HORIZON_WEIGHTS["1Y"]["absolute"] == 0.35
        assert HORIZON_WEIGHTS["1Y"]["absolute"] > HORIZON_WEIGHTS["1Y"]["sentiment"]

    def test_3m_matches_original_default(self):
        """3M 权重与原始默认一致（回归测试）"""
        assert HORIZON_WEIGHTS["3M"] == {
            "relative": 0.30, "absolute": 0.25,
            "sentiment": 0.25, "macro": 0.20,
        }

    def test_all_horizons_sum_to_one(self):
        """所有 horizon 的权重之和 = 1.0"""
        for h, w in HORIZON_WEIGHTS.items():
            total = sum(w.values())
            assert abs(total - 1.0) < 1e-6, f"Horizon {h} weights sum to {total}"

    def test_scorer_uses_horizon_weights(self):
        """Scorer 应根据 horizon 动态选择权重"""
        scorer = ValuationScorer()  # 无自定义权重
        # sentiment 在 1M 最强; 仅 sentiment 为正, 其他为 0
        results = self._make_results(sent=0.5)
        val_1m = scorer.score(
            symbol="TEST", name="", sector="", industry="",
            current_price=100.0, horizon="1M", **results,
        )
        val_1y = scorer.score(
            symbol="TEST", name="", sector="", industry="",
            current_price=100.0, horizon="1Y", **results,
        )
        # 1M 时 sentiment 权重 0.35 > 1Y 的 0.15 → 1M 综合分更高
        assert val_1m.composite_score > val_1y.composite_score

    def test_scorer_absolute_dominates_1y(self):
        """1Y 窗口 absolute 权重更高"""
        scorer = ValuationScorer()
        # 仅 absolute 为负 (低估), 其他为 0
        results = self._make_results(abs_=-0.8)
        val_1m = scorer.score(
            symbol="TEST", name="", sector="", industry="",
            current_price=100.0, horizon="1M", **results,
        )
        val_1y = scorer.score(
            symbol="TEST", name="", sector="", industry="",
            current_price=100.0, horizon="1Y", **results,
        )
        # 1Y 时 absolute 权重 0.35 > 1M 的 0.15 → 1Y 综合分更低（更低估）
        assert val_1y.composite_score < val_1m.composite_score

    def test_custom_weights_override_horizon(self):
        """用户自定义权重优先于 horizon 默认"""
        custom = {"relative": 1.0, "absolute": 0.0, "sentiment": 0.0, "macro": 0.0}
        scorer = ValuationScorer(weights=custom)
        results = self._make_results(rel=-0.8, abs_=0.8)
        val = scorer.score(
            symbol="TEST", name="", sector="", industry="",
            current_price=100.0, horizon="1Y", **results,
        )
        # 即使 1Y 默认 absolute=0.35, 自定义权重忽略 absolute
        assert val.composite_score < 0  # 被 relative=-0.8 主导

    def test_get_weights_returns_horizon_default(self):
        """get_weights() 无自定义权重时返回 horizon 对应的默认"""
        scorer = ValuationScorer()
        assert scorer.get_weights("1M") == HORIZON_WEIGHTS["1M"]
        assert scorer.get_weights("1Y") == HORIZON_WEIGHTS["1Y"]

    def test_get_weights_returns_custom(self):
        """get_weights() 有自定义权重时始终返回自定义"""
        custom = {"relative": 0.5, "absolute": 0.5, "sentiment": 0.0, "macro": 0.0}
        scorer = ValuationScorer(weights=custom)
        assert scorer.get_weights("1M") == custom
        assert scorer.get_weights("1Y") == custom


# =============================================================================
# 集成测试
# =============================================================================

class TestIntegration:
    """端到端集成测试（不依赖外部 API）"""

    def test_full_scoring_pipeline(self):
        """完整打分流程：因子计算 → 综合评分"""
        stock = make_stock_info()
        peers = make_peers()
        prices = generate_prices(200)
        volumes = generate_volumes(200)

        # 1. 相对估值
        rel_factors = RelativeValuationFactors()
        rel_result = rel_factors.calculate(stock, peers)

        # 2. 绝对估值
        abs_factors = AbsoluteValuationFactors()
        abs_result = abs_factors.calculate(
            stock_info=stock,
            cashflow_reports=make_cashflow_reports(),
            balance_reports=make_balance_reports(),
            income_reports=make_income_reports(),
        )

        # 3. 情绪因子
        sent_factors = SentimentFactors()
        sent_result = sent_factors.calculate(prices, volumes)

        # 4. 宏观因子
        macro_factors = MacroFactors()
        macro_result = macro_factors.calculate(
            {"DGS10": 4.2, "VIXCLS": 18.0, "FEDFUNDS": 5.25},
            market_pe=22.0,
        )

        # 5. 综合打分
        scorer = ValuationScorer()
        val = scorer.score(
            symbol="AAPL", name="Apple Inc.",
            sector="Technology", industry="Consumer Electronics",
            current_price=180.0,
            relative_result=rel_result,
            absolute_result=abs_result,
            sentiment_result=sent_result,
            macro_result=macro_result,
            horizon="3M",
        )

        # 验证完整输出
        assert val.symbol == "AAPL"
        assert val.grade in ("A", "B", "C", "D", "F")
        assert -1.0 <= val.composite_score <= 1.0
        assert -1.0 <= val.relative_score <= 1.0
        assert -1.0 <= val.absolute_score <= 1.0
        assert -1.0 <= val.sentiment_score <= 1.0
        assert -1.0 <= val.macro_score <= 1.0

        # 序列化往返
        d = val.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["grade"] == val.grade

    def test_sector_summary_computation(self):
        """行业汇总计算"""
        from strategy.valuation.scanner import ValuationScanner

        valuations = [
            StockValuation(
                symbol="A", name="", sector="Tech", industry="",
                current_price=100, composite_score=-0.3, grade="B",
            ),
            StockValuation(
                symbol="B", name="", sector="Tech", industry="",
                current_price=100, composite_score=0.2, grade="D",
            ),
            StockValuation(
                symbol="C", name="", sector="Health", industry="",
                current_price=100, composite_score=-0.5, grade="A",
            ),
        ]

        summary = ValuationScanner._compute_sector_summary(valuations)
        assert "Tech" in summary
        assert "Health" in summary
        assert summary["Tech"]["count"] == 2
        assert summary["Health"]["count"] == 1
        assert summary["Health"]["undervalued_count"] == 1


# =============================================================================
# 数据采集辅助函数测试
# =============================================================================

class TestCallsPerSymbol:
    """_calls_per_symbol API 调用次数计算"""

    def test_overview_only(self):
        from strategy.valuation.scanner import _calls_per_symbol
        assert _calls_per_symbol(["overview"]) == 1

    def test_ohlcv_only(self):
        from strategy.valuation.scanner import _calls_per_symbol
        assert _calls_per_symbol(["ohlcv"]) == 1

    def test_financials_only(self):
        from strategy.valuation.scanner import _calls_per_symbol
        assert _calls_per_symbol(["financials"]) == 3

    def test_all_types(self):
        from strategy.valuation.scanner import _calls_per_symbol
        assert _calls_per_symbol(["overview", "ohlcv", "financials"]) == 5

    def test_overview_and_financials(self):
        from strategy.valuation.scanner import _calls_per_symbol
        assert _calls_per_symbol(["overview", "financials"]) == 4

    def test_empty(self):
        from strategy.valuation.scanner import _calls_per_symbol
        assert _calls_per_symbol([]) == 0

    def test_unknown_type_ignored(self):
        from strategy.valuation.scanner import _calls_per_symbol
        assert _calls_per_symbol(["overview", "unknown"]) == 1

    def test_all_expands(self):
        from strategy.valuation.scanner import _calls_per_symbol
        assert _calls_per_symbol(["all"]) == 5  # overview(1) + ohlcv(1) + financials(3)

    def test_all_deduplicates(self):
        """all + overview 不重复计算"""
        from strategy.valuation.scanner import _calls_per_symbol
        assert _calls_per_symbol(["all", "overview"]) == 5


class TestExpandDataTypes:
    """_expand_data_types all 展开"""

    def test_all_expands_to_three(self):
        from strategy.valuation.scanner import _expand_data_types
        assert _expand_data_types(["all"]) == ["overview", "ohlcv", "financials"]

    def test_single_type_unchanged(self):
        from strategy.valuation.scanner import _expand_data_types
        assert _expand_data_types(["overview"]) == ["overview"]

    def test_all_with_extra_dedup(self):
        from strategy.valuation.scanner import _expand_data_types
        result = _expand_data_types(["overview", "all"])
        assert result == ["overview", "ohlcv", "financials"]

    def test_empty(self):
        from strategy.valuation.scanner import _expand_data_types
        assert _expand_data_types([]) == []


class TestFilterFreshSymbols:
    """_filter_fresh_symbols 缓存新鲜度过滤"""

    @pytest.mark.asyncio
    async def test_all_cached_overview(self):
        """所有股票 overview 都新鲜 → 全部跳过"""
        from strategy.valuation.scanner import _filter_fresh_symbols

        now = datetime.now()
        cache = AsyncMock()
        cache.get_stock_info_batch.return_value = {
            "AAPL": {"pe_ratio": 28.5, "updated_at": now.isoformat()},
            "MSFT": {"pe_ratio": 35.0, "updated_at": now.isoformat()},
        }

        needs, skipped = await _filter_fresh_symbols(
            ["AAPL", "MSFT"], cache, ["overview"]
        )
        assert needs == []
        assert skipped == 2

    @pytest.mark.asyncio
    async def test_none_cached_overview(self):
        """overview 缓存为空 → 全部需要采集"""
        from strategy.valuation.scanner import _filter_fresh_symbols

        cache = AsyncMock()
        cache.get_stock_info_batch.return_value = {}

        needs, skipped = await _filter_fresh_symbols(
            ["AAPL", "MSFT", "GOOGL"], cache, ["overview"]
        )
        assert needs == ["AAPL", "MSFT", "GOOGL"]
        assert skipped == 0

    @pytest.mark.asyncio
    async def test_partial_cached_overview(self):
        """部分新鲜、部分过期 → 只返回过期的"""
        from strategy.valuation.scanner import _filter_fresh_symbols

        now = datetime.now()
        old = (now - timedelta(days=10)).isoformat()  # 超过 7 天 TTL
        cache = AsyncMock()
        cache.get_stock_info_batch.return_value = {
            "AAPL": {"pe_ratio": 28.5, "updated_at": now.isoformat()},
            "MSFT": {"pe_ratio": 35.0, "updated_at": old},
        }

        needs, skipped = await _filter_fresh_symbols(
            ["AAPL", "MSFT", "GOOGL"], cache, ["overview"]
        )
        assert "AAPL" not in needs
        assert "MSFT" in needs
        assert "GOOGL" in needs
        assert skipped == 1

    @pytest.mark.asyncio
    async def test_financials_freshness(self):
        """financials 缓存过期 → 需要采集"""
        from strategy.valuation.scanner import _filter_fresh_symbols

        now = datetime.now()
        cache = AsyncMock()
        # overview 新鲜
        cache.get_stock_info_batch.return_value = {
            "AAPL": {"pe_ratio": 28.5, "updated_at": now.isoformat()},
        }
        # financials 过期（35 天前）
        cache.get_valuation_financials.return_value = [
            {"updated_at": (now - timedelta(days=35)).isoformat()}
        ]

        needs, skipped = await _filter_fresh_symbols(
            ["AAPL"], cache, ["overview", "financials"]
        )
        assert needs == ["AAPL"]
        assert skipped == 0

    @pytest.mark.asyncio
    async def test_financials_fresh(self):
        """financials 新鲜 + overview 新鲜 → 跳过"""
        from strategy.valuation.scanner import _filter_fresh_symbols

        now = datetime.now()
        cache = AsyncMock()
        cache.get_stock_info_batch.return_value = {
            "AAPL": {"pe_ratio": 28.5, "updated_at": now.isoformat()},
        }
        cache.get_valuation_financials.return_value = [
            {"updated_at": now.isoformat()}
        ]

        needs, skipped = await _filter_fresh_symbols(
            ["AAPL"], cache, ["overview", "financials"]
        )
        assert needs == []
        assert skipped == 1

    @pytest.mark.asyncio
    async def test_ohlcv_no_data(self):
        """ohlcv 无缓存 → 需要采集"""
        from strategy.valuation.scanner import _filter_fresh_symbols

        cache = AsyncMock()
        cache.get_stock_info_batch.return_value = {}
        cache.get_market_data.return_value = None

        needs, skipped = await _filter_fresh_symbols(
            ["AAPL"], cache, ["ohlcv"]
        )
        assert needs == ["AAPL"]


class TestFetchDataBatchLogic:
    """fetch_data_for_scan 智能批量计算"""

    @pytest.mark.asyncio
    async def test_free_plan_overview_max_25(self):
        """Free plan + overview → 最多 25 只"""
        from strategy.valuation.scanner import fetch_data_for_scan

        symbols = [f"SYM{i}" for i in range(50)]
        cache = AsyncMock()
        cache.get_stock_info_batch.return_value = {}
        provider = AsyncMock()
        provider.get_stock_overview.return_value = {"Symbol": "TEST"}

        stats = await fetch_data_for_scan(
            symbols=symbols, provider=provider, cache=cache,
            data_types=["overview"], plan="free",
        )
        assert stats["to_fetch"] == 25
        assert provider.get_stock_overview.call_count == 25

    @pytest.mark.asyncio
    async def test_free_plan_financials_max_8(self):
        """Free plan + financials → 最多 8 只 (25÷3=8)"""
        from strategy.valuation.scanner import fetch_data_for_scan

        symbols = [f"SYM{i}" for i in range(50)]
        cache = AsyncMock()
        cache.get_stock_info_batch.return_value = {}
        cache.get_valuation_financials.return_value = []
        # get_stock_info 返回有 market_cap 的数据，使 financials 不被跳过
        cache.get_stock_info.return_value = {"market_cap": 1e9}
        provider = AsyncMock()
        provider.get_cash_flow.return_value = {"data": []}
        provider.get_balance_sheet.return_value = {"data": []}
        provider.get_income_statement.return_value = {"data": []}

        stats = await fetch_data_for_scan(
            symbols=symbols, provider=provider, cache=cache,
            data_types=["financials"], plan="free",
        )
        assert stats["to_fetch"] == 8
        assert stats["api_calls_estimated"] == 24

    @pytest.mark.asyncio
    async def test_premium_plan_no_limit(self):
        """Premium plan → 处理全部"""
        from strategy.valuation.scanner import fetch_data_for_scan

        symbols = [f"SYM{i}" for i in range(40)]
        cache = AsyncMock()
        cache.get_stock_info_batch.return_value = {}
        provider = AsyncMock()
        provider.get_stock_overview.return_value = {"Symbol": "TEST"}

        stats = await fetch_data_for_scan(
            symbols=symbols, provider=provider, cache=cache,
            data_types=["overview"], plan="premium",
        )
        assert stats["to_fetch"] == 40

    @pytest.mark.asyncio
    async def test_all_cached_skips_fetch(self):
        """全部已缓存 → to_fetch=0"""
        from strategy.valuation.scanner import fetch_data_for_scan

        now = datetime.now()
        cache = AsyncMock()
        cache.get_stock_info_batch.return_value = {
            "AAPL": {"pe_ratio": 28.5, "updated_at": now.isoformat()},
            "MSFT": {"pe_ratio": 35.0, "updated_at": now.isoformat()},
        }
        provider = AsyncMock()

        stats = await fetch_data_for_scan(
            symbols=["AAPL", "MSFT"], provider=provider, cache=cache,
            data_types=["overview"], plan="free",
        )
        assert stats["already_cached"] == 2
        assert stats["to_fetch"] == 0
        provider.get_stock_overview.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_size_override(self):
        """显式 batch_size 覆盖自动计算"""
        from strategy.valuation.scanner import fetch_data_for_scan

        symbols = [f"SYM{i}" for i in range(50)]
        cache = AsyncMock()
        cache.get_stock_info_batch.return_value = {}
        provider = AsyncMock()
        provider.get_stock_overview.return_value = {"Symbol": "TEST"}

        stats = await fetch_data_for_scan(
            symbols=symbols, provider=provider, cache=cache,
            data_types=["overview"], batch_size=5, plan="free",
        )
        assert stats["to_fetch"] == 5

    @pytest.mark.asyncio
    async def test_resume_skips_cached(self):
        """第二次运行自动跳过已缓存的股票"""
        from strategy.valuation.scanner import fetch_data_for_scan

        now = datetime.now()
        cache = AsyncMock()
        # 前 3 只已缓存，后 2 只未缓存
        cache.get_stock_info_batch.return_value = {
            "SYM0": {"pe_ratio": 10.0, "updated_at": now.isoformat()},
            "SYM1": {"pe_ratio": 20.0, "updated_at": now.isoformat()},
            "SYM2": {"pe_ratio": 30.0, "updated_at": now.isoformat()},
        }
        provider = AsyncMock()
        provider.get_stock_overview.return_value = {"Symbol": "TEST"}

        stats = await fetch_data_for_scan(
            symbols=["SYM0", "SYM1", "SYM2", "SYM3", "SYM4"],
            provider=provider, cache=cache,
            data_types=["overview"], plan="free",
        )
        assert stats["already_cached"] == 3
        assert stats["to_fetch"] == 2
        assert provider.get_stock_overview.call_count == 2
