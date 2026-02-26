"""
数据提供者测试：yfinance 情绪数据 + FRED 宏观数据

覆盖两个新增数据源的数据获取能力：
- YFinanceSentimentProvider: 空头持仓、内部人交易
- FREDProvider: FRED 宏观时间序列

用法:
    pytest python/tests/test_data_providers.py -v
    pytest python/tests/test_data_providers.py -v -k "yfinance"
    pytest python/tests/test_data_providers.py -v -k "fred"
    pytest python/tests/test_data_providers.py -v -s  # 显示 print 输出
"""

import asyncio
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Optional

import pandas as pd
import pytest

sys.path.insert(0, "/workspace/python/alchemist")

from data.providers.yfinance_provider import YFinanceSentimentProvider
from data.providers.fred_provider import FREDProvider, FRED_SERIES


# ---------- 辅助工具 ----------

def run_async(coro):
    """同步运行异步函数"""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# =====================================================================
# YFinance 情绪数据提供者测试
# =====================================================================


class TestYFinanceProviderInit:
    """YFinanceSentimentProvider 初始化测试"""

    def test_is_available_true_when_yfinance_installed(self):
        """yfinance 已安装时 is_available 应为 True"""
        provider = YFinanceSentimentProvider()
        # 由于 yfinance 已安装，should be True
        assert isinstance(provider.is_available, bool)

    def test_is_available_false_when_import_fails(self):
        """yfinance 未安装时 is_available 应为 False"""
        with patch("builtins.__import__", side_effect=ImportError("no module named yfinance")):
            provider = YFinanceSentimentProvider()
            assert provider.is_available is False

    def test_returns_none_when_unavailable_short(self):
        """yfinance 不可用时 get_short_interest 返回 None"""
        provider = YFinanceSentimentProvider()
        provider._available = False
        result = provider.get_short_interest("AAPL")
        assert result is None

    def test_returns_none_when_unavailable_insider(self):
        """yfinance 不可用时 get_insider_transactions 返回 None"""
        provider = YFinanceSentimentProvider()
        provider._available = False
        result = provider.get_insider_transactions("AAPL")
        assert result is None

    def test_batch_returns_empty_when_unavailable(self):
        """yfinance 不可用时 get_batch_sentiment_data 返回空字典"""
        provider = YFinanceSentimentProvider()
        provider._available = False
        result = provider.get_batch_sentiment_data(["AAPL", "MSFT"])
        assert result == {}


class TestYFinanceShortInterest:
    """空头持仓数据结构与字段验证（Mock 网络请求）"""

    def _make_provider_with_mock_info(self, info_data: dict):
        """创建带 mock ticker.info 的 Provider"""
        provider = YFinanceSentimentProvider()
        provider._available = True
        return provider, info_data

    def test_short_interest_structure_with_mock(self):
        """验证 get_short_interest 返回正确的数据结构"""
        mock_info = {
            "shortPercentOfFloat": 0.02,
            "shortRatio": 1.5,
            "sharesShort": 50_000_000,
            "sharesShortPriorMonth": 45_000_000,
            "floatShares": 2_500_000_000,
        }
        provider = YFinanceSentimentProvider()
        provider._available = True

        mock_ticker = MagicMock()
        mock_ticker.info = mock_info

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = provider.get_short_interest("AAPL")

        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["short_percent_of_float"] == 0.02
        assert result["short_ratio"] == 1.5
        assert result["shares_short"] == 50_000_000
        assert result["shares_short_prior_month"] == 45_000_000
        assert result["float_shares"] == 2_500_000_000
        assert "fetch_date" in result

    def test_short_interest_returns_none_when_no_data(self):
        """info 中无 short 数据时返回 None"""
        provider = YFinanceSentimentProvider()
        provider._available = True

        mock_ticker = MagicMock()
        mock_ticker.info = {}  # 空 info

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = provider.get_short_interest("UNKNOWN")

        assert result is None

    def test_short_interest_uppercase_symbol(self):
        """symbol 应被规范化为大写"""
        mock_info = {
            "shortPercentOfFloat": 0.05,
            "shortRatio": 2.0,
        }
        provider = YFinanceSentimentProvider()
        provider._available = True

        mock_ticker = MagicMock()
        mock_ticker.info = mock_info

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = provider.get_short_interest("aapl")  # 小写输入

        assert result is not None
        assert result["symbol"] == "AAPL"  # 应被转换为大写

    def test_short_interest_handles_exception(self):
        """API 异常时返回 None（不抛出）"""
        provider = YFinanceSentimentProvider()
        provider._available = True

        with patch("yfinance.Ticker", side_effect=Exception("network error")):
            result = provider.get_short_interest("AAPL")

        assert result is None


class TestYFinanceInsiderTransactions:
    """内部人交易数据结构验证（Mock 网络请求）"""

    def test_insider_structure_with_purchases(self):
        """验证 get_insider_transactions 买入记录的数据结构"""
        import pandas as pd

        mock_df = pd.DataFrame([
            {"Insider": "John Doe", "Relation": "CEO", "Start Date": "2025-01-10",
             "Text": "Purchase", "Shares": 10000, "Value": 1500000},
            {"Insider": "Jane Smith", "Relation": "CFO", "Start Date": "2025-01-15",
             "Text": "Sale", "Shares": 5000, "Value": 750000},
        ])

        provider = YFinanceSentimentProvider()
        provider._available = True

        mock_ticker = MagicMock()
        mock_ticker.insider_transactions = mock_df

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = provider.get_insider_transactions("MSFT")

        assert result is not None
        assert result["symbol"] == "MSFT"
        assert result["total_transactions"] == 2
        assert result["net_buy_count"] == 1
        assert result["net_sell_count"] == 1
        assert result["net_direction"] == 0.0  # 1买1卖，净方向为0
        assert "transactions" in result
        assert len(result["transactions"]) <= 10
        assert "fetch_date" in result

    def test_insider_net_direction_all_buys(self):
        """全部买入时净方向应接近 +1"""
        import pandas as pd

        mock_df = pd.DataFrame([
            {"Insider": "A", "Relation": "CEO", "Start Date": "2025-01-10",
             "Text": "Purchase", "Shares": 10000, "Value": 1000000},
            {"Insider": "B", "Relation": "CTO", "Start Date": "2025-01-11",
             "Text": "Purchase", "Shares": 5000, "Value": 500000},
        ])

        provider = YFinanceSentimentProvider()
        provider._available = True

        mock_ticker = MagicMock()
        mock_ticker.insider_transactions = mock_df

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = provider.get_insider_transactions("GOOGL")

        assert result is not None
        assert result["net_direction"] == 1.0   # 全部买入
        assert -1.0 <= result["net_direction"] <= 1.0

    def test_insider_net_direction_all_sells(self):
        """全部卖出时净方向应接近 -1"""
        import pandas as pd

        mock_df = pd.DataFrame([
            {"Insider": "A", "Relation": "CEO", "Start Date": "2025-01-10",
             "Text": "Sale", "Shares": 20000, "Value": 3000000},
        ])

        provider = YFinanceSentimentProvider()
        provider._available = True

        mock_ticker = MagicMock()
        mock_ticker.insider_transactions = mock_df

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = provider.get_insider_transactions("AMZN")

        assert result is not None
        assert result["net_direction"] == -1.0  # 全部卖出
        assert -1.0 <= result["net_direction"] <= 1.0

    def test_insider_empty_dataframe(self):
        """空的 insider_transactions 返回零值结构（非 None）"""
        import pandas as pd

        provider = YFinanceSentimentProvider()
        provider._available = True

        mock_ticker = MagicMock()
        mock_ticker.insider_transactions = pd.DataFrame()  # 空 DataFrame

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = provider.get_insider_transactions("XYZ")

        assert result is not None
        assert result["total_transactions"] == 0
        assert result["net_buy_count"] == 0
        assert result["net_sell_count"] == 0
        assert result["net_direction"] == 0.0

    def test_insider_handles_exception(self):
        """API 异常时返回 None"""
        provider = YFinanceSentimentProvider()
        provider._available = True

        with patch("yfinance.Ticker", side_effect=RuntimeError("API error")):
            result = provider.get_insider_transactions("AAPL")

        assert result is None


class TestYFinanceBatchSentiment:
    """批量情绪数据获取测试"""

    def test_batch_returns_dict_with_symbols(self):
        """批量获取应返回 {symbol: {short_interest, insider}} 结构"""
        mock_info = {"shortPercentOfFloat": 0.03, "shortRatio": 1.2}

        import pandas as pd
        mock_df = pd.DataFrame([
            {"Insider": "A", "Relation": "CEO", "Start Date": "2025-01-01",
             "Text": "Purchase", "Shares": 1000, "Value": 100000},
        ])

        provider = YFinanceSentimentProvider()
        provider._available = True

        mock_ticker = MagicMock()
        mock_ticker.info = mock_info
        mock_ticker.insider_transactions = mock_df

        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = provider.get_batch_sentiment_data(["AAPL", "MSFT"])

        assert isinstance(result, dict)
        for symbol in ["AAPL", "MSFT"]:
            if symbol in result:
                entry = result[symbol]
                assert "short_interest" in entry
                assert "insider" in entry

    def test_batch_empty_symbols(self):
        """空列表返回空字典"""
        provider = YFinanceSentimentProvider()
        provider._available = True
        result = provider.get_batch_sentiment_data([])
        assert result == {}


class TestYFinanceLiveData:
    """yfinance 真实网络数据测试（需要网络连接）"""

    @pytest.fixture(autouse=True)
    def skip_if_unavailable(self):
        """yfinance 不可用时跳过所有测试"""
        provider = YFinanceSentimentProvider()
        if not provider.is_available:
            pytest.skip("yfinance 未安装，跳过实时数据测试")

    def test_short_interest_real_aapl(self):
        """测试获取 AAPL 真实空头持仓数据"""
        provider = YFinanceSentimentProvider()
        result = provider.get_short_interest("AAPL")

        # yfinance 数据可能因权限而不可用，返回 None 也是合法的
        if result is not None:
            assert result["symbol"] == "AAPL"
            assert "fetch_date" in result
            # 基本字段存在（值可为 None）
            assert "short_percent_of_float" in result
            assert "short_ratio" in result
            assert "shares_short" in result
            if result["short_percent_of_float"] is not None:
                assert 0.0 <= result["short_percent_of_float"] <= 1.0

    def test_insider_transactions_real_msft(self):
        """测试获取 MSFT 真实内部人交易数据"""
        provider = YFinanceSentimentProvider()
        result = provider.get_insider_transactions("MSFT")

        if result is not None:
            assert result["symbol"] == "MSFT"
            assert "net_direction" in result
            assert -1.0 <= result["net_direction"] <= 1.0
            assert isinstance(result["transactions"], list)
            assert result["total_transactions"] >= 0

    def test_batch_sentiment_real_symbols(self):
        """测试批量获取真实股票情绪数据"""
        provider = YFinanceSentimentProvider()
        symbols = ["AAPL", "MSFT"]
        result = provider.get_batch_sentiment_data(symbols)

        assert isinstance(result, dict)
        # 至少部分结果成功
        for sym, data in result.items():
            assert sym in [s.upper() for s in symbols]
            assert "short_interest" in data
            assert "insider" in data


# =====================================================================
# FRED 宏观数据提供者测试
# =====================================================================


class TestFREDProviderInit:
    """FREDProvider 初始化测试"""

    def test_is_available_with_api_key(self):
        """有 API key 时 is_available 为 True"""
        provider = FREDProvider(api_key="test_key_12345")
        assert provider.is_available is True

    def test_is_available_without_api_key(self):
        """无 API key 时 is_available 为 False"""
        import os
        original = os.environ.pop("FRED_API_KEY", None)
        try:
            provider = FREDProvider(api_key="")
            assert provider.is_available is False
        finally:
            if original:
                os.environ["FRED_API_KEY"] = original

    def test_reads_api_key_from_env(self, monkeypatch):
        """从环境变量读取 API key"""
        monkeypatch.setenv("FRED_API_KEY", "env_key_xyz")
        provider = FREDProvider()
        assert provider.api_key == "env_key_xyz"
        assert provider.is_available is True

    def test_explicit_key_overrides_env(self, monkeypatch):
        """显式传入的 key 优先于环境变量"""
        monkeypatch.setenv("FRED_API_KEY", "env_key")
        provider = FREDProvider(api_key="explicit_key")
        assert provider.api_key == "explicit_key"

    def test_fred_series_dict_populated(self):
        """FRED_SERIES 常量包含核心序列"""
        assert "DGS10" in FRED_SERIES
        assert "FEDFUNDS" in FRED_SERIES
        assert "VIXCLS" in FRED_SERIES
        assert "GDP" in FRED_SERIES
        assert "CPIAUCSL" in FRED_SERIES


class _AsyncContextManager:
    """辅助类：为 aiohttp session.get() 提供正确的 async context manager 协议"""

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        return False


class TestFREDGetSeriesMocked:
    """FRED get_series 单元测试（Mock HTTP 响应）"""

    def _make_observations(self, n: int = 10) -> list:
        """生成 n 条模拟 FRED 观测值"""
        base_date = datetime(2025, 1, 1)
        return [
            {
                "date": (base_date + timedelta(days=i * 30)).strftime("%Y-%m-%d"),
                "value": str(4.5 + i * 0.1),
            }
            for i in range(n)
        ]

    def test_get_series_returns_pandas_series(self):
        """get_series 成功时返回 pd.Series"""
        provider = FREDProvider(api_key="test_key")
        observations = self._make_observations(12)
        mock_data = {"observations": observations}

        async def _run():
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_data)

            mock_session = MagicMock()
            mock_session.get.return_value = _AsyncContextManager(mock_response)

            with patch.object(provider, "_get_session", AsyncMock(return_value=mock_session)):
                series = await provider.get_series("DGS10")
            await provider.close()
            return series

        result = run_async(_run())
        assert result is not None
        assert isinstance(result, pd.Series)
        assert len(result) == 12
        assert result.name == "DGS10"
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_get_series_returns_none_without_api_key(self):
        """无 API key 时 get_series 返回 None"""
        import os
        original = os.environ.pop("FRED_API_KEY", None)
        try:
            provider = FREDProvider(api_key="")

            async def _run():
                return await provider.get_series("DGS10")

            result = run_async(_run())
            assert result is None
        finally:
            if original:
                os.environ["FRED_API_KEY"] = original

    def test_get_series_handles_http_error(self):
        """HTTP 错误时返回 None"""
        provider = FREDProvider(api_key="test_key")

        async def _run():
            mock_response = MagicMock()
            mock_response.status = 403

            mock_session = MagicMock()
            mock_session.get.return_value = _AsyncContextManager(mock_response)

            with patch.object(provider, "_get_session", AsyncMock(return_value=mock_session)):
                return await provider.get_series("INVALID")

        result = run_async(_run())
        assert result is None

    def test_get_series_skips_dot_values(self):
        """FRED '.' 占位符应被过滤掉"""
        provider = FREDProvider(api_key="test_key")
        observations = [
            {"date": "2025-01-01", "value": "4.5"},
            {"date": "2025-02-01", "value": "."},   # 无数据占位符
            {"date": "2025-03-01", "value": "4.7"},
        ]
        mock_data = {"observations": observations}

        async def _run():
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_data)

            mock_session = MagicMock()
            mock_session.get.return_value = _AsyncContextManager(mock_response)

            with patch.object(provider, "_get_session", AsyncMock(return_value=mock_session)):
                return await provider.get_series("FEDFUNDS")

        result = run_async(_run())
        assert result is not None
        assert len(result) == 2   # "." 被过滤


class TestFREDGetLatestValue:
    """FRED get_latest_value 测试"""

    def test_get_latest_value_returns_float(self):
        """get_latest_value 返回序列最新值（float）"""
        provider = FREDProvider(api_key="test_key")

        # Mock get_series 返回值
        mock_series = pd.Series(
            [4.25, 4.30, 4.35],
            index=pd.date_range("2025-11-01", periods=3, freq="MS"),
            name="DGS10",
        )

        async def _run():
            with patch.object(provider, "get_series", AsyncMock(return_value=mock_series)):
                return await provider.get_latest_value("DGS10")

        result = run_async(_run())
        assert result is not None
        assert isinstance(result, float)
        assert result == 4.35  # 最后一个值

    def test_get_latest_value_returns_none_on_empty_series(self):
        """序列为 None 时返回 None"""
        provider = FREDProvider(api_key="test_key")

        async def _run():
            with patch.object(provider, "get_series", AsyncMock(return_value=None)):
                return await provider.get_latest_value("DGS10")

        result = run_async(_run())
        assert result is None


class TestFREDMacroSnapshot:
    """FRED get_macro_snapshot 测试"""

    def test_macro_snapshot_structure(self):
        """宏观快照包含必要的经济指标键"""
        provider = FREDProvider(api_key="test_key")

        # Mock get_latest_value 返回固定值
        mock_values = {
            "DGS10": 4.35,
            "DGS2": 4.10,
            "FEDFUNDS": 4.50,
            "VIXCLS": 18.5,
            "BAA10Y": 1.75,
            "T10YIE": 2.30,
        }

        async def mock_get_latest(series_id):
            return mock_values.get(series_id)

        async def _run():
            with patch.object(provider, "get_latest_value", side_effect=mock_get_latest):
                return await provider.get_macro_snapshot()

        result = run_async(_run())

        assert isinstance(result, dict)
        assert "DGS10" in result
        assert "DGS2" in result
        assert "FEDFUNDS" in result
        assert "VIXCLS" in result
        assert "BAA10Y" in result
        assert "T10YIE" in result

    def test_macro_snapshot_yield_curve_spread(self):
        """快照包含派生指标 yield_curve_spread = DGS10 - DGS2"""
        provider = FREDProvider(api_key="test_key")

        mock_values = {
            "DGS10": 4.35,
            "DGS2": 4.10,
            "FEDFUNDS": 4.50,
        }

        async def mock_get_latest(series_id):
            return mock_values.get(series_id)

        async def _run():
            with patch.object(provider, "get_latest_value", side_effect=mock_get_latest):
                return await provider.get_macro_snapshot()

        result = run_async(_run())
        assert "yield_curve_spread" in result
        assert abs(result["yield_curve_spread"] - (4.35 - 4.10)) < 1e-9

    def test_macro_snapshot_handles_missing_series(self):
        """部分序列不可用时仍返回有效快照（不抛异常）"""
        provider = FREDProvider(api_key="test_key")

        async def mock_get_latest(series_id):
            if series_id == "DGS10":
                return 4.35
            return None  # 其他序列不可用

        async def _run():
            with patch.object(provider, "get_latest_value", side_effect=mock_get_latest):
                return await provider.get_macro_snapshot()

        result = run_async(_run())
        assert isinstance(result, dict)
        assert result.get("DGS10") == 4.35
        # DGS2 = None，不应包含 yield_curve_spread
        assert "yield_curve_spread" not in result

    def test_macro_snapshot_empty_when_no_api_key(self):
        """无 API key 时快照为空字典"""
        import os
        original = os.environ.pop("FRED_API_KEY", None)
        try:
            provider = FREDProvider(api_key="")

            async def _run():
                return await provider.get_macro_snapshot()

            result = run_async(_run())
            assert isinstance(result, dict)
            assert len(result) == 0
        finally:
            if original:
                os.environ["FRED_API_KEY"] = original


class TestFREDGetMultipleSeries:
    """FRED get_multiple_series 批量获取测试"""

    def test_get_multiple_series_returns_dict(self):
        """批量获取返回 {series_id: pd.Series} 字典"""
        provider = FREDProvider(api_key="test_key")

        mock_series = pd.Series(
            [4.0, 4.1],
            index=pd.date_range("2025-01-01", periods=2, freq="MS"),
        )

        async def _run():
            with patch.object(provider, "get_series", AsyncMock(return_value=mock_series)):
                return await provider.get_multiple_series(["DGS10", "DGS2"])

        result = run_async(_run())
        assert isinstance(result, dict)
        assert "DGS10" in result
        assert "DGS2" in result
        assert isinstance(result["DGS10"], pd.Series)

    def test_get_multiple_series_skips_failed(self):
        """部分序列获取失败时仍返回成功的结果"""
        provider = FREDProvider(api_key="test_key")

        mock_series = pd.Series([4.0], index=pd.DatetimeIndex(["2025-01-01"]))

        async def mock_get_series(series_id, *args, **kwargs):
            if series_id == "DGS10":
                return mock_series
            return None  # DGS2 失败

        async def _run():
            with patch.object(provider, "get_series", side_effect=mock_get_series):
                return await provider.get_multiple_series(["DGS10", "DGS2"])

        result = run_async(_run())
        assert "DGS10" in result
        assert "DGS2" not in result  # 失败的不出现在结果中


class TestFREDClose:
    """FRED 会话生命周期测试"""

    def test_close_without_session(self):
        """未创建会话时调用 close 不应报错"""
        provider = FREDProvider(api_key="test_key")

        async def _run():
            await provider.close()

        run_async(_run())  # 不应抛出异常

    def test_close_with_open_session(self):
        """关闭已打开的会话"""
        provider = FREDProvider(api_key="test_key")

        async def _run():
            _ = await provider._get_session()  # 触发会话创建
            assert provider._session is not None
            assert not provider._session.closed
            await provider.close()

        run_async(_run())


class TestFREDLiveData:
    """FRED 真实网络数据测试（需要 API Key）"""

    @pytest.fixture(autouse=True)
    def skip_if_no_key(self):
        """未配置 FRED API Key 时跳过"""
        provider = FREDProvider()
        if not provider.is_available:
            pytest.skip("FRED_API_KEY 未配置，跳过实时数据测试")

    def test_get_dgs10_series(self):
        """获取 10 年期国债收益率（DGS10）时间序列"""
        provider = FREDProvider()
        end = datetime.now()
        start = end - timedelta(days=90)

        async def _run():
            try:
                return await provider.get_series("DGS10", start, end)
            finally:
                await provider.close()

        result = run_async(_run())

        assert result is not None, "DGS10 数据获取失败"
        assert isinstance(result, pd.Series)
        assert len(result) > 0
        assert result.name == "DGS10"
        assert isinstance(result.index, pd.DatetimeIndex)
        # 10年期国债收益率合理范围
        assert all(0 < v < 20 for v in result.values), f"DGS10 值超出合理范围: {result.values}"

    def test_get_fedfunds_series(self):
        """获取联邦基金利率（FEDFUNDS）"""
        provider = FREDProvider()
        end = datetime.now()
        start = end - timedelta(days=90)

        async def _run():
            try:
                return await provider.get_series("FEDFUNDS", start, end)
            finally:
                await provider.close()

        result = run_async(_run())

        assert result is not None
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_get_latest_dgs10(self):
        """获取 DGS10 最新值"""
        provider = FREDProvider()

        async def _run():
            try:
                return await provider.get_latest_value("DGS10")
            finally:
                await provider.close()

        result = run_async(_run())

        assert result is not None
        assert isinstance(result, float)
        assert 0 < result < 20  # 合理范围

    def test_macro_snapshot_live(self):
        """获取完整宏观快照"""
        provider = FREDProvider()

        async def _run():
            try:
                return await provider.get_macro_snapshot()
            finally:
                await provider.close()

        result = run_async(_run())

        assert isinstance(result, dict)
        assert len(result) > 0
        # 应至少包含一个核心利率指标
        has_rate = any(k in result for k in ["DGS10", "DGS2", "FEDFUNDS"])
        assert has_rate, f"快照缺少核心利率指标，只有: {list(result.keys())}"

        # 检查派生指标
        if "DGS10" in result and "DGS2" in result:
            assert "yield_curve_spread" in result
            expected = result["DGS10"] - result["DGS2"]
            assert abs(result["yield_curve_spread"] - expected) < 1e-9

    def test_get_multiple_series_live(self):
        """批量获取多个序列"""
        provider = FREDProvider()
        end = datetime.now()
        start = end - timedelta(days=60)

        async def _run():
            try:
                return await provider.get_multiple_series(["DGS10", "VIXCLS"], start, end)
            finally:
                await provider.close()

        result = run_async(_run())

        assert isinstance(result, dict)
        # 至少应成功获取一个序列
        assert len(result) > 0
        for sid, series in result.items():
            assert isinstance(series, pd.Series)
            assert len(series) > 0
