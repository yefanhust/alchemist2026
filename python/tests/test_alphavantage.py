"""
Alpha Vantage 美股数据连通性测试

测试通过 Alpha Vantage API 获取美股市场数据的连通性，验证各接口返回字段的完整性与正确性。
API Key 通过 conftest.py 自动从 config/config.yaml 加载，无需手动设置环境变量。

限流说明：
    AlphaVantageProvider 内置类级别共享限流器，从 config/config.yaml 读取
    alphavantage.calls_per_minute / alphavantage.calls_per_day 配置，
    所有 provider 实例共享同一限流器，测试无需额外处理限流。

用法:
    pytest python/tests/test_alphavantage.py -v
    pytest python/tests/test_alphavantage.py -v -k test_get_daily
"""

import asyncio
import pytest
from datetime import datetime, timedelta

from data.providers.alphavantage import AlphaVantageProvider
from data.providers import DataInterval
from data.models import MarketData, OHLCV
from alchemist.utils.config import get_config

# 美股测试标的
SYMBOL_AAPL = "AAPL"    # Apple Inc.
SYMBOL_MSFT = "MSFT"    # Microsoft Corp.
SYMBOL_GOOGL = "GOOGL"  # Alphabet Inc.


@pytest.fixture
def provider():
    """创建 AlphaVantageProvider 实例，从配置文件获取 API key"""
    config = get_config()
    api_key = config.alphavantage.api_key
    plan = config.alphavantage.plan
    return AlphaVantageProvider(api_key=api_key, plan=plan)


def run_async(coro):
    """同步运行异步函数的辅助方法"""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------- 基础初始化 ----------

class TestAlphaVantageInit:
    """Provider 初始化与基本属性测试"""

    def test_provider_name(self, provider):
        """测试 Provider 名称"""
        assert provider.name == "alphavantage"

    def test_provider_api_key(self, provider):
        """测试 API Key 已加载"""
        assert provider.api_key is not None
        assert len(provider.api_key) > 0

    def test_supported_intervals(self, provider):
        """测试支持的数据间隔"""
        supported = provider.supported_intervals
        assert DataInterval.DAILY in supported
        assert DataInterval.WEEKLY in supported
        assert DataInterval.MONTHLY in supported


# ---------- 股票搜索 ----------

class TestAlphaVantageSearch:
    """股票搜索功能测试"""

    def test_search_symbols_apple(self, provider):
        """测试搜索 Apple 返回结果及字段完整性"""
        async def _test():
            try:
                return await provider.search_symbols("Apple")
            finally:
                await provider.close()

        results = run_async(_test())

        assert isinstance(results, list)
        assert len(results) > 0, "搜索 'Apple' 应返回至少一个结果"

        # 验证返回字段
        first = results[0]
        assert "symbol" in first, "搜索结果应包含 symbol 字段"
        assert "name" in first, "搜索结果应包含 name 字段"
        assert "region" in first, "搜索结果应包含 region 字段"

        # 应能搜到 AAPL
        symbols = [r["symbol"] for r in results]
        assert "AAPL" in symbols, "搜索 'Apple' 应包含 AAPL"

        print(f"\n搜索 'Apple' 返回 {len(results)} 个结果:")
        for r in results:
            print(f"  {r['symbol']} - {r['name']} ({r.get('region')}, {r.get('currency')})")

    def test_search_symbols_microsoft(self, provider):
        """测试搜索 Microsoft 返回结果"""
        async def _test():
            try:
                return await provider.search_symbols("Microsoft")
            finally:
                await provider.close()

        results = run_async(_test())

        assert isinstance(results, list)
        assert len(results) > 0, "搜索 'Microsoft' 应返回至少一个结果"

        symbols = [r["symbol"] for r in results]
        assert "MSFT" in symbols, "搜索 'Microsoft' 应包含 MSFT"


# ---------- 实时报价 ----------

class TestAlphaVantageQuote:
    """实时报价 (get_quote) 测试"""

    def test_get_quote_aapl(self, provider):
        """测试获取 AAPL 实时报价及字段验证"""
        async def _test():
            try:
                return await provider.get_quote(SYMBOL_AAPL)
            finally:
                await provider.close()

        quote = run_async(_test())

        if quote is None:
            pytest.skip(f"GLOBAL_QUOTE 未返回 {SYMBOL_AAPL} 数据（可能超出 API 限额）")

        # 验证核心字段
        assert "price" in quote, "报价应包含 price 字段"
        assert "volume" in quote, "报价应包含 volume 字段"
        assert quote["price"] > 0, f"{SYMBOL_AAPL} 价格应大于 0"
        assert quote["volume"] >= 0, f"{SYMBOL_AAPL} 成交量应 >= 0"

        print(f"\n{SYMBOL_AAPL} 报价:")
        for k, v in quote.items():
            print(f"  {k}: {v}")

    def test_get_quote_msft(self, provider):
        """测试获取 MSFT 实时报价"""
        async def _test():
            try:
                return await provider.get_quote(SYMBOL_MSFT)
            finally:
                await provider.close()

        quote = run_async(_test())

        if quote is None:
            pytest.skip(f"GLOBAL_QUOTE 未返回 {SYMBOL_MSFT} 数据")

        assert quote["price"] > 0
        print(f"\n{SYMBOL_MSFT} 报价: price={quote['price']}, volume={quote['volume']}")


# ---------- 最新价格 ----------

class TestAlphaVantageLatestPrice:
    """最新价格 (get_latest_price) 测试"""

    def test_get_latest_price_aapl(self, provider):
        """测试获取 AAPL 最新价格"""
        async def _test():
            try:
                return await provider.get_latest_price(SYMBOL_AAPL)
            finally:
                await provider.close()

        price = run_async(_test())

        assert price is not None, f"获取 {SYMBOL_AAPL} 最新价格不应为 None"
        assert isinstance(price, (int, float))
        assert price > 0, f"{SYMBOL_AAPL} 最新价格应大于 0"

        print(f"\n{SYMBOL_AAPL} 最新价格: {price}")


# ---------- 日线历史数据 ----------

class TestAlphaVantageDaily:
    """日线历史数据测试"""

    def test_get_daily_data_aapl(self, provider):
        """测试获取 AAPL 日线数据及 OHLCV 字段验证"""
        async def _test():
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                return await provider.get_historical_data(
                    symbol=SYMBOL_AAPL,
                    start_date=start_date,
                    end_date=end_date,
                    interval=DataInterval.DAILY,
                )
            finally:
                await provider.close()

        data = run_async(_test())

        # MarketData 基本验证
        assert isinstance(data, MarketData)
        assert data.symbol == SYMBOL_AAPL
        assert not data.is_empty, f"获取 {SYMBOL_AAPL} 日线数据为空"
        assert len(data) > 0

        # 时间范围验证
        assert data.start_date is not None
        assert data.end_date is not None
        assert data.start_date <= data.end_date

        # 逐条验证 OHLCV 字段
        for ohlcv in data.data:
            assert isinstance(ohlcv, OHLCV)
            assert isinstance(ohlcv.timestamp, datetime)
            assert ohlcv.open > 0, "open 应大于 0"
            assert ohlcv.high > 0, "high 应大于 0"
            assert ohlcv.low > 0, "low 应大于 0"
            assert ohlcv.close > 0, "close 应大于 0"
            assert ohlcv.high >= ohlcv.low, "high 应 >= low"
            assert ohlcv.high >= ohlcv.open, "high 应 >= open"
            assert ohlcv.high >= ohlcv.close, "high 应 >= close"
            assert ohlcv.low <= ohlcv.open, "low 应 <= open"
            assert ohlcv.low <= ohlcv.close, "low 应 <= close"
            assert ohlcv.volume >= 0, "volume 应 >= 0"

        # OHLCV 派生属性验证
        sample = data.data[0]
        assert sample.typical_price == pytest.approx(
            (sample.high + sample.low + sample.close) / 3
        )
        assert sample.range == pytest.approx(sample.high - sample.low)
        assert sample.body == pytest.approx(sample.close - sample.open)
        assert isinstance(sample.is_bullish, bool)

        # latest 属性
        assert data.latest is not None
        assert data.latest_price == data.latest.close

        print(f"\n{SYMBOL_AAPL} 日线数据:")
        print(f"  数据条数: {len(data)}")
        print(f"  时间范围: {data.start_date} ~ {data.end_date}")
        print(f"  最新收盘价: {data.latest_price}")
        print(f"  最近 5 条:")
        for ohlcv in data.data[-5:]:
            print(f"    {ohlcv.timestamp.date()} O:{ohlcv.open} H:{ohlcv.high} "
                  f"L:{ohlcv.low} C:{ohlcv.close} V:{ohlcv.volume}")

    def test_get_daily_data_googl(self, provider):
        """测试获取 GOOGL 日线数据"""
        async def _test():
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                return await provider.get_historical_data(
                    symbol=SYMBOL_GOOGL,
                    start_date=start_date,
                    end_date=end_date,
                    interval=DataInterval.DAILY,
                )
            finally:
                await provider.close()

        data = run_async(_test())

        assert isinstance(data, MarketData)
        assert data.symbol == SYMBOL_GOOGL
        assert not data.is_empty
        assert len(data) > 0

        print(f"\n{SYMBOL_GOOGL} 日线数据: {len(data)} 条, "
              f"最新收盘价: {data.latest_price}")


# ---------- 周线 / 月线历史数据 ----------

class TestAlphaVantageWeeklyMonthly:
    """周线与月线数据测试"""

    def test_get_weekly_data(self, provider):
        """测试获取 AAPL 周线数据"""
        async def _test():
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=90)
                return await provider.get_historical_data(
                    symbol=SYMBOL_AAPL,
                    start_date=start_date,
                    end_date=end_date,
                    interval=DataInterval.WEEKLY,
                )
            finally:
                await provider.close()

        data = run_async(_test())

        assert isinstance(data, MarketData)
        assert not data.is_empty, f"获取 {SYMBOL_AAPL} 周线数据为空"
        assert len(data) > 0

        print(f"\n{SYMBOL_AAPL} 周线数据: {len(data)} 条, "
              f"范围: {data.start_date} ~ {data.end_date}")

    def test_get_monthly_data(self, provider):
        """测试获取 MSFT 月线数据"""
        async def _test():
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
                return await provider.get_historical_data(
                    symbol=SYMBOL_MSFT,
                    start_date=start_date,
                    end_date=end_date,
                    interval=DataInterval.MONTHLY,
                )
            finally:
                await provider.close()

        data = run_async(_test())

        assert isinstance(data, MarketData)
        assert not data.is_empty, f"获取 {SYMBOL_MSFT} 月线数据为空"
        assert len(data) > 0

        print(f"\n{SYMBOL_MSFT} 月线数据: {len(data)} 条, "
              f"范围: {data.start_date} ~ {data.end_date}")


# ---------- 数据转换 ----------

class TestAlphaVantageDataConversion:
    """数据格式转换测试"""

    def test_to_dataframe(self, provider):
        """测试 MarketData 转换为 Pandas DataFrame"""
        async def _test():
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                return await provider.get_historical_data(
                    symbol=SYMBOL_AAPL,
                    start_date=start_date,
                    end_date=end_date,
                    interval=DataInterval.DAILY,
                )
            finally:
                await provider.close()

        data = run_async(_test())
        df = data.to_dataframe()

        assert not df.empty, "DataFrame 不应为空"

        # 验证 DataFrame 列名
        expected_columns = ["open", "high", "low", "close", "volume"]
        for col in expected_columns:
            assert col in df.columns, f"DataFrame 应包含 '{col}' 列"

        # 验证数据类型
        assert df["open"].dtype in ("float64", "float32")
        assert df["volume"].dtype in ("float64", "float32", "int64")

        # 验证行数与原始数据一致
        assert len(df) == len(data)

        print(f"\n{SYMBOL_AAPL} DataFrame (最后 5 行):")
        print(df.tail())

    def test_to_numpy(self, provider):
        """测试 MarketData 转换为 NumPy 数组"""
        async def _test():
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                return await provider.get_historical_data(
                    symbol=SYMBOL_AAPL,
                    start_date=start_date,
                    end_date=end_date,
                    interval=DataInterval.DAILY,
                )
            finally:
                await provider.close()

        data = run_async(_test())
        arrays = data.to_numpy()

        assert arrays is not None, "NumPy 转换结果不应为 None"
        print(f"\n{SYMBOL_AAPL} NumPy 数组 keys: {list(arrays.keys()) if isinstance(arrays, dict) else type(arrays)}")

    def test_ohlcv_to_dict(self, provider):
        """测试单条 OHLCV 转换为字典"""
        async def _test():
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=10)
                return await provider.get_historical_data(
                    symbol=SYMBOL_MSFT,
                    start_date=start_date,
                    end_date=end_date,
                    interval=DataInterval.DAILY,
                )
            finally:
                await provider.close()

        data = run_async(_test())
        assert not data.is_empty

        sample = data.data[0]
        d = sample.to_dict()

        assert isinstance(d, dict)
        assert "timestamp" in d
        assert "open" in d
        assert "high" in d
        assert "low" in d
        assert "close" in d
        assert "volume" in d

        print(f"\nOHLCV to_dict: {d}")


# ---------- MarketData 切片与计算 ----------

class TestAlphaVantageMarketDataOps:
    """MarketData 操作测试（slice, returns 等）"""

    def test_slice_by_date(self, provider):
        """测试按日期切片"""
        async def _test():
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=60)
                return await provider.get_historical_data(
                    symbol=SYMBOL_AAPL,
                    start_date=start_date,
                    end_date=end_date,
                    interval=DataInterval.DAILY,
                )
            finally:
                await provider.close()

        data = run_async(_test())
        assert not data.is_empty

        # 切片最近 10 天
        slice_start = datetime.now() - timedelta(days=10)
        slice_end = datetime.now()
        sliced = data.slice(start=slice_start, end=slice_end)

        assert isinstance(sliced, MarketData)
        assert len(sliced) <= len(data), "切片后数据条数应 <= 原数据"

        print(f"\n原数据 {len(data)} 条, 切片后 {len(sliced)} 条")

    def test_returns_calculation(self, provider):
        """测试收益率计算"""
        async def _test():
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                return await provider.get_historical_data(
                    symbol=SYMBOL_AAPL,
                    start_date=start_date,
                    end_date=end_date,
                    interval=DataInterval.DAILY,
                )
            finally:
                await provider.close()

        data = run_async(_test())
        assert len(data) >= 2, "需要至少 2 条数据才能计算收益率"

        returns = data.returns()
        assert returns is not None, "收益率计算结果不应为 None"

        print(f"\n{SYMBOL_AAPL} 收益率序列 (前 5 个): {returns[:5] if hasattr(returns, '__getitem__') else returns}")
