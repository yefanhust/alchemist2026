"""
SQLiteCache 单元测试

覆盖范围：
- 基本 CRUD 操作 (get/set/delete/exists/clear)
- TTL 过期机制
- 批量操作 (get_many/set_many/delete_many)
- 缓存键模式匹配
- MarketData 序列化/反序列化
- 通用对象 (pickle) 序列化
- 市场数据专用表 (save_market_data/get_market_data)
- 数据持久化（重建实例后数据仍存在）
- 缓存命中/未命中流程（模拟 DataProvider）
- 缓存 vs 远端数据源性能对比
- get_or_set 缓存穿透保护
- 过期清理 (cleanup_expired)
- 统计信息 (stats)
- 并发访问安全性
- 边界与异常场景
"""

import asyncio
import os
import pickle
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest
from data.cache.base import CacheBackend
from data.cache.sqlite_cache import SQLiteCache
from data.models import OHLCV, MarketData
from data.providers import DataInterval, DataProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_async(coro):
    """在同步上下文中执行协程"""
    return asyncio.run(coro)


def make_ohlcv(
    day: int = 1,
    month: int = 1,
    year: int = 2025,
    open_: float = 100.0,
    high: float = 110.0,
    low: float = 90.0,
    close: float = 105.0,
    volume: float = 1_000_000,
    adjusted_close: float = 105.0,
) -> OHLCV:
    """快速构建 OHLCV 对象"""
    return OHLCV(
        timestamp=datetime(year, month, day),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        adjusted_close=adjusted_close,
    )


def make_market_data(symbol: str = "AAPL", days: int = 30) -> MarketData:
    """构建包含 N 天数据的 MarketData"""
    data = []
    for i in range(days):
        base = 150.0 + i * 0.5
        data.append(
            OHLCV(
                timestamp=datetime(2025, 1, 1) + timedelta(days=i),
                open=base,
                high=base + 5.0,
                low=base - 3.0,
                close=base + 2.0,
                volume=1_000_000 + i * 10_000,
                adjusted_close=base + 2.0,
            )
        )
    return MarketData(
        symbol=symbol,
        data=data,
        metadata={"source": "test", "interval": "1d"},
    )


class FakeRemoteProvider(DataProvider):
    """
    模拟远端数据源

    每次 get_historical_data 人为引入延迟以模拟网络 I/O，
    并记录实际调用次数，用于验证缓存是否生效。
    """

    def __init__(self, cache_backend=None, latency: float = 0.05):
        super().__init__(cache_backend)
        self.latency = latency
        self.api_call_count = 0

    @property
    def name(self) -> str:
        return "fake_remote"

    @property
    def supported_intervals(self):
        return [DataInterval.DAILY]

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: DataInterval = DataInterval.DAILY,
    ) -> MarketData:
        end_date = end_date or datetime.now()
        cache_key = self._generate_cache_key(symbol, start_date, end_date, interval)

        # 先查缓存
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                return cached

        # 模拟网络延迟
        await asyncio.sleep(self.latency)
        self.api_call_count += 1

        market_data = make_market_data(symbol, days=30)

        # 写缓存
        if self.cache and not market_data.is_empty:
            await self.cache.set(cache_key, market_data)

        return market_data

    async def get_latest_price(self, symbol: str) -> Optional[float]:
        return 150.0

    async def get_quote(self, symbol):
        return {"price": 150.0}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path):
    """返回临时数据库路径"""
    return str(tmp_path / "test_cache.db")


@pytest.fixture
def cache(tmp_db):
    """创建全新的 SQLiteCache 实例"""
    return SQLiteCache(db_path=tmp_db)


@pytest.fixture
def market_data():
    """30 天 AAPL 测试数据"""
    return make_market_data("AAPL", days=30)


# =========================================================================
# 1. 基本 CRUD 操作
# =========================================================================


class TestBasicCRUD:
    """基础读/写/删/存在性/清空操作"""

    def test_set_and_get(self, cache):
        """写入后应能读取到相同的值"""

        async def _test():
            await cache.set("key1", "value1")
            result = await cache.get("key1")
            assert result == "value1"

        run_async(_test())

    def test_get_nonexistent_key(self, cache):
        """读取不存在的键应返回 None"""

        async def _test():
            result = await cache.get("nonexistent")
            assert result is None

        run_async(_test())

    def test_set_overwrite(self, cache):
        """对同一个键多次写入，最后一次应覆盖之前的值"""

        async def _test():
            await cache.set("key1", "old")
            await cache.set("key1", "new")
            result = await cache.get("key1")
            assert result == "new"

        run_async(_test())

    def test_delete_existing_key(self, cache):
        """删除存在的键应返回 True，之后 get 返回 None"""

        async def _test():
            await cache.set("key1", "value1")
            deleted = await cache.delete("key1")
            assert deleted is True
            assert await cache.get("key1") is None

        run_async(_test())

    def test_delete_nonexistent_key(self, cache):
        """删除不存在的键应返回 False"""

        async def _test():
            deleted = await cache.delete("nonexistent")
            assert deleted is False

        run_async(_test())

    def test_exists(self, cache):
        """exists 正确反映键的存在状态"""

        async def _test():
            assert await cache.exists("key1") is False
            await cache.set("key1", "val")
            assert await cache.exists("key1") is True

        run_async(_test())

    def test_clear(self, cache):
        """clear 清空全部缓存并返回删除的条目数"""

        async def _test():
            await cache.set("a", 1)
            await cache.set("b", 2)
            await cache.set("c", 3)
            count = await cache.clear()
            assert count == 3
            assert await cache.get("a") is None
            assert await cache.get("b") is None
            assert await cache.get("c") is None

        run_async(_test())

    def test_clear_empty_cache(self, cache):
        """清空空缓存应返回 0"""

        async def _test():
            count = await cache.clear()
            assert count == 0

        run_async(_test())


# =========================================================================
# 2. 数据类型序列化
# =========================================================================


class TestSerialization:
    """验证不同数据类型的序列化与反序列化"""

    def test_string(self, cache):
        async def _test():
            await cache.set("s", "hello world")
            assert await cache.get("s") == "hello world"

        run_async(_test())

    def test_integer(self, cache):
        async def _test():
            await cache.set("i", 42)
            assert await cache.get("i") == 42

        run_async(_test())

    def test_float(self, cache):
        async def _test():
            await cache.set("f", 3.14159)
            result = await cache.get("f")
            assert abs(result - 3.14159) < 1e-10

        run_async(_test())

    def test_dict(self, cache):
        async def _test():
            data = {"nested": {"key": [1, 2, 3]}, "flag": True}
            await cache.set("d", data)
            assert await cache.get("d") == data

        run_async(_test())

    def test_list(self, cache):
        async def _test():
            data = [1, "two", 3.0, None, True]
            await cache.set("l", data)
            assert await cache.get("l") == data

        run_async(_test())

    def test_bytes(self, cache):
        async def _test():
            data = b"\x00\x01\x02\xff"
            await cache.set("b", data)
            assert await cache.get("b") == data

        run_async(_test())

    def test_market_data_serialization(self, cache, market_data):
        """MarketData 走 JSON 序列化路径，反序列化后数据完全一致"""

        async def _test():
            await cache.set("md", market_data)
            restored = await cache.get("md")

            assert isinstance(restored, MarketData)
            assert restored.symbol == market_data.symbol
            assert len(restored) == len(market_data)
            assert restored.metadata == market_data.metadata

            # 逐条比较 OHLCV
            for orig, rest in zip(market_data.data, restored.data):
                assert orig.timestamp == rest.timestamp
                assert orig.open == rest.open
                assert orig.high == rest.high
                assert orig.low == rest.low
                assert orig.close == rest.close
                assert orig.volume == rest.volume
                assert orig.adjusted_close == rest.adjusted_close

        run_async(_test())

    def test_empty_market_data(self, cache):
        """空 MarketData 的序列化/反序列化"""

        async def _test():
            empty = MarketData(symbol="EMPTY", data=[], metadata={})
            await cache.set("empty_md", empty)
            restored = await cache.get("empty_md")
            assert isinstance(restored, MarketData)
            assert restored.symbol == "EMPTY"
            assert restored.is_empty

        run_async(_test())

    def test_serialize_data_type_tag(self, cache, market_data):
        """验证 MarketData 和非 MarketData 使用不同的 data_type 标签"""

        async def _test():
            # MarketData → data_type = "market_data"
            blob_md, dtype_md = cache._serialize(market_data)
            assert dtype_md == "market_data"

            # 普通对象 → data_type = "pickle"
            blob_gen, dtype_gen = cache._serialize({"key": "value"})
            assert dtype_gen == "pickle"

        run_async(_test())


# =========================================================================
# 3. TTL 过期机制
# =========================================================================


class TestTTL:
    """验证缓存过期行为"""

    def test_ttl_not_expired(self, cache):
        """TTL 未到期时应正常返回"""

        async def _test():
            await cache.set("k", "v", ttl=timedelta(seconds=60))
            result = await cache.get("k")
            assert result == "v"

        run_async(_test())

    def test_ttl_expired(self, cache):
        """TTL 过期后 get 应返回 None"""

        async def _test():
            await cache.set("k", "v", ttl=timedelta(milliseconds=50))
            await asyncio.sleep(0.1)  # 等待过期
            result = await cache.get("k")
            assert result is None

        run_async(_test())

    def test_ttl_auto_delete_on_get(self, cache):
        """过期条目在 get 时被自动删除"""

        async def _test():
            await cache.set("k", "v", ttl=timedelta(milliseconds=50))
            await asyncio.sleep(0.1)

            # get 触发删除
            assert await cache.get("k") is None

            # 确认确实从数据库中移除
            keys = await cache.keys("k")
            assert "k" not in keys

        run_async(_test())

    def test_no_ttl_never_expires(self, cache):
        """不设 TTL 的条目不会过期"""

        async def _test():
            await cache.set("k", "v")  # 无 TTL
            await asyncio.sleep(0.1)
            assert await cache.get("k") == "v"

        run_async(_test())

    def test_default_ttl(self, tmp_db):
        """通过 default_ttl 参数设置全局过期时间"""

        async def _test():
            c = SQLiteCache(db_path=tmp_db, default_ttl=timedelta(milliseconds=50))
            await c.set("k", "v")  # 未显式指定 ttl，使用 default_ttl
            assert await c.get("k") == "v"

            await asyncio.sleep(0.1)
            assert await c.get("k") is None

        run_async(_test())

    def test_explicit_ttl_overrides_default(self, tmp_db):
        """显式传入 ttl 应覆盖 default_ttl"""

        async def _test():
            c = SQLiteCache(db_path=tmp_db, default_ttl=timedelta(milliseconds=50))
            # 显式设置较长的 TTL
            await c.set("k", "v", ttl=timedelta(seconds=60))
            await asyncio.sleep(0.1)
            assert await c.get("k") == "v"  # 仍然有效

        run_async(_test())

    def test_cleanup_expired(self, cache):
        """cleanup_expired 批量清理过期条目"""

        async def _test():
            # 写入 5 条即将过期的 + 2 条永不过期的
            for i in range(5):
                await cache.set(f"exp_{i}", i, ttl=timedelta(milliseconds=50))
            await cache.set("perm_a", "a")
            await cache.set("perm_b", "b")

            await asyncio.sleep(0.1)

            cleaned = await cache.cleanup_expired()
            assert cleaned == 5

            # 永不过期的仍然存在
            assert await cache.get("perm_a") == "a"
            assert await cache.get("perm_b") == "b"

        run_async(_test())


# =========================================================================
# 4. 批量操作
# =========================================================================


class TestBatchOperations:
    """批量读写删"""

    def test_set_many_and_get_many(self, cache):
        async def _test():
            mapping = {"k1": "v1", "k2": "v2", "k3": "v3"}
            count = await cache.set_many(mapping)
            assert count == 3

            result = await cache.get_many(["k1", "k2", "k3", "missing"])
            assert result == {"k1": "v1", "k2": "v2", "k3": "v3"}

        run_async(_test())

    def test_delete_many(self, cache):
        async def _test():
            await cache.set_many({"a": 1, "b": 2, "c": 3})
            deleted = await cache.delete_many(["a", "c", "nonexistent"])
            assert deleted == 2  # a 和 c 成功删除
            assert await cache.get("b") == 2

        run_async(_test())


# =========================================================================
# 5. 缓存键模式匹配
# =========================================================================


class TestKeyPatterns:
    """keys() 通配符匹配"""

    def test_wildcard_all(self, cache):
        async def _test():
            await cache.set_many({"a:1": 1, "a:2": 2, "b:1": 3})
            all_keys = await cache.keys("*")
            assert set(all_keys) == {"a:1", "a:2", "b:1"}

        run_async(_test())

    def test_wildcard_prefix(self, cache):
        async def _test():
            await cache.set_many({"a:1": 1, "a:2": 2, "b:1": 3})
            keys = await cache.keys("a:*")
            assert set(keys) == {"a:1", "a:2"}

        run_async(_test())

    def test_wildcard_single_char(self, cache):
        async def _test():
            await cache.set_many({"abc": 1, "adc": 2, "aec": 3, "abcd": 4})
            keys = await cache.keys("a?c")
            assert set(keys) == {"abc", "adc", "aec"}

        run_async(_test())

    def test_no_match(self, cache):
        async def _test():
            await cache.set("foo", 1)
            keys = await cache.keys("bar*")
            assert keys == []

        run_async(_test())


# =========================================================================
# 6. 市场数据专用表
# =========================================================================


class TestMarketDataTable:
    """save_market_data / get_market_data 专用表操作"""

    def test_save_and_retrieve(self, cache, market_data):
        """保存后按 symbol + interval 检索"""

        async def _test():
            saved = await cache.save_market_data("AAPL", "1d", market_data)
            assert saved == len(market_data)

            retrieved = await cache.get_market_data("AAPL", "1d")
            assert retrieved is not None
            assert len(retrieved) == len(market_data)
            assert retrieved.symbol == "AAPL"
            assert retrieved.metadata.get("source") == "cache"

        run_async(_test())

    def test_retrieve_with_date_range(self, cache, market_data):
        """按日期范围过滤"""

        async def _test():
            await cache.save_market_data("AAPL", "1d", market_data)

            start = datetime(2025, 1, 10)
            end = datetime(2025, 1, 20)
            retrieved = await cache.get_market_data("AAPL", "1d", start, end)

            assert retrieved is not None
            for ohlcv in retrieved.data:
                assert start <= ohlcv.timestamp <= end

        run_async(_test())

    def test_retrieve_nonexistent_symbol(self, cache):
        """查询不存在的 symbol 应返回 None"""

        async def _test():
            result = await cache.get_market_data("ZZZZ", "1d")
            assert result is None

        run_async(_test())

    def test_save_empty_market_data(self, cache):
        """保存空 MarketData 应返回 0"""

        async def _test():
            empty = MarketData(symbol="EMPTY")
            saved = await cache.save_market_data("EMPTY", "1d", empty)
            assert saved == 0

        run_async(_test())

    def test_upsert_on_conflict(self, cache, market_data):
        """重复写入同一 (symbol, interval, timestamp) 应更新而非报错"""

        async def _test():
            await cache.save_market_data("AAPL", "1d", market_data)
            # 再次写入相同数据不应抛异常
            saved = await cache.save_market_data("AAPL", "1d", market_data)
            assert saved == len(market_data)

            # 数据条数不应翻倍
            retrieved = await cache.get_market_data("AAPL", "1d")
            assert len(retrieved) == len(market_data)

        run_async(_test())

    def test_multiple_symbols(self, cache):
        """同一张表存储多个 symbol 的数据，互不干扰"""

        async def _test():
            aapl = make_market_data("AAPL", days=10)
            msft = make_market_data("MSFT", days=20)

            await cache.save_market_data("AAPL", "1d", aapl)
            await cache.save_market_data("MSFT", "1d", msft)

            r_aapl = await cache.get_market_data("AAPL", "1d")
            r_msft = await cache.get_market_data("MSFT", "1d")

            assert len(r_aapl) == 10
            assert len(r_msft) == 20

        run_async(_test())

    def test_multiple_intervals(self, cache):
        """同一 symbol 不同 interval 的数据互不干扰"""

        async def _test():
            daily = make_market_data("AAPL", days=30)
            weekly = make_market_data("AAPL", days=10)

            await cache.save_market_data("AAPL", "1d", daily)
            await cache.save_market_data("AAPL", "1w", weekly)

            r_daily = await cache.get_market_data("AAPL", "1d")
            r_weekly = await cache.get_market_data("AAPL", "1w")

            assert len(r_daily) == 30
            assert len(r_weekly) == 10

        run_async(_test())

    def test_ohlcv_values_preserved(self, cache):
        """逐字段验证专用表写入的 OHLCV 精度"""

        async def _test():
            ohlcv = make_ohlcv(
                day=15,
                open_=123.456,
                high=130.789,
                low=118.123,
                close=127.654,
                volume=9_876_543,
                adjusted_close=127.654,
            )
            md = MarketData(symbol="TSLA", data=[ohlcv])
            await cache.save_market_data("TSLA", "1d", md)

            retrieved = await cache.get_market_data("TSLA", "1d")
            r = retrieved.data[0]
            assert abs(r.open - 123.456) < 1e-6
            assert abs(r.high - 130.789) < 1e-6
            assert abs(r.low - 118.123) < 1e-6
            assert abs(r.close - 127.654) < 1e-6
            assert abs(r.volume - 9_876_543) < 1e-6
            assert abs(r.adjusted_close - 127.654) < 1e-6

        run_async(_test())


# =========================================================================
# 7. 数据持久化
# =========================================================================


class TestPersistence:
    """重建 SQLiteCache 实例后数据仍然可读"""

    def test_cache_entries_persist(self, tmp_db):
        """通用缓存表数据持久化"""

        async def _test():
            # 写入
            c1 = SQLiteCache(db_path=tmp_db)
            await c1.set("persist_key", {"data": [1, 2, 3]})

            # 用全新实例读取
            c2 = SQLiteCache(db_path=tmp_db)
            result = await c2.get("persist_key")
            assert result == {"data": [1, 2, 3]}

        run_async(_test())

    def test_market_data_persist(self, tmp_db):
        """市场数据专用表持久化"""

        async def _test():
            md = make_market_data("AAPL", days=15)

            c1 = SQLiteCache(db_path=tmp_db)
            await c1.save_market_data("AAPL", "1d", md)

            c2 = SQLiteCache(db_path=tmp_db)
            retrieved = await c2.get_market_data("AAPL", "1d")
            assert retrieved is not None
            assert len(retrieved) == 15

        run_async(_test())

    def test_market_data_json_persist(self, tmp_db):
        """MarketData 通过 cache_entries 表 (JSON 序列化) 的持久化"""

        async def _test():
            md = make_market_data("GOOGL", days=20)

            c1 = SQLiteCache(db_path=tmp_db)
            await c1.set("alphavantage:GOOGL:1d:20250101:20250201", md)

            c2 = SQLiteCache(db_path=tmp_db)
            restored = await c2.get("alphavantage:GOOGL:1d:20250101:20250201")
            assert isinstance(restored, MarketData)
            assert restored.symbol == "GOOGL"
            assert len(restored) == 20

            # 逐条对比
            for orig, rest in zip(md.data, restored.data):
                assert orig.close == rest.close

        run_async(_test())

    def test_ttl_persists(self, tmp_db):
        """TTL 信息持久化到磁盘，新实例仍能识别过期"""

        async def _test():
            c1 = SQLiteCache(db_path=tmp_db)
            await c1.set("short_lived", "temp", ttl=timedelta(milliseconds=50))

            await asyncio.sleep(0.1)

            c2 = SQLiteCache(db_path=tmp_db)
            assert await c2.get("short_lived") is None

        run_async(_test())

    def test_db_file_created(self, tmp_db):
        """首次操作后数据库文件应被创建"""

        async def _test():
            assert not Path(tmp_db).exists()
            c = SQLiteCache(db_path=tmp_db)
            await c.set("trigger_init", True)
            assert Path(tmp_db).exists()

        run_async(_test())


# =========================================================================
# 8. 缓存命中/未命中流程（端到端集成）
# =========================================================================


class TestCacheIntegration:
    """模拟完整的 DataProvider + SQLiteCache 交互"""

    def test_first_call_misses_cache(self, cache):
        """首次调用应穿透缓存，调用远端 API"""

        async def _test():
            provider = FakeRemoteProvider(cache_backend=cache)
            start = datetime(2025, 1, 1)
            end = datetime(2025, 1, 31)

            data = await provider.get_historical_data("AAPL", start, end)
            assert provider.api_call_count == 1
            assert not data.is_empty

        run_async(_test())

    def test_second_call_hits_cache(self, cache):
        """第二次相同请求应命中缓存，不再调用远端 API"""

        async def _test():
            provider = FakeRemoteProvider(cache_backend=cache)
            start = datetime(2025, 1, 1)
            end = datetime(2025, 1, 31)

            data1 = await provider.get_historical_data("AAPL", start, end)
            data2 = await provider.get_historical_data("AAPL", start, end)

            assert provider.api_call_count == 1  # 只调了一次远端
            assert len(data1) == len(data2)

        run_async(_test())

    def test_different_params_miss_cache(self, cache):
        """不同参数应各自独立缓存"""

        async def _test():
            provider = FakeRemoteProvider(cache_backend=cache)

            await provider.get_historical_data("AAPL", datetime(2025, 1, 1), datetime(2025, 1, 31))
            await provider.get_historical_data("MSFT", datetime(2025, 1, 1), datetime(2025, 1, 31))
            await provider.get_historical_data("AAPL", datetime(2025, 2, 1), datetime(2025, 2, 28))

            # 三次不同参数 → 三次 API 调用
            assert provider.api_call_count == 3

        run_async(_test())

    def test_cached_data_content_matches(self, cache):
        """缓存返回的数据与首次 API 返回的数据内容一致"""

        async def _test():
            provider = FakeRemoteProvider(cache_backend=cache)
            start = datetime(2025, 1, 1)
            end = datetime(2025, 1, 31)

            data1 = await provider.get_historical_data("AAPL", start, end)
            data2 = await provider.get_historical_data("AAPL", start, end)

            assert data1.symbol == data2.symbol
            assert len(data1) == len(data2)
            for o1, o2 in zip(data1.data, data2.data):
                assert o1.timestamp == o2.timestamp
                assert o1.open == o2.open
                assert o1.close == o2.close

        run_async(_test())

    def test_no_cache_always_calls_api(self):
        """不注入缓存时每次都应调用远端 API"""

        async def _test():
            provider = FakeRemoteProvider(cache_backend=None)
            start = datetime(2025, 1, 1)
            end = datetime(2025, 1, 31)

            await provider.get_historical_data("AAPL", start, end)
            await provider.get_historical_data("AAPL", start, end)

            assert provider.api_call_count == 2

        run_async(_test())


# =========================================================================
# 9. 性能对比
# =========================================================================


class TestPerformance:
    """缓存 vs 远端数据源的响应时间对比"""

    def test_cache_much_faster_than_remote(self, cache):
        """从缓存读取应显著快于（模拟的）远端调用"""

        async def _test():
            provider = FakeRemoteProvider(cache_backend=cache, latency=0.1)
            start = datetime(2025, 1, 1)
            end = datetime(2025, 1, 31)

            # 首次：走远端
            t0 = time.perf_counter()
            await provider.get_historical_data("AAPL", start, end)
            remote_time = time.perf_counter() - t0

            # 第二次：走缓存
            t0 = time.perf_counter()
            await provider.get_historical_data("AAPL", start, end)
            cache_time = time.perf_counter() - t0

            # 缓存应至少快 5 倍
            assert cache_time < remote_time / 5, f"缓存 ({cache_time:.4f}s) 未显著快于远端 ({remote_time:.4f}s)"

        run_async(_test())

    def test_batch_cache_performance(self, cache):
        """批量缓存读取的吞吐量"""

        async def _test():
            # 预热：写入 100 条
            for i in range(100):
                await cache.set(f"perf_{i}", {"index": i, "data": list(range(50))})

            # 读取
            t0 = time.perf_counter()
            for i in range(100):
                result = await cache.get(f"perf_{i}")
                assert result is not None
            elapsed = time.perf_counter() - t0

            # 100 次读取应在 2 秒内完成
            assert elapsed < 2.0, f"100 次缓存读取耗时 {elapsed:.3f}s，超过 2 秒"

        run_async(_test())

    def test_market_data_cache_vs_serialize_overhead(self, cache):
        """大量 MarketData 缓存的序列化/反序列化开销可接受"""

        async def _test():
            large_md = make_market_data("AAPL", days=365)

            # 写入
            t0 = time.perf_counter()
            await cache.set("large_md", large_md)
            write_time = time.perf_counter() - t0

            # 读取
            t0 = time.perf_counter()
            restored = await cache.get("large_md")
            read_time = time.perf_counter() - t0

            assert len(restored) == 365
            # 读写各应在 1 秒内
            assert write_time < 1.0, f"写入 365 天数据耗时 {write_time:.3f}s"
            assert read_time < 1.0, f"读取 365 天数据耗时 {read_time:.3f}s"

        run_async(_test())


# =========================================================================
# 10. get_or_set 缓存穿透保护
# =========================================================================


class TestGetOrSet:
    """get_or_set 的行为验证"""

    def test_get_or_set_miss_calls_factory(self, cache):
        """缓存未命中时应调用 factory 并缓存结果"""

        async def _test():
            call_count = 0

            def factory():
                nonlocal call_count
                call_count += 1
                return "generated_value"

            result = await cache.get_or_set("k", factory)
            assert result == "generated_value"
            assert call_count == 1

            # 第二次不应再调用 factory
            result2 = await cache.get_or_set("k", factory)
            assert result2 == "generated_value"
            assert call_count == 1

        run_async(_test())

    def test_get_or_set_hit_skips_factory(self, cache):
        """缓存命中时不调用 factory"""

        async def _test():
            await cache.set("k", "existing")

            called = False

            def factory():
                nonlocal called
                called = True
                return "new"

            result = await cache.get_or_set("k", factory)
            assert result == "existing"
            assert called is False

        run_async(_test())

    def test_get_or_set_async_factory(self, cache):
        """支持异步 factory 函数"""

        async def _test():
            async def async_factory():
                return {"async": True}

            result = await cache.get_or_set("k", async_factory)
            assert result == {"async": True}

        run_async(_test())

    def test_get_or_set_with_ttl(self, cache):
        """get_or_set 支持 TTL 参数"""

        async def _test():
            await cache.get_or_set("k", lambda: "temp", ttl=timedelta(milliseconds=50))
            assert await cache.get("k") == "temp"

            await asyncio.sleep(0.1)
            assert await cache.get("k") is None

        run_async(_test())

    def test_get_or_set_non_callable(self, cache):
        """factory 为非 callable 时直接作为默认值"""

        async def _test():
            result = await cache.get_or_set("k", "default_value")
            assert result == "default_value"

        run_async(_test())


# =========================================================================
# 11. 统计信息
# =========================================================================


class TestStats:
    """stats() 返回的缓存统计"""

    def test_empty_stats(self, cache):
        async def _test():
            stats = await cache.stats()
            assert stats["total_entries"] == 0
            assert stats["expired_entries"] == 0
            assert stats["market_data_entries"] == 0
            assert stats["unique_symbols"] == 0

        run_async(_test())

    def test_stats_after_inserts(self, cache):
        async def _test():
            await cache.set("a", 1)
            await cache.set("b", 2)

            md = make_market_data("AAPL", days=10)
            await cache.save_market_data("AAPL", "1d", md)

            stats = await cache.stats()
            assert stats["total_entries"] == 2
            assert stats["market_data_entries"] == 10
            assert stats["unique_symbols"] == 1

        run_async(_test())

    def test_stats_expired_count(self, cache):
        async def _test():
            await cache.set("exp1", 1, ttl=timedelta(milliseconds=50))
            await cache.set("exp2", 2, ttl=timedelta(milliseconds=50))
            await cache.set("perm", 3)

            await asyncio.sleep(0.1)

            stats = await cache.stats()
            assert stats["expired_entries"] == 2
            assert stats["total_entries"] == 3  # 未清理前仍计入总数

        run_async(_test())

    def test_stats_db_size(self, cache):
        async def _test():
            await cache.set("data", "x" * 1000)
            stats = await cache.stats()
            assert stats["db_size_bytes"] > 0
            assert stats["db_size_mb"] >= 0
            assert "db_path" in stats

        run_async(_test())


# =========================================================================
# 12. 并发访问
# =========================================================================


class TestConcurrency:
    """多个协程同时操作缓存"""

    def test_concurrent_writes(self, cache):
        """并发写入不应丢数据或抛异常"""

        async def _test():
            async def writer(i):
                await cache.set(f"concurrent_{i}", i)

            await asyncio.gather(*[writer(i) for i in range(50)])

            # 验证所有数据均已写入
            for i in range(50):
                val = await cache.get(f"concurrent_{i}")
                assert val == i

        run_async(_test())

    def test_concurrent_read_write(self, cache):
        """读写混合并发不应出错"""

        async def _test():
            await cache.set("shared", "initial")

            results = []

            async def reader():
                val = await cache.get("shared")
                results.append(val)

            async def writer():
                await cache.set("shared", "updated")

            tasks = [reader() for _ in range(10)] + [writer() for _ in range(5)]
            await asyncio.gather(*tasks)

            # 最终值应为 "updated"
            final = await cache.get("shared")
            assert final == "updated"

            # 所有读取结果应为 "initial" 或 "updated"
            for r in results:
                assert r in ("initial", "updated")

        run_async(_test())

    def test_concurrent_initialization(self, tmp_db):
        """多个 SQLiteCache 实例并发初始化同一数据库不应报错"""

        async def _test():
            caches = [SQLiteCache(db_path=tmp_db) for _ in range(5)]

            async def init_and_write(c, i):
                await c.set(f"init_{i}", i)

            await asyncio.gather(*[init_and_write(c, i) for i, c in enumerate(caches)])

            # 用新实例验证
            verify = SQLiteCache(db_path=tmp_db)
            for i in range(5):
                assert await verify.get(f"init_{i}") == i

        run_async(_test())


# =========================================================================
# 13. 上下文管理器
# =========================================================================


class TestContextManager:
    """async with 用法"""

    def test_async_context_manager(self, tmp_db):
        async def _test():
            async with SQLiteCache(db_path=tmp_db) as c:
                await c.set("ctx_key", "ctx_value")
                assert await c.get("ctx_key") == "ctx_value"

        run_async(_test())


# =========================================================================
# 14. 边界与异常场景
# =========================================================================


class TestEdgeCases:
    """边界条件和异常处理"""

    def test_very_long_key(self, cache):
        """超长缓存键"""

        async def _test():
            long_key = "k" * 10_000
            await cache.set(long_key, "value")
            assert await cache.get(long_key) == "value"

        run_async(_test())

    def test_empty_string_value(self, cache):
        """空字符串作为值（不应被误判为 None）"""

        async def _test():
            await cache.set("empty_str", "")
            result = await cache.get("empty_str")
            # pickle 序列化后空字符串是合法的值
            assert result == ""

        run_async(_test())

    def test_none_value(self, cache):
        """None 作为值"""

        async def _test():
            await cache.set("none_val", None)
            # 当前实现中 get 返回 None 表示不存在，
            # 所以存 None 可能与"不存在"混淆，这是一个已知行为
            result = await cache.get("none_val")
            # 无论返回 None 还是有值，不应抛异常
            assert result is None

        run_async(_test())

    def test_large_market_data(self, cache):
        """大量 OHLCV 数据（模拟 5 年日线 ≈ 1260 条）"""

        async def _test():
            large_md = make_market_data("SPY", days=1260)
            await cache.set("spy_5y", large_md)

            restored = await cache.get("spy_5y")
            assert len(restored) == 1260
            assert restored.symbol == "SPY"

        run_async(_test())

    def test_special_characters_in_key(self, cache):
        """缓存键包含特殊字符"""

        async def _test():
            special_keys = [
                "alphavantage:600519.SHH:1d:20250101:20250201",
                "key with spaces",
                "key/with/slashes",
                "中文键名",
                "emoji_🔑",
            ]
            for i, key in enumerate(special_keys):
                await cache.set(key, i)

            for i, key in enumerate(special_keys):
                assert await cache.get(key) == i

        run_async(_test())

    def test_repr(self, tmp_db):
        """__repr__ 输出包含数据库路径"""
        c = SQLiteCache(db_path=tmp_db)
        r = repr(c)
        assert "SQLiteCache" in r
        assert tmp_db in r

    def test_is_subclass_of_cache_backend(self, cache):
        """SQLiteCache 是 CacheBackend 的子类"""
        assert isinstance(cache, CacheBackend)

    def test_db_directory_auto_created(self, tmp_path):
        """db_path 的父目录不存在时应自动创建"""
        nested = tmp_path / "a" / "b" / "c" / "cache.db"
        c = SQLiteCache(db_path=str(nested))
        assert nested.parent.exists()

    def test_clear_also_clears_market_data_table(self, cache):
        """clear() 应同时清空 cache_entries 和 market_data 两张表"""

        async def _test():
            await cache.set("generic", "value")
            md = make_market_data("AAPL", days=5)
            await cache.save_market_data("AAPL", "1d", md)

            await cache.clear()

            assert await cache.get("generic") is None
            assert await cache.get_market_data("AAPL", "1d") is None

        run_async(_test())
