"""
缓存服务层
封装 SQLiteCache 操作，提供业务逻辑
"""

import re
from datetime import datetime
from typing import Optional, List, Dict, Any

import aiosqlite

from data.cache.sqlite_cache import SQLiteCache
from data.models import MarketData


class CacheService:
    """缓存数据服务"""

    def __init__(self, cache: SQLiteCache):
        self.cache = cache

    async def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return await self.cache.stats()

    async def get_symbols(self) -> List[Dict[str, Any]]:
        """
        获取所有 Symbol 及其数据范围

        从 cache_entries 表解析缓存键格式:
        alphavantage:{symbol}:{interval}:{start_date}:{end_date}
        """
        symbols_data = []
        seen = set()

        # 首先尝试从 market_data 表获取
        async with aiosqlite.connect(self.cache.db_path) as db:
            cursor = await db.execute("""
                SELECT
                    symbol,
                    interval,
                    COUNT(*) as count,
                    MIN(timestamp) as start_date,
                    MAX(timestamp) as end_date
                FROM market_data
                GROUP BY symbol, interval
                ORDER BY symbol
            """)
            rows = await cursor.fetchall()

            for row in rows:
                key = f"{row[0]}:{row[1]}"
                if key not in seen:
                    seen.add(key)
                    symbols_data.append({
                        "symbol": row[0],
                        "interval": row[1],
                        "count": row[2],
                        "start_date": row[3],
                        "end_date": row[4],
                        "source": "market_data",
                    })

        # 然后从 cache_entries 表解析
        keys = await self.cache.keys("*")
        for key in keys:
            # 解析键格式: provider:symbol:interval:start:end
            match = re.match(r"(\w+):(\w+):(\w+):(\d+):(\d+)", key)
            if match:
                provider, symbol, interval, start, end = match.groups()
                cache_key = f"{symbol}:{interval}"
                if cache_key not in seen:
                    seen.add(cache_key)
                    # 获取实际数据来统计条数
                    data = await self.cache.get(key)
                    count = len(data.data) if data and hasattr(data, 'data') else 0
                    start_date = data.start_date.isoformat() if data and data.start_date else None
                    end_date = data.end_date.isoformat() if data and data.end_date else None

                    symbols_data.append({
                        "symbol": symbol,
                        "interval": interval,
                        "count": count,
                        "start_date": start_date,
                        "end_date": end_date,
                        "source": "cache_entries",
                    })

        return symbols_data

    async def get_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[MarketData]:
        """
        获取 OHLCV 数据

        首先尝试从 market_data 表获取，然后从 cache_entries 表获取
        """
        # 首先尝试从 market_data 表获取
        data = await self.cache.get_market_data(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
        )

        if data and not data.is_empty:
            return data

        # 如果 market_data 表没有数据，从 cache_entries 表获取
        keys = await self.cache.keys(f"*:{symbol}:{interval}:*")

        for key in keys:
            cached_data = await self.cache.get(key)
            if cached_data and isinstance(cached_data, MarketData):
                # 应用日期过滤
                if start_date or end_date:
                    filtered_data = []
                    for ohlcv in cached_data.data:
                        if start_date and ohlcv.timestamp < start_date:
                            continue
                        if end_date and ohlcv.timestamp > end_date:
                            continue
                        filtered_data.append(ohlcv)

                    return MarketData(
                        symbol=cached_data.symbol,
                        data=filtered_data,
                        metadata=cached_data.metadata,
                    )
                return cached_data

        return None

    async def get_all_cached_keys(self) -> List[str]:
        """获取所有缓存键"""
        return await self.cache.keys("*")

    async def cleanup_expired(self) -> int:
        """清理过期缓存"""
        return await self.cache.cleanup_expired()
