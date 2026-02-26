"""
SQLite 缓存后端
使用 SQLite 数据库实现持久化缓存
"""

import asyncio
import json
import os
import pickle
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any, Dict, List
import aiosqlite
from loguru import logger

from data.cache.base import CacheBackend
from data.models import MarketData, OHLCV


class SQLiteCache(CacheBackend):
    """
    SQLite 缓存后端
    
    特点：
    - 持久化存储
    - 支持过期时间
    - 适合中等规模数据
    - 无需外部依赖
    
    表结构：
    - cache_entries: 通用缓存表
    - market_data: 市场数据专用表（优化查询）
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        default_ttl: Optional[timedelta] = None,
    ):
        """
        初始化 SQLite 缓存

        Args:
            db_path: 数据库文件路径（默认使用项目根目录下 data/cache/market_data.db）
            default_ttl: 默认过期时间
        """
        super().__init__(default_ttl)

        if db_path is None:
            from utils.config import get_data_cache_path
            db_path = get_data_cache_path()

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def _ensure_initialized(self) -> None:
        """确保数据库已初始化"""
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            async with aiosqlite.connect(self.db_path) as db:
                # 创建通用缓存表
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        value BLOB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        data_type TEXT
                    )
                """)
                
                # 创建市场数据表（优化时序数据存储）
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL NOT NULL,
                        adjusted_close REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, interval, timestamp)
                    )
                """)
                
                # 创建股票信息表（存储 Sector/Industry 等元数据）
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS stock_info (
                        symbol TEXT PRIMARY KEY,
                        name TEXT,
                        exchange TEXT,
                        asset_type TEXT,
                        sector TEXT,
                        industry TEXT,
                        country TEXT,
                        currency TEXT,
                        description TEXT,
                        market_cap REAL,
                        pe_ratio REAL,
                        dividend_yield REAL,
                        eps REAL,
                        beta REAL,
                        high_52week REAL,
                        low_52week REAL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # 估值扩展字段迁移（stock_info 表）
                valuation_columns = [
                    ("pb_ratio", "REAL"),
                    ("ps_ratio", "REAL"),
                    ("peg_ratio", "REAL"),
                    ("forward_pe", "REAL"),
                    ("ev_to_ebitda", "REAL"),
                    ("book_value", "REAL"),
                    ("revenue_per_share", "REAL"),
                    ("profit_margin", "REAL"),
                    ("operating_margin", "REAL"),
                    ("return_on_equity", "REAL"),
                    ("shares_outstanding", "REAL"),
                    ("price_to_fcf", "REAL"),
                    ("dividend_per_share", "REAL"),
                    ("payout_ratio", "REAL"),
                ]
                for col_name, col_type in valuation_columns:
                    try:
                        await db.execute(
                            f"ALTER TABLE stock_info ADD COLUMN {col_name} {col_type}"
                        )
                    except Exception:
                        pass  # 列已存在

                # 创建估值财务报表表（DCF 用）
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS valuation_financials (
                        symbol TEXT NOT NULL,
                        fiscal_year TEXT NOT NULL,
                        report_type TEXT NOT NULL,
                        data TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (symbol, fiscal_year, report_type)
                    )
                """)

                # 创建估值扫描结果表
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS valuation_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scan_date TIMESTAMP NOT NULL,
                        lookback_period TEXT NOT NULL,
                        universe TEXT,
                        result_json TEXT NOT NULL,
                        weights_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # 创建索引
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_stock_info_sector
                    ON stock_info(sector)
                """)

                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_stock_info_industry
                    ON stock_info(industry)
                """)

                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_market_data_symbol_interval
                    ON market_data(symbol, interval)
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_market_data_timestamp 
                    ON market_data(timestamp)
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cache_expires 
                    ON cache_entries(expires_at)
                """)
                
                await db.commit()
            
            self._initialized = True
            logger.debug(f"SQLite 缓存初始化完成: {self.db_path}")
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        await self._ensure_initialized()
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT value, data_type, expires_at 
                FROM cache_entries 
                WHERE key = ?
                """,
                (key,)
            )
            row = await cursor.fetchone()
            
            if row is None:
                return None
            
            value_blob, data_type, expires_at = row
            
            # 检查过期
            if expires_at:
                expires_dt = datetime.fromisoformat(expires_at)
                if datetime.now() > expires_dt:
                    await self.delete(key)
                    return None
            
            # 反序列化
            return self._deserialize(value_blob, data_type)
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None,
    ) -> bool:
        """设置缓存值"""
        await self._ensure_initialized()
        
        ttl = ttl or self.default_ttl
        expires_at = None
        if ttl:
            expires_at = (datetime.now() + ttl).isoformat()
        
        # 序列化
        value_blob, data_type = self._serialize(value)
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT OR REPLACE INTO cache_entries (key, value, data_type, expires_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (key, value_blob, data_type, expires_at)
                )
                await db.commit()
            return True
            
        except Exception as e:
            logger.error(f"缓存设置失败: {key}, {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        await self._ensure_initialized()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    "DELETE FROM cache_entries WHERE key = ?",
                    (key,)
                )
                await db.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"缓存删除失败: {key}, {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        return await self.get(key) is not None
    
    async def clear(self) -> int:
        """清空所有缓存"""
        await self._ensure_initialized()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM cache_entries")
                count = (await cursor.fetchone())[0]
                
                await db.execute("DELETE FROM cache_entries")
                await db.execute("DELETE FROM market_data")
                await db.commit()
                
                return count
                
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")
            return 0
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """获取匹配的缓存键"""
        await self._ensure_initialized()
        
        # 转换通配符为 SQL LIKE 模式
        sql_pattern = pattern.replace("*", "%").replace("?", "_")
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT key FROM cache_entries WHERE key LIKE ?",
                (sql_pattern,)
            )
            rows = await cursor.fetchall()
            return [row[0] for row in rows]
    
    async def cleanup_expired(self) -> int:
        """清理过期缓存"""
        await self._ensure_initialized()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    """
                    DELETE FROM cache_entries 
                    WHERE expires_at IS NOT NULL AND expires_at < ?
                    """,
                    (datetime.now().isoformat(),)
                )
                await db.commit()
                
                count = cursor.rowcount
                if count > 0:
                    logger.info(f"清理了 {count} 条过期缓存")
                return count
                
        except Exception as e:
            logger.error(f"清理过期缓存失败: {e}")
            return 0
    
    async def stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        await self._ensure_initialized()
        
        async with aiosqlite.connect(self.db_path) as db:
            # 总条目数
            cursor = await db.execute("SELECT COUNT(*) FROM cache_entries")
            total_entries = (await cursor.fetchone())[0]
            
            # 过期条目数
            cursor = await db.execute(
                """
                SELECT COUNT(*) FROM cache_entries 
                WHERE expires_at IS NOT NULL AND expires_at < ?
                """,
                (datetime.now().isoformat(),)
            )
            expired_entries = (await cursor.fetchone())[0]
            
            # 市场数据条目数（market_data 表行数 + cache_entries 中的 market_data 条目）
            cursor = await db.execute("SELECT COUNT(*) FROM market_data")
            market_data_rows = (await cursor.fetchone())[0]

            cursor = await db.execute(
                "SELECT COUNT(*) FROM cache_entries WHERE data_type = 'market_data'"
            )
            market_data_cache = (await cursor.fetchone())[0]
            market_data_entries = market_data_rows + market_data_cache
            
            # 市场数据覆盖的股票数（合并 market_data 表和 cache_entries 中的 symbol）
            cursor = await db.execute("""
                SELECT COUNT(*) FROM (
                    SELECT DISTINCT symbol FROM market_data
                    UNION
                    SELECT DISTINCT
                        CASE
                            WHEN INSTR(key, ':') > 0
                            THEN SUBSTR(key,
                                INSTR(key, ':') + 1,
                                INSTR(SUBSTR(key, INSTR(key, ':') + 1), ':') - 1
                            )
                        END AS symbol
                    FROM cache_entries
                    WHERE data_type = 'market_data'
                )
            """)
            unique_symbols = (await cursor.fetchone())[0]
            
            # 数据库文件大小
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            return {
                "total_entries": total_entries,
                "expired_entries": expired_entries,
                "market_data_entries": market_data_entries,
                "unique_symbols": unique_symbols,
                "db_size_bytes": db_size,
                "db_size_mb": round(db_size / 1024 / 1024, 2),
                "db_path": str(self.db_path),
            }
    
    # ========== 市场数据专用方法 ==========
    
    async def save_market_data(
        self,
        symbol: str,
        interval: str,
        data: MarketData,
    ) -> int:
        """
        保存市场数据到专用表
        
        Args:
            symbol: 资产代码
            interval: 数据间隔
            data: 市场数据
            
        Returns:
            保存的记录数
        """
        await self._ensure_initialized()
        
        if data.is_empty:
            return 0
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                records = [
                    (
                        symbol,
                        interval,
                        ohlcv.timestamp.isoformat(),
                        ohlcv.open,
                        ohlcv.high,
                        ohlcv.low,
                        ohlcv.close,
                        ohlcv.volume,
                        ohlcv.adjusted_close,
                    )
                    for ohlcv in data.data
                ]
                
                await db.executemany(
                    """
                    INSERT OR REPLACE INTO market_data 
                    (symbol, interval, timestamp, open, high, low, close, volume, adjusted_close)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    records
                )
                await db.commit()
                
                return len(records)
                
        except Exception as e:
            logger.error(f"保存市场数据失败: {symbol}, {e}")
            return 0
    
    async def get_market_data(
        self,
        symbol: str,
        interval: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[MarketData]:
        """
        从专用表获取市场数据
        
        Args:
            symbol: 资产代码
            interval: 数据间隔
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            MarketData 对象
        """
        await self._ensure_initialized()
        
        query = """
            SELECT timestamp, open, high, low, close, volume, adjusted_close
            FROM market_data
            WHERE symbol = ? AND interval = ?
        """
        params = [symbol, interval]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        query += " ORDER BY timestamp ASC"
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                
                if not rows:
                    return None
                
                ohlcv_list = [
                    OHLCV(
                        timestamp=datetime.fromisoformat(row[0]),
                        open=row[1],
                        high=row[2],
                        low=row[3],
                        close=row[4],
                        volume=row[5],
                        adjusted_close=row[6],
                    )
                    for row in rows
                ]
                
                return MarketData(
                    symbol=symbol,
                    data=ohlcv_list,
                    metadata={"source": "cache", "interval": interval},
                )
                
        except Exception as e:
            logger.error(f"获取市场数据失败: {symbol}, {e}")
            return None
    
    # ========== 股票信息方法 ==========

    async def save_stock_info(
        self,
        symbol: str,
        info: Dict[str, Any],
    ) -> bool:
        """
        保存股票信息到专用表

        Args:
            symbol: 股票代码
            info: 股票信息字典，包含 sector, industry 等字段

        Returns:
            是否保存成功
        """
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT OR REPLACE INTO stock_info
                    (symbol, name, exchange, asset_type, sector, industry,
                     country, currency, description, market_cap, pe_ratio,
                     dividend_yield, eps, beta, high_52week, low_52week,
                     pb_ratio, ps_ratio, peg_ratio, forward_pe, ev_to_ebitda,
                     book_value, revenue_per_share, profit_margin, operating_margin,
                     return_on_equity, shares_outstanding, price_to_fcf,
                     dividend_per_share, payout_ratio, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol.upper(),
                        info.get("name"),
                        info.get("exchange"),
                        info.get("asset_type"),
                        info.get("sector"),
                        info.get("industry"),
                        info.get("country"),
                        info.get("currency"),
                        info.get("description"),
                        info.get("market_cap"),
                        info.get("pe_ratio"),
                        info.get("dividend_yield"),
                        info.get("eps"),
                        info.get("beta"),
                        info.get("high_52week"),
                        info.get("low_52week"),
                        info.get("pb_ratio"),
                        info.get("ps_ratio"),
                        info.get("peg_ratio"),
                        info.get("forward_pe"),
                        info.get("ev_to_ebitda"),
                        info.get("book_value"),
                        info.get("revenue_per_share"),
                        info.get("profit_margin"),
                        info.get("operating_margin"),
                        info.get("return_on_equity"),
                        info.get("shares_outstanding"),
                        info.get("price_to_fcf"),
                        info.get("dividend_per_share"),
                        info.get("payout_ratio"),
                        datetime.now().isoformat(),
                    )
                )
                await db.commit()
            return True

        except Exception as e:
            logger.error(f"保存股票信息失败: {symbol}, {e}")
            return False

    async def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取股票信息

        Args:
            symbol: 股票代码

        Returns:
            股票信息字典，包含 sector, industry 等字段
        """
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM stock_info WHERE symbol = ?",
                    (symbol.upper(),)
                )
                row = await cursor.fetchone()

                if row is None:
                    return None

                return dict(row)

        except Exception as e:
            logger.error(f"获取股票信息失败: {symbol}, {e}")
            return None

    async def get_stock_info_batch(
        self,
        symbols: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        批量获取股票信息

        Args:
            symbols: 股票代码列表

        Returns:
            {symbol: info_dict} 映射
        """
        await self._ensure_initialized()

        if not symbols:
            return {}

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                placeholders = ",".join("?" * len(symbols))
                cursor = await db.execute(
                    f"SELECT * FROM stock_info WHERE symbol IN ({placeholders})",
                    [s.upper() for s in symbols]
                )
                rows = await cursor.fetchall()

                return {row["symbol"]: dict(row) for row in rows}

        except Exception as e:
            logger.error(f"批量获取股票信息失败: {e}")
            return {}

    async def search_stocks_by_sector(
        self,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        按行业搜索股票

        Args:
            sector: 行业大类（如 Technology, Healthcare）
            industry: 细分行业（如 Software, Biotechnology）

        Returns:
            匹配的股票信息列表
        """
        await self._ensure_initialized()

        query = "SELECT * FROM stock_info WHERE 1=1"
        params = []

        if sector:
            query += " AND LOWER(sector) LIKE ?"
            params.append(f"%{sector.lower()}%")

        if industry:
            query += " AND LOWER(industry) LIKE ?"
            params.append(f"%{industry.lower()}%")

        query += " ORDER BY symbol"

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"搜索股票失败: {e}")
            return []

    async def get_all_sectors(self) -> List[str]:
        """
        获取所有已缓存的行业大类

        Returns:
            行业大类列表
        """
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    """
                    SELECT DISTINCT sector FROM stock_info
                    WHERE sector IS NOT NULL AND sector != ''
                    ORDER BY sector
                    """
                )
                rows = await cursor.fetchall()
                return [row[0] for row in rows]

        except Exception as e:
            logger.error(f"获取行业列表失败: {e}")
            return []

    async def get_stock_info_stats(self) -> Dict[str, Any]:
        """
        获取股票信息缓存统计

        Returns:
            统计信息
        """
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                # 总条目数
                cursor = await db.execute("SELECT COUNT(*) FROM stock_info")
                total = (await cursor.fetchone())[0]

                # 有 sector 信息的条目数
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM stock_info WHERE sector IS NOT NULL AND sector != ''"
                )
                with_sector = (await cursor.fetchone())[0]

                # 不同 sector 数量
                cursor = await db.execute(
                    "SELECT COUNT(DISTINCT sector) FROM stock_info WHERE sector IS NOT NULL AND sector != ''"
                )
                sector_count = (await cursor.fetchone())[0]

                # 不同 industry 数量
                cursor = await db.execute(
                    "SELECT COUNT(DISTINCT industry) FROM stock_info WHERE industry IS NOT NULL AND industry != ''"
                )
                industry_count = (await cursor.fetchone())[0]

                return {
                    "total_stocks": total,
                    "with_sector_info": with_sector,
                    "unique_sectors": sector_count,
                    "unique_industries": industry_count,
                }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

    async def get_cached_stocks_paginated(
        self,
        page: int = 1,
        per_page: int = 50,
        search: str = "",
        sector: str = "",
    ) -> Dict[str, Any]:
        """
        分页获取缓存的股票列表

        Args:
            page: 页码（从1开始）
            per_page: 每页条数
            search: 搜索关键词（匹配 symbol 或 name）
            sector: 行业过滤

        Returns:
            {"total", "page", "per_page", "total_pages", "stocks": [...], "sectors": [...]}
        """
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row

                # 构建查询条件
                conditions = []
                params: list = []

                if search:
                    conditions.append(
                        "(UPPER(symbol) LIKE ? OR UPPER(name) LIKE ?)"
                    )
                    term = f"%{search.upper()}%"
                    params.extend([term, term])

                if sector:
                    conditions.append("LOWER(sector) = ?")
                    params.append(sector.lower())

                where = (" AND ".join(conditions)) if conditions else "1=1"

                # 总数
                cursor = await db.execute(
                    f"SELECT COUNT(*) FROM stock_info WHERE {where}", params
                )
                total = (await cursor.fetchone())[0]
                total_pages = max(1, (total + per_page - 1) // per_page)

                # 分页数据
                offset = (page - 1) * per_page
                cursor = await db.execute(
                    f"""
                    SELECT symbol, name, sector, industry, market_cap,
                           pe_ratio, updated_at
                    FROM stock_info
                    WHERE {where}
                    ORDER BY market_cap DESC NULLS LAST, symbol
                    LIMIT ? OFFSET ?
                    """,
                    params + [per_page, offset],
                )
                rows = await cursor.fetchall()
                stocks = [dict(row) for row in rows]

                # 检查每只股票是否有 OHLCV 和财报数据
                for s in stocks:
                    sym = s["symbol"]
                    c = await db.execute(
                        "SELECT COUNT(*) FROM market_data WHERE symbol = ?",
                        (sym,),
                    )
                    s["ohlcv_count"] = (await c.fetchone())[0]

                    c = await db.execute(
                        "SELECT COUNT(DISTINCT report_type) FROM valuation_financials WHERE symbol = ?",
                        (sym,),
                    )
                    s["financials_types"] = (await c.fetchone())[0]

                # 所有行业列表
                cursor = await db.execute(
                    """
                    SELECT DISTINCT sector FROM stock_info
                    WHERE sector IS NOT NULL AND sector != ''
                    ORDER BY sector
                    """
                )
                sectors = [row[0] for row in await cursor.fetchall()]

                return {
                    "total": total,
                    "page": page,
                    "per_page": per_page,
                    "total_pages": total_pages,
                    "stocks": stocks,
                    "sectors": sectors,
                }

        except Exception as e:
            logger.error(f"分页获取股票列表失败: {e}")
            return {
                "total": 0, "page": 1, "per_page": per_page,
                "total_pages": 0, "stocks": [], "sectors": [],
            }

    async def get_stock_detail(self, symbol: str) -> Dict[str, Any]:
        """
        获取单只股票的全部缓存数据（overview + OHLCV摘要 + 财报）

        Args:
            symbol: 股票代码

        Returns:
            {"overview": {...}, "ohlcv_summary": {...}, "financials": {...}}
        """
        await self._ensure_initialized()
        sym = symbol.upper()

        # Overview
        overview = await self.get_stock_info(sym)

        # OHLCV 摘要
        ohlcv_summary = None
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute(
                    """
                    SELECT COUNT(*) as cnt,
                           MIN(timestamp) as start_date,
                           MAX(timestamp) as end_date
                    FROM market_data WHERE symbol = ?
                    """,
                    (sym,),
                )
                row = await cursor.fetchone()
                if row and row[0] > 0:
                    # 获取最新一条数据
                    cursor2 = await db.execute(
                        """
                        SELECT close, volume FROM market_data
                        WHERE symbol = ?
                        ORDER BY timestamp DESC LIMIT 1
                        """,
                        (sym,),
                    )
                    latest = await cursor2.fetchone()
                    ohlcv_summary = {
                        "count": row[0],
                        "start_date": row[1],
                        "end_date": row[2],
                        "latest_close": latest[0] if latest else None,
                        "latest_volume": latest[1] if latest else None,
                    }
        except Exception as e:
            logger.error(f"获取 OHLCV 摘要失败: {sym}, {e}")

        # 财报数据
        financials = {}
        for rtype in ("income", "balance", "cashflow"):
            reports = await self.get_valuation_financials(sym, rtype)
            if reports:
                financials[rtype] = reports

        return {
            "overview": overview,
            "ohlcv_summary": ohlcv_summary,
            "financials": financials,
        }

    # ========== 序列化/反序列化 ==========
    
    def _serialize(self, value: Any) -> tuple:
        """序列化值"""
        if isinstance(value, MarketData):
            # 特殊处理 MarketData
            data = {
                "symbol": value.symbol,
                "data": [ohlcv.to_dict() for ohlcv in value.data],
                "metadata": value.metadata,
            }
            return (json.dumps(data).encode(), "market_data")
        else:
            # 通用 pickle 序列化
            return (pickle.dumps(value), "pickle")
    
    def _deserialize(self, value_blob: bytes, data_type: str) -> Any:
        """反序列化值"""
        if data_type == "market_data":
            data = json.loads(value_blob.decode())
            return MarketData(
                symbol=data["symbol"],
                data=[OHLCV.from_dict(d) for d in data["data"]],
                metadata=data.get("metadata", {}),
            )
        else:
            return pickle.loads(value_blob)
    
    # ========== 估值财务报表方法 ==========

    async def save_valuation_financials(
        self,
        symbol: str,
        fiscal_year: str,
        report_type: str,
        data: Dict[str, Any],
    ) -> bool:
        """
        保存财务报表数据（供 DCF 模型使用）

        Args:
            symbol: 股票代码
            fiscal_year: 财年（如 "2024"）
            report_type: 报表类型 ("income" / "balance" / "cashflow")
            data: 报表数据字典
        """
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT OR REPLACE INTO valuation_financials
                    (symbol, fiscal_year, report_type, data, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        symbol.upper(),
                        fiscal_year,
                        report_type,
                        json.dumps(data),
                        datetime.now().isoformat(),
                    )
                )
                await db.commit()
            return True

        except Exception as e:
            logger.error(f"保存财务报表失败: {symbol}/{fiscal_year}/{report_type}, {e}")
            return False

    async def get_valuation_financials(
        self,
        symbol: str,
        report_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取财务报表数据

        Args:
            symbol: 股票代码
            report_type: 报表类型（可选，不传则返回所有类型）

        Returns:
            财务报表列表
        """
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                if report_type:
                    cursor = await db.execute(
                        "SELECT * FROM valuation_financials WHERE symbol = ? AND report_type = ? ORDER BY fiscal_year DESC",
                        (symbol.upper(), report_type)
                    )
                else:
                    cursor = await db.execute(
                        "SELECT * FROM valuation_financials WHERE symbol = ? ORDER BY fiscal_year DESC",
                        (symbol.upper(),)
                    )
                rows = await cursor.fetchall()
                results = []
                for row in rows:
                    d = dict(row)
                    d["data"] = json.loads(d["data"])
                    results.append(d)
                return results

        except Exception as e:
            logger.error(f"获取财务报表失败: {symbol}, {e}")
            return []

    async def save_valuation_result(
        self,
        scan_date: datetime,
        horizon: str,
        universe: str,
        result: Dict[str, Any],
        weights: Dict[str, float],
    ) -> bool:
        """保存估值扫描结果"""
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO valuation_results
                    (scan_date, lookback_period, universe, result_json, weights_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        scan_date.isoformat(),
                        horizon,
                        universe,
                        json.dumps(result),
                        json.dumps(weights),
                    )
                )
                await db.commit()
            return True

        except Exception as e:
            logger.error(f"保存估值结果失败: {e}")
            return False

    async def get_valuation_results(
        self,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """获取历史估值扫描结果"""
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM valuation_results ORDER BY scan_date DESC LIMIT ?",
                    (limit,)
                )
                rows = await cursor.fetchall()
                results = []
                for row in rows:
                    d = dict(row)
                    d["result_json"] = json.loads(d["result_json"])
                    d["weights_json"] = json.loads(d["weights_json"]) if d["weights_json"] else {}
                    results.append(d)
                return results

        except Exception as e:
            logger.error(f"获取估值结果失败: {e}")
            return []

    async def __aenter__(self):
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def __repr__(self):
        return f"SQLiteCache({self.db_path})"
