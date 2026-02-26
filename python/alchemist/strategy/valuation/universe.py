"""
股票池管理

提供全市场股票列表、预设股票池、过滤条件。
"""

import re
from typing import Any, Dict, List, Optional

from loguru import logger

# 合法美股代码：1-5 位大写字母（排除优先股 -P-、权证 -W- 等）
_VALID_TICKER = re.compile(r"^[A-Z]{1,5}$")


# S&P 500 核心成分股（前100，完整列表过长，使用 LISTING_STATUS 获取全量）
SP500_CORE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B",
    "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "AVGO", "CVX",
    "MRK", "LLY", "ABBV", "PEP", "KO", "COST", "ADBE", "WMT", "MCD",
    "CSCO", "CRM", "ACN", "TMO", "ABT", "DHR", "BAC", "NFLX", "CMCSA",
    "PFE", "LIN", "NKE", "TXN", "PM", "ORCL", "AMD", "WFC", "INTC",
    "UPS", "UNP", "RTX", "QCOM", "MS", "NEE", "ELV", "LOW", "INTU",
    "SPGI", "BMY", "HON", "ISRG", "AMAT", "GS", "BLK", "DE", "GILD",
    "MDT", "SYK", "ADP", "VRTX", "REGN", "CB", "BKNG", "MDLZ", "ADI",
    "TJX", "PLD", "TMUS", "SCHW", "CI", "AMT", "MMC", "ETN", "SO",
    "ZTS", "DUK", "MO", "CME", "BDX", "SLB", "EOG", "CL", "PYPL",
    "LRCX", "APD", "ITW", "NOC", "GD", "SNPS", "ICE", "SHW", "KLAC",
]

NASDAQ100_CORE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "AVGO",
    "ADBE", "CSCO", "NFLX", "COST", "AMD", "INTC", "QCOM", "INTU",
    "AMAT", "BKNG", "ADI", "ISRG", "VRTX", "REGN", "GILD", "MDLZ",
    "PYPL", "LRCX", "KLAC", "SNPS", "MRVL", "PANW", "FTNT", "CDNS",
    "MNST", "ADP", "ORLY", "MELI", "NXPI", "ASML", "PCAR", "KDP",
    "CTAS", "ROST", "IDXX", "CPRT", "ODFL", "WDAY", "FAST", "FANG",
    "AEP", "EXC", "XEL", "DXCM", "EA", "VRSK", "CTSH", "MCHP",
    "PAYX", "GEHC", "ON", "ANSS", "CDW", "TTD", "ZS", "CRWD",
]


class StockUniverse:
    """股票池管理器"""

    def __init__(self, cache=None, provider=None):
        """
        Args:
            cache: SQLiteCache 实例
            provider: AlphaVantageProvider 实例（用于 universe=all 时拉取 LISTING_STATUS）
        """
        self.cache = cache
        self.provider = provider

    async def get_universe(
        self,
        universe_name: str = "sp500",
        custom_symbols: Optional[List[str]] = None,
    ) -> List[str]:
        """
        获取股票池

        Args:
            universe_name: 预设名称 ("sp500" / "nasdaq100" / "all" / "custom")
            custom_symbols: 自定义符号列表（universe_name="custom" 时使用）

        Returns:
            股票代码列表
        """
        if universe_name == "custom" and custom_symbols:
            return [s.upper() for s in custom_symbols]

        if universe_name == "sp500":
            return SP500_CORE.copy()

        if universe_name == "nasdaq100":
            return NASDAQ100_CORE.copy()

        if universe_name == "all":
            return await self._get_all_us_stocks()

        logger.warning(f"未知股票池: {universe_name}，使用 SP500")
        return SP500_CORE.copy()

    async def _get_all_us_stocks(self) -> List[str]:
        """
        获取全市场美股列表

        优先从缓存读取；缓存为空时调用 LISTING_STATUS API 拉取并写入缓存。
        """
        if self.cache is None:
            logger.warning("无缓存，返回 SP500 作为默认股票池")
            return SP500_CORE.copy()

        try:
            # 1. 先查缓存
            import aiosqlite
            async with aiosqlite.connect(self.cache.db_path) as db:
                cursor = await db.execute(
                    """
                    SELECT symbol FROM stock_info
                    WHERE country IN ('United States', 'USA')
                      AND asset_type = 'Common Stock'
                      AND LENGTH(symbol) BETWEEN 1 AND 5
                      AND symbol NOT GLOB '*[^A-Z]*'
                    ORDER BY market_cap DESC NULLS LAST
                    """
                )
                rows = await cursor.fetchall()
                symbols = [row[0] for row in rows]

            if symbols:
                logger.info(f"从缓存获取 {len(symbols)} 只美股")
                return symbols

            # 2. 缓存为空，尝试调用 LISTING_STATUS API
            if self.provider is not None:
                return await self._fetch_listing_status()

            logger.info("缓存中无全市场数据且无 provider，返回 SP500")
            return SP500_CORE.copy()

        except Exception as e:
            logger.warning(f"获取全市场数据失败: {e}，使用 SP500")
            return SP500_CORE.copy()

    async def _fetch_listing_status(self) -> List[str]:
        """
        调用 LISTING_STATUS API 获取全市场列表，
        过滤为 Common Stock 并写入 stock_info 缓存。
        """
        logger.info("缓存为空，正在通过 LISTING_STATUS API 获取全市场股票列表...")
        listings = await self.provider.get_listing_status(state="active")

        # 过滤: 仅普通股、美国主要交易所、合法代码格式（1-5位大写字母）
        major_exchanges = {"NYSE", "NASDAQ", "NYSE ARCA", "NYSE MKT", "BATS"}
        symbols = []
        for item in listings:
            sym = item.get("symbol", "")
            asset_type = item.get("asset_type", "")
            exchange = item.get("exchange", "")
            if (
                asset_type == "Stock"
                and _VALID_TICKER.match(sym)
                and exchange in major_exchanges
            ):
                symbols.append(sym)

                # 写入 stock_info 缓存（基础信息，后续 overview 会补全）
                if self.cache is not None:
                    await self.cache.save_stock_info(sym, {
                        "name": item.get("name", ""),
                        "exchange": exchange,
                        "asset_type": "Common Stock",
                        "country": "United States",
                        "ipo_date": item.get("ipo_date", ""),
                    })

        logger.info(f"LISTING_STATUS: {len(listings)} 条记录，过滤后 {len(symbols)} 只美股")
        return symbols

    async def filter_universe(
        self,
        symbols: List[str],
        min_market_cap: float = 1e8,
        min_avg_volume: float = 1e5,
        sectors: Optional[List[str]] = None,
        exchanges: Optional[List[str]] = None,
    ) -> List[str]:
        """
        过滤股票池

        Args:
            symbols: 原始符号列表
            min_market_cap: 最低市值（默认1亿美元）
            min_avg_volume: 最低日均成交量
            sectors: 限制行业（可选）
            exchanges: 限制交易所（可选）

        Returns:
            过滤后的符号列表
        """
        if self.cache is None:
            return symbols

        try:
            stock_infos = await self.cache.get_stock_info_batch(symbols)
        except Exception:
            return symbols

        filtered = []
        for symbol in symbols:
            info = stock_infos.get(symbol.upper())
            if info is None:
                # 无缓存数据的股票保留（可能尚未采集）
                filtered.append(symbol)
                continue

            # 市值过滤
            mc = info.get("market_cap")
            if mc is not None and mc < min_market_cap:
                continue

            # 行业过滤
            if sectors:
                s = info.get("sector", "")
                if s and s not in sectors:
                    continue

            # 交易所过滤
            if exchanges:
                ex = info.get("exchange", "")
                if ex and ex not in exchanges:
                    continue

            filtered.append(symbol)

        logger.info(f"股票池过滤: {len(symbols)} → {len(filtered)}")
        return filtered

    async def get_data_coverage(
        self,
        symbols: List[str],
    ) -> Dict[str, Any]:
        """
        检查数据覆盖率

        Returns:
            {
                "total": int,
                "with_overview": int,
                "with_financials": int,
                "with_ohlcv": int,
                "coverage_pct": float,
                "missing_overview": [symbols],
            }
        """
        if self.cache is None:
            return {"total": len(symbols), "coverage_pct": 0}

        stock_infos = await self.cache.get_stock_info_batch(symbols)

        with_overview = 0
        missing_overview = []
        for s in symbols:
            info = stock_infos.get(s.upper())
            if info and info.get("pe_ratio") is not None:
                with_overview += 1
            else:
                missing_overview.append(s)

        # 检查财务报表覆盖
        with_financials = 0
        for s in symbols:
            reports = await self.cache.get_valuation_financials(s.upper(), "cashflow")
            if reports:
                with_financials += 1

        return {
            "total": len(symbols),
            "with_overview": with_overview,
            "with_financials": with_financials,
            "coverage_pct": round(with_overview / len(symbols) * 100, 1) if symbols else 0,
            "financials_pct": round(with_financials / len(symbols) * 100, 1) if symbols else 0,
            "missing_overview": missing_overview[:50],  # 最多显示50个
        }
