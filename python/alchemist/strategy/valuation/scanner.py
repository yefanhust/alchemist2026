"""
估值扫描引擎

协调数据采集、因子计算、综合打分的完整流程。
"""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from strategy.valuation.factors.relative import RelativeValuationFactors
from strategy.valuation.factors.absolute import AbsoluteValuationFactors
from strategy.valuation.factors.sentiment import SentimentFactors
from strategy.valuation.factors.macro import MacroFactors
from strategy.valuation.models import (
    ScanResult,
    StockValuation,
    get_horizon_days,
)
from strategy.valuation.scorer import ValuationScorer
from strategy.valuation.universe import StockUniverse


class ValuationScanner:
    """
    全市场估值扫描引擎

    协调四步估值框架的完整流程：
    1. 股票池获取与过滤
    2. 数据完备性检查
    3. 四维因子计算
    4. 综合打分与排名
    """

    def __init__(
        self,
        cache=None,
        alpha_vantage_provider=None,
        fred_provider=None,
        yfinance_provider=None,
        weights: Optional[Dict[str, float]] = None,
        dcf_config: Optional[Dict[str, Any]] = None,
    ):
        self.cache = cache
        self.av_provider = alpha_vantage_provider
        self.fred_provider = fred_provider
        self.yf_provider = yfinance_provider

        self.universe = StockUniverse(cache, provider=alpha_vantage_provider)
        self.relative_factors = RelativeValuationFactors()
        self.absolute_factors = AbsoluteValuationFactors(dcf_config)
        self.sentiment_factors = SentimentFactors()
        self.macro_factors = MacroFactors()
        self.scorer = ValuationScorer(weights)

    async def scan(
        self,
        horizon: str = "3M",
        universe_name: str = "sp500",
        custom_symbols: Optional[List[str]] = None,
        top_n: int = 50,
        min_market_cap: float = 1e8,
    ) -> ScanResult:
        """
        执行估值扫描

        Args:
            horizon: 时间窗口 ("1M" / "3M" / "6M" / "1Y")
            universe_name: 股票池名称
            custom_symbols: 自定义符号列表
            top_n: 返回前 N 只
            min_market_cap: 最低市值

        Returns:
            ScanResult 扫描结果
        """
        start_time = time.time()
        horizon_days = get_horizon_days(horizon)

        logger.info(f"开始估值扫描: period={horizon}, universe={universe_name}, top_n={top_n}")

        # 1. 获取股票池
        symbols = await self.universe.get_universe(universe_name, custom_symbols)
        symbols = await self.universe.filter_universe(
            symbols, min_market_cap=min_market_cap
        )
        logger.info(f"股票池: {len(symbols)} 只")

        # 2. 加载全部 stock_info（基本面数据）
        all_stock_info = {}
        if self.cache:
            all_stock_info = await self.cache.get_stock_info_batch(symbols)

        # 3. 宏观因子（全局统一）
        macro_result = await self._compute_macro()

        # 4. 按行业分组（用于相对估值的行业对比）
        industry_groups = self._group_by_industry(all_stock_info)

        # 5. 批量获取情绪数据（yfinance）
        sentiment_data = {}
        if self.yf_provider and self.yf_provider.is_available:
            try:
                sentiment_data = self.yf_provider.get_batch_sentiment_data(symbols)
            except Exception as e:
                logger.warning(f"批量获取情绪数据失败: {e}")

        # 6. 逐只股票计算
        valuations: List[StockValuation] = []
        for symbol in symbols:
            try:
                val = await self._evaluate_single(
                    symbol=symbol,
                    stock_info=all_stock_info.get(symbol.upper(), {}),
                    industry_groups=industry_groups,
                    macro_result=macro_result,
                    sentiment_data=sentiment_data.get(symbol.upper(), {}),
                    horizon_days=horizon_days,
                    horizon=horizon,
                )
                if val is not None:
                    valuations.append(val)
            except Exception as e:
                logger.debug(f"评估 {symbol} 失败: {e}")
                continue

        # 7. 排序
        valuations.sort(key=lambda v: v.composite_score)

        # 低估（分数最低的）
        undervalued = [v for v in valuations if v.grade in ("A", "B")][:top_n]
        # 高估（分数最高的）
        overvalued = [v for v in reversed(valuations) if v.grade in ("D", "F")][:top_n]

        # 8. 行业汇总
        sector_summary = self._compute_sector_summary(valuations)

        duration = time.time() - start_time

        result = ScanResult(
            scan_date=datetime.now(),
            horizon=horizon,
            total_scanned=len(valuations),
            most_undervalued=undervalued,
            most_overvalued=overvalued,
            sector_summary=sector_summary,
            macro_context=macro_result.get("details", {}),
            weights_used=self.scorer.get_weights(horizon),
            scan_duration_seconds=duration,
        )

        # 9. 持久化结果
        if self.cache:
            await self.cache.save_valuation_result(
                scan_date=datetime.now(),
                horizon=horizon,
                universe=universe_name,
                result=result.to_dict(),
                weights=self.scorer.get_weights(horizon),
            )

        logger.info(
            f"扫描完成: {len(valuations)} 只股票, "
            f"低估 {len(undervalued)} 只, 高估 {len(overvalued)} 只, "
            f"耗时 {duration:.1f}s"
        )

        return result

    async def evaluate_stock(
        self,
        symbol: str,
        horizon: str = "3M",
    ) -> Optional[StockValuation]:
        """
        评估单只股票

        Args:
            symbol: 股票代码
            horizon: 时间窗口

        Returns:
            StockValuation 或 None
        """
        horizon_days = get_horizon_days(horizon)

        # 获取基本面数据
        stock_info = {}
        if self.cache:
            stock_info = await self.cache.get_stock_info(symbol.upper()) or {}

        # 获取行业同行
        industry_groups = {}
        if stock_info.get("industry") and self.cache:
            peers = await self.cache.search_stocks_by_sector(
                industry=stock_info["industry"]
            )
            industry_groups[stock_info["industry"]] = peers

        # 宏观
        macro_result = await self._compute_macro()

        # 情绪
        sentiment_data = {}
        if self.yf_provider and self.yf_provider.is_available:
            try:
                sentiment_data = self.yf_provider.get_batch_sentiment_data([symbol])
                sentiment_data = sentiment_data.get(symbol.upper(), {})
            except Exception:
                pass

        return await self._evaluate_single(
            symbol=symbol,
            stock_info=stock_info,
            industry_groups=industry_groups,
            macro_result=macro_result,
            sentiment_data=sentiment_data,
            horizon_days=horizon_days,
            horizon=horizon,
        )

    async def _evaluate_single(
        self,
        symbol: str,
        stock_info: Dict[str, Any],
        industry_groups: Dict[str, List],
        macro_result: Dict[str, Any],
        sentiment_data: Dict[str, Any],
        horizon_days: int,
        horizon: str,
    ) -> Optional[StockValuation]:
        """评估单只股票"""
        if not stock_info:
            return None

        # 估算当前价格
        pe = stock_info.get("pe_ratio")
        eps = stock_info.get("eps")
        current_price = 0.0
        if pe and eps and pe > 0 and eps > 0:
            current_price = pe * eps
        elif stock_info.get("high_52week") and stock_info.get("low_52week"):
            current_price = (stock_info["high_52week"] + stock_info["low_52week"]) / 2

        if current_price <= 0:
            return None

        # --- 第1步：相对估值 ---
        industry = stock_info.get("industry", "")
        peers = industry_groups.get(industry, [])
        relative_result = self.relative_factors.calculate(stock_info, peers)

        # --- 第2步：绝对估值 ---
        absolute_result = {"absolute_score": 0.0, "dcf_result": {}, "ri_result": {}}
        if self.cache:
            cashflow_reports = await self._get_financial_reports(symbol, "cashflow")
            balance_reports = await self._get_financial_reports(symbol, "balance")
            income_reports = await self._get_financial_reports(symbol, "income")

            if cashflow_reports:
                risk_free_rate = macro_result.get("details", {}).get("treasury_10y", 4.0) / 100
                absolute_result = self.absolute_factors.calculate(
                    stock_info=stock_info,
                    cashflow_reports=cashflow_reports,
                    balance_reports=balance_reports,
                    income_reports=income_reports,
                    risk_free_rate=risk_free_rate,
                )

        # --- 第3步：情绪因子 ---
        prices, volumes = await self._get_price_data(symbol, horizon_days + 60)
        sentiment_result = {"sentiment_score": 0.0, "factors": {}, "details": {}}
        if len(prices) > 20:
            sentiment_result = self.sentiment_factors.calculate(
                prices=prices,
                volumes=volumes,
                horizon_days=horizon_days,
                short_data=sentiment_data.get("short_interest"),
                insider_data=sentiment_data.get("insider"),
            )

        # --- 第4步：综合打分 ---
        return self.scorer.score(
            symbol=symbol.upper(),
            name=stock_info.get("name", ""),
            sector=stock_info.get("sector", ""),
            industry=industry,
            current_price=current_price,
            relative_result=relative_result,
            absolute_result=absolute_result,
            sentiment_result=sentiment_result,
            macro_result=macro_result,
            horizon=horizon,
        )

    async def _compute_macro(self) -> Dict[str, Any]:
        """计算宏观因子"""
        macro_data = {}

        if self.fred_provider and self.fred_provider.is_available:
            try:
                macro_data = await self.fred_provider.get_macro_snapshot()
            except Exception as e:
                logger.warning(f"获取宏观数据失败: {e}")

        # 市场 PE（S&P 500 大致平均值，可从数据中动态计算）
        market_pe = None
        if self.cache:
            try:
                spy_info = await self.cache.get_stock_info("SPY")
                if spy_info and spy_info.get("pe_ratio"):
                    market_pe = spy_info["pe_ratio"]
            except Exception:
                pass

        return self.macro_factors.calculate(macro_data, market_pe)

    async def _get_financial_reports(
        self, symbol: str, report_type: str
    ) -> List[Dict[str, Any]]:
        """从缓存获取财务报表"""
        if self.cache is None:
            return []

        try:
            reports = await self.cache.get_valuation_financials(
                symbol.upper(), report_type
            )
            return [r["data"] for r in reports]
        except Exception:
            return []

    async def _get_price_data(
        self, symbol: str, days: int
    ) -> tuple:
        """从缓存获取价格数据"""
        prices = np.array([])
        volumes = np.array([])

        if self.cache is None:
            return prices, volumes

        try:
            start_date = datetime.now() - timedelta(days=int(days * 1.5))
            market_data = await self.cache.get_market_data(
                symbol.upper(), "1d", start_date
            )
            if market_data and market_data.data:
                prices = np.array([bar.close for bar in market_data.data])
                volumes = np.array([bar.volume for bar in market_data.data])
        except Exception:
            pass

        return prices, volumes

    def _group_by_industry(
        self, all_stock_info: Dict[str, Dict]
    ) -> Dict[str, List[Dict]]:
        """按行业分组"""
        groups: Dict[str, List] = {}
        for symbol, info in all_stock_info.items():
            industry = info.get("industry", "")
            if industry:
                if industry not in groups:
                    groups[industry] = []
                groups[industry].append(info)
        return groups

    @staticmethod
    def _compute_sector_summary(
        valuations: List[StockValuation],
    ) -> Dict[str, Dict[str, Any]]:
        """计算行业汇总"""
        sector_data: Dict[str, List[float]] = {}
        sector_counts: Dict[str, int] = {}

        for v in valuations:
            sector = v.sector or "Unknown"
            if sector not in sector_data:
                sector_data[sector] = []
                sector_counts[sector] = 0
            sector_data[sector].append(v.composite_score)
            sector_counts[sector] += 1

        summary = {}
        for sector, scores in sector_data.items():
            arr = np.array(scores)
            summary[sector] = {
                "count": sector_counts[sector],
                "avg_score": round(float(np.mean(arr)), 4),
                "median_score": round(float(np.median(arr)), 4),
                "undervalued_count": int(np.sum(arr < -0.15)),
                "overvalued_count": int(np.sum(arr > 0.15)),
            }

        return dict(sorted(summary.items(), key=lambda x: x[1]["avg_score"]))


# 每种数据类型消耗的最大 API 调用次数（financials: cash_flow 失败则跳过后续，实际可能仅 1 次）
_API_CALLS_PER_TYPE = {"overview": 1, "ohlcv": 1, "financials": 3}

# "all" 展开为所有单项
_ALL_DATA_TYPES = list(_API_CALLS_PER_TYPE.keys())

# Free plan 每日调用上限
_FREE_DAILY_LIMIT = 25


def _expand_data_types(data_types: List[str]) -> List[str]:
    """将 data_types 中的 "all" 展开为所有单项，去重保序"""
    result = []
    for dt in data_types:
        items = _ALL_DATA_TYPES if dt == "all" else [dt]
        for item in items:
            if item not in result:
                result.append(item)
    return result


def _calls_per_symbol(data_types: List[str]) -> int:
    """根据 data_types 计算单只股票需要的 API 调用次数"""
    expanded = _expand_data_types(data_types)
    return sum(_API_CALLS_PER_TYPE.get(dt, 0) for dt in expanded)


async def _filter_fresh_symbols(
    symbols: List[str],
    cache,
    data_types: List[str],
    overview_ttl_days: int = 7,
    financials_ttl_days: int = 30,
    ohlcv_ttl_days: int = 3,
) -> tuple:
    """
    过滤掉所有请求数据类型均已有新鲜缓存的股票。

    Returns:
        (需要采集的股票列表, 跳过数量)
    """
    now = datetime.now()
    needs_fetch = []
    skipped = 0

    # overview 批量查询（单条 SQL，高效）
    stock_info_batch = {}
    if "overview" in data_types:
        stock_info_batch = await cache.get_stock_info_batch(symbols)

    for symbol in symbols:
        all_fresh = True  # 所有请求的类型都新鲜才跳过

        if "overview" in data_types:
            info = stock_info_batch.get(symbol.upper())
            if info and info.get("pe_ratio") is not None:
                updated = info.get("updated_at", "")
                if updated:
                    try:
                        if (now - datetime.fromisoformat(updated)).days < overview_ttl_days:
                            pass  # 新鲜
                        else:
                            all_fresh = False
                    except (ValueError, TypeError):
                        all_fresh = False
                else:
                    all_fresh = False
            else:
                all_fresh = False

        if all_fresh and "financials" in data_types:
            reports = await cache.get_valuation_financials(symbol.upper(), "cashflow")
            if reports:
                updated = reports[0].get("updated_at", "")
                if updated:
                    try:
                        if (now - datetime.fromisoformat(updated)).days < financials_ttl_days:
                            pass  # 新鲜
                        else:
                            all_fresh = False
                    except (ValueError, TypeError):
                        all_fresh = False
                else:
                    all_fresh = False
            else:
                all_fresh = False

        if all_fresh and "ohlcv" in data_types:
            cutoff = now - timedelta(days=ohlcv_ttl_days)
            md = await cache.get_market_data(symbol.upper(), "1d", start_date=cutoff)
            if not md or not md.data:
                all_fresh = False

        if all_fresh:
            skipped += 1
        else:
            needs_fetch.append(symbol)

    return needs_fetch, skipped


async def fetch_data_for_scan(
    symbols: List[str],
    provider,
    cache,
    data_types: Optional[List[str]] = None,
    batch_size: int = 0,
    plan: str = "free",
) -> Dict[str, Any]:
    """
    增量采集数据（独立于扫描的数据采集函数）

    自动跳过已有新鲜缓存的股票，根据 plan 和 data_types 智能计算批量大小。

    Args:
        symbols: 要采集的股票列表
        provider: AlphaVantageProvider
        cache: SQLiteCache
        data_types: 要采集的数据类型 ["overview", "ohlcv", "financials", "all"]
        batch_size: 每批符号数 (0=根据 plan 自动计算)
        plan: API 计划 ("free" / "premium")

    Returns:
        采集统计信息
    """
    if data_types is None:
        data_types = ["overview"]

    # "all" → ["overview", "ohlcv", "financials"]
    data_types = _expand_data_types(data_types)

    stats = {
        "total_symbols": len(symbols),
        "already_cached": 0,
        "to_fetch": 0,
        "api_calls_estimated": 0,
        "overview_fetched": 0,
        "ohlcv_fetched": 0,
        "financials_fetched": 0,
        "errors": 0,
    }

    # 1. 过滤已有新鲜缓存的股票
    symbols_to_fetch, cached_count = await _filter_fresh_symbols(
        symbols, cache, data_types,
    )
    stats["already_cached"] = cached_count

    if not symbols_to_fetch:
        logger.info("所有股票数据均在缓存有效期内，无需采集")
        return stats

    # 2. 智能计算批量大小
    calls_per_sym = _calls_per_symbol(data_types)

    if plan == "free" and calls_per_sym > 0:
        auto_batch = _FREE_DAILY_LIMIT // calls_per_sym
        if batch_size > 0:
            # 用户显式指定，取与自动值的较小者
            effective_batch = min(batch_size, auto_batch, len(symbols_to_fetch))
        else:
            effective_batch = min(auto_batch, len(symbols_to_fetch))
    else:
        # Premium plan：无每日上限，rate limiter 控制节奏
        if batch_size > 0:
            effective_batch = min(batch_size, len(symbols_to_fetch))
        else:
            effective_batch = len(symbols_to_fetch)

    batch_symbols = symbols_to_fetch[:effective_batch]
    stats["to_fetch"] = len(batch_symbols)
    stats["api_calls_estimated"] = len(batch_symbols) * calls_per_sym

    logger.info(
        f"增量采集: {len(symbols)} 总计, {cached_count} 已缓存, "
        f"{len(symbols_to_fetch)} 待采集, 本次批量={len(batch_symbols)} "
        f"(~{stats['api_calls_estimated']} API 调用)"
    )

    # 3. 采集（缓存过滤已在前面完成，无需内层检查）
    for i, symbol in enumerate(batch_symbols):
        logger.info(f"[{i + 1}/{len(batch_symbols)}] 采集 {symbol}")

        try:
            # overview
            has_valid_overview = False
            if "overview" in data_types:
                overview = await provider.get_stock_overview(symbol)
                if overview:
                    stats["overview_fetched"] += 1
                    has_valid_overview = True

            # ohlcv
            if "ohlcv" in data_types:
                start = datetime.now() - timedelta(days=400)
                from data.providers import DataInterval
                data = await provider.get_historical_data(
                    symbol, start, interval=DataInterval.DAILY
                )
                if data and not data.is_empty:
                    stats["ohlcv_fetched"] += 1

            # financials — 需有有效 overview（有 market_cap），否则跳过
            if "financials" in data_types:
                if not has_valid_overview and "overview" not in data_types:
                    # overview 未在本轮采集，检查缓存
                    existing = await cache.get_stock_info(symbol)
                    if existing and existing.get("market_cap"):
                        has_valid_overview = True

                if has_valid_overview:
                    cf = await provider.get_cash_flow(symbol)
                    if cf:
                        stats["financials_fetched"] += 1
                        await provider.get_balance_sheet(symbol)
                        await provider.get_income_statement(symbol)
                else:
                    stats["skipped_financials"] = stats.get("skipped_financials", 0) + 1

        except Exception as e:
            logger.warning(f"采集 {symbol} 失败: {e}")
            stats["errors"] += 1

    return stats
