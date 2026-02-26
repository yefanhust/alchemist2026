"""
估值扫描 API 路由
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Query
from loguru import logger

from web.schemas.valuation import (
    ScanRequest,
    ScanResultResponse,
    StockValuationResponse,
    DataStatusResponse,
)

router = APIRouter()


def _get_scanner(request: Request):
    """获取或创建扫描器实例"""
    from data.cache.sqlite_cache import SQLiteCache
    from data.providers.fred_provider import FREDProvider
    from data.providers.yfinance_provider import YFinanceSentimentProvider
    from strategy.valuation.scanner import ValuationScanner

    cache_service = request.app.state.cache_service
    cache = cache_service.cache if cache_service else None

    # FRED
    fred = FREDProvider(cache_backend=cache)

    # yfinance
    yf = YFinanceSentimentProvider()

    return ValuationScanner(
        cache=cache,
        fred_provider=fred,
        yfinance_provider=yf,
    )


@router.post("/scan")
async def scan_valuation(request: Request, body: ScanRequest):
    """
    执行估值扫描

    根据指定的时间窗口和股票池，计算所有股票的综合估值分数，
    返回最被高估和最被低估的股票排名。
    """
    scanner = _get_scanner(request)

    if body.weights:
        scanner.scorer.custom_weights = body.weights

    try:
        result = await scanner.scan(
            horizon=body.horizon,
            universe_name=body.universe,
            custom_symbols=body.custom_symbols,
            top_n=body.top_n,
        )
        return result.to_dict()

    except Exception as e:
        logger.error(f"估值扫描失败: {e}")
        raise HTTPException(status_code=500, detail=f"扫描失败: {str(e)}")


@router.get("/results")
async def get_latest_results(
    request: Request,
    limit: int = Query(default=1, ge=1, le=10),
):
    """获取最近的扫描结果"""
    cache_service = request.app.state.cache_service
    if not cache_service:
        raise HTTPException(status_code=500, detail="缓存服务不可用")

    results = await cache_service.cache.get_valuation_results(limit)
    return results


@router.get("/stock/{symbol}")
async def get_stock_valuation(
    request: Request,
    symbol: str,
    horizon: str = Query(default="3M", description="投资时间窗口: 1M/3M/6M/1Y"),
):
    """
    获取单只股票的详细估值分析

    包含四维因子得分、DCF 三情景分析、关键指标。
    """
    scanner = _get_scanner(request)

    try:
        result = await scanner.evaluate_stock(symbol, horizon)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"无法评估 {symbol}，可能缺少基本面数据。"
                       f"请先运行 valuation-fetch 采集数据。"
            )
        return result.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"评估 {symbol} 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-status")
async def get_data_status(
    request: Request,
    universe: str = Query(default="sp500"),
):
    """
    查看数据采集状态

    显示指定股票池的数据覆盖率（OVERVIEW、财务报表、OHLCV）。
    """
    scanner = _get_scanner(request)

    symbols = await scanner.universe.get_universe(universe)
    coverage = await scanner.universe.get_data_coverage(symbols)
    return coverage


@router.get("/history")
async def get_scan_history(
    request: Request,
    limit: int = Query(default=5, ge=1, le=20),
):
    """获取历史扫描记录"""
    cache_service = request.app.state.cache_service
    if not cache_service:
        raise HTTPException(status_code=500, detail="缓存服务不可用")

    results = await cache_service.cache.get_valuation_results(limit)
    return {
        "count": len(results),
        "results": [
            {
                "id": r.get("id"),
                "scan_date": r.get("scan_date"),
                "lookback_period": r.get("lookback_period"),
                "universe": r.get("universe"),
                "total_scanned": r.get("result_json", {}).get("total_scanned", 0),
            }
            for r in results
        ]
    }


@router.get("/cached-stocks")
async def get_cached_stocks(
    request: Request,
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=50, ge=10, le=200),
    search: str = Query(default=""),
    sector: str = Query(default=""),
):
    """
    分页获取缓存的全部股票列表

    支持按 symbol/name 搜索、按行业过滤。
    """
    cache_service = request.app.state.cache_service
    if not cache_service:
        raise HTTPException(status_code=500, detail="缓存服务不可用")

    return await cache_service.cache.get_cached_stocks_paginated(
        page=page, per_page=per_page, search=search, sector=sector,
    )


@router.get("/cached-stock/{symbol}")
async def get_cached_stock_detail(
    request: Request,
    symbol: str,
):
    """
    获取单只股票的全部缓存数据

    包含 overview（基本面快照）、OHLCV 摘要、财务报表（income/balance/cashflow）。
    """
    cache_service = request.app.state.cache_service
    if not cache_service:
        raise HTTPException(status_code=500, detail="缓存服务不可用")

    detail = await cache_service.cache.get_stock_detail(symbol)
    if not detail.get("overview"):
        raise HTTPException(status_code=404, detail=f"未找到 {symbol} 的缓存数据")
    return detail
