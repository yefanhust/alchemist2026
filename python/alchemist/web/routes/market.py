"""
市场数据 API 路由
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Query

from web.schemas.market import OHLCVResponse, SymbolListResponse

router = APIRouter()


@router.get("/symbols", response_model=SymbolListResponse)
async def get_symbols(request: Request) -> SymbolListResponse:
    """
    获取所有 Symbol 列表

    返回缓存中所有资产代码及其数据范围
    """
    service = request.app.state.cache_service
    symbols = await service.get_symbols()
    return SymbolListResponse(symbols=symbols, total=len(symbols))


@router.get("/ohlcv/{symbol}", response_model=OHLCVResponse)
async def get_ohlcv(
    request: Request,
    symbol: str,
    interval: str = Query(default="1d", description="数据间隔 (1d, 1h, etc.)"),
    start_date: Optional[str] = Query(default=None, description="开始日期 (ISO 格式, 如 2024-01-01)"),
    end_date: Optional[str] = Query(default=None, description="结束日期 (ISO 格式, 如 2024-12-31)"),
) -> OHLCVResponse:
    """
    获取 OHLCV 数据

    返回指定资产的 K 线数据，支持日期范围过滤
    """
    service = request.app.state.cache_service

    # 解析日期
    start_dt = None
    end_dt = None

    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"无效的开始日期格式: {start_date}，请使用 ISO 格式 (如 2024-01-01)"
            )

    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"无效的结束日期格式: {end_date}，请使用 ISO 格式 (如 2024-12-31)"
            )

    # 获取数据
    data = await service.get_ohlcv(
        symbol=symbol.upper(),
        interval=interval,
        start_date=start_dt,
        end_date=end_dt,
    )

    if data is None or data.is_empty:
        raise HTTPException(
            status_code=404,
            detail=f"未找到 {symbol} 的数据 (interval={interval})"
        )

    return OHLCVResponse(
        symbol=data.symbol,
        interval=interval,
        count=len(data),
        data=[ohlcv.to_dict() for ohlcv in data.data],
    )
