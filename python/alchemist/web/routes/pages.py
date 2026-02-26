"""
前端页面路由
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """首页仪表盘"""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Alchemist2026 - 数据可视化"}
    )


@router.get("/chart/candlestick", response_class=HTMLResponse)
async def candlestick_chart(request: Request):
    """K线图页面"""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "charts/candlestick.html",
        {"request": request, "title": "K线图"}
    )


@router.get("/chart/comparison", response_class=HTMLResponse)
async def comparison_chart(request: Request):
    """多 Symbol 对比页面"""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "charts/comparison.html",
        {"request": request, "title": "多资产对比"}
    )


@router.get("/chart/gold-backtest", response_class=HTMLResponse)
async def gold_backtest_chart(request: Request):
    """黄金策略回测页面"""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "charts/gold_backtest.html",
        {"request": request, "title": "黄金策略回测"}
    )


@router.get("/chart/valuation", response_class=HTMLResponse)
async def valuation_chart(request: Request):
    """估值扫描页面"""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        "charts/valuation.html",
        {"request": request, "title": "估值扫描"}
    )
