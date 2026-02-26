"""
估值扫描 Pydantic 模型
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ScanRequest(BaseModel):
    """扫描请求"""
    horizon: str = Field(default="3M", description="投资时间窗口: 1M/3M/6M/1Y (预期回归合理估值的时间)")
    universe: str = Field(default="sp500", description="股票池: sp500/nasdaq100/all/custom")
    custom_symbols: Optional[List[str]] = Field(default=None, description="自定义股票列表")
    top_n: int = Field(default=50, ge=1, le=200, description="返回前N只")
    weights: Optional[Dict[str, float]] = Field(default=None, description="自定义权重")


class StockValuationResponse(BaseModel):
    """单只股票估值"""
    symbol: str
    name: str
    sector: str
    industry: str
    current_price: float
    composite_score: float
    grade: str
    relative_score: float
    absolute_score: float
    sentiment_score: float
    macro_score: float
    dcf_intrinsic_values: Dict[str, Optional[float]]
    safety_margin: Optional[float]
    key_metrics: Dict[str, Any]
    scan_date: Optional[str]
    horizon: str


class SectorSummary(BaseModel):
    """行业汇总"""
    count: int
    avg_score: float
    median_score: float
    undervalued_count: int
    overvalued_count: int


class ScanResultResponse(BaseModel):
    """扫描结果"""
    scan_date: str
    horizon: str
    total_scanned: int
    most_undervalued: List[StockValuationResponse]
    most_overvalued: List[StockValuationResponse]
    sector_summary: Dict[str, SectorSummary]
    macro_context: Dict[str, Any]
    weights_used: Dict[str, float]
    scan_duration_seconds: float


class DataStatusResponse(BaseModel):
    """数据采集状态"""
    total: int
    with_overview: int
    with_financials: int
    coverage_pct: float
    financials_pct: float
    missing_overview: List[str]
