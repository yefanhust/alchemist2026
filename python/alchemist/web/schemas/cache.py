"""
缓存相关 Pydantic 模型
"""

from typing import Any, Dict
from pydantic import BaseModel


class CacheStatsResponse(BaseModel):
    """缓存统计响应"""
    total_entries: int
    expired_entries: int
    market_data_entries: int
    unique_symbols: int
    db_size_bytes: int
    db_size_mb: float
    db_path: str
