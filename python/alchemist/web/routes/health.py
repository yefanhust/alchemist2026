"""
健康检查路由
"""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health_check(request: Request) -> Dict[str, Any]:
    """
    健康检查端点

    返回服务状态和数据库连接状态
    """
    # 检查缓存服务
    cache_service = request.app.state.cache_service
    db_status = "unknown"

    try:
        stats = await cache_service.get_stats()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "database": db_status,
        "timestamp": datetime.now().isoformat(),
    }
