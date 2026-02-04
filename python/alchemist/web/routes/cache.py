"""
缓存 API 路由
"""

from typing import List

from fastapi import APIRouter, Request

from web.schemas.cache import CacheStatsResponse

router = APIRouter()


@router.get("/stats", response_model=CacheStatsResponse)
async def get_cache_stats(request: Request) -> CacheStatsResponse:
    """
    获取缓存统计信息

    返回缓存条目数、过期条目数、数据库大小等
    """
    service = request.app.state.cache_service
    stats = await service.get_stats()
    return CacheStatsResponse(**stats)


@router.get("/keys")
async def get_cache_keys(request: Request) -> dict:
    """
    获取所有缓存键

    返回缓存中所有的键列表
    """
    service = request.app.state.cache_service
    keys = await service.get_all_cached_keys()
    return {"keys": keys, "total": len(keys)}


@router.post("/cleanup")
async def cleanup_expired(request: Request) -> dict:
    """
    清理过期缓存

    返回清理的条目数
    """
    service = request.app.state.cache_service
    count = await service.cleanup_expired()
    return {"cleaned": count, "message": f"已清理 {count} 条过期缓存"}
