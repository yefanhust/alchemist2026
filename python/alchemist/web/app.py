"""
FastAPI 应用实例
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.config import WebConfig
from web.services.cache_service import CacheService
from data.cache.sqlite_cache import SQLiteCache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 初始化缓存服务
    config = WebConfig()
    cache = SQLiteCache(db_path=config.db_path)
    await cache._ensure_initialized()
    app.state.cache_service = CacheService(cache)
    app.state.config = config

    yield

    # 清理资源（如果需要）


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title="Alchemist2026 Web",
        description="量化交易系统数据可视化服务",
        version="0.1.0",
        lifespan=lifespan,
    )

    # 获取模块路径
    module_path = Path(__file__).parent

    # 静态文件
    static_path = module_path / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=static_path), name="static")

    # 模板
    templates_path = module_path / "templates"
    if templates_path.exists():
        app.state.templates = Jinja2Templates(directory=templates_path)

    # 注册路由
    from web.routes import health, cache, market, pages

    app.include_router(health.router, tags=["Health"])
    app.include_router(cache.router, prefix="/api/cache", tags=["Cache"])
    app.include_router(market.router, prefix="/api/market", tags=["Market"])
    app.include_router(pages.router, tags=["Pages"])

    return app


# 创建应用实例
app = create_app()
