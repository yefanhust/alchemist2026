"""
FastAPI 应用实例
"""

import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from web.config import WebConfig
from web.services.cache_service import CacheService
from web.auth import IPBanManager, AuthManager, COOKIE_NAME
from data.cache.sqlite_cache import SQLiteCache


# 不需要认证的路径前缀
PUBLIC_PATHS = ("/login", "/static", "/health", "/favicon.ico")


class AuthMiddleware(BaseHTTPMiddleware):
    """认证中间件：未登录时重定向到登录页"""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # 放行公开路径
        if any(path.startswith(p) for p in PUBLIC_PATHS):
            return await call_next(request)

        # 如果未配置密码，跳过认证
        auth_manager: AuthManager | None = getattr(
            request.app.state, "auth_manager", None
        )
        if auth_manager is None:
            return await call_next(request)

        # 验证 session cookie
        token = request.cookies.get(COOKIE_NAME)
        if token and auth_manager.validate_session_token(token):
            return await call_next(request)

        # 未认证，重定向到登录页
        return RedirectResponse(url="/login", status_code=303)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 初始化缓存服务
    config = WebConfig()
    cache = SQLiteCache(db_path=config.db_path)
    await cache._ensure_initialized()
    app.state.cache_service = CacheService(cache)
    app.state.config = config

    # 初始化认证（仅在设置了密码时启用）
    if config.auth_password:
        secret_key = config.secret_key or secrets.token_hex(32)
        ban_manager = IPBanManager(config.auth_db_path)
        app.state.auth_manager = AuthManager(
            password=config.auth_password,
            secret_key=secret_key,
            ban_manager=ban_manager,
        )
        # 启动时清理过期封禁记录
        ban_manager.cleanup_expired()
    else:
        app.state.auth_manager = None

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

    # 认证中间件
    app.add_middleware(AuthMiddleware)

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
    from web.routes import health, cache, market, pages, gold_backtest, auth, valuation

    app.include_router(auth.router, tags=["Auth"])
    app.include_router(health.router, tags=["Health"])
    app.include_router(cache.router, prefix="/api/cache", tags=["Cache"])
    app.include_router(market.router, prefix="/api/market", tags=["Market"])
    app.include_router(gold_backtest.router, prefix="/api/gold-backtest", tags=["GoldBacktest"])
    app.include_router(valuation.router, prefix="/api/valuation", tags=["Valuation"])
    app.include_router(pages.router, tags=["Pages"])

    return app


# 创建应用实例
app = create_app()
