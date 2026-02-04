"""
Web 模块
提供数据可视化和 API 服务
"""

from web.app import create_app
from web.config import WebConfig

__all__ = [
    "create_app",
    "WebConfig",
]
