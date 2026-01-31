"""
缓存系统模块
"""

from data.cache.base import CacheBackend
from data.cache.sqlite_cache import SQLiteCache

__all__ = ["CacheBackend", "SQLiteCache"]
