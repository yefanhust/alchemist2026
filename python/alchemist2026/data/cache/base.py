"""
缓存后端基类
定义缓存系统的统一接口
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Any, Dict


class CacheBackend(ABC):
    """
    缓存后端抽象基类
    
    定义缓存操作的统一接口。
    所有缓存实现（SQLite, Redis, 文件等）都应实现此接口。
    """
    
    def __init__(self, default_ttl: Optional[timedelta] = None):
        """
        初始化缓存后端
        
        Args:
            default_ttl: 默认过期时间（None 表示永不过期）
        """
        self.default_ttl = default_ttl
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，不存在或已过期返回 None
        """
        pass
    
    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None,
    ) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（None 使用默认值）
            
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        检查缓存是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            是否存在且未过期
        """
        pass
    
    @abstractmethod
    async def clear(self) -> int:
        """
        清空所有缓存
        
        Returns:
            删除的条目数
        """
        pass
    
    @abstractmethod
    async def keys(self, pattern: str = "*") -> list:
        """
        获取匹配的缓存键
        
        Args:
            pattern: 匹配模式
            
        Returns:
            匹配的键列表
        """
        pass
    
    async def get_or_set(
        self,
        key: str,
        default_factory,
        ttl: Optional[timedelta] = None,
    ) -> Any:
        """
        获取缓存，不存在则设置默认值
        
        Args:
            key: 缓存键
            default_factory: 默认值工厂函数
            ttl: 过期时间
            
        Returns:
            缓存值
        """
        value = await self.get(key)
        if value is not None:
            return value
        
        # 生成默认值
        if callable(default_factory):
            value = await default_factory() if asyncio.iscoroutinefunction(default_factory) else default_factory()
        else:
            value = default_factory
        
        await self.set(key, value, ttl)
        return value
    
    async def get_many(self, keys: list) -> Dict[str, Any]:
        """
        批量获取缓存
        
        Args:
            keys: 缓存键列表
            
        Returns:
            {key: value} 字典
        """
        result = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        return result
    
    async def set_many(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[timedelta] = None,
    ) -> int:
        """
        批量设置缓存
        
        Args:
            mapping: {key: value} 字典
            ttl: 过期时间
            
        Returns:
            成功设置的数量
        """
        count = 0
        for key, value in mapping.items():
            if await self.set(key, value, ttl):
                count += 1
        return count
    
    async def delete_many(self, keys: list) -> int:
        """
        批量删除缓存
        
        Args:
            keys: 缓存键列表
            
        Returns:
            成功删除的数量
        """
        count = 0
        for key in keys:
            if await self.delete(key):
                count += 1
        return count
    
    @abstractmethod
    async def cleanup_expired(self) -> int:
        """
        清理过期缓存
        
        Returns:
            清理的条目数
        """
        pass
    
    @abstractmethod
    async def stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        pass
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        pass


import asyncio  # 在文件顶部添加，这里为了不修改原有代码放在这里
