"""
Alpha Vantage 数据提供者
实现从 Alpha Vantage API 获取市场数据
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import aiohttp
from loguru import logger

from data.providers import DataProvider, DataInterval
from data.models import MarketData, OHLCV
from data.cache.base import CacheBackend


class RateLimiter:
    """
    API 限流器
    
    Alpha Vantage 免费版限制：5次/分钟
    """
    
    def __init__(self, calls_per_minute: int = 5):
        self.calls_per_minute = calls_per_minute
        self.call_times: List[datetime] = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """获取请求许可"""
        async with self._lock:
            now = datetime.now()
            
            # 清理旧记录
            one_minute_ago = now - timedelta(minutes=1)
            self.call_times = [t for t in self.call_times if t > one_minute_ago]
            
            # 检查是否需要等待
            if len(self.call_times) >= self.calls_per_minute:
                wait_until = self.call_times[0] + timedelta(minutes=1)
                wait_seconds = (wait_until - now).total_seconds()
                if wait_seconds > 0:
                    logger.debug(f"API 限流，等待 {wait_seconds:.1f} 秒")
                    await asyncio.sleep(wait_seconds)
            
            self.call_times.append(datetime.now())


class AlphaVantageProvider(DataProvider):
    """
    Alpha Vantage 数据提供者
    
    使用 Alpha Vantage API 获取股票、ETF 等市场数据。
    支持日线、周线、月线及分钟数据。
    
    API 文档: https://www.alphavantage.co/documentation/
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    # 时间间隔映射
    INTERVAL_MAP = {
        DataInterval.MINUTE_1: ("TIME_SERIES_INTRADAY", "1min"),
        DataInterval.MINUTE_5: ("TIME_SERIES_INTRADAY", "5min"),
        DataInterval.MINUTE_15: ("TIME_SERIES_INTRADAY", "15min"),
        DataInterval.MINUTE_30: ("TIME_SERIES_INTRADAY", "30min"),
        DataInterval.HOUR_1: ("TIME_SERIES_INTRADAY", "60min"),
        DataInterval.DAILY: ("TIME_SERIES_DAILY_ADJUSTED", None),
        DataInterval.WEEKLY: ("TIME_SERIES_WEEKLY_ADJUSTED", None),
        DataInterval.MONTHLY: ("TIME_SERIES_MONTHLY_ADJUSTED", None),
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_backend: Optional[CacheBackend] = None,
        calls_per_minute: int = 5,
    ):
        """
        初始化 Alpha Vantage 提供者
        
        Args:
            api_key: API 密钥（可通过环境变量 ALPHAVANTAGE_API_KEY 设置）
            cache_backend: 缓存后端
            calls_per_minute: 每分钟最大请求数
        """
        super().__init__(cache_backend)
        
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Alpha Vantage API key 未设置。"
                "请通过参数传入或设置环境变量 ALPHAVANTAGE_API_KEY"
            )
        
        self.rate_limiter = RateLimiter(calls_per_minute)
        self._session: Optional[aiohttp.ClientSession] = None
    
    @property
    def name(self) -> str:
        return "alphavantage"
    
    @property
    def supported_intervals(self) -> List[DataInterval]:
        return list(self.INTERVAL_MAP.keys())
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP 会话"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self) -> None:
        """关闭 HTTP 会话"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _make_request(self, params: Dict[str, str]) -> Dict[str, Any]:
        """
        发送 API 请求
        
        Args:
            params: 请求参数
            
        Returns:
            响应数据
        """
        # 等待限流
        await self.rate_limiter.acquire()
        
        # 添加 API Key
        params["apikey"] = self.api_key
        
        session = await self._get_session()
        
        try:
            async with session.get(self.BASE_URL, params=params) as response:
                if response.status != 200:
                    logger.error(f"API 请求失败: {response.status}")
                    raise Exception(f"API 请求失败: {response.status}")
                
                data = await response.json()
                
                # 检查错误响应
                if "Error Message" in data:
                    logger.error(f"API 错误: {data['Error Message']}")
                    raise Exception(f"API 错误: {data['Error Message']}")
                
                if "Note" in data:
                    # API 限制提示
                    logger.warning(f"API 提示: {data['Note']}")
                
                return data
                
        except aiohttp.ClientError as e:
            logger.error(f"网络请求错误: {e}")
            raise
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: DataInterval = DataInterval.DAILY,
    ) -> MarketData:
        """
        获取历史数据
        
        Args:
            symbol: 资产代码
            start_date: 开始日期
            end_date: 结束日期
            interval: 数据间隔
            
        Returns:
            MarketData 对象
        """
        symbol = symbol.upper()
        end_date = end_date or datetime.now()
        
        # 生成缓存键
        cache_key = self._generate_cache_key(symbol, start_date, end_date, interval)
        
        # 尝试从缓存获取
        if self.cache:
            cached_data = await self.cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"从缓存获取 {symbol} 数据")
                return cached_data
        
        # 从 API 获取数据
        logger.info(f"从 Alpha Vantage 获取 {symbol} 数据")
        
        func, av_interval = self.INTERVAL_MAP[interval]
        
        params = {
            "function": func,
            "symbol": symbol,
            "outputsize": "full",  # 获取完整数据
        }
        
        if av_interval:
            params["interval"] = av_interval
        
        data = await self._make_request(params)
        
        # 解析数据
        market_data = self._parse_response(symbol, data, interval)
        
        # 过滤日期范围
        market_data = market_data.slice(start_date, end_date)
        
        # 保存到缓存
        if self.cache and not market_data.is_empty:
            await self.cache.set(cache_key, market_data)
        
        return market_data
    
    def _parse_response(
        self,
        symbol: str,
        data: Dict[str, Any],
        interval: DataInterval,
    ) -> MarketData:
        """
        解析 API 响应
        
        Args:
            symbol: 资产代码
            data: API 响应数据
            interval: 数据间隔
            
        Returns:
            MarketData 对象
        """
        # 确定时间序列键
        time_series_key = None
        for key in data.keys():
            if "Time Series" in key:
                time_series_key = key
                break
        
        if not time_series_key:
            logger.warning(f"响应中未找到时间序列数据: {list(data.keys())}")
            return MarketData(symbol=symbol)
        
        time_series = data[time_series_key]
        ohlcv_list = []
        
        for timestamp_str, values in time_series.items():
            try:
                # 解析时间戳
                if " " in timestamp_str:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                else:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d")
                
                # 解析 OHLCV 数据
                ohlcv = OHLCV(
                    timestamp=timestamp,
                    open=float(values.get("1. open", 0)),
                    high=float(values.get("2. high", 0)),
                    low=float(values.get("3. low", 0)),
                    close=float(values.get("4. close", 0)),
                    volume=float(values.get("5. volume", values.get("6. volume", 0))),
                    adjusted_close=float(values.get("5. adjusted close", values.get("4. close", 0))),
                )
                ohlcv_list.append(ohlcv)
                
            except (ValueError, KeyError) as e:
                logger.warning(f"解析数据点失败: {timestamp_str}, {e}")
                continue
        
        # 按时间排序
        ohlcv_list.sort(key=lambda x: x.timestamp)
        
        return MarketData(
            symbol=symbol,
            data=ohlcv_list,
            metadata={"source": "alphavantage", "interval": interval.value},
        )
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        获取最新价格
        
        Args:
            symbol: 资产代码
            
        Returns:
            最新价格
        """
        quote = await self.get_quote(symbol)
        if quote:
            return quote.get("price")
        return None
    
    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取实时报价
        
        Args:
            symbol: 资产代码
            
        Returns:
            报价信息
        """
        symbol = symbol.upper()
        
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
        }
        
        data = await self._make_request(params)
        
        global_quote = data.get("Global Quote", {})
        if not global_quote:
            return None
        
        return {
            "symbol": global_quote.get("01. symbol"),
            "open": float(global_quote.get("02. open", 0)),
            "high": float(global_quote.get("03. high", 0)),
            "low": float(global_quote.get("04. low", 0)),
            "price": float(global_quote.get("05. price", 0)),
            "volume": float(global_quote.get("06. volume", 0)),
            "latest_trading_day": global_quote.get("07. latest trading day"),
            "previous_close": float(global_quote.get("08. previous close", 0)),
            "change": float(global_quote.get("09. change", 0)),
            "change_percent": global_quote.get("10. change percent", "0%").replace("%", ""),
        }
    
    async def search_symbols(self, keywords: str) -> List[Dict[str, Any]]:
        """
        搜索资产代码
        
        Args:
            keywords: 搜索关键词
            
        Returns:
            匹配的资产列表
        """
        params = {
            "function": "SYMBOL_SEARCH",
            "keywords": keywords,
        }
        
        data = await self._make_request(params)
        
        matches = data.get("bestMatches", [])
        
        return [
            {
                "symbol": m.get("1. symbol"),
                "name": m.get("2. name"),
                "type": m.get("3. type"),
                "region": m.get("4. region"),
                "currency": m.get("8. currency"),
            }
            for m in matches
        ]
    
    def __repr__(self):
        return f"AlphaVantageProvider(api_key=***)"
