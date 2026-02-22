"""
Alpha Vantage 数据提供者
实现从 Alpha Vantage API 获取市场数据
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger


async def _countdown_sleep(total_seconds: float, reason: str = "API 限流") -> None:
    """带进度条倒计时的异步等待"""
    total = max(1, int(total_seconds + 0.5))
    bar_width = 20
    for elapsed in range(total):
        remaining = total - elapsed
        filled = int(bar_width * elapsed / total)
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
        sys.stderr.write(f"\r\u23f3 {reason} | {bar} | {remaining}s ")
        sys.stderr.flush()
        await asyncio.sleep(1)
    bar = "\u2588" * bar_width
    sys.stderr.write(f"\r\u2705 {reason} | {bar} | done       \n")
    sys.stderr.flush()


from data.cache.base import CacheBackend
from data.models import OHLCV, MarketData
from data.providers import DataInterval, DataProvider


class RateLimiter:
    """
    API 限流器

    支持按秒或按分钟的滑动窗口限流，以及可选的每日上限。

    - ALPHAVANTAGE_CALLS_PER_SECOND  → 1 秒窗口（免费版）
    - ALPHAVANTAGE_CALLS_PER_MINUTE  → 60 秒窗口（Premium）
    - ALPHAVANTAGE_CALLS_PER_DAY     → 每日上限，0 表示不限（Premium）

    两个窗口变量互斥：优先使用 per_second，若为 0 则使用 per_minute。
    """

    def __init__(
        self,
        calls_per_second: int = 0,
        calls_per_minute: int = 0,
        calls_per_day: int = 0,
    ):
        if calls_per_second > 0:
            self.window_size = 1.0  # 秒
            self.max_calls = calls_per_second
        elif calls_per_minute > 0:
            self.window_size = 60.0  # 秒
            self.max_calls = calls_per_minute
        else:
            self.window_size = 0.0
            self.max_calls = 0

        self.calls_per_day = calls_per_day  # 0 = 不限制
        self.call_times: List[float] = []  # unix timestamps
        self.daily_call_count: int = 0
        self._daily_reset_date: str = ""
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """获取请求许可，必要时阻塞等待"""
        async with self._lock:
            now_ts = datetime.now().timestamp()
            today_str = datetime.now().strftime("%Y-%m-%d")

            # ---------- 每日限额 ----------
            if today_str != self._daily_reset_date:
                self._daily_reset_date = today_str
                self.daily_call_count = 0

            if self.calls_per_day > 0 and self.daily_call_count >= self.calls_per_day:
                logger.error(f"已达每日 API 调用上限 ({self.calls_per_day} 次)，" "请明天再试或升级 API plan")
                raise Exception(f"已达每日 API 调用上限 ({self.calls_per_day} 次)")

            # ---------- 滑动窗口限额 ----------
            if self.max_calls > 0:
                cutoff = now_ts - self.window_size
                self.call_times = [t for t in self.call_times if t > cutoff]

                if len(self.call_times) >= self.max_calls:
                    wait_seconds = self.call_times[0] + self.window_size - now_ts + 0.1
                    if wait_seconds > 0:
                        logger.debug(f"API 限流，等待 {wait_seconds:.1f} 秒")
                        if wait_seconds >= 1.0:
                            await _countdown_sleep(wait_seconds, "API 速率限制")
                        else:
                            await asyncio.sleep(wait_seconds)

            self.call_times.append(datetime.now().timestamp())
            self.daily_call_count += 1


class AlphaVantageProvider(DataProvider):
    """
    Alpha Vantage 数据提供者

    使用 Alpha Vantage API 获取股票、ETF 等市场数据。
    支持日线、周线、月线及分钟数据。

    API 文档: https://www.alphavantage.co/documentation/
    """

    BASE_URL = "https://www.alphavantage.co/query"

    # Alpha Vantage 不支持的交易所后缀
    # 这些交易所的 symbol 会导致 "Invalid API call" 错误
    UNSUPPORTED_EXCHANGES = {".HK", ".HKG"}

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

    # Plan 预设: {plan_name: (calls_per_second, calls_per_minute, calls_per_day)}
    PLAN_PRESETS: Dict[str, tuple] = {
        "free": (1, 0, 25),  # 1 次/秒, 25 次/天
        "premium": (0, 75, 0),  # 75 次/分钟, 无每日上限
    }

    # 类级别共享限流器，按 API key 隔离。
    # 同一个 API key 的所有 provider 实例共享同一个 RateLimiter，
    # 保证跨实例也不会超出速率限制。
    _rate_limiters: Dict[str, RateLimiter] = {}

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_backend: Optional[CacheBackend] = None,
        plan: Optional[str] = None,
    ):
        """
        初始化 Alpha Vantage 提供者

        Args:
            api_key: API 密钥（可通过环境变量 ALPHAVANTAGE_API_KEY 设置）
            cache_backend: 缓存后端
            plan: API plan 类型 "free" / "premium"
                  （默认读取环境变量 ALPHAVANTAGE_PLAN，未设置则为 "free"）
                  自动应用对应的限流策略：
                    free    → 1 次/秒, 25 次/天
                    premium → 75 次/分钟, 无每日上限
        """
        super().__init__(cache_backend)

        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Alpha Vantage API key 未设置。" "请通过参数传入或设置环境变量 ALPHAVANTAGE_API_KEY")

        if plan is None:
            plan = os.getenv("ALPHAVANTAGE_PLAN", "free").lower()

        preset = self.PLAN_PRESETS.get(plan)
        if preset is None:
            raise ValueError(f"未知的 Alpha Vantage plan: '{plan}'，" f"可选值: {', '.join(self.PLAN_PRESETS.keys())}")

        calls_per_second, calls_per_minute, calls_per_day = preset
        self.plan = plan

        # 获取或创建该 API key 对应的共享限流器
        if self.api_key not in self._rate_limiters:
            self._rate_limiters[self.api_key] = RateLimiter(
                calls_per_second=calls_per_second,
                calls_per_minute=calls_per_minute,
                calls_per_day=calls_per_day,
            )
            logger.info(
                f"Alpha Vantage plan={plan} 限流: "
                f"{calls_per_second or calls_per_minute} 次/"
                f"{'秒' if calls_per_second else '分钟'}"
                f"{f', {calls_per_day} 次/天' if calls_per_day else ''}"
            )
        self.rate_limiter = self._rate_limiters[self.api_key]

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

    async def _make_request(self, params: Dict[str, str], max_retries: int = 3) -> Dict[str, Any]:
        """
        发送 API 请求（含限流重试）

        Args:
            params: 请求参数
            max_retries: 遇到限流时最大重试次数

        Returns:
            响应数据
        """
        for attempt in range(max_retries + 1):
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

                    # 检查限流响应（Information 或 Note）
                    rate_limit_msg = data.get("Information") or data.get("Note")
                    if rate_limit_msg:
                        if attempt < max_retries:
                            rl = self.rate_limiter
                            wait_time = rl.window_size / max(rl.max_calls, 1) + 1
                            logger.warning(
                                f"API 限流，{wait_time:.0f}秒后重试 " f"({attempt + 1}/{max_retries}): {rate_limit_msg}"
                            )
                            if wait_time >= 1.0:
                                await _countdown_sleep(
                                    wait_time,
                                    f"API 限流重试 {attempt + 1}/{max_retries}",
                                )
                            else:
                                await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.error(f"API 限流，已达最大重试次数: {rate_limit_msg}")
                            raise Exception(f"API 限流: {rate_limit_msg}")

                    return data

            except aiohttp.ClientError as e:
                logger.error(f"网络请求错误: {e}")
                raise

        # 不应到达此处
        raise Exception("API 请求失败: 超过最大重试次数")

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

        # 检查是否为不支持的交易所
        for suffix in self.UNSUPPORTED_EXCHANGES:
            if symbol.endswith(suffix.upper()):
                raise ValueError(
                    f"Alpha Vantage 不支持交易所后缀 '{suffix}'（symbol={symbol}）。"
                    f"港股(HKEX)不在 Alpha Vantage 支持范围内，"
                    f"请使用其他数据源（如 Yahoo Finance）获取港股数据，"
                    f"或使用对应的美股 ADR 代码。"
                )

        # 生成缓存键
        cache_key = self._generate_cache_key(symbol, start_date, end_date, interval)

        # 尝试从缓存获取
        if self.cache:
            cached_data = await self.cache.get(cache_key)
            if cached_data is not None:
                logger.info(f"从缓存获取 {symbol} 数据 ({len(cached_data)} 条)")
                # 确保 market_data 表也有数据（供 Web 仪表盘查询）
                if hasattr(self.cache, "save_market_data"):
                    await self.cache.save_market_data(symbol, interval.value, cached_data)
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

        try:
            data = await self._make_request(params)
        except Exception as e:
            # 部分 _ADJUSTED 接口对非美股可能不可用，回退到非 adjusted 版本
            if "Invalid API call" in str(e) and func.endswith("_ADJUSTED"):
                fallback_func = func.replace("_ADJUSTED", "")
                logger.warning(f"{func} 不支持 {symbol}，回退到 {fallback_func}")
                params["function"] = fallback_func
                data = await self._make_request(params)
            else:
                raise

        # 解析数据
        market_data = self._parse_response(symbol, data, interval)

        # 过滤日期范围
        market_data = market_data.slice(start_date, end_date)

        # 保存到缓存
        if self.cache and not market_data.is_empty:
            await self.cache.set(cache_key, market_data)
            # 同步写入 market_data 表，供 Web 仪表盘直接查询
            if hasattr(self.cache, "save_market_data"):
                await self.cache.save_market_data(symbol, interval.value, market_data)

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

        # 检查是否为不支持的交易所
        for suffix in self.UNSUPPORTED_EXCHANGES:
            if symbol.endswith(suffix.upper()):
                logger.warning(f"Alpha Vantage 不支持交易所后缀 '{suffix}'（symbol={symbol}）")
                return None

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

    async def get_forex_daily(
        self,
        from_symbol: str,
        to_symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> MarketData:
        """
        获取外汇日线数据

        Args:
            from_symbol: 基础货币（如 EUR）
            to_symbol: 报价货币（如 USD）
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            MarketData 对象（volume 为 0）
        """
        pair = f"{from_symbol.upper()}/{to_symbol.upper()}"
        end_date = end_date or datetime.now()

        # 尝试缓存
        cache_key = self._generate_cache_key(
            f"FX_{from_symbol}_{to_symbol}", start_date, end_date, DataInterval.DAILY
        )
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                logger.info(f"从缓存获取 {pair} 外汇数据 ({len(cached)} 条)")
                return cached

        logger.info(f"从 Alpha Vantage 获取 {pair} 外汇数据")
        params = {
            "function": "FX_DAILY",
            "from_symbol": from_symbol.upper(),
            "to_symbol": to_symbol.upper(),
            "outputsize": "full",
        }
        data = await self._make_request(params)

        # FX_DAILY 返回 "Time Series FX (Daily)" 键，复用解析逻辑
        market_data = self._parse_response(pair, data, DataInterval.DAILY)
        market_data = market_data.slice(start_date, end_date)

        if self.cache and not market_data.is_empty:
            await self.cache.set(cache_key, market_data)

        return market_data

    async def get_economic_indicator(
        self,
        function: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: str = "daily",
        maturity: Optional[str] = None,
    ) -> MarketData:
        """
        获取经济指标数据（TREASURY_YIELD、CPI、INFLATION 等）

        Alpha Vantage 经济指标返回格式：
        {"name": "...", "interval": "...", "data": [{"date": "2024-01-02", "value": "4.25"}, ...]}

        将 value 填入 OHLCV 的 close 字段。

        Args:
            function: API 函数名（如 TREASURY_YIELD）
            start_date: 开始日期
            end_date: 结束日期
            interval: 数据间隔（daily / weekly / monthly）
            maturity: 到期期限（仅 TREASURY_YIELD 使用，如 10year）

        Returns:
            MarketData 对象
        """
        end_date = end_date or datetime.now()
        symbol = f"{function}:{maturity}" if maturity else function

        cache_key = self._generate_cache_key(symbol, start_date, end_date, DataInterval.DAILY)
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                logger.info(f"从缓存获取 {symbol} 经济指标 ({len(cached)} 条)")
                return cached

        logger.info(f"从 Alpha Vantage 获取 {symbol} 经济指标")
        params = {"function": function, "interval": interval}
        if maturity:
            params["maturity"] = maturity

        data = await self._make_request(params)

        # 解析经济指标格式
        ohlcv_list = []
        for entry in data.get("data", []):
            try:
                date_str = entry.get("date", "")
                value_str = entry.get("value", "")
                if value_str == "." or not value_str:
                    continue
                value = float(value_str)
                timestamp = datetime.strptime(date_str, "%Y-%m-%d")
                ohlcv_list.append(OHLCV(
                    timestamp=timestamp,
                    open=value,
                    high=value,
                    low=value,
                    close=value,
                    volume=0,
                    adjusted_close=value,
                ))
            except (ValueError, KeyError) as e:
                logger.warning(f"解析经济指标数据点失败: {entry}, {e}")
                continue

        ohlcv_list.sort(key=lambda x: x.timestamp)
        market_data = MarketData(
            symbol=symbol,
            data=ohlcv_list,
            metadata={"source": "alphavantage", "type": "economic_indicator"},
        )
        market_data = market_data.slice(start_date, end_date)

        if self.cache and not market_data.is_empty:
            await self.cache.set(cache_key, market_data)

        return market_data

    async def get_treasury_yield(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        maturity: str = "10year",
        interval: str = "daily",
    ) -> MarketData:
        """
        获取美国国债收益率数据

        Args:
            start_date: 开始日期
            end_date: 结束日期
            maturity: 到期期限（3month, 2year, 5year, 7year, 10year, 30year）
            interval: 数据间隔（daily, weekly, monthly）

        Returns:
            MarketData 对象（close 字段为收益率百分比值，如 4.25 表示 4.25%）
        """
        return await self.get_economic_indicator(
            function="TREASURY_YIELD",
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            maturity=maturity,
        )

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
