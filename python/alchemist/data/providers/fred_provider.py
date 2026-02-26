"""
FRED (Federal Reserve Economic Data) 数据提供者
获取宏观经济指标用于估值分析
"""

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
from loguru import logger


# FRED 常用宏观序列
FRED_SERIES = {
    "FEDFUNDS": "联邦基金利率",
    "DGS10": "10年期国债收益率",
    "DGS2": "2年期国债收益率",
    "CPIAUCSL": "CPI (城市消费者价格指数)",
    "VIXCLS": "VIX 恐慌指数",
    "BAA10Y": "穆迪Baa企业债与10年国债利差",
    "GDP": "美国GDP",
    "GDPC1": "实际GDP",
    "WILSHIRE5000PRICEINDEX": "Wilshire 5000总市场指数",
    "UNRATE": "失业率",
    "T10YIE": "10年期盈亏平衡通胀率",
}


class FREDProvider:
    """
    FRED 数据提供者

    通过 FRED API 获取美国宏观经济数据。
    用于估值分析中的宏观环境评估（Fed模型、巴菲特指标、20法则等）。

    API 文档: https://fred.stlouisfed.org/docs/api/fred/
    """

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_backend=None,
    ):
        """
        初始化 FRED 提供者

        Args:
            api_key: FRED API key（可通过环境变量 FRED_API_KEY 设置）
            cache_backend: 缓存后端（SQLiteCache）
        """
        self.api_key = api_key or os.getenv("FRED_API_KEY", "")
        self.cache = cache_backend
        self._session: Optional[aiohttp.ClientSession] = None

        if not self.api_key:
            logger.warning(
                "FRED API key 未设置。宏观数据将不可用。"
                "请设置环境变量 FRED_API_KEY 或在 config.yaml 中配置。"
                "免费注册: https://fred.stlouisfed.org/docs/api/api_key.html"
            )

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_series(
        self,
        series_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[pd.Series]:
        """
        获取 FRED 时间序列数据

        Args:
            series_id: FRED 序列 ID（如 "DGS10", "FEDFUNDS"）
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            pandas Series（index=日期, values=数值），失败返回 None
        """
        if not self.api_key:
            logger.warning(f"FRED API key 未设置，无法获取 {series_id}")
            return None

        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=365 * 5))

        # 检查缓存
        cache_key = f"fred:{series_id}:{start_date.strftime('%Y%m%d')}:{end_date.strftime('%Y%m%d')}"
        if self.cache:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"从缓存获取 FRED {series_id}")
                return cached

        # 请求 FRED API
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date.strftime("%Y-%m-%d"),
            "observation_end": end_date.strftime("%Y-%m-%d"),
            "sort_order": "asc",
        }

        try:
            session = await self._get_session()
            async with session.get(self.BASE_URL, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"FRED API 请求失败: {resp.status}")
                    return None

                data = await resp.json()

        except Exception as e:
            logger.error(f"FRED API 网络错误: {e}")
            return None

        observations = data.get("observations", [])
        if not observations:
            logger.warning(f"FRED {series_id} 无数据")
            return None

        # 构建 Series
        dates = []
        values = []
        for obs in observations:
            date_str = obs.get("date", "")
            value_str = obs.get("value", "")
            if value_str == "." or not value_str:
                continue
            try:
                dates.append(pd.Timestamp(date_str))
                values.append(float(value_str))
            except (ValueError, TypeError):
                continue

        if not dates:
            return None

        series = pd.Series(values, index=pd.DatetimeIndex(dates), name=series_id)

        # 缓存（TTL=1天）
        if self.cache:
            await self.cache.set(cache_key, series, ttl=timedelta(days=1))

        logger.info(f"FRED {series_id}: {len(series)} 条数据 ({series.index[0].date()} ~ {series.index[-1].date()})")
        return series

    async def get_multiple_series(
        self,
        series_ids: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, pd.Series]:
        """
        批量获取多个 FRED 序列

        Returns:
            {series_id: pd.Series}
        """
        results = {}
        for sid in series_ids:
            series = await self.get_series(sid, start_date, end_date)
            if series is not None:
                results[sid] = series
        return results

    async def get_latest_value(self, series_id: str) -> Optional[float]:
        """
        获取序列最新值

        Args:
            series_id: FRED 序列 ID

        Returns:
            最新数值
        """
        end = datetime.now()
        start = end - timedelta(days=30)
        series = await self.get_series(series_id, start, end)
        if series is not None and len(series) > 0:
            return float(series.iloc[-1])
        return None

    async def get_macro_snapshot(self) -> Dict[str, Any]:
        """
        获取宏观经济快照（用于估值分析）

        Returns:
            宏观指标字典
        """
        snapshot = {}
        key_series = ["DGS10", "DGS2", "FEDFUNDS", "VIXCLS", "BAA10Y", "T10YIE"]

        for sid in key_series:
            value = await self.get_latest_value(sid)
            if value is not None:
                snapshot[sid] = value

        # 派生指标
        dgs10 = snapshot.get("DGS10")
        dgs2 = snapshot.get("DGS2")
        if dgs10 is not None and dgs2 is not None:
            snapshot["yield_curve_spread"] = dgs10 - dgs2  # 收益率曲线斜率

        return snapshot
