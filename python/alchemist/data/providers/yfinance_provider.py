"""
yfinance 数据提供者（情绪数据专用）
获取 Alpha Vantage 不提供的 Short Interest 和 Insider Trading 数据
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger


class YFinanceSentimentProvider:
    """
    yfinance 情绪数据提供者

    仅用于获取 Alpha Vantage 不提供的情绪数据：
    - Short Interest（空头持仓）
    - Insider Trading（内部人交易）

    yfinance 无需 API key，无速率限制。
    """

    def __init__(self):
        try:
            import yfinance  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False
            logger.warning("yfinance 未安装，情绪数据（空头持仓/内部人交易）不可用。"
                           "请运行: pip install yfinance")

    @property
    def is_available(self) -> bool:
        return self._available

    def get_short_interest(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取空头持仓数据

        Args:
            symbol: 股票代码

        Returns:
            空头持仓信息字典
        """
        if not self._available:
            return None

        import yfinance as yf

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}

            short_percent = info.get("shortPercentOfFloat")
            short_ratio = info.get("shortRatio")
            shares_short = info.get("sharesShort")
            shares_short_prior = info.get("sharesShortPriorMonth")
            float_shares = info.get("floatShares")

            if short_percent is None and short_ratio is None:
                return None

            return {
                "symbol": symbol.upper(),
                "short_percent_of_float": short_percent,
                "short_ratio": short_ratio,  # 空头回补天数
                "shares_short": shares_short,
                "shares_short_prior_month": shares_short_prior,
                "float_shares": float_shares,
                "fetch_date": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.debug(f"获取 {symbol} 空头持仓失败: {e}")
            return None

    def get_insider_transactions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取内部人交易数据

        Args:
            symbol: 股票代码

        Returns:
            内部人交易汇总
        """
        if not self._available:
            return None

        import yfinance as yf

        try:
            ticker = yf.Ticker(symbol)

            # 获取内部人交易记录
            insider_df = ticker.insider_transactions
            if insider_df is None or insider_df.empty:
                return {
                    "symbol": symbol.upper(),
                    "total_transactions": 0,
                    "net_buy_count": 0,
                    "net_sell_count": 0,
                    "net_direction": 0.0,
                    "transactions": [],
                    "fetch_date": datetime.now().isoformat(),
                }

            buy_count = 0
            sell_count = 0
            buy_value = 0.0
            sell_value = 0.0

            transactions = []
            for _, row in insider_df.head(20).iterrows():
                text = str(row.get("Text", "")).lower()
                shares = row.get("Shares", 0) or 0
                value = row.get("Value", 0) or 0

                if "purchase" in text or "buy" in text:
                    buy_count += 1
                    buy_value += abs(float(value))
                elif "sale" in text or "sell" in text:
                    sell_count += 1
                    sell_value += abs(float(value))

                transactions.append({
                    "insider": str(row.get("Insider", "")),
                    "relation": str(row.get("Relation", "")),
                    "date": str(row.get("Start Date", "")),
                    "text": str(row.get("Text", "")),
                    "shares": int(shares) if shares else 0,
                    "value": float(value) if value else 0,
                })

            total = buy_count + sell_count
            net_direction = 0.0
            if total > 0:
                # +1 = 全部买入(看多), -1 = 全部卖出(看空)
                net_direction = (buy_count - sell_count) / total

            return {
                "symbol": symbol.upper(),
                "total_transactions": total,
                "net_buy_count": buy_count,
                "net_sell_count": sell_count,
                "buy_value": buy_value,
                "sell_value": sell_value,
                "net_direction": net_direction,
                "transactions": transactions[:10],
                "fetch_date": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.debug(f"获取 {symbol} 内部人交易失败: {e}")
            return None

    def get_batch_sentiment_data(
        self,
        symbols: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        批量获取情绪数据

        Args:
            symbols: 股票代码列表

        Returns:
            {symbol: {"short_interest": {...}, "insider": {...}}}
        """
        results = {}
        for symbol in symbols:
            short_data = self.get_short_interest(symbol)
            insider_data = self.get_insider_transactions(symbol)

            if short_data or insider_data:
                results[symbol.upper()] = {
                    "short_interest": short_data,
                    "insider": insider_data,
                }

        logger.info(f"批量获取情绪数据完成: {len(results)}/{len(symbols)}")
        return results
