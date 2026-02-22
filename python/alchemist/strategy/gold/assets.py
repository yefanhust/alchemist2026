"""
黄金相关资产配置
定义黄金投资策略涉及的各类资产
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class GoldAssets:
    """
    黄金相关资产配置

    包含各类黄金相关投资标的的定义。
    """

    # 美股黄金 ETF
    gold_etf: List[str] = field(default_factory=lambda: [
        "GLD",   # SPDR Gold Shares
        "IAU",   # iShares Gold Trust
        "GDX",   # VanEck Gold Miners ETF
        "GDXJ",  # VanEck Junior Gold Miners ETF
    ])

    # 黄金矿业股
    gold_miners: List[str] = field(default_factory=lambda: [
        "NEM",   # Newmont Corporation
        "GOLD",  # Barrick Gold
        "AEM",   # Agnico Eagle Mines
        "KL",    # Kirkland Lake Gold
    ])

    # 亚洲黄金 ETF
    asian_gold: List[str] = field(default_factory=lambda: [
        "518880.SH",  # 华安黄金 ETF
        "159937.SZ",  # 博时黄金 ETF
    ])

    # 外汇黄金交易对
    forex_gold_pairs: List[str] = field(default_factory=lambda: [
        "XAU/USD",  # 黄金/美元
        "XAU/EUR",  # 黄金/欧元
        "XAU/JPY",  # 黄金/日元
    ])

    @property
    def all_symbols(self) -> List[str]:
        """获取所有资产符号"""
        return (
            self.gold_etf +
            self.gold_miners +
            self.asian_gold +
            self.forex_gold_pairs
        )

    @property
    def primary_etf(self) -> str:
        """主要黄金 ETF（用于基准计算）"""
        return self.gold_etf[0] if self.gold_etf else "GLD"

    def get_category(self, symbol: str) -> str:
        """
        获取资产类别

        Args:
            symbol: 资产符号

        Returns:
            类别名称
        """
        if symbol in self.gold_etf:
            return "gold_etf"
        elif symbol in self.gold_miners:
            return "gold_miners"
        elif symbol in self.asian_gold:
            return "asian_gold"
        elif symbol in self.forex_gold_pairs:
            return "forex_gold_pairs"
        return "unknown"


# 默认资产配置实例
DEFAULT_GOLD_ASSETS = GoldAssets()


# 跨市场关联资产 — 可通过 Alpha Vantage 股票 API 获取的 ETF
CROSS_MARKET_ETF_SYMBOLS = {
    "gold_miners": "GDX",              # 黄金矿业 ETF（情绪因子）
    "sp500": "SPY",                    # 标普500 ETF（跨市场 + 宏观）
    "usd_index": "UUP",               # 美元指数基金（跨市场 + 宏观）
    "inflation_expectations": "TIP",   # 通胀保值债券 ETF（宏观）
    "treasury": "TLT",                 # 20+ 年期国债 ETF（宏观风险偏好）
    "vix": "VIXY",                     # VIX 短期期货 ETF（宏观风险偏好）
}

# 需要通过 Alpha Vantage 特殊端点获取的数据
CROSS_MARKET_SPECIAL_DATA = {
    "treasury_yield": {"type": "treasury_yield", "maturity": "10year"},
    "eur_usd": {"type": "forex", "from": "EUR", "to": "USD"},
    "usd_jpy": {"type": "forex", "from": "USD", "to": "JPY"},
}

# 已确认无法通过 Alpha Vantage 获取的数据（不再引用）:
# - shanghai_gold (AU9999.SGE) — 中国交易所
# - tokyo_gold (TOCOM:GOLD) — 日本交易所
# - gold_etf_aum — 无 API 提供
# - bitcoin (BTC-USD) — 优先级低，暂不实现
