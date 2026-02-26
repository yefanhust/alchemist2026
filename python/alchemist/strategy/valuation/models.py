"""
估值扫描数据模型
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class StockValuation:
    """单只股票的估值评估结果"""

    symbol: str
    name: str
    sector: str
    industry: str
    current_price: float

    # 综合评分 -1(极度低估) ~ +1(极度高估)
    composite_score: float
    # 字母等级 A(强烈低估) B(低估) C(合理) D(高估) F(强烈高估)
    grade: str

    # 四维度分项分数
    relative_score: float = 0.0     # 相对估值
    absolute_score: float = 0.0     # 绝对估值(DCF)
    sentiment_score: float = 0.0    # 情绪面
    macro_score: float = 0.0        # 宏观环境

    # DCF 三情景内在价值
    dcf_intrinsic_values: Dict[str, Optional[float]] = field(default_factory=lambda: {
        "optimistic": None,
        "neutral": None,
        "pessimistic": None,
    })

    # 安全边际 (intrinsic_value - price) / intrinsic_value
    safety_margin: Optional[float] = None

    # 关键指标原始值
    key_metrics: Dict[str, Any] = field(default_factory=dict)

    scan_date: Optional[datetime] = None
    horizon: str = "3M"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "sector": self.sector,
            "industry": self.industry,
            "current_price": self.current_price,
            "composite_score": round(self.composite_score, 4),
            "grade": self.grade,
            "relative_score": round(self.relative_score, 4),
            "absolute_score": round(self.absolute_score, 4),
            "sentiment_score": round(self.sentiment_score, 4),
            "macro_score": round(self.macro_score, 4),
            "dcf_intrinsic_values": self.dcf_intrinsic_values,
            "safety_margin": round(self.safety_margin, 4) if self.safety_margin is not None else None,
            "key_metrics": self.key_metrics,
            "scan_date": self.scan_date.isoformat() if self.scan_date else None,
            "horizon": self.horizon,
        }


@dataclass
class ScanResult:
    """估值扫描完整结果"""

    scan_date: datetime
    horizon: str
    total_scanned: int

    # 排名列表
    most_undervalued: List[StockValuation] = field(default_factory=list)  # Grade A/B
    most_overvalued: List[StockValuation] = field(default_factory=list)   # Grade D/F

    # 行业聚合 {sector: {"avg_score": x, "count": n, "most_undervalued": sym, ...}}
    sector_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 宏观快照
    macro_context: Dict[str, float] = field(default_factory=dict)

    # 使用的权重
    weights_used: Dict[str, float] = field(default_factory=dict)

    # 扫描耗时
    scan_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scan_date": self.scan_date.isoformat(),
            "horizon": self.horizon,
            "total_scanned": self.total_scanned,
            "most_undervalued": [s.to_dict() for s in self.most_undervalued],
            "most_overvalued": [s.to_dict() for s in self.most_overvalued],
            "sector_summary": self.sector_summary,
            "macro_context": self.macro_context,
            "weights_used": self.weights_used,
            "scan_duration_seconds": round(self.scan_duration_seconds, 2),
        }


# ========== 评分等级 ==========

GRADE_THRESHOLDS = {
    "A": (-1.0, -0.5),    # 强烈低估
    "B": (-0.5, -0.15),   # 低估
    "C": (-0.15, 0.15),   # 合理
    "D": (0.15, 0.5),     # 高估
    "F": (0.5, 1.0),      # 强烈高估
}


def score_to_grade(score: float) -> str:
    """将综合分数映射为字母等级"""
    score = max(-1.0, min(1.0, score))
    for grade, (low, high) in GRADE_THRESHOLDS.items():
        if low <= score < high:
            return grade
    return "F" if score >= 0.5 else "A"


# ========== 投资时间窗口 ==========
# 前瞻性回归窗口：预期标的在此期间内回归合理估值
# 同时用于 momentum/volatility 的历史回看周期

HORIZON_DAYS = {
    "1M": 21,       # 1个月 ≈ 21个交易日
    "3M": 63,       # 3个月
    "6M": 126,      # 半年
    "1Y": 252,      # 1年
}


def get_horizon_days(horizon: str) -> int:
    """获取投资时间窗口对应的交易日数"""
    return HORIZON_DAYS.get(horizon.upper(), 63)


# 向后兼容别名
get_lookback_days = get_horizon_days


# ========== 时间窗口对应的因子权重 ==========
# 短期窗口：情绪/技术面权重高；长期窗口：基本面(DCF)权重高

HORIZON_WEIGHTS = {
    "1M": {"relative": 0.30, "absolute": 0.15, "sentiment": 0.35, "macro": 0.20},
    "3M": {"relative": 0.30, "absolute": 0.25, "sentiment": 0.25, "macro": 0.20},
    "6M": {"relative": 0.25, "absolute": 0.30, "sentiment": 0.20, "macro": 0.25},
    "1Y": {"relative": 0.25, "absolute": 0.35, "sentiment": 0.15, "macro": 0.25},
}
