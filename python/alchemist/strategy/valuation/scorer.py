"""
综合估值打分器（第四步：整合分析）

将相对估值、绝对估值、情绪面、宏观面四维因子加权综合，
输出 A-F 字母评级和综合分数。
"""

from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

from strategy.valuation.models import HORIZON_WEIGHTS, StockValuation, score_to_grade


class ValuationScorer:
    """
    四维度综合估值评分器

    default_weights:
        relative:  0.30  — 相对估值（PE/PB/PS 等行业对比）
        absolute:  0.25  — 绝对估值（DCF + 剩余收益）
        sentiment: 0.25  — 市场情绪（RSI/空头/内部人）
        macro:     0.20  — 宏观环境（Fed模型/20法则/巴菲特指标）
    """

    DEFAULT_WEIGHTS = {
        "relative": 0.30,
        "absolute": 0.25,
        "sentiment": 0.25,
        "macro": 0.20,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # weights=None → 使用 horizon 动态默认；用户显式指定 → 始终使用自定义
        self.custom_weights = weights

    def get_weights(self, horizon: str = "3M") -> Dict[str, float]:
        """获取实际使用的权重（用户自定义或 horizon 动态默认）"""
        return self.custom_weights or HORIZON_WEIGHTS.get(
            horizon.upper(), HORIZON_WEIGHTS["3M"]
        )

    def score(
        self,
        symbol: str,
        name: str,
        sector: str,
        industry: str,
        current_price: float,
        relative_result: Dict[str, Any],
        absolute_result: Dict[str, Any],
        sentiment_result: Dict[str, Any],
        macro_result: Dict[str, Any],
        horizon: str = "3M",
    ) -> StockValuation:
        """
        计算综合估值评分

        Args:
            symbol: 股票代码
            name: 公司名称
            sector: 行业大类
            industry: 细分行业
            current_price: 当前价格
            relative_result: 相对估值因子结果
            absolute_result: 绝对估值因子结果
            sentiment_result: 情绪因子结果
            macro_result: 宏观因子结果
            horizon: 投资时间窗口 (1M/3M/6M/1Y)

        Returns:
            StockValuation 对象
        """
        # 确定实际权重：用户自定义优先，否则按 horizon 动态选择
        weights = self.custom_weights or HORIZON_WEIGHTS.get(
            horizon.upper(), HORIZON_WEIGHTS["3M"]
        )

        scores = {
            "relative": relative_result.get("relative_score", 0.0),
            "absolute": absolute_result.get("absolute_score", 0.0),
            "sentiment": sentiment_result.get("sentiment_score", 0.0),
            "macro": macro_result.get("macro_score", 0.0),
        }

        # 加权综合
        composite = 0.0
        total_weight = 0.0
        for dim, weight in weights.items():
            if dim in scores:
                composite += scores[dim] * weight
                total_weight += weight

        if total_weight > 0:
            composite /= total_weight

        composite = float(np.clip(composite, -1, 1))
        grade = score_to_grade(composite)

        # DCF 内在价值
        dcf_result = absolute_result.get("dcf_result", {})
        dcf_values = dcf_result.get("intrinsic_values", {})
        safety_margins = dcf_result.get("safety_margins", {})
        safety_margin = safety_margins.get("neutral")

        # 汇集关键指标
        key_metrics = {}
        # 相对估值指标
        rel_details = relative_result.get("details", {})
        for k in ["pe_ratio", "pb_ratio", "ps_ratio", "peg_ratio",
                   "ev_to_ebitda", "pe_industry_median"]:
            if k in rel_details and rel_details[k] is not None:
                key_metrics[k] = rel_details[k]

        # 情绪指标
        sent_details = sentiment_result.get("details", {})
        for k in ["rsi", "short_percent", "short_ratio",
                   "insider_net_direction", "momentum_return"]:
            if k in sent_details and sent_details[k] is not None:
                key_metrics[k] = sent_details[k]

        # 宏观指标
        macro_details = macro_result.get("details", {})
        for k in ["treasury_10y", "vix", "rule_of_20_sum"]:
            if k in macro_details and macro_details[k] is not None:
                key_metrics[k] = macro_details[k]

        # DCF 信息
        if dcf_result.get("valid"):
            key_metrics["fcf_growth_rate"] = dcf_result.get("fcf_growth_rate")
            key_metrics["wacc"] = dcf_result.get("wacc")

        from datetime import datetime

        return StockValuation(
            symbol=symbol,
            name=name or "",
            sector=sector or "",
            industry=industry or "",
            current_price=current_price,
            composite_score=composite,
            grade=grade,
            relative_score=scores["relative"],
            absolute_score=scores["absolute"],
            sentiment_score=scores["sentiment"],
            macro_score=scores["macro"],
            dcf_intrinsic_values={
                "optimistic": dcf_values.get("optimistic"),
                "neutral": dcf_values.get("neutral"),
                "pessimistic": dcf_values.get("pessimistic"),
            },
            safety_margin=safety_margin,
            key_metrics=key_metrics,
            scan_date=datetime.now(),
            horizon=horizon,
        )
