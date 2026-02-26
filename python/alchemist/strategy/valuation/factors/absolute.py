"""
绝对估值因子（第二步：深度剖析）

DCF (现金流折现) + 剩余收益模型
计算内在价值，与当前股价对比得出安全边际。
"""

from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger


def _safe_float(val) -> Optional[float]:
    """安全转换为 float"""
    if val is None:
        return None
    try:
        if isinstance(val, str):
            if val in ("None", "-", "", "null"):
                return None
            return float(val)
        return float(val)
    except (ValueError, TypeError):
        return None


class DCFModel:
    """
    现金流折现模型

    基于公司历史自由现金流，预测未来FCF并折现，计算内在价值。
    支持乐观/中性/悲观三种情景。
    """

    def __init__(
        self,
        projection_years: int = 5,
        terminal_growth_rate: float = 0.025,
        equity_risk_premium: float = 0.05,
    ):
        self.projection_years = projection_years
        self.terminal_growth_rate = terminal_growth_rate
        self.equity_risk_premium = equity_risk_premium

    def calculate_intrinsic_value(
        self,
        cashflow_reports: List[Dict[str, Any]],
        balance_reports: List[Dict[str, Any]],
        stock_info: Dict[str, Any],
        risk_free_rate: float = 0.04,
        scenarios: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        计算 DCF 内在价值

        Args:
            cashflow_reports: 年度现金流量表列表（按年份降序）
            balance_reports: 年度资产负债表列表
            stock_info: 股票基本面信息
            risk_free_rate: 无风险利率（默认4%，应从FRED获取DGS10）
            scenarios: 情景列表 ["optimistic", "neutral", "pessimistic"]

        Returns:
            {
                "intrinsic_values": {scenario: per_share_value},
                "safety_margins": {scenario: margin},
                "fcf_history": [...],
                "wacc": float,
                ...
            }
        """
        if scenarios is None:
            scenarios = ["optimistic", "neutral", "pessimistic"]

        result = {
            "intrinsic_values": {},
            "safety_margins": {},
            "fcf_history": [],
            "fcf_growth_rate": None,
            "wacc": None,
            "current_price": None,
            "valid": False,
        }

        # 1. 提取历史 FCF
        fcf_list = self._extract_fcf(cashflow_reports)
        result["fcf_history"] = fcf_list

        if len(fcf_list) < 2:
            logger.debug("FCF 历史数据不足，无法计算 DCF")
            return result

        # 过滤正 FCF
        positive_fcf = [f for f in fcf_list if f > 0]
        if len(positive_fcf) < 2:
            logger.debug("正 FCF 不足，公司现金流不佳")
            return result

        # 2. 计算 FCF 增长率
        growth_rates = []
        for i in range(1, len(fcf_list)):
            if fcf_list[i - 1] > 0 and fcf_list[i] > 0:
                g = (fcf_list[i] - fcf_list[i - 1]) / fcf_list[i - 1]
                if -1.0 < g < 5.0:  # 排除极端值
                    growth_rates.append(g)

        if not growth_rates:
            return result

        base_growth = float(np.median(growth_rates))
        # 限制增长率范围
        base_growth = max(-0.2, min(0.5, base_growth))
        result["fcf_growth_rate"] = base_growth

        # 3. 计算 WACC
        beta = stock_info.get("beta") or 1.0
        wacc = risk_free_rate + float(beta) * self.equity_risk_premium
        wacc = max(0.06, min(0.20, wacc))  # 限制范围 6%-20%
        result["wacc"] = wacc

        # 4. 获取流通股数和当前价格
        shares = _safe_float(stock_info.get("shares_outstanding"))
        pe = _safe_float(stock_info.get("pe_ratio"))
        eps = _safe_float(stock_info.get("eps"))

        if shares is None or shares <= 0:
            return result

        current_price = None
        if pe and eps and pe > 0 and eps > 0:
            current_price = pe * eps
        result["current_price"] = current_price

        # 5. 最近一年的 FCF 作为基础
        latest_fcf = fcf_list[-1] if fcf_list[-1] > 0 else positive_fcf[-1]

        # 6. 三种情景
        scenario_multipliers = {
            "optimistic": 1.2,
            "neutral": 1.0,
            "pessimistic": 0.5,
        }

        for scenario in scenarios:
            mult = scenario_multipliers.get(scenario, 1.0)
            scenario_growth = base_growth * mult
            # 悲观情景下增长率可能为负
            scenario_growth = max(-0.1, scenario_growth)

            intrinsic = self._dcf_valuation(
                latest_fcf, scenario_growth, wacc, shares
            )
            result["intrinsic_values"][scenario] = round(intrinsic, 2)

            if current_price and current_price > 0:
                margin = (intrinsic - current_price) / intrinsic
                result["safety_margins"][scenario] = round(margin, 4)

        result["valid"] = True
        return result

    def _extract_fcf(self, cashflow_reports: List[Dict[str, Any]]) -> List[float]:
        """从现金流量表提取 FCF = operatingCashflow - capitalExpenditures"""
        fcf_list = []
        for report in cashflow_reports:
            ocf = _safe_float(report.get("operatingCashflow"))
            capex = _safe_float(report.get("capitalExpenditures"))

            if ocf is not None and capex is not None:
                fcf = ocf - abs(capex)
                fcf_list.append(fcf)

        # 按时间正序（oldest first）
        fcf_list.reverse()
        return fcf_list

    def _dcf_valuation(
        self,
        latest_fcf: float,
        growth_rate: float,
        wacc: float,
        shares: float,
    ) -> float:
        """计算 DCF 内在价值每股"""
        if wacc <= self.terminal_growth_rate:
            return 0.0

        # 预测未来 FCF
        pv_fcf = 0.0
        projected_fcf = latest_fcf
        for year in range(1, self.projection_years + 1):
            projected_fcf *= (1 + growth_rate)
            pv_fcf += projected_fcf / ((1 + wacc) ** year)

        # 终值
        terminal_value = (
            projected_fcf * (1 + self.terminal_growth_rate)
            / (wacc - self.terminal_growth_rate)
        )
        pv_terminal = terminal_value / ((1 + wacc) ** self.projection_years)

        total_value = pv_fcf + pv_terminal
        per_share = total_value / shares if shares > 0 else 0

        return max(0, per_share)


class ResidualIncomeModel:
    """
    剩余收益模型

    公司价值 = 账面价值 + Σ(剩余收益 / (1+r)^t)
    剩余收益 = (ROE - 权益成本率) × 账面价值
    适用于分红不稳定或 FCF 波动大的公司。
    """

    def __init__(self, projection_years: int = 5):
        self.projection_years = projection_years

    def calculate(
        self,
        stock_info: Dict[str, Any],
        income_reports: List[Dict[str, Any]],
        balance_reports: List[Dict[str, Any]],
        risk_free_rate: float = 0.04,
        equity_risk_premium: float = 0.05,
    ) -> Dict[str, Any]:
        """
        计算剩余收益模型内在价值

        Returns:
            {"intrinsic_value": float, "safety_margin": float, "valid": bool}
        """
        result = {"intrinsic_value": None, "safety_margin": None, "valid": False}

        book_value = _safe_float(stock_info.get("book_value"))
        roe = _safe_float(stock_info.get("return_on_equity"))
        shares = _safe_float(stock_info.get("shares_outstanding"))
        beta = _safe_float(stock_info.get("beta")) or 1.0

        if book_value is None or roe is None or shares is None:
            return result
        if book_value <= 0 or shares <= 0:
            return result

        # 权益成本率
        cost_of_equity = risk_free_rate + beta * equity_risk_premium
        cost_of_equity = max(0.06, min(0.20, cost_of_equity))

        # 总账面价值
        total_book = book_value * shares

        # 剩余收益折现
        residual_pv = 0.0
        current_book = total_book
        for year in range(1, self.projection_years + 1):
            residual_income = (roe - cost_of_equity) * current_book
            residual_pv += residual_income / ((1 + cost_of_equity) ** year)
            current_book *= (1 + roe)  # 假设利润再投资

        intrinsic_total = total_book + residual_pv
        intrinsic_per_share = intrinsic_total / shares

        result["intrinsic_value"] = round(max(0, intrinsic_per_share), 2)

        # 安全边际
        pe = _safe_float(stock_info.get("pe_ratio"))
        eps = _safe_float(stock_info.get("eps"))
        if pe and eps and pe > 0 and eps > 0:
            current_price = pe * eps
            if intrinsic_per_share > 0:
                result["safety_margin"] = round(
                    (intrinsic_per_share - current_price) / intrinsic_per_share, 4
                )

        result["valid"] = True
        return result


class AbsoluteValuationFactors:
    """绝对估值因子计算器"""

    def __init__(self, dcf_config: Optional[Dict[str, Any]] = None):
        config = dcf_config or {}
        self.dcf_model = DCFModel(
            projection_years=config.get("projection_years", 5),
            terminal_growth_rate=config.get("terminal_growth_rate", 0.025),
            equity_risk_premium=config.get("equity_risk_premium", 0.05),
        )
        self.ri_model = ResidualIncomeModel()

    def calculate(
        self,
        stock_info: Dict[str, Any],
        cashflow_reports: List[Dict[str, Any]],
        balance_reports: List[Dict[str, Any]],
        income_reports: List[Dict[str, Any]],
        risk_free_rate: float = 0.04,
    ) -> Dict[str, Any]:
        """
        计算绝对估值因子

        Returns:
            {
                "absolute_score": float,  # -1(低估) ~ +1(高估)
                "dcf_result": {...},
                "ri_result": {...},
            }
        """
        # DCF
        dcf_result = self.dcf_model.calculate_intrinsic_value(
            cashflow_reports=cashflow_reports,
            balance_reports=balance_reports,
            stock_info=stock_info,
            risk_free_rate=risk_free_rate,
        )

        # 剩余收益
        ri_result = self.ri_model.calculate(
            stock_info=stock_info,
            income_reports=income_reports,
            balance_reports=balance_reports,
            risk_free_rate=risk_free_rate,
        )

        # 综合安全边际 → absolute_score
        absolute_score = self._compute_score(dcf_result, ri_result)

        return {
            "absolute_score": absolute_score,
            "dcf_result": dcf_result,
            "ri_result": ri_result,
        }

    def _compute_score(
        self,
        dcf_result: Dict[str, Any],
        ri_result: Dict[str, Any],
    ) -> float:
        """
        根据安全边际计算绝对估值分数

        安全边际 > 50% → -0.8 (强烈低估)
        安全边际 > 20% → -0.4 (低估)
        安全边际 ~0%  →  0.0 (合理)
        安全边际 < -20% → +0.4 (高估)
        安全边际 < -50% → +0.8 (强烈高估)
        """
        margins = []

        # DCF 中性情景的安全边际
        if dcf_result.get("valid"):
            neutral_margin = dcf_result.get("safety_margins", {}).get("neutral")
            if neutral_margin is not None:
                margins.append(neutral_margin)

        # 剩余收益模型安全边际
        if ri_result.get("valid"):
            ri_margin = ri_result.get("safety_margin")
            if ri_margin is not None:
                margins.append(ri_margin)

        if not margins:
            return 0.0

        avg_margin = float(np.mean(margins))

        # 安全边际为正 → 低估 (负分)，为负 → 高估 (正分)
        score = -float(np.tanh(avg_margin * 2))
        return float(np.clip(score, -1, 1))
