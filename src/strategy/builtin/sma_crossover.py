"""
SMA 交叉策略
经典的双均线交叉策略
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np

from ..base import IndicatorBasedStrategy, Signal, SignalType
from ...core.asset import Asset
from ...core.portfolio import Portfolio
from ...data.models import MarketData


class SMACrossoverStrategy(IndicatorBasedStrategy):
    """
    SMA 交叉策略
    
    当短期均线上穿长期均线时买入，
    当短期均线下穿长期均线时卖出。
    
    参数:
        fast_period: 短期均线周期（默认 10）
        slow_period: 长期均线周期（默认 30）
    """
    
    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化策略
        
        Args:
            fast_period: 短期均线周期
            slow_period: 长期均线周期
            params: 额外参数
        """
        default_params = {
            "fast_period": fast_period,
            "slow_period": slow_period,
        }
        if params:
            default_params.update(params)
        
        super().__init__("SMA_Crossover", default_params)
        
        self.fast_period = default_params["fast_period"]
        self.slow_period = default_params["slow_period"]
        
        # 记录上一次的均线状态，用于检测交叉
        self._last_state: Dict[str, str] = {}  # symbol -> "above" | "below"
    
    @property
    def required_history(self) -> int:
        """需要的最小历史数据量"""
        return self.slow_period + 2
    
    def generate_signals(
        self,
        asset: Asset,
        data: MarketData,
        portfolio: Portfolio,
    ) -> List[Signal]:
        """
        生成交易信号
        
        Args:
            asset: 交易资产
            data: 市场数据
            portfolio: 当前投资组合
            
        Returns:
            信号列表
        """
        if len(data) < self.required_history:
            return []
        
        # 获取收盘价
        closes = np.array([d.close for d in data.data])
        
        # 计算均线
        fast_sma = self._calculate_sma(closes, self.fast_period)
        slow_sma = self._calculate_sma(closes, self.slow_period)
        
        # 对齐长度
        min_len = min(len(fast_sma), len(slow_sma))
        if min_len < 2:
            return []
        
        fast_sma = fast_sma[-min_len:]
        slow_sma = slow_sma[-min_len:]
        
        # 当前和前一时刻的相对位置
        current_diff = fast_sma[-1] - slow_sma[-1]
        prev_diff = fast_sma[-2] - slow_sma[-2]
        
        # 确定当前状态
        current_state = "above" if current_diff > 0 else "below"
        prev_state = self._last_state.get(asset.symbol, current_state)
        
        signals = []
        timestamp = data.latest.timestamp
        
        # 检测交叉
        if prev_state == "below" and current_state == "above":
            # 金叉：短期均线上穿长期均线 -> 买入信号
            signal_strength = min(abs(current_diff) / slow_sma[-1] * 100, 1.0)
            signals.append(Signal(
                signal_type=SignalType.BUY,
                asset=asset,
                timestamp=timestamp,
                strength=signal_strength,
                metadata={
                    "strategy_id": self.name,
                    "reason": "golden_cross",
                    "fast_sma": float(fast_sma[-1]),
                    "slow_sma": float(slow_sma[-1]),
                },
            ))
        
        elif prev_state == "above" and current_state == "below":
            # 死叉：短期均线下穿长期均线 -> 卖出信号
            signal_strength = min(abs(current_diff) / slow_sma[-1] * 100, 1.0)
            signals.append(Signal(
                signal_type=SignalType.SELL,
                asset=asset,
                timestamp=timestamp,
                strength=signal_strength,
                metadata={
                    "strategy_id": self.name,
                    "reason": "death_cross",
                    "fast_sma": float(fast_sma[-1]),
                    "slow_sma": float(slow_sma[-1]),
                },
            ))
        
        # 更新状态
        self._last_state[asset.symbol] = current_state
        
        return signals
    
    def reset(self) -> None:
        """重置策略状态"""
        super().reset()
        self._last_state.clear()
    
    def __repr__(self):
        return f"SMACrossoverStrategy(fast={self.fast_period}, slow={self.slow_period})"
