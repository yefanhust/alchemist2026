"""
均值回归策略
基于价格偏离均值的程度进行交易
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np

from strategy.base import IndicatorBasedStrategy, Signal, SignalType
from core.asset import Asset
from core.portfolio import Portfolio
from data.models import MarketData


class MeanReversionStrategy(IndicatorBasedStrategy):
    """
    均值回归策略
    
    假设价格会回归均值。当价格偏离均值超过一定阈值时，
    预期会回归，因此反向交易。
    
    使用布林带来衡量偏离程度：
    - 价格触及下轨时买入（预期上涨）
    - 价格触及上轨时卖出（预期下跌）
    - 价格回归中轨时平仓
    
    参数:
        period: 均值计算周期（默认 20）
        std_multiplier: 标准差倍数（默认 2.0）
        entry_threshold: 入场阈值（触及带宽的百分比，默认 0.95）
        exit_threshold: 出场阈值（回归至带宽的百分比，默认 0.3）
    """
    
    def __init__(
        self,
        period: int = 20,
        std_multiplier: float = 2.0,
        entry_threshold: float = 0.95,
        exit_threshold: float = 0.3,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化策略
        
        Args:
            period: 均值计算周期
            std_multiplier: 标准差倍数
            entry_threshold: 入场阈值
            exit_threshold: 出场阈值
            params: 额外参数
        """
        default_params = {
            "period": period,
            "std_multiplier": std_multiplier,
            "entry_threshold": entry_threshold,
            "exit_threshold": exit_threshold,
        }
        if params:
            default_params.update(params)
        
        super().__init__("Mean_Reversion", default_params)
        
        self.period = default_params["period"]
        self.std_multiplier = default_params["std_multiplier"]
        self.entry_threshold = default_params["entry_threshold"]
        self.exit_threshold = default_params["exit_threshold"]
        
        # 记录持仓状态
        self._positions: Dict[str, str] = {}  # symbol -> "long" | "short" | None
    
    @property
    def required_history(self) -> int:
        """需要的最小历史数据量"""
        return self.period + 1
    
    def _calculate_bollinger_bands(
        self,
        data: np.ndarray,
    ) -> tuple:
        """
        计算布林带
        
        Args:
            data: 价格数据
            
        Returns:
            (upper, middle, lower) 元组
        """
        # 计算移动平均
        sma = self._calculate_sma(data, self.period)
        
        if len(sma) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # 计算标准差
        std = np.zeros(len(sma))
        for i in range(len(sma)):
            window_start = len(data) - len(sma) + i - self.period + 1
            window_end = len(data) - len(sma) + i + 1
            std[i] = np.std(data[window_start:window_end])
        
        upper = sma + self.std_multiplier * std
        lower = sma - self.std_multiplier * std
        
        return upper, sma, lower
    
    def _calculate_position_in_band(
        self,
        price: float,
        upper: float,
        middle: float,
        lower: float,
    ) -> float:
        """
        计算价格在布林带中的相对位置
        
        Returns:
            位置值：-1（下轨）到 1（上轨），0 为中轨
        """
        band_width = upper - lower
        if band_width == 0:
            return 0
        
        # 归一化到 [-1, 1]
        position = 2 * (price - lower) / band_width - 1
        return np.clip(position, -1, 1)
    
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
        current_price = closes[-1]
        
        # 计算布林带
        upper, middle, lower = self._calculate_bollinger_bands(closes)
        
        if len(upper) == 0:
            return []
        
        # 获取最新的布林带值
        current_upper = upper[-1]
        current_middle = middle[-1]
        current_lower = lower[-1]
        
        # 计算当前价格在布林带中的位置
        band_position = self._calculate_position_in_band(
            current_price, current_upper, current_middle, current_lower
        )
        
        signals = []
        timestamp = data.latest.timestamp
        current_position = self._positions.get(asset.symbol)
        
        # 检查入场条件
        if current_position is None:
            if band_position <= -self.entry_threshold:
                # 价格触及下轨，超卖，买入
                signal_strength = abs(band_position)
                signals.append(Signal(
                    signal_type=SignalType.BUY,
                    asset=asset,
                    timestamp=timestamp,
                    strength=signal_strength,
                    target_price=current_middle,  # 目标回归到中轨
                    stop_loss=current_lower * 0.98,  # 止损在下轨下方
                    metadata={
                        "strategy_id": self.name,
                        "reason": "oversold",
                        "band_position": float(band_position),
                        "upper": float(current_upper),
                        "middle": float(current_middle),
                        "lower": float(current_lower),
                    },
                ))
                self._positions[asset.symbol] = "long"
            
            elif band_position >= self.entry_threshold:
                # 价格触及上轨，超买，卖出
                signal_strength = abs(band_position)
                signals.append(Signal(
                    signal_type=SignalType.SELL,
                    asset=asset,
                    timestamp=timestamp,
                    strength=signal_strength,
                    target_price=current_middle,
                    stop_loss=current_upper * 1.02,
                    metadata={
                        "strategy_id": self.name,
                        "reason": "overbought",
                        "band_position": float(band_position),
                        "upper": float(current_upper),
                        "middle": float(current_middle),
                        "lower": float(current_lower),
                    },
                ))
                self._positions[asset.symbol] = "short"
        
        # 检查出场条件
        elif current_position == "long":
            if abs(band_position) <= self.exit_threshold:
                # 价格回归中轨，平多
                signals.append(Signal(
                    signal_type=SignalType.EXIT_LONG,
                    asset=asset,
                    timestamp=timestamp,
                    strength=1.0 - abs(band_position),
                    metadata={
                        "strategy_id": self.name,
                        "reason": "mean_reversion",
                        "band_position": float(band_position),
                    },
                ))
                self._positions[asset.symbol] = None
            
            elif band_position >= self.entry_threshold:
                # 反转信号：从多头转为空头
                signals.append(Signal(
                    signal_type=SignalType.EXIT_LONG,
                    asset=asset,
                    timestamp=timestamp,
                    strength=1.0,
                    metadata={
                        "strategy_id": self.name,
                        "reason": "reversal",
                    },
                ))
                signals.append(Signal(
                    signal_type=SignalType.SELL,
                    asset=asset,
                    timestamp=timestamp,
                    strength=abs(band_position),
                    metadata={
                        "strategy_id": self.name,
                        "reason": "overbought",
                    },
                ))
                self._positions[asset.symbol] = "short"
        
        elif current_position == "short":
            if abs(band_position) <= self.exit_threshold:
                # 价格回归中轨，平空
                signals.append(Signal(
                    signal_type=SignalType.EXIT_SHORT,
                    asset=asset,
                    timestamp=timestamp,
                    strength=1.0 - abs(band_position),
                    metadata={
                        "strategy_id": self.name,
                        "reason": "mean_reversion",
                        "band_position": float(band_position),
                    },
                ))
                self._positions[asset.symbol] = None
            
            elif band_position <= -self.entry_threshold:
                # 反转信号：从空头转为多头
                signals.append(Signal(
                    signal_type=SignalType.EXIT_SHORT,
                    asset=asset,
                    timestamp=timestamp,
                    strength=1.0,
                    metadata={
                        "strategy_id": self.name,
                        "reason": "reversal",
                    },
                ))
                signals.append(Signal(
                    signal_type=SignalType.BUY,
                    asset=asset,
                    timestamp=timestamp,
                    strength=abs(band_position),
                    metadata={
                        "strategy_id": self.name,
                        "reason": "oversold",
                    },
                ))
                self._positions[asset.symbol] = "long"
        
        return signals
    
    def reset(self) -> None:
        """重置策略状态"""
        super().reset()
        self._positions.clear()
    
    def __repr__(self):
        return (
            f"MeanReversionStrategy(period={self.period}, "
            f"std={self.std_multiplier}, entry={self.entry_threshold})"
        )
