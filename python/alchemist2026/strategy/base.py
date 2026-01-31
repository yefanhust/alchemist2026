"""
策略基类模块
定义交易策略的统一接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

from core.asset import Asset
from core.portfolio import Portfolio
from core.order import Order, OrderType, OrderSide
from data.models import MarketData


class SignalType(Enum):
    """信号类型"""
    BUY = "buy"           # 买入信号
    SELL = "sell"         # 卖出信号
    HOLD = "hold"         # 持有信号
    EXIT_LONG = "exit_long"   # 平多
    EXIT_SHORT = "exit_short"  # 平空


@dataclass
class Signal:
    """
    交易信号
    
    由策略生成，指示交易方向和强度。
    """
    signal_type: SignalType
    asset: Asset
    timestamp: datetime
    strength: float = 1.0      # 信号强度 [0, 1]
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_entry(self) -> bool:
        """是否为入场信号"""
        return self.signal_type in (SignalType.BUY, SignalType.SELL)
    
    @property
    def is_exit(self) -> bool:
        """是否为出场信号"""
        return self.signal_type in (SignalType.EXIT_LONG, SignalType.EXIT_SHORT)
    
    def to_order(
        self,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
    ) -> Optional[Order]:
        """
        转换为订单
        
        Args:
            quantity: 交易数量
            order_type: 订单类型
            
        Returns:
            Order 对象
        """
        if self.signal_type == SignalType.HOLD:
            return None
        
        if self.signal_type in (SignalType.BUY,):
            side = OrderSide.BUY
        else:
            side = OrderSide.SELL
        
        return Order(
            asset=self.asset,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=self.target_price if order_type == OrderType.LIMIT else None,
            strategy_id=self.metadata.get("strategy_id"),
            metadata={
                "signal_strength": self.strength,
                "stop_loss": self.stop_loss,
                "take_profit": self.take_profit,
            },
        )


class Strategy(ABC):
    """
    策略抽象基类
    
    定义交易策略的统一接口。
    所有策略都应继承此类并实现核心方法。
    """
    
    def __init__(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化策略
        
        Args:
            name: 策略名称
            params: 策略参数
        """
        self.name = name
        self.params = params or {}
        self.signals: List[Signal] = []
        self._initialized = False
    
    @property
    @abstractmethod
    def required_history(self) -> int:
        """
        需要的最小历史数据量
        
        Returns:
            数据点数量
        """
        pass
    
    @abstractmethod
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
        pass
    
    def initialize(self, assets: List[Asset], data: Dict[str, MarketData]) -> None:
        """
        初始化策略
        
        在回测开始前调用，可用于预计算指标等。
        
        Args:
            assets: 资产列表
            data: 历史数据
        """
        self._initialized = True
    
    def on_data(
        self,
        timestamp: datetime,
        data: Dict[str, MarketData],
        portfolio: Portfolio,
    ) -> List[Signal]:
        """
        处理新数据
        
        每个时间步调用，生成交易信号。
        
        Args:
            timestamp: 当前时间戳
            data: 当前市场数据
            portfolio: 当前投资组合
            
        Returns:
            信号列表
        """
        all_signals = []
        
        for symbol, market_data in data.items():
            if len(market_data) < self.required_history:
                continue
            
            asset = Asset(symbol=symbol)
            signals = self.generate_signals(asset, market_data, portfolio)
            all_signals.extend(signals)
        
        self.signals.extend(all_signals)
        return all_signals
    
    def on_order_filled(self, order: Order) -> None:
        """
        订单成交回调
        
        Args:
            order: 成交的订单
        """
        pass
    
    def on_order_rejected(self, order: Order, reason: str) -> None:
        """
        订单拒绝回调
        
        Args:
            order: 被拒绝的订单
            reason: 拒绝原因
        """
        pass
    
    def reset(self) -> None:
        """重置策略状态"""
        self.signals = []
        self._initialized = False
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """获取策略参数"""
        return self.params.get(key, default)
    
    def set_param(self, key: str, value: Any) -> None:
        """设置策略参数"""
        self.params[key] = value
    
    def summary(self) -> Dict[str, Any]:
        """获取策略摘要"""
        signal_counts = {}
        for signal in self.signals:
            st = signal.signal_type.value
            signal_counts[st] = signal_counts.get(st, 0) + 1
        
        return {
            "name": self.name,
            "params": self.params,
            "required_history": self.required_history,
            "total_signals": len(self.signals),
            "signal_counts": signal_counts,
        }
    
    def __repr__(self):
        return f"Strategy({self.name})"


class IndicatorBasedStrategy(Strategy):
    """
    基于指标的策略基类
    
    提供技术指标计算的通用功能。
    """
    
    def __init__(
        self,
        name: str,
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, params)
        self._indicator_cache: Dict[str, np.ndarray] = {}
    
    def _calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        计算简单移动平均
        
        Args:
            data: 价格数据
            period: 周期
            
        Returns:
            SMA 数组
        """
        if len(data) < period:
            return np.array([])
        
        return np.convolve(data, np.ones(period) / period, mode='valid')
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        计算指数移动平均
        
        Args:
            data: 价格数据
            period: 周期
            
        Returns:
            EMA 数组
        """
        if len(data) < period:
            return np.array([])
        
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        
        return ema
    
    def _calculate_rsi(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """
        计算相对强弱指标
        
        Args:
            data: 价格数据
            period: 周期
            
        Returns:
            RSI 数组
        """
        if len(data) < period + 1:
            return np.array([])
        
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.zeros(len(deltas))
        avg_loss = np.zeros(len(deltas))
        
        # 初始平均
        avg_gain[period - 1] = np.mean(gains[:period])
        avg_loss[period - 1] = np.mean(losses[:period])
        
        # 平滑平均
        for i in range(period, len(deltas)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi[period - 1:]
    
    def _calculate_macd(
        self,
        data: np.ndarray,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> tuple:
        """
        计算 MACD
        
        Args:
            data: 价格数据
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
            
        Returns:
            (MACD, Signal, Histogram) 元组
        """
        if len(data) < slow_period:
            return np.array([]), np.array([]), np.array([])
        
        fast_ema = self._calculate_ema(data, fast_period)
        slow_ema = self._calculate_ema(data, slow_period)
        
        # 对齐长度
        macd_line = fast_ema[-len(slow_ema):] - slow_ema
        signal_line = self._calculate_ema(macd_line, signal_period)
        
        # 对齐长度
        min_len = len(signal_line)
        macd_line = macd_line[-min_len:]
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
