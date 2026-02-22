"""
模拟引擎模块
事件驱动的交易模拟引擎
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
import asyncio
from loguru import logger

from core.asset import Asset
from core.portfolio import Portfolio
from core.order import Order, OrderType
from data.models import MarketData, OHLCV
from strategy.base import Strategy, Signal, SignalType
from simulation.broker import VirtualBroker, BrokerConfig


class EventType(Enum):
    """事件类型"""
    MARKET_DATA = "market_data"      # 市场数据更新
    SIGNAL = "signal"                # 交易信号
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_REJECTED = "order_rejected"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    DAY_END = "day_end"


@dataclass
class Event:
    """事件对象"""
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)


class SimulationEngine:
    """
    模拟引擎
    
    事件驱动的交易模拟引擎，协调数据、策略、订单和组合。
    
    工作流程：
    1. 接收市场数据
    2. 通知策略生成信号
    3. 将信号转换为订单
    4. 通过券商执行订单
    5. 更新组合状态
    6. 触发相关回调
    """
    
    def __init__(
        self,
        portfolio: Portfolio,
        broker: Optional[VirtualBroker] = None,
        strategies: Optional[List[Strategy]] = None,
        position_sizer: Optional[Callable[[Signal, Portfolio, float], float]] = None,
    ):
        """
        初始化模拟引擎
        
        Args:
            portfolio: 投资组合
            broker: 虚拟券商
            strategies: 策略列表
            position_sizer: 仓位计算函数
        """
        self.portfolio = portfolio
        self.broker = broker or VirtualBroker()
        self.strategies = strategies or []
        self.position_sizer = position_sizer or self._default_position_sizer
        
        # 事件队列
        self.event_queue: List[Event] = []
        
        # 事件监听器
        self.event_listeners: Dict[EventType, List[Callable]] = {
            event_type: [] for event_type in EventType
        }
        
        # 市场数据缓存
        self.market_data: Dict[str, MarketData] = {}
        self.current_prices: Dict[str, float] = {}
        
        # 运行状态
        self.is_running = False
        self.current_timestamp: Optional[datetime] = None
        
        # 设置券商回调
        self.broker.on_fill = self._on_order_filled
        self.broker.on_reject = self._on_order_rejected
    
    def add_strategy(self, strategy: Strategy) -> None:
        """添加策略"""
        self.strategies.append(strategy)
    
    def remove_strategy(self, strategy: Strategy) -> None:
        """移除策略"""
        if strategy in self.strategies:
            self.strategies.remove(strategy)
    
    def add_listener(
        self,
        event_type: EventType,
        callback: Callable[[Event], None],
    ) -> None:
        """添加事件监听器"""
        self.event_listeners[event_type].append(callback)
    
    def remove_listener(
        self,
        event_type: EventType,
        callback: Callable[[Event], None],
    ) -> None:
        """移除事件监听器"""
        if callback in self.event_listeners[event_type]:
            self.event_listeners[event_type].remove(callback)
    
    def _emit_event(self, event: Event) -> None:
        """触发事件"""
        self.event_queue.append(event)
        
        for callback in self.event_listeners[event.event_type]:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"事件处理错误: {e}")
    
    def _default_position_sizer(
        self,
        signal: Signal,
        portfolio: Portfolio,
        current_price: float,
    ) -> float:
        """
        默认仓位计算：使用净值的10%

        Args:
            signal: 交易信号
            portfolio: 投资组合
            current_price: 当前价格

        Returns:
            交易数量
        """
        # 使用净值而非现金，避免做空时仓位无限膨胀
        net_value = portfolio.total_value(self.current_prices)
        # 确保净值为正，避免负净值时开仓
        if net_value <= 0:
            return 0

        # 使用初始资本和当前净值的较小值，防止杠杆过大
        base_value = min(net_value, portfolio.initial_capital)
        position_value = base_value * 0.1 * signal.strength
        quantity = position_value / current_price

        # 四舍五入到整数
        return max(1, int(quantity))
    
    def on_market_data(
        self,
        symbol: str,
        ohlcv: OHLCV,
    ) -> None:
        """
        处理市场数据更新

        Args:
            symbol: 资产代码
            ohlcv: OHLCV 数据
        """
        self.current_timestamp = ohlcv.timestamp
        self.current_prices[symbol] = ohlcv.close

        # 更新市场数据缓存
        if symbol not in self.market_data:
            self.market_data[symbol] = MarketData(symbol=symbol)
        self.market_data[symbol].append(ohlcv)

        # 触发事件
        self._emit_event(Event(
            event_type=EventType.MARKET_DATA,
            timestamp=ohlcv.timestamp,
            data={"symbol": symbol, "ohlcv": ohlcv},
        ))

        # 只处理与当前 symbol 匹配的待处理订单，
        # 避免订单被错误地用其他 symbol 的价格成交
        self.broker.process_orders(self.portfolio, ohlcv, symbol=symbol)
    
    def process_strategies(self) -> List[Signal]:
        """
        运行所有策略，收集信号
        
        Returns:
            信号列表
        """
        all_signals = []
        
        for strategy in self.strategies:
            signals = strategy.on_data(
                self.current_timestamp,
                self.market_data,
                self.portfolio,
            )
            all_signals.extend(signals)
        
        return all_signals
    
    def process_signals(self, signals: List[Signal]) -> List[Order]:
        """
        处理交易信号，生成订单
        
        Args:
            signals: 信号列表
            
        Returns:
            订单列表
        """
        orders = []
        
        for signal in signals:
            if signal.signal_type == SignalType.HOLD:
                continue
            
            current_price = self.current_prices.get(signal.asset.symbol)
            if current_price is None:
                logger.warning(f"无法获取 {signal.asset.symbol} 的价格")
                continue
            
            # 触发信号事件
            self._emit_event(Event(
                event_type=EventType.SIGNAL,
                timestamp=signal.timestamp,
                data={"signal": signal},
            ))
            
            # 计算仓位
            quantity = self.position_sizer(signal, self.portfolio, current_price)
            
            if quantity <= 0:
                continue
            
            # 处理出场信号
            if signal.is_exit:
                position = self.portfolio.positions.get(signal.asset.symbol)
                if position and not position.is_flat:
                    quantity = abs(position.quantity)
            
            # 生成订单
            order = signal.to_order(quantity, OrderType.MARKET)
            if order:
                orders.append(order)
                
                # 提交订单
                success = self.broker.submit_order(
                    order,
                    self.portfolio,
                    current_price,
                )
                
                if success:
                    self._emit_event(Event(
                        event_type=EventType.ORDER_SUBMITTED,
                        timestamp=self.current_timestamp,
                        data={"order": order},
                    ))
        
        return orders
    
    def _on_order_filled(self, order: Order) -> None:
        """订单成交回调"""
        self._emit_event(Event(
            event_type=EventType.ORDER_FILLED,
            timestamp=self.current_timestamp,
            data={"order": order},
        ))
        
        # 通知策略
        for strategy in self.strategies:
            strategy.on_order_filled(order)
    
    def _on_order_rejected(self, order: Order, reason: str) -> None:
        """订单拒绝回调"""
        self._emit_event(Event(
            event_type=EventType.ORDER_REJECTED,
            timestamp=self.current_timestamp,
            data={"order": order, "reason": reason},
        ))
        
        # 通知策略
        for strategy in self.strategies:
            strategy.on_order_rejected(order, reason)
    
    def on_day_end(self) -> None:
        """日终处理"""
        # 记录组合快照（使用模拟时间而非真实时间）
        self.portfolio.record_daily(self.current_prices, timestamp=self.current_timestamp)
        
        self._emit_event(Event(
            event_type=EventType.DAY_END,
            timestamp=self.current_timestamp,
            data={"prices": self.current_prices.copy()},
        ))
    
    def step(self, symbol: str, ohlcv: OHLCV) -> None:
        """
        执行单步模拟
        
        Args:
            symbol: 资产代码
            ohlcv: OHLCV 数据
        """
        # 处理市场数据
        self.on_market_data(symbol, ohlcv)
        
        # 运行策略
        signals = self.process_strategies()
        
        # 处理信号
        self.process_signals(signals)
    
    def run(
        self,
        data: Dict[str, MarketData],
        on_step: Optional[Callable[[datetime], None]] = None,
    ) -> None:
        """
        运行完整模拟
        
        Args:
            data: 市场数据字典 {symbol: MarketData}
            on_step: 每步回调（用于进度显示等）
        """
        self.is_running = True
        
        # 初始化策略
        for strategy in self.strategies:
            assets = [Asset(symbol=s) for s in data.keys()]
            strategy.initialize(assets, data)
        
        # 收集所有时间戳
        all_timestamps = set()
        symbol_data = {}  # {timestamp: {symbol: ohlcv}}
        
        for symbol, market_data in data.items():
            for ohlcv in market_data.data:
                all_timestamps.add(ohlcv.timestamp)
                if ohlcv.timestamp not in symbol_data:
                    symbol_data[ohlcv.timestamp] = {}
                symbol_data[ohlcv.timestamp][symbol] = ohlcv
        
        # 按时间排序
        sorted_timestamps = sorted(all_timestamps)
        
        # 逐时间步处理
        last_date = None

        for timestamp in sorted_timestamps:
            current_date = timestamp.date()

            # 在处理新一天的数据之前，先完成前一天的日终快照
            # （此时 current_prices 仍为前一天的收盘价，保证快照正确）
            if last_date is not None and current_date != last_date:
                self.on_day_end()
            last_date = current_date

            # 处理每个资产的数据
            for symbol, ohlcv in symbol_data[timestamp].items():
                self.step(symbol, ohlcv)

            # 步骤回调
            if on_step:
                on_step(timestamp)
        
        # 最后一天的日终处理
        if last_date is not None:
            self.on_day_end()
        
        self.is_running = False
    
    def reset(self) -> None:
        """重置引擎状态"""
        self.market_data.clear()
        self.current_prices.clear()
        self.event_queue.clear()
        self.current_timestamp = None
        self.broker.reset()
        
        for strategy in self.strategies:
            strategy.reset()
    
    def summary(self) -> Dict[str, Any]:
        """获取引擎摘要"""
        return {
            "portfolio": self.portfolio.summary(self.current_prices),
            "broker": self.broker.summary(),
            "strategies": [s.summary() for s in self.strategies],
            "symbols": list(self.market_data.keys()),
            "total_events": len(self.event_queue),
        }
