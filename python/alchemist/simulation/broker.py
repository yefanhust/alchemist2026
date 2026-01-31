"""
虚拟券商模块
模拟订单执行和成交
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
import random

from core.order import Order, OrderType, OrderSide, OrderStatus
from core.portfolio import Portfolio
from data.models import OHLCV


@dataclass
class BrokerConfig:
    """券商配置"""
    commission_rate: float = 0.001      # 手续费率
    min_commission: float = 1.0         # 最低手续费
    slippage_rate: float = 0.0005       # 滑点率
    slippage_mode: str = "percentage"   # 滑点模式: percentage, fixed, random
    partial_fill_enabled: bool = False  # 是否允许部分成交
    short_selling_enabled: bool = True  # 是否允许做空
    margin_rate: float = 0.5            # 保证金率（做空时）


class VirtualBroker:
    """
    虚拟券商
    
    模拟真实券商的订单处理和成交机制。
    
    功能：
    - 订单验证
    - 手续费计算
    - 滑点模拟
    - 成交撮合
    - 部分成交模拟
    """
    
    def __init__(
        self,
        config: Optional[BrokerConfig] = None,
        on_fill: Optional[Callable[[Order], None]] = None,
        on_reject: Optional[Callable[[Order, str], None]] = None,
    ):
        """
        初始化虚拟券商
        
        Args:
            config: 券商配置
            on_fill: 成交回调
            on_reject: 拒绝回调
        """
        self.config = config or BrokerConfig()
        self.on_fill = on_fill
        self.on_reject = on_reject
        
        # 待处理订单队列
        self.pending_orders: List[Order] = []
        
        # 成交历史
        self.fill_history: List[Dict[str, Any]] = []
    
    def calculate_commission(self, order: Order, fill_price: float) -> float:
        """
        计算手续费
        
        Args:
            order: 订单
            fill_price: 成交价格
            
        Returns:
            手续费金额
        """
        notional = order.quantity * fill_price
        commission = notional * self.config.commission_rate
        return max(commission, self.config.min_commission)
    
    def calculate_slippage(
        self,
        order: Order,
        market_price: float,
    ) -> float:
        """
        计算滑点后的成交价格
        
        Args:
            order: 订单
            market_price: 市场价格
            
        Returns:
            考虑滑点后的成交价格
        """
        if self.config.slippage_rate == 0:
            return market_price
        
        if self.config.slippage_mode == "percentage":
            slippage = market_price * self.config.slippage_rate
        elif self.config.slippage_mode == "fixed":
            slippage = self.config.slippage_rate
        elif self.config.slippage_mode == "random":
            slippage = market_price * self.config.slippage_rate * random.random()
        else:
            slippage = 0
        
        # 买入时价格上滑，卖出时价格下滑
        if order.side == OrderSide.BUY:
            return market_price + slippage
        else:
            return market_price - slippage
    
    def validate_order(
        self,
        order: Order,
        portfolio: Portfolio,
        current_price: float,
    ) -> tuple:
        """
        验证订单
        
        Args:
            order: 订单
            portfolio: 投资组合
            current_price: 当前价格
            
        Returns:
            (是否有效, 拒绝原因)
        """
        # 检查订单数量
        if order.quantity <= 0:
            return False, "订单数量必须大于0"
        
        # 检查买入资金
        if order.side == OrderSide.BUY:
            estimated_cost = order.quantity * current_price * (1 + self.config.slippage_rate)
            commission = self.calculate_commission(order, current_price)
            total_cost = estimated_cost + commission
            
            if total_cost > portfolio.cash:
                return False, f"资金不足: 需要 {total_cost:.2f}, 可用 {portfolio.cash:.2f}"
        
        # 检查卖出持仓
        elif order.side == OrderSide.SELL:
            position = portfolio.positions.get(order.asset.symbol)
            
            if not self.config.short_selling_enabled:
                if position is None or position.quantity < order.quantity:
                    return False, "持仓不足且不允许做空"
            else:
                # 做空需要检查保证金
                if position is None or position.quantity < order.quantity:
                    short_qty = order.quantity - (position.quantity if position else 0)
                    margin_required = short_qty * current_price * self.config.margin_rate
                    if margin_required > portfolio.cash:
                        return False, f"保证金不足: 需要 {margin_required:.2f}"
        
        return True, ""
    
    def submit_order(
        self,
        order: Order,
        portfolio: Portfolio,
        current_price: float,
    ) -> bool:
        """
        提交订单
        
        Args:
            order: 订单
            portfolio: 投资组合
            current_price: 当前价格
            
        Returns:
            是否成功提交
        """
        # 验证订单
        is_valid, reject_reason = self.validate_order(order, portfolio, current_price)
        
        if not is_valid:
            order.reject(reject_reason)
            if self.on_reject:
                self.on_reject(order, reject_reason)
            return False
        
        # 提交订单
        order.submit()
        self.pending_orders.append(order)
        
        return True
    
    def process_orders(
        self,
        portfolio: Portfolio,
        market_data: OHLCV,
    ) -> List[Order]:
        """
        处理待处理订单
        
        Args:
            portfolio: 投资组合
            market_data: 当前市场数据
            
        Returns:
            已成交的订单列表
        """
        filled_orders = []
        remaining_orders = []
        
        for order in self.pending_orders:
            if not order.is_active:
                continue
            
            # 检查是否可以成交
            fill_price = self._get_fill_price(order, market_data)
            
            if fill_price is not None:
                # 应用滑点
                fill_price = self.calculate_slippage(order, fill_price)
                
                # 计算手续费
                commission = self.calculate_commission(order, fill_price)
                
                # 执行成交
                success = portfolio.execute_order(
                    order,
                    fill_price,
                    commission,
                    slippage=0,  # 已经在 fill_price 中考虑
                )
                
                if success:
                    filled_orders.append(order)
                    
                    # 记录成交
                    self.fill_history.append({
                        "order_id": order.order_id,
                        "symbol": order.asset.symbol,
                        "side": order.side.value,
                        "quantity": order.filled_quantity,
                        "price": fill_price,
                        "commission": commission,
                        "timestamp": market_data.timestamp,
                    })
                    
                    # 触发回调
                    if self.on_fill:
                        self.on_fill(order)
                else:
                    # 成交失败，保留在队列中
                    remaining_orders.append(order)
            else:
                # 无法成交，保留在队列中
                remaining_orders.append(order)
        
        self.pending_orders = remaining_orders
        return filled_orders
    
    def _get_fill_price(
        self,
        order: Order,
        market_data: OHLCV,
    ) -> Optional[float]:
        """
        获取成交价格
        
        Args:
            order: 订单
            market_data: 市场数据
            
        Returns:
            成交价格，如果无法成交返回 None
        """
        if order.order_type == OrderType.MARKET:
            # 市价单：使用开盘价模拟（假设在开盘时成交）
            return market_data.open
        
        elif order.order_type == OrderType.LIMIT:
            # 限价单：检查价格是否触及
            if order.side == OrderSide.BUY:
                if market_data.low <= order.limit_price:
                    return min(order.limit_price, market_data.open)
            else:
                if market_data.high >= order.limit_price:
                    return max(order.limit_price, market_data.open)
        
        elif order.order_type == OrderType.STOP:
            # 止损单：检查是否触发
            if order.side == OrderSide.BUY:
                if market_data.high >= order.stop_price:
                    return max(order.stop_price, market_data.open)
            else:
                if market_data.low <= order.stop_price:
                    return min(order.stop_price, market_data.open)
        
        elif order.order_type == OrderType.STOP_LIMIT:
            # 止损限价单：先检查止损触发，再检查限价
            triggered = False
            if order.side == OrderSide.BUY:
                triggered = market_data.high >= order.stop_price
            else:
                triggered = market_data.low <= order.stop_price
            
            if triggered:
                if order.side == OrderSide.BUY:
                    if market_data.low <= order.limit_price:
                        return min(order.limit_price, market_data.open)
                else:
                    if market_data.high >= order.limit_price:
                        return max(order.limit_price, market_data.open)
        
        return None
    
    def cancel_order(self, order: Order) -> bool:
        """
        取消订单
        
        Args:
            order: 订单
            
        Returns:
            是否成功取消
        """
        if order in self.pending_orders:
            order.cancel()
            self.pending_orders.remove(order)
            return True
        return False
    
    def cancel_all_orders(self) -> int:
        """
        取消所有订单
        
        Returns:
            取消的订单数量
        """
        count = len(self.pending_orders)
        for order in self.pending_orders:
            order.cancel()
        self.pending_orders.clear()
        return count
    
    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        获取待处理订单
        
        Args:
            symbol: 资产代码（可选，筛选）
            
        Returns:
            订单列表
        """
        if symbol:
            return [o for o in self.pending_orders if o.asset.symbol == symbol]
        return self.pending_orders.copy()
    
    def reset(self) -> None:
        """重置券商状态"""
        self.pending_orders.clear()
        self.fill_history.clear()
    
    def summary(self) -> Dict[str, Any]:
        """获取券商摘要"""
        total_commission = sum(f["commission"] for f in self.fill_history)
        total_volume = sum(f["quantity"] * f["price"] for f in self.fill_history)
        
        return {
            "pending_orders": len(self.pending_orders),
            "total_fills": len(self.fill_history),
            "total_commission": total_commission,
            "total_volume": total_volume,
            "config": {
                "commission_rate": self.config.commission_rate,
                "slippage_rate": self.config.slippage_rate,
                "short_selling": self.config.short_selling_enabled,
            },
        }
