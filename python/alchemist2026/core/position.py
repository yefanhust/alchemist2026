"""
持仓管理模块
追踪和管理资产持仓
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

from core.asset import Asset
from core.order import Order, OrderSide


class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"     # 多头
    SHORT = "short"   # 空头
    FLAT = "flat"     # 空仓


@dataclass
class Trade:
    """
    单笔交易记录
    """
    order_id: str
    asset: Asset
    side: OrderSide
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    
    @property
    def value(self) -> float:
        """交易价值"""
        return self.quantity * self.price
    
    @property
    def cost(self) -> float:
        """交易成本（含手续费）"""
        return self.value + self.commission


@dataclass
class Position:
    """
    持仓类
    
    表示某个资产的持仓状态。
    
    Attributes:
        asset: 持仓资产
        quantity: 持仓数量（正数为多头，负数为空头）
        avg_cost: 平均成本价
        realized_pnl: 已实现盈亏
        trades: 交易历史
    """
    
    asset: Asset
    quantity: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    opened_at: Optional[datetime] = None
    
    @property
    def side(self) -> PositionSide:
        """持仓方向"""
        if self.quantity > 0:
            return PositionSide.LONG
        elif self.quantity < 0:
            return PositionSide.SHORT
        else:
            return PositionSide.FLAT
    
    @property
    def is_flat(self) -> bool:
        """是否空仓"""
        return abs(self.quantity) < 1e-10
    
    @property
    def market_value(self) -> float:
        """市值（基于成本价）"""
        return abs(self.quantity) * self.avg_cost
    
    def unrealized_pnl(self, current_price: float) -> float:
        """
        计算未实现盈亏
        
        Args:
            current_price: 当前市场价格
            
        Returns:
            未实现盈亏
        """
        if self.is_flat:
            return 0.0
        
        if self.quantity > 0:  # 多头
            return (current_price - self.avg_cost) * self.quantity
        else:  # 空头
            return (self.avg_cost - current_price) * abs(self.quantity)
    
    def total_pnl(self, current_price: float) -> float:
        """
        计算总盈亏
        
        Args:
            current_price: 当前市场价格
            
        Returns:
            总盈亏（已实现 + 未实现）
        """
        return self.realized_pnl + self.unrealized_pnl(current_price)
    
    def return_rate(self, current_price: float) -> float:
        """
        计算收益率
        
        Args:
            current_price: 当前市场价格
            
        Returns:
            收益率（百分比）
        """
        if self.avg_cost == 0 or self.is_flat:
            return 0.0
        
        return (current_price - self.avg_cost) / self.avg_cost * 100
    
    def update(self, order: Order) -> None:
        """
        根据订单更新持仓
        
        Args:
            order: 已成交的订单
        """
        if order.filled_quantity <= 0:
            return
        
        # 记录交易
        trade = Trade(
            order_id=order.order_id,
            asset=self.asset,
            side=order.side,
            quantity=order.filled_quantity,
            price=order.filled_price,
            commission=order.commission,
            timestamp=datetime.now(),
        )
        self.trades.append(trade)
        self.total_commission += order.commission
        
        # 计算持仓变化
        if order.side == OrderSide.BUY:
            self._process_buy(order.filled_quantity, order.filled_price)
        else:
            self._process_sell(order.filled_quantity, order.filled_price)
    
    def _process_buy(self, quantity: float, price: float) -> None:
        """处理买入"""
        if self.quantity >= 0:
            # 加仓或开多仓
            total_cost = self.avg_cost * self.quantity + price * quantity
            self.quantity += quantity
            self.avg_cost = total_cost / self.quantity if self.quantity > 0 else 0
            
            if self.opened_at is None:
                self.opened_at = datetime.now()
        else:
            # 空头平仓
            close_qty = min(quantity, abs(self.quantity))
            pnl = (self.avg_cost - price) * close_qty  # 空头盈亏
            self.realized_pnl += pnl
            
            remaining_buy = quantity - close_qty
            self.quantity += close_qty
            
            if remaining_buy > 0:
                # 反手做多
                self.quantity = remaining_buy
                self.avg_cost = price
                self.opened_at = datetime.now()
            elif self.is_flat:
                self.avg_cost = 0
                self.opened_at = None
    
    def _process_sell(self, quantity: float, price: float) -> None:
        """处理卖出"""
        if self.quantity <= 0:
            # 加空仓或开空仓
            total_cost = self.avg_cost * abs(self.quantity) + price * quantity
            self.quantity -= quantity
            self.avg_cost = total_cost / abs(self.quantity) if self.quantity != 0 else 0
            
            if self.opened_at is None:
                self.opened_at = datetime.now()
        else:
            # 多头平仓
            close_qty = min(quantity, self.quantity)
            pnl = (price - self.avg_cost) * close_qty  # 多头盈亏
            self.realized_pnl += pnl
            
            remaining_sell = quantity - close_qty
            self.quantity -= close_qty
            
            if remaining_sell > 0:
                # 反手做空
                self.quantity = -remaining_sell
                self.avg_cost = price
                self.opened_at = datetime.now()
            elif self.is_flat:
                self.avg_cost = 0
                self.opened_at = None
    
    def close(self, price: float) -> float:
        """
        平仓
        
        Args:
            price: 平仓价格
            
        Returns:
            平仓盈亏
        """
        if self.is_flat:
            return 0.0
        
        pnl = self.unrealized_pnl(price)
        self.realized_pnl += pnl
        self.quantity = 0
        self.avg_cost = 0
        self.opened_at = None
        
        return pnl
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "asset": self.asset.to_dict(),
            "quantity": self.quantity,
            "side": self.side.value,
            "avg_cost": self.avg_cost,
            "market_value": self.market_value,
            "realized_pnl": self.realized_pnl,
            "total_commission": self.total_commission,
            "trade_count": len(self.trades),
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
        }
    
    def __repr__(self):
        return (
            f"Position({self.asset.symbol}, {self.side.value}, "
            f"qty={self.quantity:.2f}, avg={self.avg_cost:.2f})"
        )
