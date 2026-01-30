"""
订单系统模块
定义订单类型、状态和订单对象
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from uuid import uuid4

from .asset import Asset


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"           # 市价单
    LIMIT = "limit"             # 限价单
    STOP = "stop"               # 止损单
    STOP_LIMIT = "stop_limit"   # 止损限价单
    TRAILING_STOP = "trailing_stop"  # 追踪止损单


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"         # 待处理
    SUBMITTED = "submitted"     # 已提交
    PARTIAL = "partial"         # 部分成交
    FILLED = "filled"           # 完全成交
    CANCELLED = "cancelled"     # 已取消
    REJECTED = "rejected"       # 被拒绝
    EXPIRED = "expired"         # 已过期


class TimeInForce(Enum):
    """订单有效期"""
    DAY = "day"                 # 当日有效
    GTC = "gtc"                 # 撤销前有效
    IOC = "ioc"                 # 立即成交或取消
    FOK = "fok"                 # 全部成交或取消


@dataclass
class Order:
    """
    订单类
    
    表示一个交易订单。
    
    Attributes:
        asset: 交易资产
        side: 买卖方向
        quantity: 订单数量
        order_type: 订单类型
        limit_price: 限价（限价单使用）
        stop_price: 止损价（止损单使用）
        time_in_force: 订单有效期
        status: 订单状态
        filled_quantity: 已成交数量
        filled_price: 成交均价
        commission: 手续费
    """
    
    asset: Asset
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    
    # 内部状态
    order_id: str = field(default_factory=lambda: str(uuid4()))
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    
    # 策略标记
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """验证订单参数"""
        if self.quantity <= 0:
            raise ValueError("订单数量必须大于0")
        
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("限价单必须指定限价")
        
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError("止损单必须指定止损价")
        
        if self.order_type == OrderType.STOP_LIMIT and self.limit_price is None:
            raise ValueError("止损限价单必须指定限价")
        
        # 规范化数量
        self.quantity = self.asset.round_quantity(self.quantity)
    
    @property
    def remaining_quantity(self) -> float:
        """未成交数量"""
        return self.quantity - self.filled_quantity
    
    @property
    def is_active(self) -> bool:
        """订单是否活跃"""
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL)
    
    @property
    def is_completed(self) -> bool:
        """订单是否已完成"""
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED)
    
    @property
    def total_value(self) -> float:
        """订单总价值（基于成交价）"""
        return self.filled_quantity * self.filled_price
    
    def fill(self, quantity: float, price: float, commission: float = 0.0) -> None:
        """
        执行订单成交
        
        Args:
            quantity: 成交数量
            price: 成交价格
            commission: 手续费
        """
        if quantity <= 0:
            raise ValueError("成交数量必须大于0")
        
        if quantity > self.remaining_quantity:
            raise ValueError("成交数量超过剩余数量")
        
        # 计算新的成交均价
        total_value = self.filled_quantity * self.filled_price + quantity * price
        self.filled_quantity += quantity
        self.filled_price = total_value / self.filled_quantity if self.filled_quantity > 0 else 0
        
        # 累加手续费
        self.commission += commission
        
        # 更新状态
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_at = datetime.now()
        else:
            self.status = OrderStatus.PARTIAL
    
    def cancel(self) -> None:
        """取消订单"""
        if self.is_completed:
            raise ValueError("已完成的订单无法取消")
        self.status = OrderStatus.CANCELLED
    
    def reject(self, reason: str = "") -> None:
        """拒绝订单"""
        self.status = OrderStatus.REJECTED
        self.metadata["reject_reason"] = reason
    
    def submit(self) -> None:
        """提交订单"""
        if self.status != OrderStatus.PENDING:
            raise ValueError("只有待处理的订单可以提交")
        self.status = OrderStatus.SUBMITTED
        self.submitted_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "order_id": self.order_id,
            "asset": self.asset.to_dict(),
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "commission": self.commission,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "strategy_id": self.strategy_id,
            "metadata": self.metadata,
        }
    
    def __repr__(self):
        return (
            f"Order({self.order_id[:8]}..., {self.asset.symbol}, "
            f"{self.side.value}, {self.quantity}, {self.status.value})"
        )
