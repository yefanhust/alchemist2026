"""
投资组合管理模块
管理多资产投资组合
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

from .asset import Asset
from .order import Order, OrderSide, OrderStatus
from .position import Position


@dataclass
class PortfolioSnapshot:
    """投资组合快照"""
    timestamp: datetime
    cash: float
    positions_value: float
    total_value: float
    unrealized_pnl: float
    realized_pnl: float


class Portfolio:
    """
    投资组合类
    
    管理现金、持仓、订单和绩效追踪。
    
    Attributes:
        initial_capital: 初始资金
        cash: 当前现金
        positions: 持仓字典
        orders: 订单历史
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        name: str = "default",
        currency: str = "USD",
    ):
        """
        初始化投资组合
        
        Args:
            initial_capital: 初始资金
            name: 组合名称
            currency: 基础货币
        """
        self.name = name
        self.currency = currency
        self.initial_capital = initial_capital
        self.cash = initial_capital
        
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.snapshots: List[PortfolioSnapshot] = []
        
        self.created_at = datetime.now()
        
        # 记录初始快照
        self._record_snapshot({})
    
    def get_position(self, asset: Asset) -> Position:
        """
        获取资产持仓，如不存在则创建
        
        Args:
            asset: 资产对象
            
        Returns:
            持仓对象
        """
        if asset.symbol not in self.positions:
            self.positions[asset.symbol] = Position(asset=asset)
        return self.positions[asset.symbol]
    
    def submit_order(self, order: Order) -> bool:
        """
        提交订单
        
        Args:
            order: 订单对象
            
        Returns:
            是否成功提交
        """
        # 验证订单
        if not self._validate_order(order):
            order.reject("订单验证失败")
            self.orders.append(order)
            return False
        
        order.submit()
        self.orders.append(order)
        return True
    
    def _validate_order(self, order: Order) -> bool:
        """
        验证订单是否合法
        
        Args:
            order: 订单对象
            
        Returns:
            是否合法
        """
        if order.side == OrderSide.BUY:
            # 买入需要足够的现金
            estimated_cost = order.quantity * (order.limit_price or 0)
            # 市价单预留更多资金
            if order.limit_price is None:
                estimated_cost = order.quantity * 1e10  # 需要在执行时检查
            # 这里简化处理，实际需要考虑手续费
            return True  # 延迟到执行时检查
        else:
            # 卖出需要足够的持仓（或允许做空）
            position = self.positions.get(order.asset.symbol)
            if position is None:
                return True  # 允许做空
            return True  # 延迟到执行时检查
    
    def execute_order(
        self,
        order: Order,
        fill_price: float,
        commission: float = 0.0,
        slippage: float = 0.0,
    ) -> bool:
        """
        执行订单
        
        Args:
            order: 订单对象
            fill_price: 成交价格
            commission: 手续费
            slippage: 滑点
            
        Returns:
            是否成功执行
        """
        # 应用滑点
        if order.side == OrderSide.BUY:
            actual_price = fill_price * (1 + slippage)
        else:
            actual_price = fill_price * (1 - slippage)
        
        # 计算所需资金
        total_cost = order.quantity * actual_price + commission
        
        # 检查买入是否有足够现金
        if order.side == OrderSide.BUY and total_cost > self.cash:
            order.reject("现金不足")
            return False
        
        # 检查卖出是否有足够持仓（如果不允许做空）
        position = self.get_position(order.asset)
        if order.side == OrderSide.SELL and position.quantity < order.quantity:
            # 这里允许做空，如果需要禁止做空可以取消注释下面的代码
            # order.reject("持仓不足")
            # return False
            pass
        
        # 执行成交
        order.fill(order.quantity, actual_price, commission)
        
        # 更新持仓
        position.update(order)
        
        # 更新现金
        if order.side == OrderSide.BUY:
            self.cash -= total_cost
        else:
            self.cash += order.quantity * actual_price - commission
        
        return True
    
    def positions_value(self, prices: Dict[str, float]) -> float:
        """
        计算持仓市值
        
        Args:
            prices: 当前价格字典 {symbol: price}
            
        Returns:
            总持仓市值
        """
        total = 0.0
        for symbol, position in self.positions.items():
            if not position.is_flat:
                price = prices.get(symbol, position.avg_cost)
                total += position.quantity * price
        return total
    
    def total_value(self, prices: Dict[str, float]) -> float:
        """
        计算组合总价值
        
        Args:
            prices: 当前价格字典
            
        Returns:
            总价值
        """
        return self.cash + self.positions_value(prices)
    
    def unrealized_pnl(self, prices: Dict[str, float]) -> float:
        """
        计算未实现盈亏
        
        Args:
            prices: 当前价格字典
            
        Returns:
            未实现盈亏
        """
        total = 0.0
        for symbol, position in self.positions.items():
            if not position.is_flat:
                price = prices.get(symbol, position.avg_cost)
                total += position.unrealized_pnl(price)
        return total
    
    def realized_pnl(self) -> float:
        """
        计算已实现盈亏
        
        Returns:
            已实现盈亏
        """
        return sum(p.realized_pnl for p in self.positions.values())
    
    def total_commission(self) -> float:
        """
        计算总手续费
        
        Returns:
            总手续费
        """
        return sum(p.total_commission for p in self.positions.values())
    
    def returns(self, prices: Dict[str, float]) -> float:
        """
        计算收益率
        
        Args:
            prices: 当前价格字典
            
        Returns:
            收益率（百分比）
        """
        if self.initial_capital == 0:
            return 0.0
        
        total = self.total_value(prices)
        return (total - self.initial_capital) / self.initial_capital * 100
    
    def _record_snapshot(self, prices: Dict[str, float]) -> None:
        """记录组合快照"""
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            cash=self.cash,
            positions_value=self.positions_value(prices),
            total_value=self.total_value(prices),
            unrealized_pnl=self.unrealized_pnl(prices),
            realized_pnl=self.realized_pnl(),
        )
        self.snapshots.append(snapshot)
    
    def record_daily(self, prices: Dict[str, float]) -> None:
        """记录每日快照（供回测使用）"""
        self._record_snapshot(prices)
    
    def get_active_orders(self) -> List[Order]:
        """获取活跃订单"""
        return [o for o in self.orders if o.is_active]
    
    def get_filled_orders(self) -> List[Order]:
        """获取已成交订单"""
        return [o for o in self.orders if o.status == OrderStatus.FILLED]
    
    def cancel_all_orders(self) -> int:
        """
        取消所有活跃订单
        
        Returns:
            取消的订单数量
        """
        count = 0
        for order in self.get_active_orders():
            order.cancel()
            count += 1
        return count
    
    def close_all_positions(self, prices: Dict[str, float]) -> float:
        """
        平仓所有持仓
        
        Args:
            prices: 当前价格字典
            
        Returns:
            平仓总盈亏
        """
        total_pnl = 0.0
        for symbol, position in self.positions.items():
            if not position.is_flat:
                price = prices.get(symbol, position.avg_cost)
                pnl = position.close(price)
                self.cash += position.quantity * price  # 这里已经平仓，quantity 为 0
                total_pnl += pnl
        return total_pnl
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        将快照历史转换为 DataFrame
        
        Returns:
            快照数据表
        """
        if not self.snapshots:
            return pd.DataFrame()
        
        data = [
            {
                "timestamp": s.timestamp,
                "cash": s.cash,
                "positions_value": s.positions_value,
                "total_value": s.total_value,
                "unrealized_pnl": s.unrealized_pnl,
                "realized_pnl": s.realized_pnl,
            }
            for s in self.snapshots
        ]
        
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df
    
    def summary(self, prices: Dict[str, float]) -> Dict[str, Any]:
        """
        获取组合摘要
        
        Args:
            prices: 当前价格字典
            
        Returns:
            摘要字典
        """
        return {
            "name": self.name,
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "positions_value": self.positions_value(prices),
            "total_value": self.total_value(prices),
            "returns_pct": self.returns(prices),
            "unrealized_pnl": self.unrealized_pnl(prices),
            "realized_pnl": self.realized_pnl(),
            "total_commission": self.total_commission(),
            "position_count": sum(1 for p in self.positions.values() if not p.is_flat),
            "order_count": len(self.orders),
            "filled_order_count": len(self.get_filled_orders()),
        }
    
    def __repr__(self):
        return f"Portfolio({self.name}, cash={self.cash:.2f}, positions={len(self.positions)})"
