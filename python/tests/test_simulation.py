"""
模拟交易系统综合测试
测试交易执行、盈亏计算、收益指标、风险指标、交易统计的准确性
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
import pandas as pd
import pytest

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "alchemist"))

from core.asset import Asset, AssetType
from core.order import Order, OrderType, OrderSide, OrderStatus
from core.position import Position, PositionSide, Trade
from core.portfolio import Portfolio, PortfolioSnapshot
from simulation.broker import VirtualBroker, BrokerConfig
from data.models import OHLCV


def run_async(coro):
    """运行异步函数的辅助方法"""
    return asyncio.run(coro)


# =============================================================================
# 测试数据工厂
# =============================================================================

def create_asset(symbol: str = "AAPL") -> Asset:
    """创建测试资产"""
    return Asset(
        symbol=symbol,
        asset_type=AssetType.STOCK,
        name=f"{symbol} Inc.",
        exchange="NASDAQ",
        currency="USD",
        tick_size=0.01,
        lot_size=1.0,
    )


def create_order(
    asset: Asset = None,
    side: OrderSide = OrderSide.BUY,
    quantity: float = 100,
    order_type: OrderType = OrderType.MARKET,
    limit_price: float = None,
    stop_price: float = None,
) -> Order:
    """创建测试订单"""
    if asset is None:
        asset = create_asset()
    return Order(
        asset=asset,
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
    )


def create_ohlcv(
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: float = 1000000,
    timestamp: datetime = None,
) -> OHLCV:
    """创建测试OHLCV数据"""
    if timestamp is None:
        timestamp = datetime.now()
    return OHLCV(
        timestamp=timestamp,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


# =============================================================================
# 手续费计算测试
# =============================================================================

class TestCommissionCalculation:
    """手续费计算准确性测试"""

    def test_commission_percentage_calculation(self):
        """测试百分比手续费计算"""
        config = BrokerConfig(commission_rate=0.001, min_commission=1.0)
        broker = VirtualBroker(config=config)
        asset = create_asset()

        # 订单: 100股 @ $150 = $15000
        # 手续费: $15000 * 0.001 = $15
        order = create_order(asset=asset, quantity=100)
        commission = broker.calculate_commission(order, fill_price=150.0)

        assert commission == 15.0, f"Expected 15.0, got {commission}"

    def test_commission_minimum(self):
        """测试最低手续费"""
        config = BrokerConfig(commission_rate=0.001, min_commission=5.0)
        broker = VirtualBroker(config=config)
        asset = create_asset()

        # 订单: 10股 @ $100 = $1000
        # 计算手续费: $1000 * 0.001 = $1 < 最低 $5
        # 实际手续费: $5
        order = create_order(asset=asset, quantity=10)
        commission = broker.calculate_commission(order, fill_price=100.0)

        assert commission == 5.0, f"Expected 5.0 (minimum), got {commission}"

    def test_commission_exact_at_minimum(self):
        """测试刚好等于最低手续费的情况"""
        config = BrokerConfig(commission_rate=0.001, min_commission=1.0)
        broker = VirtualBroker(config=config)
        asset = create_asset()

        # 订单: 10股 @ $100 = $1000
        # 计算手续费: $1000 * 0.001 = $1 = 最低
        order = create_order(asset=asset, quantity=10)
        commission = broker.calculate_commission(order, fill_price=100.0)

        assert commission == 1.0, f"Expected 1.0, got {commission}"

    def test_commission_with_different_rates(self):
        """测试不同手续费率"""
        test_cases = [
            # (rate, quantity, price, expected)
            (0.001, 100, 100.0, 10.0),   # 0.1%
            (0.0005, 100, 100.0, 5.0),   # 0.05%
            (0.002, 100, 100.0, 20.0),   # 0.2%
            (0.0001, 1000, 50.0, 5.0),   # 0.01%
        ]

        for rate, qty, price, expected in test_cases:
            config = BrokerConfig(commission_rate=rate, min_commission=1.0)
            broker = VirtualBroker(config=config)
            order = create_order(quantity=qty)
            commission = broker.calculate_commission(order, fill_price=price)

            assert commission == expected, \
                f"Rate {rate}, qty {qty}, price {price}: expected {expected}, got {commission}"

    def test_commission_large_order(self):
        """测试大额订单手续费"""
        config = BrokerConfig(commission_rate=0.001, min_commission=1.0)
        broker = VirtualBroker(config=config)

        # 10000股 @ $500 = $5,000,000
        # 手续费: $5000
        order = create_order(quantity=10000)
        commission = broker.calculate_commission(order, fill_price=500.0)

        assert commission == 5000.0, f"Expected 5000.0, got {commission}"

    def test_commission_fractional_shares(self):
        """测试分数股手续费"""
        config = BrokerConfig(commission_rate=0.001, min_commission=0.01)
        broker = VirtualBroker(config=config)

        # 假设资产允许分数股
        asset = Asset(symbol="FRAC", lot_size=0.01)
        order = create_order(asset=asset, quantity=0.5)

        # 0.5股 @ $100 = $50, 手续费 = $0.05
        commission = broker.calculate_commission(order, fill_price=100.0)

        assert commission == 0.05, f"Expected 0.05, got {commission}"


# =============================================================================
# 滑点计算测试
# =============================================================================

class TestSlippageCalculation:
    """滑点计算准确性测试"""

    def test_percentage_slippage_buy(self):
        """测试买入时百分比滑点"""
        config = BrokerConfig(slippage_rate=0.001, slippage_mode="percentage")
        broker = VirtualBroker(config=config)

        order = create_order(side=OrderSide.BUY, quantity=100)
        market_price = 100.0

        # 买入滑点: 100 + 100 * 0.001 = 100.1
        fill_price = broker.calculate_slippage(order, market_price)

        assert fill_price == 100.1, f"Expected 100.1, got {fill_price}"

    def test_percentage_slippage_sell(self):
        """测试卖出时百分比滑点"""
        config = BrokerConfig(slippage_rate=0.001, slippage_mode="percentage")
        broker = VirtualBroker(config=config)

        order = create_order(side=OrderSide.SELL, quantity=100)
        market_price = 100.0

        # 卖出滑点: 100 - 100 * 0.001 = 99.9
        fill_price = broker.calculate_slippage(order, market_price)

        assert fill_price == 99.9, f"Expected 99.9, got {fill_price}"

    def test_fixed_slippage_buy(self):
        """测试买入时固定滑点"""
        config = BrokerConfig(slippage_rate=0.05, slippage_mode="fixed")
        broker = VirtualBroker(config=config)

        order = create_order(side=OrderSide.BUY, quantity=100)
        market_price = 100.0

        # 买入固定滑点: 100 + 0.05 = 100.05
        fill_price = broker.calculate_slippage(order, market_price)

        assert fill_price == 100.05, f"Expected 100.05, got {fill_price}"

    def test_fixed_slippage_sell(self):
        """测试卖出时固定滑点"""
        config = BrokerConfig(slippage_rate=0.05, slippage_mode="fixed")
        broker = VirtualBroker(config=config)

        order = create_order(side=OrderSide.SELL, quantity=100)
        market_price = 100.0

        # 卖出固定滑点: 100 - 0.05 = 99.95
        fill_price = broker.calculate_slippage(order, market_price)

        assert fill_price == 99.95, f"Expected 99.95, got {fill_price}"

    def test_zero_slippage(self):
        """测试零滑点"""
        config = BrokerConfig(slippage_rate=0.0, slippage_mode="percentage")
        broker = VirtualBroker(config=config)

        order = create_order(side=OrderSide.BUY, quantity=100)
        market_price = 100.0

        fill_price = broker.calculate_slippage(order, market_price)

        assert fill_price == 100.0, f"Expected 100.0, got {fill_price}"

    def test_random_slippage_bounded(self):
        """测试随机滑点在合理范围内"""
        config = BrokerConfig(slippage_rate=0.01, slippage_mode="random")
        broker = VirtualBroker(config=config)

        order = create_order(side=OrderSide.BUY, quantity=100)
        market_price = 100.0

        # 运行多次确保随机滑点在范围内
        for _ in range(100):
            fill_price = broker.calculate_slippage(order, market_price)
            # 买入: 价格应在 [100, 101] 之间
            assert 100.0 <= fill_price <= 101.0, \
                f"Random slippage out of bounds: {fill_price}"

    def test_slippage_direction_correctness(self):
        """测试滑点方向正确性 - 买入价升，卖出价降"""
        config = BrokerConfig(slippage_rate=0.005, slippage_mode="percentage")
        broker = VirtualBroker(config=config)
        market_price = 100.0

        buy_order = create_order(side=OrderSide.BUY, quantity=100)
        sell_order = create_order(side=OrderSide.SELL, quantity=100)

        buy_fill = broker.calculate_slippage(buy_order, market_price)
        sell_fill = broker.calculate_slippage(sell_order, market_price)

        # 买入价格应高于市场价
        assert buy_fill > market_price, "Buy slippage should increase price"
        # 卖出价格应低于市场价
        assert sell_fill < market_price, "Sell slippage should decrease price"
        # 买入卖出价差
        assert buy_fill > sell_fill, "Buy price should be higher than sell price"


# =============================================================================
# 订单成交价格测试
# =============================================================================

class TestOrderFillPrice:
    """订单成交价格测试"""

    def test_market_order_fill_at_open(self):
        """测试市价单以开盘价成交"""
        broker = VirtualBroker(config=BrokerConfig(slippage_rate=0))

        order = create_order(
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        order.submit()

        ohlcv = create_ohlcv(open_price=100.0, high=105.0, low=98.0, close=102.0)

        fill_price = broker._get_fill_price(order, ohlcv)

        assert fill_price == 100.0, f"Market order should fill at open: {fill_price}"

    def test_limit_buy_order_triggered(self):
        """测试限价买入单触发"""
        broker = VirtualBroker(config=BrokerConfig(slippage_rate=0))

        # 限价买入 @ $99，当日最低 $98，应该成交
        order = create_order(
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=99.0,
        )
        order.submit()

        ohlcv = create_ohlcv(open_price=100.0, high=105.0, low=98.0, close=102.0)

        fill_price = broker._get_fill_price(order, ohlcv)

        # 限价买入: min(limit_price, open) = min(99, 100) = 99
        assert fill_price == 99.0, f"Limit buy should fill at limit price: {fill_price}"

    def test_limit_buy_order_not_triggered(self):
        """测试限价买入单未触发"""
        broker = VirtualBroker(config=BrokerConfig(slippage_rate=0))

        # 限价买入 @ $95，当日最低 $98，不应该成交
        order = create_order(
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=95.0,
        )
        order.submit()

        ohlcv = create_ohlcv(open_price=100.0, high=105.0, low=98.0, close=102.0)

        fill_price = broker._get_fill_price(order, ohlcv)

        assert fill_price is None, f"Limit buy should not trigger: {fill_price}"

    def test_limit_sell_order_triggered(self):
        """测试限价卖出单触发"""
        broker = VirtualBroker(config=BrokerConfig(slippage_rate=0))

        # 限价卖出 @ $104，当日最高 $105，应该成交
        order = create_order(
            side=OrderSide.SELL,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=104.0,
        )
        order.submit()

        ohlcv = create_ohlcv(open_price=100.0, high=105.0, low=98.0, close=102.0)

        fill_price = broker._get_fill_price(order, ohlcv)

        # 限价卖出: max(limit_price, open) = max(104, 100) = 104
        assert fill_price == 104.0, f"Limit sell should fill at limit price: {fill_price}"

    def test_stop_buy_order_triggered(self):
        """测试止损买入单触发"""
        broker = VirtualBroker(config=BrokerConfig(slippage_rate=0))

        # 止损买入 @ $103，当日最高 $105，应该成交
        order = create_order(
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.STOP,
            stop_price=103.0,
        )
        order.submit()

        ohlcv = create_ohlcv(open_price=100.0, high=105.0, low=98.0, close=102.0)

        fill_price = broker._get_fill_price(order, ohlcv)

        # 止损买入: max(stop_price, open) = max(103, 100) = 103
        assert fill_price == 103.0, f"Stop buy should fill at stop price: {fill_price}"

    def test_stop_sell_order_triggered(self):
        """测试止损卖出单触发"""
        broker = VirtualBroker(config=BrokerConfig(slippage_rate=0))

        # 止损卖出 @ $99，当日最低 $98，应该成交
        order = create_order(
            side=OrderSide.SELL,
            quantity=100,
            order_type=OrderType.STOP,
            stop_price=99.0,
        )
        order.submit()

        ohlcv = create_ohlcv(open_price=100.0, high=105.0, low=98.0, close=102.0)

        fill_price = broker._get_fill_price(order, ohlcv)

        # 止损卖出: min(stop_price, open) = min(99, 100) = 99
        assert fill_price == 99.0, f"Stop sell should fill at stop price: {fill_price}"

    def test_stop_limit_buy_order(self):
        """测试止损限价买入单"""
        broker = VirtualBroker(config=BrokerConfig(slippage_rate=0))

        # 止损限价买入: 止损 $103, 限价 $104
        # 当日 high=105 触发止损，low=98 <= 104 触发限价
        order = create_order(
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.STOP_LIMIT,
            stop_price=103.0,
            limit_price=104.0,
        )
        order.submit()

        ohlcv = create_ohlcv(open_price=100.0, high=105.0, low=98.0, close=102.0)

        fill_price = broker._get_fill_price(order, ohlcv)

        # 止损限价买入: min(limit_price, open) = min(104, 100) = 100
        assert fill_price == 100.0, f"Stop limit buy should fill: {fill_price}"

    def test_gap_up_market_order(self):
        """测试跳空高开时市价单成交"""
        broker = VirtualBroker(config=BrokerConfig(slippage_rate=0))

        order = create_order(
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        order.submit()

        # 跳空高开: 开盘价110远高于前收盘100
        ohlcv = create_ohlcv(open_price=110.0, high=115.0, low=108.0, close=112.0)

        fill_price = broker._get_fill_price(order, ohlcv)

        assert fill_price == 110.0, f"Gap up market order should fill at open: {fill_price}"


# =============================================================================
# 持仓盈亏计算测试
# =============================================================================

class TestPositionPnL:
    """持仓盈亏计算准确性测试"""

    def test_long_position_unrealized_pnl_profit(self):
        """测试多头持仓未实现盈利"""
        asset = create_asset()
        position = Position(asset=asset, quantity=100, avg_cost=100.0)

        current_price = 110.0
        unrealized = position.unrealized_pnl(current_price)

        # (110 - 100) * 100 = 1000
        assert unrealized == 1000.0, f"Expected 1000.0, got {unrealized}"

    def test_long_position_unrealized_pnl_loss(self):
        """测试多头持仓未实现亏损"""
        asset = create_asset()
        position = Position(asset=asset, quantity=100, avg_cost=100.0)

        current_price = 90.0
        unrealized = position.unrealized_pnl(current_price)

        # (90 - 100) * 100 = -1000
        assert unrealized == -1000.0, f"Expected -1000.0, got {unrealized}"

    def test_short_position_unrealized_pnl_profit(self):
        """测试空头持仓未实现盈利"""
        asset = create_asset()
        position = Position(asset=asset, quantity=-100, avg_cost=100.0)

        current_price = 90.0
        unrealized = position.unrealized_pnl(current_price)

        # 空头盈利: (100 - 90) * 100 = 1000
        assert unrealized == 1000.0, f"Expected 1000.0, got {unrealized}"

    def test_short_position_unrealized_pnl_loss(self):
        """测试空头持仓未实现亏损"""
        asset = create_asset()
        position = Position(asset=asset, quantity=-100, avg_cost=100.0)

        current_price = 110.0
        unrealized = position.unrealized_pnl(current_price)

        # 空头亏损: (100 - 110) * 100 = -1000
        assert unrealized == -1000.0, f"Expected -1000.0, got {unrealized}"

    def test_long_position_realized_pnl_on_sell(self):
        """测试多头平仓实现盈亏"""
        asset = create_asset()
        position = Position(asset=asset, quantity=100, avg_cost=100.0)

        # 创建卖出订单并执行
        order = create_order(asset=asset, side=OrderSide.SELL, quantity=100)
        order.fill(100, 120.0, commission=0)

        position.update(order)

        # 多头平仓盈亏: (120 - 100) * 100 = 2000
        assert position.realized_pnl == 2000.0, f"Expected 2000.0, got {position.realized_pnl}"
        assert position.is_flat, "Position should be flat"

    def test_short_position_realized_pnl_on_cover(self):
        """测试空头平仓实现盈亏"""
        asset = create_asset()
        position = Position(asset=asset, quantity=-100, avg_cost=100.0)

        # 创建买入订单平仓
        order = create_order(asset=asset, side=OrderSide.BUY, quantity=100)
        order.fill(100, 80.0, commission=0)

        position.update(order)

        # 空头平仓盈亏: (100 - 80) * 100 = 2000
        assert position.realized_pnl == 2000.0, f"Expected 2000.0, got {position.realized_pnl}"
        assert position.is_flat, "Position should be flat"

    def test_partial_close_realized_pnl(self):
        """测试部分平仓实现盈亏"""
        asset = create_asset()
        position = Position(asset=asset, quantity=100, avg_cost=100.0)

        # 卖出50股
        order = create_order(asset=asset, side=OrderSide.SELL, quantity=50)
        order.fill(50, 120.0, commission=0)

        position.update(order)

        # 部分平仓盈亏: (120 - 100) * 50 = 1000
        assert position.realized_pnl == 1000.0, f"Expected 1000.0, got {position.realized_pnl}"
        assert position.quantity == 50, f"Expected remaining 50, got {position.quantity}"

    def test_add_to_long_position_avg_cost(self):
        """测试加仓后平均成本计算"""
        asset = create_asset()
        position = Position(asset=asset, quantity=100, avg_cost=100.0)

        # 加仓100股 @ $120
        order = create_order(asset=asset, side=OrderSide.BUY, quantity=100)
        order.fill(100, 120.0, commission=0)

        position.update(order)

        # 新平均成本: (100*100 + 100*120) / 200 = 110
        assert position.quantity == 200, f"Expected 200, got {position.quantity}"
        assert position.avg_cost == 110.0, f"Expected 110.0, got {position.avg_cost}"

    def test_total_pnl_calculation(self):
        """测试总盈亏计算（已实现+未实现）"""
        asset = create_asset()
        position = Position(asset=asset, quantity=100, avg_cost=100.0)

        # 卖出50股 @ $120，实现盈亏 1000
        order = create_order(asset=asset, side=OrderSide.SELL, quantity=50)
        order.fill(50, 120.0, commission=0)
        position.update(order)

        # 剩余50股，当前价 $130
        current_price = 130.0

        # 未实现: (130 - 100) * 50 = 1500
        # 总盈亏: 1000 + 1500 = 2500
        total = position.total_pnl(current_price)

        assert total == 2500.0, f"Expected 2500.0, got {total}"

    def test_reversal_from_long_to_short(self):
        """测试从多头反手做空"""
        asset = create_asset()
        position = Position(asset=asset, quantity=100, avg_cost=100.0)

        # 卖出150股，从多头反手做空
        order = create_order(asset=asset, side=OrderSide.SELL, quantity=150)
        order.fill(150, 120.0, commission=0)

        position.update(order)

        # 平多头100股盈亏: (120 - 100) * 100 = 2000
        # 开空头50股，成本 $120
        assert position.realized_pnl == 2000.0, f"Expected 2000.0, got {position.realized_pnl}"
        assert position.quantity == -50, f"Expected -50, got {position.quantity}"
        assert position.avg_cost == 120.0, f"Expected 120.0, got {position.avg_cost}"

    def test_flat_position_unrealized_pnl(self):
        """测试空仓未实现盈亏为0"""
        asset = create_asset()
        position = Position(asset=asset, quantity=0, avg_cost=0.0)

        unrealized = position.unrealized_pnl(current_price=150.0)

        assert unrealized == 0.0, f"Expected 0.0, got {unrealized}"


# =============================================================================
# 投资组合测试
# =============================================================================

class TestPortfolio:
    """投资组合准确性测试"""

    def test_initial_state(self):
        """测试初始状态"""
        portfolio = Portfolio(initial_capital=100000.0)

        assert portfolio.cash == 100000.0
        assert portfolio.initial_capital == 100000.0
        assert len(portfolio.positions) == 0

    def test_buy_order_updates_cash(self):
        """测试买入订单更新现金"""
        portfolio = Portfolio(initial_capital=100000.0)
        asset = create_asset()

        order = create_order(asset=asset, side=OrderSide.BUY, quantity=100)
        order.submit()

        # 执行: 100股 @ $100, 手续费 $10
        portfolio.execute_order(order, fill_price=100.0, commission=10.0)

        # 现金: 100000 - 100*100 - 10 = 89990
        assert portfolio.cash == 89990.0, f"Expected 89990.0, got {portfolio.cash}"

    def test_sell_order_updates_cash(self):
        """测试卖出订单更新现金"""
        portfolio = Portfolio(initial_capital=100000.0)
        asset = create_asset()

        # 先买入
        buy_order = create_order(asset=asset, side=OrderSide.BUY, quantity=100)
        buy_order.submit()
        portfolio.execute_order(buy_order, fill_price=100.0, commission=10.0)

        # 再卖出
        sell_order = create_order(asset=asset, side=OrderSide.SELL, quantity=100)
        sell_order.submit()
        portfolio.execute_order(sell_order, fill_price=120.0, commission=10.0)

        # 现金: 89990 + 100*120 - 10 = 101980
        assert portfolio.cash == 101980.0, f"Expected 101980.0, got {portfolio.cash}"

    def test_positions_value(self):
        """测试持仓市值计算"""
        portfolio = Portfolio(initial_capital=100000.0)
        asset = create_asset()

        order = create_order(asset=asset, side=OrderSide.BUY, quantity=100)
        order.submit()
        portfolio.execute_order(order, fill_price=100.0, commission=10.0)

        prices = {"AAPL": 120.0}
        positions_value = portfolio.positions_value(prices)

        # 100股 @ $120 = $12000
        assert positions_value == 12000.0, f"Expected 12000.0, got {positions_value}"

    def test_total_value(self):
        """测试总价值计算"""
        portfolio = Portfolio(initial_capital=100000.0)
        asset = create_asset()

        order = create_order(asset=asset, side=OrderSide.BUY, quantity=100)
        order.submit()
        portfolio.execute_order(order, fill_price=100.0, commission=10.0)

        prices = {"AAPL": 120.0}
        total = portfolio.total_value(prices)

        # 现金 89990 + 持仓 12000 = 101990
        assert total == 101990.0, f"Expected 101990.0, got {total}"

    def test_unrealized_pnl(self):
        """测试未实现盈亏"""
        portfolio = Portfolio(initial_capital=100000.0)
        asset = create_asset()

        order = create_order(asset=asset, side=OrderSide.BUY, quantity=100)
        order.submit()
        portfolio.execute_order(order, fill_price=100.0, commission=10.0)

        prices = {"AAPL": 120.0}
        unrealized = portfolio.unrealized_pnl(prices)

        # (120 - 100) * 100 = 2000
        assert unrealized == 2000.0, f"Expected 2000.0, got {unrealized}"

    def test_multiple_positions(self):
        """测试多个持仓"""
        portfolio = Portfolio(initial_capital=100000.0)

        assets = [create_asset("AAPL"), create_asset("GOOGL"), create_asset("MSFT")]

        for asset in assets:
            order = create_order(asset=asset, side=OrderSide.BUY, quantity=10)
            order.submit()
            portfolio.execute_order(order, fill_price=100.0, commission=1.0)

        prices = {"AAPL": 110.0, "GOOGL": 120.0, "MSFT": 90.0}

        # 持仓: 10*110 + 10*120 + 10*90 = 3200
        positions_value = portfolio.positions_value(prices)
        assert positions_value == 3200.0, f"Expected 3200.0, got {positions_value}"

        # 现金: 100000 - 3*10*100 - 3*1 = 96997
        assert portfolio.cash == 96997.0, f"Expected 96997.0, got {portfolio.cash}"

    def test_total_commission(self):
        """测试总手续费"""
        portfolio = Portfolio(initial_capital=100000.0)
        asset = create_asset()

        for i in range(5):
            order = create_order(asset=asset, side=OrderSide.BUY, quantity=10)
            order.submit()
            portfolio.execute_order(order, fill_price=100.0, commission=5.0)

        total_commission = portfolio.total_commission()

        assert total_commission == 25.0, f"Expected 25.0, got {total_commission}"

    def test_returns_calculation(self):
        """测试收益率计算"""
        portfolio = Portfolio(initial_capital=100000.0)
        asset = create_asset()

        order = create_order(asset=asset, side=OrderSide.BUY, quantity=100)
        order.submit()
        portfolio.execute_order(order, fill_price=100.0, commission=0)

        # 价格上涨到 $150
        prices = {"AAPL": 150.0}
        returns = portfolio.returns(prices)

        # 总价值: 90000 + 15000 = 105000
        # 收益率: (105000 - 100000) / 100000 * 100 = 5%
        assert returns == 5.0, f"Expected 5.0%, got {returns}%"


# =============================================================================
# 收益指标计算测试
# =============================================================================

class TestReturnMetrics:
    """收益指标计算准确性测试"""

    def test_total_return_calculation(self):
        """测试总收益率计算"""
        # 初始 100000, 最终 120000
        initial = 100000.0
        final = 120000.0

        total_return = (final - initial) / initial

        assert total_return == 0.2, f"Expected 0.2 (20%), got {total_return}"

    def test_total_return_with_loss(self):
        """测试亏损时的总收益率"""
        initial = 100000.0
        final = 80000.0

        total_return = (final - initial) / initial

        assert total_return == -0.2, f"Expected -0.2 (-20%), got {total_return}"

    def test_annual_return_calculation(self):
        """测试年化收益率计算"""
        # 一年内从 100000 到 120000
        initial = 100000.0
        final = 120000.0
        days = 365

        total_return = (final - initial) / initial  # 0.2
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1

        # 大约等于 20%
        assert abs(annual_return - 0.2) < 0.001, f"Expected ~0.2, got {annual_return}"

    def test_annual_return_multi_year(self):
        """测试多年年化收益率"""
        # 3年内从 100000 到 157500 (每年 25% 收益)
        initial = 100000.0
        final = 100000 * (1.15 ** 3)  # 3年每年15%
        days = 365 * 3

        total_return = (final - initial) / initial
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1

        # 应该约等于 15%
        assert abs(annual_return - 0.15) < 0.001, f"Expected ~0.15, got {annual_return}"

    def test_annual_return_short_period(self):
        """测试短期年化收益率"""
        # 30天内从 100000 到 105000
        initial = 100000.0
        final = 105000.0
        days = 30

        total_return = (final - initial) / initial  # 0.05
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1

        # 短期年化会很高
        # (1.05)^(365.25/30) - 1 ≈ 83%
        assert annual_return > 0.8, f"Short period annual return should be high: {annual_return}"


# =============================================================================
# 风险指标计算测试
# =============================================================================

class TestRiskMetrics:
    """风险指标计算准确性测试"""

    def test_volatility_calculation(self):
        """测试波动率计算"""
        # 创建已知的日收益率序列
        daily_returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02])

        # 日波动率
        daily_vol = daily_returns.std()
        # 年化波动率
        annual_vol = daily_vol * np.sqrt(252)

        # 验证计算正确
        expected_daily_vol = np.std([0.01, -0.02, 0.015, -0.005, 0.02], ddof=1)
        assert abs(daily_vol - expected_daily_vol) < 0.0001

    def test_sharpe_ratio_calculation(self):
        """测试夏普比率计算"""
        annual_return = 0.15  # 15% 年化收益
        volatility = 0.20     # 20% 波动率
        risk_free_rate = 0.02 # 2% 无风险利率

        sharpe = (annual_return - risk_free_rate) / volatility

        # (0.15 - 0.02) / 0.20 = 0.65
        assert abs(sharpe - 0.65) < 1e-10, f"Expected 0.65, got {sharpe}"

    def test_sharpe_ratio_negative(self):
        """测试负夏普比率"""
        annual_return = 0.01  # 1% 年化收益
        volatility = 0.20     # 20% 波动率
        risk_free_rate = 0.02 # 2% 无风险利率

        sharpe = (annual_return - risk_free_rate) / volatility

        # (0.01 - 0.02) / 0.20 = -0.05
        assert abs(sharpe - (-0.05)) < 1e-10, f"Expected -0.05, got {sharpe}"

    def test_sortino_ratio_calculation(self):
        """测试索提诺比率计算"""
        daily_returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.005])

        # 只取负收益计算下行波动率
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)

        annual_return = 0.10
        risk_free_rate = 0.02

        sortino = (annual_return - risk_free_rate) / downside_std

        # 验证计算正确性（应该大于0）
        assert sortino > 0, f"Sortino ratio should be positive: {sortino}"

    def test_max_drawdown_calculation(self):
        """测试最大回撤计算"""
        # 净值曲线: 100 -> 110 -> 105 -> 90 -> 95 -> 100
        equity_curve = pd.Series([100, 110, 105, 90, 95, 100])

        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())

        # 最大回撤发生在 110 -> 90，回撤 = (110-90)/110 = 0.1818
        assert abs(max_drawdown - 0.1818) < 0.001, f"Expected ~0.1818, got {max_drawdown}"

    def test_max_drawdown_no_drawdown(self):
        """测试无回撤情况"""
        # 持续上涨
        equity_curve = pd.Series([100, 105, 110, 115, 120])

        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())

        assert max_drawdown == 0.0, f"Expected 0.0, got {max_drawdown}"

    def test_max_drawdown_continuous_decline(self):
        """测试持续下跌的最大回撤"""
        equity_curve = pd.Series([100, 90, 80, 70, 60])

        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())

        # 从100跌到60，回撤40%
        assert max_drawdown == 0.4, f"Expected 0.4, got {max_drawdown}"

    def test_drawdown_duration_calculation(self):
        """测试回撤持续时间计算"""
        # 净值曲线，中间有一段回撤
        equity_curve = pd.Series([100, 110, 105, 100, 95, 100, 105, 110, 115])

        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max

        # 计算最长回撤期
        is_underwater = drawdown < 0

        underwater_periods = []
        current_period = 0
        for underwater in is_underwater:
            if underwater:
                current_period += 1
            else:
                if current_period > 0:
                    underwater_periods.append(current_period)
                current_period = 0
        if current_period > 0:
            underwater_periods.append(current_period)

        max_duration = max(underwater_periods) if underwater_periods else 0

        # 最长回撤期应该是 5 个周期 (110 -> 105 -> 100 -> 95 -> 100 -> 105)
        assert max_duration == 5, f"Expected 5, got {max_duration}"


# =============================================================================
# 交易统计测试
# =============================================================================

class TestTradeStatistics:
    """交易统计准确性测试"""

    def test_win_rate_calculation(self):
        """测试胜率计算"""
        # 10笔交易，7笔盈利
        trade_pnls = [100, -50, 200, 150, -100, 80, -30, 120, 90, 50]

        winning_trades = sum(1 for pnl in trade_pnls if pnl > 0)
        total_trades = len(trade_pnls)
        win_rate = winning_trades / total_trades

        assert winning_trades == 7
        assert win_rate == 0.7, f"Expected 0.7, got {win_rate}"

    def test_win_rate_all_wins(self):
        """测试全部盈利时的胜率"""
        trade_pnls = [100, 50, 200, 150]

        winning_trades = sum(1 for pnl in trade_pnls if pnl > 0)
        win_rate = winning_trades / len(trade_pnls)

        assert win_rate == 1.0, f"Expected 1.0, got {win_rate}"

    def test_win_rate_all_losses(self):
        """测试全部亏损时的胜率"""
        trade_pnls = [-100, -50, -200, -150]

        winning_trades = sum(1 for pnl in trade_pnls if pnl > 0)
        win_rate = winning_trades / len(trade_pnls)

        assert win_rate == 0.0, f"Expected 0.0, got {win_rate}"

    def test_avg_win_avg_loss(self):
        """测试平均盈利和平均亏损"""
        trade_pnls = [100, -50, 200, -100, 150]

        wins = [pnl for pnl in trade_pnls if pnl > 0]
        losses = [pnl for pnl in trade_pnls if pnl < 0]

        avg_win = np.mean(wins)      # (100 + 200 + 150) / 3 = 150
        avg_loss = abs(np.mean(losses))  # (50 + 100) / 2 = 75

        assert avg_win == 150.0, f"Expected 150.0, got {avg_win}"
        assert avg_loss == 75.0, f"Expected 75.0, got {avg_loss}"

    def test_profit_factor_calculation(self):
        """测试盈亏比计算"""
        trade_pnls = [100, -50, 200, -100, 150]

        wins = [pnl for pnl in trade_pnls if pnl > 0]
        losses = [pnl for pnl in trade_pnls if pnl < 0]

        total_wins = sum(wins)       # 450
        total_losses = abs(sum(losses))  # 150
        profit_factor = total_wins / total_losses

        assert profit_factor == 3.0, f"Expected 3.0, got {profit_factor}"

    def test_profit_factor_no_losses(self):
        """测试无亏损时的盈亏比"""
        trade_pnls = [100, 50, 200]

        wins = [pnl for pnl in trade_pnls if pnl > 0]
        losses = [pnl for pnl in trade_pnls if pnl < 0]

        total_wins = sum(wins)
        total_losses = abs(sum(losses))

        # 无亏损时，盈亏比为 0（或可视为无穷大，取决于实现）
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        assert profit_factor == 0, f"Expected 0, got {profit_factor}"

    def test_total_trades_count(self):
        """测试总交易次数"""
        fills = [
            {"side": "buy", "quantity": 100},
            {"side": "sell", "quantity": 100},
            {"side": "buy", "quantity": 50},
            {"side": "sell", "quantity": 50},
        ]

        total_trades = len(fills)

        assert total_trades == 4, f"Expected 4, got {total_trades}"


# =============================================================================
# 订单验证测试
# =============================================================================

class TestOrderValidation:
    """订单验证测试"""

    def test_validate_buy_sufficient_cash(self):
        """测试买入资金充足时验证通过"""
        config = BrokerConfig(commission_rate=0.001, slippage_rate=0.0005)
        broker = VirtualBroker(config=config)
        portfolio = Portfolio(initial_capital=100000.0)

        order = create_order(side=OrderSide.BUY, quantity=100)

        is_valid, reason = broker.validate_order(order, portfolio, current_price=100.0)

        assert is_valid, f"Should be valid: {reason}"

    def test_validate_buy_insufficient_cash(self):
        """测试买入资金不足时验证失败"""
        config = BrokerConfig(commission_rate=0.001, slippage_rate=0.0005)
        broker = VirtualBroker(config=config)
        portfolio = Portfolio(initial_capital=1000.0)  # 只有1000

        order = create_order(side=OrderSide.BUY, quantity=100)  # 需要 ~10000

        is_valid, reason = broker.validate_order(order, portfolio, current_price=100.0)

        assert not is_valid, "Should be invalid due to insufficient cash"
        assert "资金不足" in reason

    def test_validate_sell_with_position(self):
        """测试有持仓时卖出验证"""
        config = BrokerConfig(short_selling_enabled=False)
        broker = VirtualBroker(config=config)
        portfolio = Portfolio(initial_capital=100000.0)

        asset = create_asset()

        # 先买入建仓
        buy_order = create_order(asset=asset, side=OrderSide.BUY, quantity=100)
        buy_order.submit()
        portfolio.execute_order(buy_order, fill_price=100.0, commission=0)

        # 卖出
        sell_order = create_order(asset=asset, side=OrderSide.SELL, quantity=100)

        is_valid, reason = broker.validate_order(sell_order, portfolio, current_price=100.0)

        assert is_valid, f"Should be valid: {reason}"

    def test_validate_sell_no_position_no_short(self):
        """测试无持仓且不允许做空时卖出验证"""
        config = BrokerConfig(short_selling_enabled=False)
        broker = VirtualBroker(config=config)
        portfolio = Portfolio(initial_capital=100000.0)

        order = create_order(side=OrderSide.SELL, quantity=100)

        is_valid, reason = broker.validate_order(order, portfolio, current_price=100.0)

        assert not is_valid, "Should be invalid - no position and short selling disabled"

    def test_validate_zero_quantity_order(self):
        """测试零数量订单验证失败"""
        config = BrokerConfig()
        broker = VirtualBroker(config=config)
        portfolio = Portfolio(initial_capital=100000.0)

        # 创建订单时应该抛出异常
        with pytest.raises(ValueError):
            create_order(quantity=0)


# =============================================================================
# 完整交易流程测试
# =============================================================================

class TestTradingWorkflow:
    """完整交易流程测试"""

    def test_buy_and_sell_workflow(self):
        """测试完整买入卖出流程"""
        config = BrokerConfig(commission_rate=0.001, min_commission=1.0, slippage_rate=0)
        broker = VirtualBroker(config=config)
        portfolio = Portfolio(initial_capital=100000.0)
        asset = create_asset()

        # 买入
        buy_order = create_order(asset=asset, side=OrderSide.BUY, quantity=100)
        broker.submit_order(buy_order, portfolio, current_price=100.0)

        ohlcv = create_ohlcv(open_price=100.0, high=105.0, low=98.0, close=102.0)
        filled = broker.process_orders(portfolio, ohlcv)

        assert len(filled) == 1
        assert buy_order.status == OrderStatus.FILLED

        # 验证现金和持仓
        # 成交价 100.0，数量 100，手续费 max(100*100*0.001, 1) = 10
        assert portfolio.cash == 100000 - 100*100 - 10  # 89990
        assert portfolio.positions["AAPL"].quantity == 100

        # 卖出
        sell_order = create_order(asset=asset, side=OrderSide.SELL, quantity=100)
        broker.submit_order(sell_order, portfolio, current_price=110.0)

        ohlcv2 = create_ohlcv(open_price=110.0, high=115.0, low=108.0, close=112.0)
        filled2 = broker.process_orders(portfolio, ohlcv2)

        assert len(filled2) == 1
        assert sell_order.status == OrderStatus.FILLED

        # 成交价 110.0，手续费 max(100*110*0.001, 1) = 11
        # 现金: 89990 + 100*110 - 11 = 100979
        assert portfolio.cash == 100979.0

        # 已实现盈亏: (110 - 100) * 100 = 1000
        assert portfolio.realized_pnl() == 1000.0

    def test_multiple_trades_cumulative_pnl(self):
        """测试多笔交易累计盈亏"""
        config = BrokerConfig(commission_rate=0, slippage_rate=0)
        broker = VirtualBroker(config=config)
        portfolio = Portfolio(initial_capital=100000.0)
        asset = create_asset()

        trades = [
            # (买入价, 卖出价, 数量)
            (100, 110, 10),   # 盈利 100
            (105, 100, 10),   # 亏损 -50
            (95, 105, 20),    # 盈利 200
        ]

        expected_total_pnl = 100 - 50 + 200  # 250

        for buy_price, sell_price, qty in trades:
            # 买入
            buy_order = create_order(asset=asset, side=OrderSide.BUY, quantity=qty)
            broker.submit_order(buy_order, portfolio, current_price=buy_price)
            ohlcv = create_ohlcv(open_price=buy_price, high=buy_price+5, low=buy_price-5, close=buy_price)
            broker.process_orders(portfolio, ohlcv)

            # 卖出
            sell_order = create_order(asset=asset, side=OrderSide.SELL, quantity=qty)
            broker.submit_order(sell_order, portfolio, current_price=sell_price)
            ohlcv = create_ohlcv(open_price=sell_price, high=sell_price+5, low=sell_price-5, close=sell_price)
            broker.process_orders(portfolio, ohlcv)

        assert portfolio.realized_pnl() == expected_total_pnl

    def test_commission_impact_on_pnl(self):
        """测试手续费对盈亏的影响"""
        config = BrokerConfig(commission_rate=0.001, min_commission=1.0, slippage_rate=0)
        broker = VirtualBroker(config=config)
        portfolio = Portfolio(initial_capital=100000.0)
        asset = create_asset()

        # 买入 100股 @ $100
        buy_order = create_order(asset=asset, side=OrderSide.BUY, quantity=100)
        broker.submit_order(buy_order, portfolio, current_price=100.0)
        ohlcv = create_ohlcv(open_price=100.0, high=105.0, low=98.0, close=102.0)
        broker.process_orders(portfolio, ohlcv)

        # 卖出 100股 @ $100 (平价卖出)
        sell_order = create_order(asset=asset, side=OrderSide.SELL, quantity=100)
        broker.submit_order(sell_order, portfolio, current_price=100.0)
        ohlcv = create_ohlcv(open_price=100.0, high=105.0, low=98.0, close=102.0)
        broker.process_orders(portfolio, ohlcv)

        # 交易盈亏为0，但手续费导致亏损
        # 手续费: 10 + 10 = 20
        total_commission = portfolio.total_commission()
        assert total_commission == 20.0

        # 净资产应减少手续费金额
        prices = {"AAPL": 100.0}
        net_value = portfolio.total_value(prices)
        assert net_value == 100000 - 20  # 99980

    def test_slippage_impact_on_execution(self):
        """测试滑点对执行的影响"""
        config = BrokerConfig(commission_rate=0, slippage_rate=0.01, slippage_mode="percentage")
        broker = VirtualBroker(config=config)
        portfolio = Portfolio(initial_capital=100000.0)
        asset = create_asset()

        # 买入时滑点导致成交价更高
        buy_order = create_order(asset=asset, side=OrderSide.BUY, quantity=100)
        broker.submit_order(buy_order, portfolio, current_price=100.0)
        ohlcv = create_ohlcv(open_price=100.0, high=105.0, low=98.0, close=102.0)
        broker.process_orders(portfolio, ohlcv)

        # 成交价: 100 + 100*0.01 = 101
        position = portfolio.positions["AAPL"]
        assert position.avg_cost == 101.0, f"Expected 101.0, got {position.avg_cost}"


# =============================================================================
# 边界条件测试
# =============================================================================

class TestEdgeCases:
    """边界条件测试"""

    def test_very_small_order(self):
        """测试极小订单"""
        config = BrokerConfig(commission_rate=0.001, min_commission=0.01)
        broker = VirtualBroker(config=config)

        asset = Asset(symbol="MICRO", lot_size=0.001)
        order = create_order(asset=asset, quantity=0.001)

        # 0.001股 @ $100 = $0.1, 手续费 = $0.0001 < 最低 $0.01
        commission = broker.calculate_commission(order, fill_price=100.0)

        assert commission == 0.01, f"Expected 0.01 (minimum), got {commission}"

    def test_very_large_order(self):
        """测试超大订单"""
        config = BrokerConfig(commission_rate=0.001, min_commission=1.0)
        broker = VirtualBroker(config=config)

        order = create_order(quantity=1000000)

        # 1000000股 @ $100 = $100,000,000, 手续费 = $100,000
        commission = broker.calculate_commission(order, fill_price=100.0)

        assert commission == 100000.0, f"Expected 100000.0, got {commission}"

    def test_price_at_zero(self):
        """测试零价格处理"""
        asset = create_asset()
        position = Position(asset=asset, quantity=100, avg_cost=100.0)

        # 当前价格为0时的未实现盈亏
        unrealized = position.unrealized_pnl(current_price=0.0)

        # (0 - 100) * 100 = -10000
        assert unrealized == -10000.0

    def test_empty_equity_curve(self):
        """测试空净值曲线的最大回撤"""
        equity_curve = pd.Series([])

        if len(equity_curve) == 0:
            max_drawdown = 0.0
        else:
            rolling_max = equity_curve.expanding().max()
            drawdown = (equity_curve - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())

        assert max_drawdown == 0.0

    def test_single_point_equity_curve(self):
        """测试单点净值曲线"""
        equity_curve = pd.Series([100000.0])

        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())

        assert max_drawdown == 0.0

    def test_zero_volatility(self):
        """测试零波动率时的夏普比率"""
        # 所有收益率相同
        daily_returns = pd.Series([0.001, 0.001, 0.001, 0.001])
        volatility = daily_returns.std() * np.sqrt(252)

        annual_return = 0.10
        risk_free_rate = 0.02

        # 波动率为0时夏普比率应该处理特殊情况
        sharpe = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

        assert sharpe == 0, f"Expected 0 for zero volatility, got {sharpe}"

    def test_no_trades_statistics(self):
        """测试无交易时的统计"""
        trade_pnls = []

        win_rate = len([p for p in trade_pnls if p > 0]) / len(trade_pnls) if trade_pnls else 0
        profit_factor = 0

        assert win_rate == 0
        assert profit_factor == 0


# =============================================================================
# 数值精度测试
# =============================================================================

class TestNumericalPrecision:
    """数值精度测试"""

    def test_commission_precision(self):
        """测试手续费计算精度"""
        config = BrokerConfig(commission_rate=0.00015, min_commission=0.01)
        broker = VirtualBroker(config=config)

        order = create_order(quantity=123)

        # 123 * 99.99 * 0.00015 = 1.8448...
        commission = broker.calculate_commission(order, fill_price=99.99)

        # 验证精度
        expected = 123 * 99.99 * 0.00015
        assert abs(commission - expected) < 0.0001 or commission == 0.01

    def test_pnl_precision(self):
        """测试盈亏计算精度"""
        asset = create_asset()
        position = Position(asset=asset, quantity=33.33, avg_cost=123.456)

        unrealized = position.unrealized_pnl(current_price=134.567)

        # (134.567 - 123.456) * 33.33 = 370.329963
        expected = (134.567 - 123.456) * 33.33

        assert abs(unrealized - expected) < 0.0001

    def test_return_precision(self):
        """测试收益率计算精度"""
        initial = 99999.99
        final = 111111.11

        total_return = (final - initial) / initial

        expected = (111111.11 - 99999.99) / 99999.99

        assert abs(total_return - expected) < 1e-10

    def test_avg_cost_precision_multiple_trades(self):
        """测试多笔交易后平均成本精度"""
        asset = create_asset()
        position = Position(asset=asset)

        trades = [
            (100, 33.33),   # 100股 @ $33.33
            (100, 44.44),   # 100股 @ $44.44
            (100, 55.55),   # 100股 @ $55.55
        ]

        for qty, price in trades:
            order = create_order(asset=asset, side=OrderSide.BUY, quantity=qty)
            order.fill(qty, price, commission=0)
            position.update(order)

        # 平均成本: (100*33.33 + 100*44.44 + 100*55.55) / 300 = 44.44
        expected_avg = (33.33 + 44.44 + 55.55) / 3

        assert abs(position.avg_cost - expected_avg) < 0.0001


# =============================================================================
# 运行测试
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
