# 模拟交易系统

本模块实现了事件驱动的交易模拟引擎，用于策略回测和模拟交易。

## 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| SimulationEngine | `engine.py` | 事件驱动引擎，协调数据、策略、订单和组合 |
| Backtester | `backtest.py` | 回测器，运行历史回测并计算绩效指标 |
| VirtualBroker | `broker.py` | 虚拟券商，处理订单执行、手续费、滑点 |

## Portfolio 资金分配机制

### 概述

当运行多标的回测时（如 `--symbol AAPL,MSFT,GOOGL`），系统使用**单一共享资金池**模式：

```
┌─────────────────────────────────────────────────┐
│              Portfolio (100,000)                │
│                                                 │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│   │  AAPL   │  │  MSFT   │  │  GOOGL  │  Cash   │
│   │ Position│  │ Position│  │ Position│         │
│   └─────────┘  └─────────┘  └─────────┘         │
│                                                 │
│   资金按信号触发时动态分配，非预先均分                │
└─────────────────────────────────────────────────┘
```

### 默认仓位计算器

资金分配由 `SimulationEngine._default_position_sizer` 控制：

```python
def _default_position_sizer(self, signal, portfolio, current_price):
    # 获取组合净值
    net_value = portfolio.total_value(self.current_prices)

    # 使用初始资本和当前净值的较小值，防止杠杆过大
    base_value = min(net_value, portfolio.initial_capital)

    # 每笔交易分配 10% × 信号强度
    position_value = base_value * 0.1 * signal.strength

    # 计算股数
    quantity = position_value / current_price
    return max(1, int(quantity))
```

### 分配规则

| 规则 | 说明 |
|------|------|
| 分配时机 | 策略产生信号时才分配，不预先划分 |
| 单笔交易规模 | `净值 × 10% × signal.strength` |
| 信号强度范围 | 0.0 ~ 1.0 |
| 实际分配比例 | 每笔交易约 0% ~ 10% |

### 示例场景

假设初始资金 100,000，运行 3 个标的的均值回归策略：

**Day 1**: AAPL 触发买入信号（strength=0.8）
```
分配金额 = 100,000 × 10% × 0.8 = $8,000
买入 AAPL 股数 = 8,000 / 股价
剩余现金 = $92,000
```

**Day 2**: MSFT 触发买入信号（strength=0.6）
```
当前净值 ≈ $100,500（假设 AAPL 小幅上涨）
分配金额 = 100,000 × 10% × 0.6 = $6,000  # 使用 min(净值, 初始资金)
买入 MSFT 股数 = 6,000 / 股价
剩余现金 ≈ $86,000
```

**Day 5**: AAPL 再次触发买入信号（strength=0.9）
```
分配金额 = 100,000 × 10% × 0.9 = $9,000
继续买入 AAPL
```

### 关键特点

1. **非等权分配**：资金不会预先平分给各标的
2. **按需分配**：哪个标的产生信号就分配给哪个
3. **可累积持仓**：同一标的可多次买入，累积仓位
4. **无平衡机制**：没有自动重平衡，可能导致单一标的占比过高

### 信号强度计算

不同策略的信号强度计算方式：

**Mean Reversion（均值回归）**
```python
# 信号强度 = 价格在布林带中的偏离程度
signal_strength = abs(band_position)  # band_position ∈ [-1, 1]
```
- 价格越接近布林带边缘，强度越高

**SMA Crossover（均线交叉）**
```python
# 信号强度 = 快慢均线差距相对于价格的比例
signal_strength = min(abs(fast_sma - slow_sma) / slow_sma * 100, 1.0)
```
- 均线差距越大，强度越高

## 自定义仓位计算器

如需自定义资金分配逻辑，可在创建 `SimulationEngine` 时传入自定义函数：

```python
def equal_weight_sizer(signal, portfolio, current_price):
    """等权重分配：每个标的最多使用 1/N 的资金"""
    num_symbols = 3  # 标的数量
    max_per_symbol = portfolio.initial_capital / num_symbols

    # 检查该标的当前持仓市值
    position = portfolio.positions.get(signal.asset.symbol)
    current_holding = 0
    if position:
        current_holding = abs(position.quantity) * current_price

    # 计算可用额度
    available = max_per_symbol - current_holding
    if available <= 0:
        return 0

    # 分配可用额度的一定比例
    position_value = available * 0.5 * signal.strength
    return max(1, int(position_value / current_price))

# 使用自定义仓位计算器
engine = SimulationEngine(
    portfolio=portfolio,
    broker=broker,
    strategies=[strategy],
    position_sizer=equal_weight_sizer,  # 传入自定义函数
)
```

## 资金流转流程

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  MarketData  │────▶│   Strategy   │────▶│    Signal    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Position   │◀────│    Broker    │◀────│    Order     │
│   Updated    │     │   Execute    │     │  (quantity)  │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Commission  │
                     │  + Slippage  │
                     └──────────────┘
```

## 相关配置

### BrokerConfig

```python
@dataclass
class BrokerConfig:
    commission_rate: float = 0.001      # 手续费率 0.1%
    min_commission: float = 1.0         # 最低手续费
    slippage_rate: float = 0.0005       # 滑点率 0.05%
    slippage_mode: str = "percentage"   # 滑点模式
    partial_fill_enabled: bool = False  # 是否允许部分成交
    short_selling_enabled: bool = True  # 是否允许做空
    margin_rate: float = 0.5            # 保证金率（做空时）
```

## 滑点 (Slippage) 详解

### 什么是滑点？

**滑点**是指订单的**预期成交价格**与**实际成交价格**之间的差异。在真实交易中，这种差异几乎总是存在的。

```
预期买入价格: $150.00
实际成交价格: $150.08  ← 多付了 $0.08（滑点）
```

### 滑点产生的原因

| 原因 | 说明 |
|------|------|
| **市场冲击** | 大额订单会影响市场价格，买入推高价格，卖出压低价格 |
| **订单簿深度** | 当前价位的挂单量不足以满足订单，需要吃掉更深的价位 |
| **网络延迟** | 下单到执行之间价格可能已经变化 |
| **流动性不足** | 交易量小的股票滑点更大 |

### 滑点计算逻辑

```python
def calculate_slippage(self, order, market_price):
    if self.config.slippage_mode == "percentage":
        slippage = market_price * self.config.slippage_rate  # 默认 0.05%

    # 买入时价格上滑（付更多钱），卖出时价格下滑（收更少钱）
    if order.side == OrderSide.BUY:
        return market_price + slippage   # 150.00 → 150.075
    else:
        return market_price - slippage   # 150.00 → 149.925
```

**核心规则**：
- **买入** → 价格上滑（你付更多钱）
- **卖出** → 价格下滑（你收更少钱）

滑点总是对交易者不利的。

### 具体示例

假设 AAPL 当前价格 $150.00，滑点率 0.05%：

| 操作 | 预期价格 | 滑点 | 实际成交价 | 影响 |
|------|---------|------|-----------|------|
| 买入 100 股 | $150.00 | +$0.075 | $150.075 | 多付 $7.50 |
| 卖出 100 股 | $150.00 | -$0.075 | $149.925 | 少收 $7.50 |

一买一卖的完整交易，滑点成本 = $7.50 × 2 = **$15.00**

### 滑点模式

系统支持三种滑点计算模式：

| 模式 | 计算方式 | 适用场景 |
|------|---------|---------|
| `percentage` | `价格 × 滑点率` | 默认模式，适合大多数股票 |
| `fixed` | 固定金额 | 特定市场（如期货每跳固定价值） |
| `random` | `价格 × 滑点率 × random()` | 模拟真实市场的不确定性 |

### 为什么回测需要模拟滑点？

如果不考虑滑点，回测结果会**过于乐观**：

```
┌─────────────────────────────────────────┐
│ 同一策略在不同滑点假设下的表现：             │
│                                         │
│ 不考虑滑点:     +15.2%  ← 过于理想         │
│ 0.05% 滑点:    +12.8%  ← 更接近现实       │
│ 0.10% 滑点:    +10.5%                    │
│ 0.20% 滑点:     +6.1%  ← 高频策略杀手      │
└─────────────────────────────────────────┘
```

**特别注意**：高频交易策略对滑点极其敏感，滑点可能完全吞噬利润。

### 滑点与手续费的区别

| 项目 | 滑点 | 手续费 |
|------|------|--------|
| 性质 | 价格偏移 | 固定费用 |
| 可预测性 | 不确定 | 确定 |
| 影响方式 | 改变成交价格 | 从账户扣除 |
| 计算基础 | 成交价格 | 成交金额 |

### 如何降低滑点影响

1. **选择流动性好的标的**：大市值股票滑点更小
2. **避免在开盘/收盘时交易**：这些时段波动大
3. **使用限价单**：控制最大可接受价格
4. **分批建仓**：大额订单分多次执行
5. **避免高频策略**：交易次数越多，滑点累积越大
