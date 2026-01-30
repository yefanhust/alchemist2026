"""
回测系统模块
提供策略历史回测功能
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
import numpy as np
import pandas as pd
from loguru import logger

from ..core.asset import Asset
from ..core.portfolio import Portfolio
from ..core.order import Order
from ..data.models import MarketData
from ..data.providers.base import DataProvider, DataInterval
from ..data.cache.sqlite_cache import SQLiteCache
from ..strategy.base import Strategy
from .engine import SimulationEngine
from .broker import VirtualBroker, BrokerConfig


@dataclass
class BacktestResult:
    """
    回测结果
    
    包含回测的完整绩效分析。
    """
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float
    
    # 收益指标
    total_return: float = 0.0           # 总收益率
    annual_return: float = 0.0          # 年化收益率
    
    # 风险指标
    volatility: float = 0.0             # 波动率
    sharpe_ratio: float = 0.0           # 夏普比率
    sortino_ratio: float = 0.0          # 索提诺比率
    max_drawdown: float = 0.0           # 最大回撤
    max_drawdown_duration: int = 0      # 最大回撤持续天数
    
    # 交易统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0          # 盈亏比
    
    # 费用
    total_commission: float = 0.0
    
    # 详细数据
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    
    def summary(self) -> str:
        """生成回测摘要报告"""
        return f"""
========== 回测报告 ==========
策略: {self.strategy_name}
回测期间: {self.start_date.strftime('%Y-%m-%d')} 至 {self.end_date.strftime('%Y-%m-%d')}

【收益概览】
初始资金: ${self.initial_capital:,.2f}
最终价值: ${self.final_value:,.2f}
总收益率: {self.total_return:.2%}
年化收益率: {self.annual_return:.2%}

【风险指标】
波动率: {self.volatility:.2%}
夏普比率: {self.sharpe_ratio:.2f}
索提诺比率: {self.sortino_ratio:.2f}
最大回撤: {self.max_drawdown:.2%}
最大回撤持续: {self.max_drawdown_duration} 天

【交易统计】
总交易次数: {self.total_trades}
盈利交易: {self.winning_trades} ({self.win_rate:.1%})
亏损交易: {self.losing_trades}
平均盈利: ${self.avg_win:,.2f}
平均亏损: ${self.avg_loss:,.2f}
盈亏比: {self.profit_factor:.2f}
总手续费: ${self.total_commission:,.2f}
==============================
        """
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "strategy_name": self.strategy_name,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "final_value": self.final_value,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_commission": self.total_commission,
        }


class Backtester:
    """
    回测器
    
    提供完整的策略回测功能。
    
    功能:
    - 历史数据获取（带缓存）
    - 策略运行
    - 绩效计算
    - 结果分析
    """
    
    def __init__(
        self,
        data_provider: DataProvider,
        cache: Optional[SQLiteCache] = None,
        broker_config: Optional[BrokerConfig] = None,
    ):
        """
        初始化回测器
        
        Args:
            data_provider: 数据提供者
            cache: 缓存后端
            broker_config: 券商配置
        """
        self.data_provider = data_provider
        self.cache = cache
        self.broker_config = broker_config or BrokerConfig()
        
        # 如果数据提供者没有缓存，设置缓存
        if self.cache and self.data_provider.cache is None:
            self.data_provider.cache = self.cache
    
    async def fetch_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: DataInterval = DataInterval.DAILY,
    ) -> Dict[str, MarketData]:
        """
        获取回测数据
        
        Args:
            symbols: 资产代码列表
            start_date: 开始日期
            end_date: 结束日期
            interval: 数据间隔
            
        Returns:
            市场数据字典
        """
        logger.info(f"获取 {len(symbols)} 个标的的历史数据...")
        
        data = await self.data_provider.get_multiple_historical_data(
            symbols, start_date, end_date, interval
        )
        
        # 记录数据状态
        for symbol, market_data in data.items():
            if market_data.is_empty:
                logger.warning(f"{symbol}: 无数据")
            else:
                logger.info(
                    f"{symbol}: {len(market_data)} 条数据, "
                    f"{market_data.start_date.strftime('%Y-%m-%d')} 至 "
                    f"{market_data.end_date.strftime('%Y-%m-%d')}"
                )
        
        return data
    
    async def run(
        self,
        strategy: Strategy,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0,
        interval: DataInterval = DataInterval.DAILY,
        show_progress: bool = True,
    ) -> BacktestResult:
        """
        运行回测
        
        Args:
            strategy: 交易策略
            symbols: 资产代码列表
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金
            interval: 数据间隔
            show_progress: 是否显示进度
            
        Returns:
            回测结果
        """
        logger.info(f"开始回测: {strategy.name}")
        logger.info(f"标的: {symbols}")
        logger.info(f"期间: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
        
        # 获取数据
        data = await self.fetch_data(symbols, start_date, end_date, interval)
        
        # 检查数据有效性
        valid_symbols = [s for s, d in data.items() if not d.is_empty]
        if not valid_symbols:
            raise ValueError("没有有效的市场数据")
        
        # 创建组件
        portfolio = Portfolio(initial_capital=initial_capital, name=strategy.name)
        broker = VirtualBroker(config=self.broker_config)
        engine = SimulationEngine(
            portfolio=portfolio,
            broker=broker,
            strategies=[strategy],
        )
        
        # 进度显示
        total_days = (end_date - start_date).days
        last_progress = 0
        
        def on_step(timestamp: datetime):
            nonlocal last_progress
            if show_progress:
                progress = (timestamp - start_date).days
                if progress - last_progress >= total_days // 20:  # 每5%更新一次
                    pct = progress / total_days * 100
                    logger.info(f"回测进度: {pct:.0f}%")
                    last_progress = progress
        
        # 运行模拟
        engine.run({s: data[s] for s in valid_symbols}, on_step=on_step)
        
        # 计算结果
        result = self._calculate_result(
            strategy=strategy,
            portfolio=portfolio,
            broker=broker,
            start_date=start_date,
            end_date=end_date,
        )
        
        logger.info("回测完成")
        logger.info(f"总收益率: {result.total_return:.2%}")
        logger.info(f"夏普比率: {result.sharpe_ratio:.2f}")
        logger.info(f"最大回撤: {result.max_drawdown:.2%}")
        
        return result
    
    def _calculate_result(
        self,
        strategy: Strategy,
        portfolio: Portfolio,
        broker: VirtualBroker,
        start_date: datetime,
        end_date: datetime,
    ) -> BacktestResult:
        """
        计算回测结果
        
        Args:
            strategy: 策略
            portfolio: 投资组合
            broker: 券商
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            回测结果
        """
        # 获取净值曲线
        equity_df = portfolio.to_dataframe()
        
        if equity_df.empty:
            return BacktestResult(
                strategy_name=strategy.name,
                start_date=start_date,
                end_date=end_date,
                initial_capital=portfolio.initial_capital,
                final_value=portfolio.initial_capital,
            )
        
        # 计算收益
        initial_capital = portfolio.initial_capital
        final_value = equity_df["total_value"].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # 计算年化收益
        days = (end_date - start_date).days
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # 计算日收益率
        equity_df["daily_return"] = equity_df["total_value"].pct_change()
        daily_returns = equity_df["daily_return"].dropna()
        
        # 波动率
        volatility = daily_returns.std() * np.sqrt(252)
        
        # 夏普比率（假设无风险利率为2%）
        risk_free_rate = 0.02
        excess_return = annual_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # 索提诺比率（只考虑下行风险）
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
        
        # 最大回撤
        equity_curve = equity_df["total_value"]
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # 最大回撤持续时间
        max_drawdown_duration = self._calculate_max_drawdown_duration(drawdown)
        
        # 交易统计
        fills = broker.fill_history
        total_trades = len(fills)
        
        # 计算每笔交易的盈亏（简化版本）
        trade_pnls = []
        for i in range(0, len(fills) - 1, 2):  # 假设买卖配对
            if i + 1 < len(fills):
                buy_fill = fills[i]
                sell_fill = fills[i + 1]
                if buy_fill["side"] == "buy" and sell_fill["side"] == "sell":
                    pnl = (sell_fill["price"] - buy_fill["price"]) * buy_fill["quantity"]
                    pnl -= buy_fill["commission"] + sell_fill["commission"]
                    trade_pnls.append(pnl)
        
        winning_trades = sum(1 for pnl in trade_pnls if pnl > 0)
        losing_trades = sum(1 for pnl in trade_pnls if pnl < 0)
        win_rate = winning_trades / len(trade_pnls) if trade_pnls else 0
        
        wins = [pnl for pnl in trade_pnls if pnl > 0]
        losses = [pnl for pnl in trade_pnls if pnl < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # 月度收益
        monthly_returns = {}
        if not equity_df.empty:
            monthly = equity_df["total_value"].resample("M").last().pct_change()
            for date, ret in monthly.items():
                if pd.notna(ret):
                    monthly_returns[date.strftime("%Y-%m")] = ret
        
        return BacktestResult(
            strategy_name=strategy.name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_value=final_value,
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_commission=portfolio.total_commission(),
            equity_curve=equity_df,
            trade_history=fills,
            monthly_returns=monthly_returns,
        )
    
    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """计算最大回撤持续时间"""
        is_underwater = drawdown < 0
        
        if not is_underwater.any():
            return 0
        
        # 找到连续underwater的最长时间
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
        
        return max(underwater_periods) if underwater_periods else 0
    
    async def optimize(
        self,
        strategy_class: type,
        param_grid: Dict[str, List[Any]],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000.0,
        metric: str = "sharpe_ratio",
    ) -> List[tuple]:
        """
        参数优化
        
        Args:
            strategy_class: 策略类
            param_grid: 参数网格
            symbols: 资产代码列表
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金
            metric: 优化目标指标
            
        Returns:
            (参数, 结果) 列表，按指标降序排列
        """
        from itertools import product
        
        # 生成参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        logger.info(f"参数优化: {len(param_combinations)} 个组合")
        
        results = []
        
        for i, params in enumerate(param_combinations):
            param_dict = dict(zip(param_names, params))
            logger.info(f"测试参数 {i+1}/{len(param_combinations)}: {param_dict}")
            
            try:
                strategy = strategy_class(**param_dict)
                result = await self.run(
                    strategy, symbols, start_date, end_date,
                    initial_capital, show_progress=False
                )
                
                metric_value = getattr(result, metric)
                results.append((param_dict, result, metric_value))
                
            except Exception as e:
                logger.warning(f"参数组合失败: {param_dict}, 错误: {e}")
        
        # 按指标排序
        results.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"最佳参数: {results[0][0]}")
        logger.info(f"最佳 {metric}: {results[0][2]:.4f}")
        
        return results
