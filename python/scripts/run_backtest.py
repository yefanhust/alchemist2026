#!/usr/bin/env python3
"""
回测运行脚本

使用方法:
    python scripts/run_backtest.py --strategy sma_crossover --symbol AAPL
    python scripts/run_backtest.py --strategy mean_reversion --symbol AAPL MSFT GOOGL
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# 注意：不要手动添加项目路径！
# 项目应该通过 'pip install -e .' 安装，这样才能正确处理相对导入
# sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.portfolio import Portfolio
from data.providers.alphavantage import AlphaVantageProvider
from data.providers.base import DataInterval
from data.cache.sqlite_cache import SQLiteCache
from strategy.builtin.sma_crossover import SMACrossoverStrategy
from strategy.builtin.mean_reversion import MeanReversionStrategy
from simulation.backtest import Backtester, BacktestResult
from simulation.broker import BrokerConfig
from utils.config import load_config
from utils.logger import setup_logger

app = typer.Typer(help="量化交易回测工具")
console = Console()


STRATEGIES = {
    "sma_crossover": SMACrossoverStrategy,
    "mean_reversion": MeanReversionStrategy,
}


def create_strategy(
    strategy_name: str,
    **kwargs,
):
    """创建策略实例"""
    if strategy_name not in STRATEGIES:
        raise ValueError(f"未知策略: {strategy_name}，可用: {list(STRATEGIES.keys())}")
    
    strategy_class = STRATEGIES[strategy_name]
    return strategy_class(**kwargs)


def display_result(result: BacktestResult):
    """显示回测结果"""
    console.print("\n[bold green]========== 回测结果 ==========[/bold green]\n")
    
    # 基本信息
    info_table = Table(title="基本信息", show_header=False)
    info_table.add_column("项目", style="cyan")
    info_table.add_column("值", style="yellow")
    
    info_table.add_row("策略", result.strategy_name)
    info_table.add_row("回测期间", f"{result.start_date.strftime('%Y-%m-%d')} 至 {result.end_date.strftime('%Y-%m-%d')}")
    info_table.add_row("初始资金", f"${result.initial_capital:,.2f}")
    info_table.add_row("最终价值", f"${result.final_value:,.2f}")
    
    console.print(info_table)
    
    # 收益指标
    returns_table = Table(title="\n收益指标")
    returns_table.add_column("指标", style="cyan")
    returns_table.add_column("值", style="yellow", justify="right")
    
    returns_table.add_row("总收益率", f"{result.total_return:.2%}")
    returns_table.add_row("年化收益率", f"{result.annual_return:.2%}")
    
    console.print(returns_table)
    
    # 风险指标
    risk_table = Table(title="\n风险指标")
    risk_table.add_column("指标", style="cyan")
    risk_table.add_column("值", style="yellow", justify="right")
    
    risk_table.add_row("波动率", f"{result.volatility:.2%}")
    risk_table.add_row("夏普比率", f"{result.sharpe_ratio:.2f}")
    risk_table.add_row("索提诺比率", f"{result.sortino_ratio:.2f}")
    risk_table.add_row("最大回撤", f"{result.max_drawdown:.2%}")
    risk_table.add_row("最大回撤持续", f"{result.max_drawdown_duration} 天")
    
    console.print(risk_table)
    
    # 交易统计
    trade_table = Table(title="\n交易统计")
    trade_table.add_column("指标", style="cyan")
    trade_table.add_column("值", style="yellow", justify="right")
    
    trade_table.add_row("总交易次数", str(result.total_trades))
    trade_table.add_row("盈利交易", f"{result.winning_trades} ({result.win_rate:.1%})")
    trade_table.add_row("亏损交易", str(result.losing_trades))
    trade_table.add_row("平均盈利", f"${result.avg_win:,.2f}")
    trade_table.add_row("平均亏损", f"${result.avg_loss:,.2f}")
    trade_table.add_row("盈亏比", f"{result.profit_factor:.2f}")
    trade_table.add_row("总手续费", f"${result.total_commission:,.2f}")
    
    console.print(trade_table)
    
    # 月度收益
    if result.monthly_returns:
        monthly_table = Table(title="\n月度收益")
        monthly_table.add_column("月份", style="cyan")
        monthly_table.add_column("收益率", style="yellow", justify="right")
        
        for month, ret in list(result.monthly_returns.items())[-12:]:  # 最近12个月
            color = "green" if ret > 0 else "red"
            monthly_table.add_row(month, f"[{color}]{ret:.2%}[/{color}]")
        
        console.print(monthly_table)


@app.command()
def backtest(
    strategy: str = typer.Option("sma_crossover", "--strategy", "-s", help="策略名称"),
    symbols: List[str] = typer.Option(["AAPL"], "--symbol", "-sym", help="股票代码"),
    start: str = typer.Option(None, "--start", help="开始日期 (YYYY-MM-DD)"),
    end: str = typer.Option(None, "--end", help="结束日期 (YYYY-MM-DD)"),
    capital: float = typer.Option(100000.0, "--capital", "-c", help="初始资金"),
    fast_period: int = typer.Option(10, "--fast", help="快速均线周期 (SMA策略)"),
    slow_period: int = typer.Option(30, "--slow", help="慢速均线周期 (SMA策略)"),
    config_file: Optional[str] = typer.Option(None, "--config", help="配置文件路径"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细输出"),
):
    """
    运行策略回测
    """
    # 设置日志
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(level=log_level)
    
    # 加载配置
    config = load_config(config_file)
    
    # 检查 API Key
    if not config.alphavantage.api_key:
        console.print("[red]错误: 未设置 Alpha Vantage API Key[/red]")
        console.print("请设置环境变量 ALPHAVANTAGE_API_KEY 或在配置文件中指定")
        raise typer.Exit(1)
    
    # 解析日期
    if end is None:
        end_date = datetime.now()
    else:
        end_date = datetime.strptime(end, "%Y-%m-%d")
    
    if start is None:
        start_date = end_date - timedelta(days=365)
    else:
        start_date = datetime.strptime(start, "%Y-%m-%d")
    
    console.print(f"\n[bold]量化交易回测系统[/bold]")
    console.print(f"策略: [cyan]{strategy}[/cyan]")
    console.print(f"标的: [cyan]{', '.join(symbols)}[/cyan]")
    console.print(f"期间: [cyan]{start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}[/cyan]")
    console.print(f"初始资金: [cyan]${capital:,.2f}[/cyan]\n")
    
    # 创建策略
    if strategy == "sma_crossover":
        strat = create_strategy(strategy, fast_period=fast_period, slow_period=slow_period)
    else:
        strat = create_strategy(strategy)
    
    # 运行回测
    async def run():
        # 创建数据提供者和缓存
        cache = SQLiteCache(db_path="./data/cache/market_data.db")
        provider = AlphaVantageProvider(
            api_key=config.alphavantage.api_key,
            cache_backend=cache,
        )
        
        # 创建回测器
        broker_config = BrokerConfig(
            commission_rate=config.broker.commission_rate,
            slippage_rate=config.broker.slippage_rate,
        )
        backtester = Backtester(
            data_provider=provider,
            cache=cache,
            broker_config=broker_config,
        )
        
        try:
            result = await backtester.run(
                strategy=strat,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                initial_capital=capital,
            )
            
            display_result(result)
            
        finally:
            await provider.close()
    
    asyncio.run(run())


@app.command()
def list_strategies():
    """列出可用策略"""
    table = Table(title="可用策略")
    table.add_column("名称", style="cyan")
    table.add_column("描述")
    
    table.add_row("sma_crossover", "双均线交叉策略")
    table.add_row("mean_reversion", "均值回归策略")
    
    console.print(table)


@app.command()
def check_gpu():
    """检查 GPU 状态"""
    from gpu import check_gpu_availability, get_gpu_memory_info
    
    info = check_gpu_availability()
    
    table = Table(title="GPU 状态")
    table.add_column("项目", style="cyan")
    table.add_column("值", style="yellow")
    
    table.add_row("CUDA 可用", str(info["cuda_available"]))
    table.add_row("CuPy 可用", str(info["cupy_available"]))
    table.add_row("PyTorch 可用", str(info["torch_available"]))
    table.add_row("GPU 数量", str(info["gpu_count"]))
    
    console.print(table)
    
    if info["devices"]:
        device_table = Table(title="\nGPU 设备")
        device_table.add_column("ID")
        device_table.add_column("名称")
        device_table.add_column("显存")
        
        for device in info["devices"]:
            device_table.add_row(
                str(device["id"]),
                device["name"],
                f"{device['total_memory_gb']:.1f} GB"
            )
        
        console.print(device_table)


def main():
    app()


if __name__ == "__main__":
    main()
