#!/usr/bin/env python3
"""
回测运行脚本

使用方法:
    python scripts/run_backtest.py backtest --strategy sma_crossover --symbol AAPL
    python scripts/run_backtest.py backtest --strategy mean_reversion --symbol AAPL,MSFT,GOOGL
    python scripts/run_backtest.py backtest --strategy mean_reversion -sym AAPL -sym MSFT -sym GOOGL
    python scripts/run_backtest.py gold-backtest --start 2024-01-01 --end 2025-12-31
    python scripts/run_backtest.py gold-backtest --no-benchmark  # 跳过基准策略比较
    python scripts/run_backtest.py list-strategies
    python scripts/run_backtest.py check-gpu
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
from strategy.builtin.weekly_dca import WeeklyDCAStrategy, WEEKDAY_NAMES
from strategy.gold.strategy import GoldTradingStrategy
from simulation.backtest import Backtester, BacktestResult
from simulation.broker import BrokerConfig
from utils.config import load_config, get_data_cache_path
from utils.logger import setup_logger
from loguru import logger

app = typer.Typer(help="量化交易回测工具")
console = Console()


STRATEGIES = {
    "sma_crossover": SMACrossoverStrategy,
    "mean_reversion": MeanReversionStrategy,
    "gold_multifactor": GoldTradingStrategy,
}

# 黄金策略跨市场 symbol 映射
# 使用 Alpha Vantage 可获取的 ETF 代替不可直接获取的指数
GOLD_CROSS_MARKET_SYMBOLS = {
    "gold_miners": "GDX",              # 黄金矿业 ETF（情绪因子）
    "sp500": "SPY",                    # 标普500 ETF（跨市场 + 宏观因子）
    "usd_index": "UUP",               # 美元指数基金（跨市场 + 宏观因子）
    "inflation_expectations": "TIP",   # 通胀保值债券 ETF（宏观因子）
    "treasury": "TLT",                 # 20+年期国债 ETF（宏观风险偏好因子）
    "vix": "VIXY",                     # VIX 短期期货 ETF（宏观风险偏好因子）
}

# 需要通过特殊 API 获取的数据（非股票 API）
GOLD_SPECIAL_DATA = {
    "treasury_yield": {"type": "treasury_yield", "maturity": "10year"},
    "eur_usd": {"type": "forex", "from": "EUR", "to": "USD"},
    "usd_jpy": {"type": "forex", "from": "USD", "to": "JPY"},
}

# 黄金回测需要获取的全部 ETF symbol
GOLD_PRIMARY_ETF = "GLD"
GOLD_ALL_SYMBOLS = [GOLD_PRIMARY_ETF] + list(GOLD_CROSS_MARKET_SYMBOLS.values())


def parse_symbols(values: List[str]) -> List[str]:
    """Parse symbols from comma-separated or multiple flag values."""
    result = []
    for value in values:
        # Split by comma and strip whitespace
        result.extend(s.strip() for s in value.split(",") if s.strip())
    return result


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
    info_table.add_row("总投入", f"${result.total_invested:,.2f}")
    info_table.add_row("总买入次数", str(result.buy_count))
    info_table.add_row("总卖出次数", str(result.sell_count))
    info_table.add_row("投资期限", result._format_invest_period())
    info_table.add_row("实际交易期", result._format_trade_period())
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
    symbols: List[str] = typer.Option(["AAPL"], "--symbol", "-sym", help="股票代码 (逗号分隔或多次使用 -sym)"),
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
    # 解析股票代码（支持逗号分隔）
    symbols = parse_symbols(symbols)

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
        cache = SQLiteCache(db_path=get_data_cache_path())
        provider = AlphaVantageProvider(
            api_key=config.alphavantage.api_key,
            cache_backend=cache,
            plan=config.alphavantage.plan,
        )

        # 创建回测器
        broker_config = BrokerConfig(
            commission_rate=config.broker.commission_rate,
            slippage_rate=config.broker.slippage_rate,
            short_selling_enabled=config.broker.short_selling_enabled,
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
def gold_backtest(
    start: str = typer.Option(None, "--start", help="开始日期 (YYYY-MM-DD)"),
    end: str = typer.Option(None, "--end", help="结束日期 (YYYY-MM-DD)"),
    capital: float = typer.Option(100000.0, "--capital", "-c", help="初始资金"),
    config_file: Optional[str] = typer.Option(None, "--config", help="配置文件路径"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细输出"),
    no_benchmark: bool = typer.Option(
        False, "--no-benchmark", help="跳过基准策略，只运行黄金多因子策略"
    ),
    optimized_params: Optional[str] = typer.Option(
        None, "--optimized-params", "-op",
        help="优化参数 YAML 路径（默认自动查找 data/output/gold_optimized_params.yaml）",
    ),
    no_optimized: bool = typer.Option(
        False, "--no-optimized", help="不加载优化参数，使用默认参数"
    ),
):
    """
    运行黄金择时增强型定投策略回测

    使用 GLD 作为主要交易标的，同时获取 GDX、SPY、UUP、TIP、TLT、VIXY
    等 ETF 数据，以及 TREASURY_YIELD、EUR/USD、USD/JPY 等特殊数据，
    用于技术面、跨市场、情绪和宏观四类因子分析。

    策略以每周定投为基础，多因子得分调节买入金额。

    默认同时运行 5 个每周定投基准策略（周一至周五各一个），
    用于与策略进行相对性能比较。
    使用 --no-benchmark 可跳过基准策略。
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

    # 解析日期（默认最近 2 年）
    if end is None:
        end_date = datetime.now()
    else:
        end_date = datetime.strptime(end, "%Y-%m-%d")

    if start is None:
        start_date = end_date - timedelta(days=730)
    else:
        start_date = datetime.strptime(start, "%Y-%m-%d")

    # 显示回测信息
    console.print(f"\n[bold]黄金择时增强型定投策略回测[/bold]")
    console.print(f"交易标的: [cyan]{GOLD_PRIMARY_ETF}[/cyan]")
    console.print(f"跨市场数据: [cyan]{', '.join(GOLD_CROSS_MARKET_SYMBOLS.values())}[/cyan]")
    console.print(f"期间: [cyan]{start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}[/cyan]")
    console.print(f"初始资金: [cyan]${capital:,.2f}[/cyan]")
    if not no_benchmark:
        console.print(f"基准策略: [cyan]每周定投 (周一至周五)[/cyan]")

    # 查找优化参数文件
    params_path = None
    if not no_optimized:
        if optimized_params:
            params_path = Path(optimized_params)
        else:
            # 自动查找默认位置
            default_path = Path(__file__).resolve().parent.parent.parent / "data" / "output" / "gold_optimized_params.yaml"
            if default_path.is_file():
                params_path = default_path

    # 创建黄金策略
    if params_path and params_path.is_file():
        console.print(f"[bold green]加载优化参数: {params_path}[/bold green]")
        strat = GoldTradingStrategy.from_optimized_params(
            yaml_path=str(params_path),
            cross_market_symbols=GOLD_CROSS_MARKET_SYMBOLS,
            end_date=end_date,
        )
    else:
        if not no_optimized and optimized_params:
            console.print(f"[yellow]警告: 未找到优化参数文件 {optimized_params}，使用默认参数[/yellow]")
        strat = GoldTradingStrategy(
            cross_market_symbols=GOLD_CROSS_MARKET_SYMBOLS,
            end_date=end_date,
        )

    # 显示因子权重（从实际策略对象读取）
    w = strat.weights
    factor_table = Table(title="\n因子权重")
    factor_table.add_column("因子", style="cyan")
    factor_table.add_column("权重", style="yellow", justify="right")
    factor_table.add_column("数据源", style="dim")
    factor_table.add_row("技术面", f"{w.technical:.0%}", f"{GOLD_PRIMARY_ETF} 价量数据")
    factor_table.add_row("跨市场", f"{w.cross_market:.0%}", "UUP, SPY, EUR/USD, USD/JPY")
    factor_table.add_row("市场情绪", f"{w.sentiment:.0%}", "GDX vs GLD 相对强度")
    factor_table.add_row("宏观", f"{w.macro:.0%}", "TREASURY_YIELD, TIP, TLT, VIXY")
    console.print(factor_table)

    buy_day_names = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五"}
    buy_day_name = buy_day_names.get(strat.position_config.buy_day, f"day{strat.position_config.buy_day}")
    console.print(f"\n策略模式: [cyan]择时增强型定投 (每{buy_day_name}买入)[/cyan]")
    console.print()

    # 运行回测
    async def run():
        cache = SQLiteCache(db_path=get_data_cache_path())
        provider = AlphaVantageProvider(
            api_key=config.alphavantage.api_key,
            cache_backend=cache,
            plan=config.alphavantage.plan,
        )

        broker_config = BrokerConfig(
            commission_rate=config.broker.commission_rate,
            slippage_rate=config.broker.slippage_rate,
            short_selling_enabled=config.broker.short_selling_enabled,
        )
        backtester = Backtester(
            data_provider=provider,
            cache=cache,
            broker_config=broker_config,
        )

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("获取市场数据并运行回测...", total=None)

                # 获取 ETF 数据（所有策略共用）
                data = await backtester.fetch_data(
                    GOLD_ALL_SYMBOLS, start_date, end_date
                )

                # 获取特殊数据（国债收益率、外汇）
                for data_key, spec in GOLD_SPECIAL_DATA.items():
                    try:
                        if spec["type"] == "treasury_yield":
                            special_data = await provider.get_treasury_yield(
                                start_date=start_date,
                                end_date=end_date,
                                maturity=spec["maturity"],
                            )
                        elif spec["type"] == "forex":
                            special_data = await provider.get_forex_daily(
                                from_symbol=spec["from"],
                                to_symbol=spec["to"],
                                start_date=start_date,
                                end_date=end_date,
                            )
                        else:
                            continue

                        if not special_data.is_empty:
                            # 将特殊数据的 ticker 加入 cross_market_symbols 映射
                            data[special_data.symbol] = special_data
                            GOLD_CROSS_MARKET_SYMBOLS[data_key] = special_data.symbol
                            logger.info(
                                f"{data_key}: {len(special_data)} 条数据 "
                                f"({special_data.symbol})"
                            )
                        else:
                            logger.warning(f"{data_key}: 无数据，跳过")
                    except Exception as e:
                        logger.warning(f"获取 {data_key} 数据失败: {e}，跳过")

                # 更新策略的跨市场映射（包含新增的特殊数据）
                strat.cross_market_symbols = GOLD_CROSS_MARKET_SYMBOLS

                # 运行黄金多因子策略
                result = await backtester.run_with_data(
                    strategy=strat,
                    data=data,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=capital,
                )

            display_result(result)

            # 显示黄金策略特有信息
            _display_gold_strategy_info(strat)

            # 运行基准策略
            if not no_benchmark:
                benchmark_results = {}
                gld_data = {GOLD_PRIMARY_ETF: data[GOLD_PRIMARY_ETF]}

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        "运行基准策略 (每周定投)...", total=5
                    )

                    for day in range(5):
                        dca_strategy = WeeklyDCAStrategy(target_day=day, end_date=end_date)
                        dca_result = await backtester.run_with_data(
                            strategy=dca_strategy,
                            data=gld_data,
                            start_date=start_date,
                            end_date=end_date,
                            initial_capital=capital,
                        )
                        benchmark_results[day] = dca_result
                        progress.advance(task)

                # 显示比较结果
                _display_benchmark_comparison(result, benchmark_results)

        finally:
            await provider.close()

    asyncio.run(run())


def _display_benchmark_comparison(
    gold_result: BacktestResult,
    benchmark_results: dict,
):
    """
    显示黄金策略与基准策略的性能比较

    Args:
        gold_result: 黄金多因子策略回测结果
        benchmark_results: 基准策略回测结果字典 {weekday: BacktestResult}
    """
    # 找到最佳基准
    best_benchmark_day = max(
        benchmark_results,
        key=lambda d: benchmark_results[d].total_return,
    )
    best_benchmark = benchmark_results[best_benchmark_day]

    # ---- 最佳基准详细信息 ----
    console.print(
        f"\n[bold green]========== 最佳基准: "
        f"定投-{WEEKDAY_NAMES[best_benchmark_day]} ==========[/bold green]\n"
    )

    bench_table = Table(title=f"最佳基准 — 每周{WEEKDAY_NAMES[best_benchmark_day]}定投", show_header=False)
    bench_table.add_column("项目", style="cyan")
    bench_table.add_column("值", style="yellow")

    bench_table.add_row("策略", best_benchmark.strategy_name)
    bench_table.add_row(
        "回测期间",
        f"{best_benchmark.start_date.strftime('%Y-%m-%d')} 至 "
        f"{best_benchmark.end_date.strftime('%Y-%m-%d')}",
    )
    bench_table.add_row("初始资金", f"${best_benchmark.initial_capital:,.2f}")
    bench_table.add_row("总投入", f"${best_benchmark.total_invested:,.2f}")
    bench_table.add_row("总买入次数", str(best_benchmark.buy_count))
    bench_table.add_row("总卖出次数", str(best_benchmark.sell_count))
    bench_table.add_row("投资期限", best_benchmark._format_invest_period())
    bench_table.add_row("实际交易期", best_benchmark._format_trade_period())
    bench_table.add_row("最终价值", f"${best_benchmark.final_value:,.2f}")
    console.print(bench_table)

    bench_ret_table = Table(title="\n收益 & 风险")
    bench_ret_table.add_column("指标", style="cyan")
    bench_ret_table.add_column("值", style="yellow", justify="right")
    bench_ret_table.add_row("总收益率", _colored_pct(best_benchmark.total_return))
    bench_ret_table.add_row("年化收益率", _colored_pct(best_benchmark.annual_return))
    bench_ret_table.add_row("波动率", f"{best_benchmark.volatility:.2%}")
    bench_ret_table.add_row("夏普比率", f"{best_benchmark.sharpe_ratio:.2f}")
    bench_ret_table.add_row("索提诺比率", f"{best_benchmark.sortino_ratio:.2f}")
    bench_ret_table.add_row("最大回撤", f"[red]{best_benchmark.max_drawdown:.2%}[/red]")
    bench_ret_table.add_row("最大回撤持续", f"{best_benchmark.max_drawdown_duration} 天")
    bench_ret_table.add_row("总手续费", f"${best_benchmark.total_commission:,.2f}")
    console.print(bench_ret_table)

    # ---- 综合比较表 ----
    console.print("\n[bold green]========== 策略性能比较 ==========[/bold green]\n")

    cmp_table = Table(title="择时增强型定投 vs 每周定投基准", expand=True)
    cmp_table.add_column("策略", style="cyan", no_wrap=True)
    cmp_table.add_column("总投入", justify="right", no_wrap=True)
    cmp_table.add_column("买/卖", justify="right", no_wrap=True)
    cmp_table.add_column("最终价值", justify="right", no_wrap=True)
    cmp_table.add_column("总收益", justify="right", no_wrap=True)
    cmp_table.add_column("年化", justify="right", no_wrap=True)
    cmp_table.add_column("投资期限", justify="right", no_wrap=True)
    cmp_table.add_column("夏普", justify="right", no_wrap=True)
    cmp_table.add_column("回撤", justify="right", no_wrap=True)
    cmp_table.add_column("相对", justify="right", no_wrap=True)

    def _add_result_row(table, label, r, ref_return, bold=False):
        """向比较表添加一行"""
        name = f"[bold]{label}[/bold]" if bold else label
        excess = r.total_return - ref_return
        ec = "green" if excess > 0 else "red"
        # 投资期限（基于回测区间，跨策略统一）
        period_str = f"{r.invest_months}月({r.start_date.strftime('%Y%m%d')}~{r.end_date.strftime('%Y%m%d')})"

        table.add_row(
            name,
            f"${r.total_invested:,.0f}",
            f"{r.buy_count}/{r.sell_count}",
            f"${r.final_value:,.0f}",
            _colored_pct(r.total_return),
            _colored_pct(r.annual_return),
            period_str,
            f"{r.sharpe_ratio:.2f}",
            f"[red]{r.max_drawdown:.1%}[/red]",
            f"[{ec}]{excess:+.1%}[/{ec}]",
        )

    # 所有行的"相对"列都以最佳基准为参照
    best_return = best_benchmark.total_return

    # 黄金策略行
    _add_result_row(
        cmp_table, "择时增强定投", gold_result, best_return, bold=True,
    )

    # 分隔线
    cmp_table.add_row(
        "", "", "", "", "", "", "", "", "", "", end_section=True,
    )

    # 基准策略行（最佳排前面）
    sorted_days = sorted(
        range(5),
        key=lambda d: benchmark_results[d].total_return,
        reverse=True,
    )
    for day in sorted_days:
        r = benchmark_results[day]
        day_name = WEEKDAY_NAMES[day]
        label = f"定投-{day_name}"
        if day == best_benchmark_day:
            label += " [green]★[/green]"
        _add_result_row(cmp_table, label, r, best_return)

    console.print(cmp_table)

    # 总结
    console.print()
    if gold_result.total_return > best_benchmark.total_return:
        diff = gold_result.total_return - best_benchmark.total_return
        console.print(
            f"[bold green]择时增强定投策略跑赢所有定投基准，"
            f"优于最佳定投({WEEKDAY_NAMES[best_benchmark_day]}) "
            f"{diff:.2%}[/bold green]"
        )
    else:
        diff = best_benchmark.total_return - gold_result.total_return
        console.print(
            f"[bold yellow]择时增强定投策略跑输最佳定投基准"
            f"({WEEKDAY_NAMES[best_benchmark_day]})，"
            f"落后 {diff:.2%}[/bold yellow]"
        )

    # 夏普比率比较
    best_sharpe_day = max(
        benchmark_results, key=lambda d: benchmark_results[d].sharpe_ratio,
    )
    best_sharpe = benchmark_results[best_sharpe_day]
    if gold_result.sharpe_ratio > best_sharpe.sharpe_ratio:
        console.print(
            f"[green]风险调整收益（夏普比率）优于所有基准: "
            f"{gold_result.sharpe_ratio:.2f} vs {best_sharpe.sharpe_ratio:.2f}[/green]"
        )
    else:
        console.print(
            f"[yellow]风险调整收益（夏普比率）低于最佳基准"
            f"({WEEKDAY_NAMES[best_sharpe_day]}): "
            f"{gold_result.sharpe_ratio:.2f} vs {best_sharpe.sharpe_ratio:.2f}[/yellow]"
        )


def _colored_pct(value: float) -> str:
    """格式化百分比并根据正负着色"""
    color = "green" if value > 0 else ("red" if value < 0 else "white")
    return f"[{color}]{value:.2%}[/{color}]"


def _display_gold_strategy_info(strategy: GoldTradingStrategy):
    """显示黄金策略特有信息"""
    console.print()

    # 信号统计
    if strategy.signals:
        signal_table = Table(title="黄金策略信号统计")
        signal_table.add_column("信号类型", style="cyan")
        signal_table.add_column("次数", style="yellow", justify="right")

        signal_counts = {}
        action_counts = {}
        for sig in strategy.signals:
            st = sig.signal_type.value
            signal_counts[st] = signal_counts.get(st, 0) + 1
            action = sig.metadata.get("tactical_action", "unknown")
            action_counts[action] = action_counts.get(action, 0) + 1

        for sig_type, count in sorted(signal_counts.items()):
            signal_table.add_row(sig_type, str(count))

        console.print(signal_table)

        # 战术动作分布
        if action_counts:
            action_table = Table(title="\n战术动作分布")
            action_table.add_column("动作", style="cyan")
            action_table.add_column("次数", style="yellow", justify="right")

            action_names = {
                "boost_buy": "加量买入 (score>0.3)",
                "normal_buy": "正常买入 (score>0.0)",
                "reduce_buy": "减量买入 (score>-0.15)",
                "skip_buy": "跳过买入 (-0.3<score≤-0.15)",
                "strong_skip": "强看空跳过 (score≤-0.3)",
                "partial_sell": "部分减仓 (score<-0.6)",
                "force_take_profit": "强制止盈 (定期落袋)",
            }
            for action, count in sorted(action_counts.items()):
                name = action_names.get(action, action)
                action_table.add_row(name, str(count))

            console.print(action_table)

        # 最后一个信号的因子得分
        last_signal = strategy.signals[-1]
        if "factor_scores" in last_signal.metadata:
            signal_ts = last_signal.timestamp.strftime("%Y-%m-%d")
            score_table = Table(title=f"\n最近信号因子得分 ({signal_ts})")
            score_table.add_column("因子", style="cyan")
            score_table.add_column("得分", style="yellow", justify="right")

            for factor, score in last_signal.metadata["factor_scores"].items():
                color = "green" if score > 0 else ("red" if score < 0 else "white")
                score_table.add_row(factor, f"[{color}]{score:+.4f}[/{color}]")

            if "composite_score" in last_signal.metadata:
                score = last_signal.metadata["composite_score"]
                color = "green" if score > 0 else ("red" if score < 0 else "white")
                score_table.add_row(
                    "[bold]综合得分[/bold]",
                    f"[bold {color}]{score:+.4f}[/bold {color}]",
                )

            console.print(score_table)
    else:
        console.print("[dim]未产生交易信号[/dim]")


@app.command()
def optimize_gold(
    start: str = typer.Option(None, "--start", help="开始日期 (YYYY-MM-DD)"),
    end: str = typer.Option(None, "--end", help="结束日期 (YYYY-MM-DD)"),
    capital: float = typer.Option(100000.0, "--capital", "-c", help="初始资金"),
    popsize: int = typer.Option(20, "--popsize", help="差分进化种群大小"),
    maxiter: int = typer.Option(50, "--maxiter", help="最大迭代代数"),
    train_ratio: float = typer.Option(0.7, "--train-ratio", help="训练集比例 (0-1)"),
    seed: int = typer.Option(42, "--seed", help="随机种子"),
    workers: int = typer.Option(-1, "--workers", "-w", help="并行 worker 数量 (1=串行, -1=全部CPU核心)"),
    config_file: Optional[str] = typer.Option(None, "--config", help="配置文件路径"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细输出"),
):
    """
    使用差分进化算法优化黄金策略参数

    自动搜索因子权重、阈值、倍率和强制止盈等全部参数的最优组合，
    目标是找到远优于基准定投策略的参数配置。

    使用 walk-forward validation 防止过拟合。
    默认使用全部 CPU 核心并行加速。
    """
    from strategy.gold.optimizer import (
        GoldStrategyOptimizer,
        PARAM_NAMES,
        build_strategy,
        compute_fitness,
    )
    from strategy.builtin.weekly_dca import WeeklyDCAStrategy

    # 设置日志
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger(level=log_level)

    # 加载配置
    config = load_config(config_file)
    if not config.alphavantage.api_key:
        console.print("[red]错误: 未设置 Alpha Vantage API Key[/red]")
        raise typer.Exit(1)

    # 解析日期（默认最近 3 年）
    if end is None:
        end_date = datetime.now()
    else:
        end_date = datetime.strptime(end, "%Y-%m-%d")
    if start is None:
        start_date = end_date - timedelta(days=1095)
    else:
        start_date = datetime.strptime(start, "%Y-%m-%d")

    console.print(f"\n[bold]黄金策略差分进化优化[/bold]")
    console.print(f"期间: [cyan]{start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}[/cyan]")
    console.print(f"初始资金: [cyan]${capital:,.2f}[/cyan]")
    import os
    n_cpus = os.cpu_count() or 1
    n_workers = n_cpus if workers == -1 else max(1, n_cpus + workers) if workers <= 0 else workers
    console.print(f"种群大小: [cyan]{popsize}[/cyan]  最大迭代: [cyan]{maxiter}[/cyan]")
    console.print(f"训练集比例: [cyan]{train_ratio:.0%}[/cyan]  随机种子: [cyan]{seed}[/cyan]")
    console.print(f"并行 Workers: [cyan]{n_workers}[/cyan] / {n_cpus} CPU 核心\n")

    async def run():
        cache = SQLiteCache(db_path=get_data_cache_path())
        provider = AlphaVantageProvider(
            api_key=config.alphavantage.api_key,
            cache_backend=cache,
            plan=config.alphavantage.plan,
        )
        broker_config = BrokerConfig(
            commission_rate=config.broker.commission_rate,
            slippage_rate=config.broker.slippage_rate,
            short_selling_enabled=config.broker.short_selling_enabled,
        )
        backtester = Backtester(
            data_provider=provider, cache=cache, broker_config=broker_config,
        )

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("获取市场数据...", total=None)
                data = await backtester.fetch_data(
                    GOLD_ALL_SYMBOLS, start_date, end_date
                )
                # 获取特殊数据
                cross_symbols = dict(GOLD_CROSS_MARKET_SYMBOLS)
                for data_key, spec in GOLD_SPECIAL_DATA.items():
                    try:
                        if spec["type"] == "treasury_yield":
                            special_data = await provider.get_treasury_yield(
                                start_date=start_date, end_date=end_date,
                                maturity=spec["maturity"],
                            )
                        elif spec["type"] == "forex":
                            special_data = await provider.get_forex_daily(
                                from_symbol=spec["from"], to_symbol=spec["to"],
                                start_date=start_date, end_date=end_date,
                            )
                        else:
                            continue
                        if not special_data.is_empty:
                            data[special_data.symbol] = special_data
                            cross_symbols[data_key] = special_data.symbol
                    except Exception as e:
                        logger.warning(f"获取 {data_key} 数据失败: {e}")

            # Checkpoint 和结果保存目录
            output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir = output_dir / "checkpoints"

            # 创建优化器
            optimizer = GoldStrategyOptimizer(
                data=data,
                cross_market_symbols=cross_symbols,
                broker_config=broker_config,
                initial_capital=capital,
                train_ratio=train_ratio,
                checkpoint_dir=str(checkpoint_dir),
            )

            console.print("[bold]开始差分进化优化...[/bold]\n")
            opt_result = optimizer.optimize(
                popsize=popsize, maxiter=maxiter, seed=seed,
                workers=workers,
            )

            # 运行基准策略（全量数据上最佳 DCA）
            gld_data = {"GLD": data["GLD"]}
            best_bench_return = -float("inf")
            best_bench_result = None
            best_bench_day = 0
            for day in range(5):
                dca = WeeklyDCAStrategy(target_day=day, end_date=end_date)
                bench = await backtester.run_with_data(
                    strategy=dca, data=gld_data,
                    start_date=start_date, end_date=end_date,
                    initial_capital=capital,
                )
                if bench.total_return > best_bench_return:
                    best_bench_return = bench.total_return
                    best_bench_result = bench
                    best_bench_day = day
            opt_result.benchmark_result = best_bench_result

            # 保存优化结果到 YAML
            result_path = output_dir / "gold_optimized_params.yaml"
            opt_result.save_params_yaml(str(result_path))
            console.print(f"\n[bold green]优化参数已保存: {result_path}[/bold green]")

            # ---- 显示结果 ----
            _display_optimization_result(opt_result, best_bench_day)

        finally:
            await provider.close()

    asyncio.run(run())


def _display_optimization_result(opt_result, best_bench_day: int):
    """显示优化结果"""
    console.print(f"\n[bold green]========== 差分进化优化结果 ==========[/bold green]\n")
    console.print(
        f"总评估次数: [cyan]{opt_result.n_evaluations}[/cyan]  "
        f"耗时: [cyan]{opt_result.elapsed_seconds:.0f}s[/cyan]  "
        f"最优适应度: [cyan]{opt_result.best_fitness:.4f}[/cyan]\n"
    )

    # 最优参数表
    param_table = Table(title="最优参数")
    param_table.add_column("参数", style="cyan")
    param_table.add_column("值", style="yellow", justify="right")
    param_table.add_column("说明", style="dim")

    param_desc = {
        "w_technical": "技术面权重",
        "w_cross_market": "跨市场权重",
        "w_sentiment": "情绪权重",
        "w_macro": "宏观权重",
        "thresh_boost": "加量买入阈值",
        "thresh_normal": "正常买入阈值",
        "thresh_reduce": "减量买入阈值",
        "thresh_skip": "跳过买入阈值",
        "thresh_sell": "触发卖出阈值",
        "boost_multiplier": "加量买入倍率",
        "reduce_multiplier": "减量买入倍率",
        "sell_fraction": "信号卖出比例",
        "buy_day": "每周买入日",
        "force_sell_interval": "强制止盈间隔(天)",
        "force_sell_fraction": "强制止盈比例",
        "force_sell_profit_thresh": "止盈盈利阈值",
    }

    day_names = {0: "周一", 1: "周二", 2: "周三", 3: "周四", 4: "周五"}

    for name, value in opt_result.best_params.items():
        desc = param_desc.get(name, "")
        if name == "buy_day":
            val_str = f"{int(value)} ({day_names.get(int(value), '')})"
        elif name in ("w_technical", "w_cross_market", "w_sentiment", "w_macro",
                       "sell_fraction", "force_sell_fraction", "force_sell_profit_thresh"):
            val_str = f"{value:.2%}"
        elif name.startswith("thresh_"):
            val_str = f"{value:+.3f}"
        elif name == "force_sell_interval":
            val_str = f"{int(value)} 天"
        else:
            val_str = f"{value:.2f}"
        param_table.add_row(name, val_str, desc)

    console.print(param_table)

    # 性能对比表
    perf_table = Table(title="\n训练集 vs 验证集 vs 全量 vs 基准", expand=True)
    perf_table.add_column("数据集", style="cyan", no_wrap=True)
    perf_table.add_column("总收益", justify="right")
    perf_table.add_column("年化", justify="right")
    perf_table.add_column("夏普", justify="right")
    perf_table.add_column("回撤", justify="right")
    perf_table.add_column("买/卖", justify="right")
    perf_table.add_column("适应度", justify="right")

    from strategy.gold.optimizer import compute_fitness

    for label, result in [
        ("训练集 (优化)", opt_result.train_result),
        ("验证集 (OOS)", opt_result.val_result),
        ("全量数据", opt_result.full_result),
        (f"基准-最佳DCA", opt_result.benchmark_result),
    ]:
        if result is None:
            continue
        fitness = compute_fitness(result)
        ret_color = "green" if result.total_return > 0 else "red"
        perf_table.add_row(
            label,
            f"[{ret_color}]{result.total_return:.2%}[/{ret_color}]",
            f"[{ret_color}]{result.annual_return:.2%}[/{ret_color}]",
            f"{result.sharpe_ratio:.2f}",
            f"[red]{result.max_drawdown:.2%}[/red]",
            f"{result.buy_count}/{result.sell_count}",
            f"{fitness:.4f}",
        )

    console.print(perf_table)

    # 总结
    console.print()
    full = opt_result.full_result
    bench = opt_result.benchmark_result
    if full and bench:
        diff = full.total_return - bench.total_return
        if diff > 0:
            console.print(
                f"[bold green]优化策略（全量数据）跑赢最佳定投基准 {diff:.2%}[/bold green]"
            )
        else:
            console.print(
                f"[bold yellow]优化策略（全量数据）跑输最佳定投基准 {abs(diff):.2%}[/bold yellow]"
            )

    # 过拟合检查
    train = opt_result.train_result
    val = opt_result.val_result
    if train and val:
        train_fitness = compute_fitness(train)
        val_fitness = compute_fitness(val)
        ratio = val_fitness / train_fitness if train_fitness > 0 else 0
        if ratio < 0.5:
            console.print(
                f"[bold red]⚠ 过拟合警告: 验证集适应度仅为训练集的 {ratio:.0%}[/bold red]"
            )
        elif ratio < 0.8:
            console.print(
                f"[yellow]验证集适应度为训练集的 {ratio:.0%}，存在轻微过拟合[/yellow]"
            )
        else:
            console.print(
                f"[green]验证集适应度为训练集的 {ratio:.0%}，泛化良好[/green]"
            )


@app.command()
def list_strategies():
    """列出可用策略"""
    table = Table(title="可用策略")
    table.add_column("名称", style="cyan")
    table.add_column("描述")

    table.add_row("sma_crossover", "双均线交叉策略")
    table.add_row("mean_reversion", "均值回归策略")
    table.add_row("gold_multifactor", "黄金择时增强型定投 (技术面+跨市场+情绪+宏观)")
    table.add_row("weekly_dca", "每周定投基准策略 (在 gold-backtest 中自动运行)")

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
