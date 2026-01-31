#!/usr/bin/env python3
"""
数据源信息查询脚本

查询各数据源的 high-level 信息，包括：
- 是否支持港股（HKEX）
- 支持的港股替代标的（US ADR 等）
- 支持的金融资产类别
- 支持的数据间隔
- 限流策略

使用方法:
    python scripts/datasource_info.py info
    python scripts/datasource_info.py info --source alphavantage
    python scripts/datasource_info.py hk-support
    python scripts/datasource_info.py search --keyword Tencent
    python scripts/datasource_info.py list-sources
"""

import asyncio
import os
from typing import Optional, List, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree

from data.providers.alphavantage import AlphaVantageProvider
from data.providers import DataProvider, DataInterval
from utils.config import load_config
from utils.logger import setup_logger

app = typer.Typer(help="数据源信息查询工具")
console = Console()


# ── 数据源注册表 ──────────────────────────────────────────────────────────────

# 每新增一个 DataProvider 实现，在此注册即可被脚本自动识别。
# key: 数据源名称（CLI --source 参数值）
# value: dict 包含 provider 构造信息及静态元数据
DATA_SOURCE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "alphavantage": {
        "display_name": "Alpha Vantage",
        "description": "全球股票、ETF、外汇、加密货币数据 (alphavantage.co)",
        "supported_asset_types": [
            ("stock",   "股票（美股、A 股、多国市场）"),
            ("etf",     "交易所交易基金（美股 ETF）"),
            ("forex",   "外汇汇率（实时 & 历史）"),
            ("crypto",  "加密货币（BTC、ETH 等数百种）"),
        ],
        "supported_exchanges": [
            ("NYSE / NASDAQ", "美国主要交易所", True),
            ("SSE (上交所)",   "代码后缀 .SHH，如 600104.SHH", True),
            ("SZSE (深交所)",  "代码后缀 .SHZ，如 000001.SHZ", True),
            ("LSE (伦交所)",   "代码后缀 .LON", True),
            ("TSE (东京)",     "代码后缀 .TYO", True),
            ("HKEX (港交所)",  "后缀 .HK / .HKG — 不支持", False),
        ],
        "hk_support": False,
        "hk_alternatives": [
            ("TCEHY",  "腾讯控股 ADR (OTC)"),
            ("BABA",   "阿里巴巴 (NYSE)"),
            ("BIDU",   "百度 (NASDAQ)"),
            ("JD",     "京东 (NASDAQ)"),
            ("PDD",    "拼多多 (NASDAQ)"),
            ("NIO",    "蔚来汽车 (NYSE)"),
            ("XPEV",   "小鹏汽车 (NYSE)"),
            ("LI",     "理想汽车 (NASDAQ)"),
            ("NTES",   "网易 (NASDAQ)"),
            ("BILI",   "哔哩哔哩 (NASDAQ)"),
        ],
    },
}

DEFAULT_SOURCE = "alphavantage"


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _get_provider(source: str, config) -> DataProvider:
    """根据数据源名称创建 provider 实例"""
    if source == "alphavantage":
        api_key = config.alphavantage.api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not api_key:
            console.print("[red]错误: 未设置 Alpha Vantage API Key[/red]")
            console.print("请设置环境变量 ALPHAVANTAGE_API_KEY 或在配置文件中指定")
            raise typer.Exit(1)
        return AlphaVantageProvider(api_key=api_key)
    else:
        console.print(f"[red]错误: 未知数据源 '{source}'[/red]")
        console.print(f"可用数据源: {', '.join(DATA_SOURCE_REGISTRY.keys())}")
        raise typer.Exit(1)


def _intervals_table(provider: DataProvider) -> Table:
    """构建支持的数据间隔表格"""
    table = Table(title="支持的数据间隔", show_header=True)
    table.add_column("间隔", style="cyan")
    table.add_column("值", style="yellow")

    for iv in provider.supported_intervals:
        table.add_row(iv.name, iv.value)
    return table


# ── CLI 命令 ──────────────────────────────────────────────────────────────────

@app.command()
def info(
    source: str = typer.Option(DEFAULT_SOURCE, "--source", "-s", help="数据源名称"),
    config_file: Optional[str] = typer.Option(None, "--config", help="配置文件路径"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细输出"),
):
    """
    显示数据源的 high-level 信息
    """
    setup_logger(level="DEBUG" if verbose else "WARNING")
    config = load_config(config_file)

    meta = DATA_SOURCE_REGISTRY.get(source)
    if meta is None:
        console.print(f"[red]未知数据源: {source}[/red]")
        raise typer.Exit(1)

    # ── 基本信息 ──
    console.print()
    console.print(Panel(
        f"[bold]{meta['display_name']}[/bold]\n{meta['description']}",
        title="数据源概览",
        border_style="blue",
    ))

    # ── 支持的资产类别 ──
    asset_table = Table(title="支持的金融资产类别")
    asset_table.add_column("类别", style="cyan")
    asset_table.add_column("说明", style="white")
    for asset_type, desc in meta["supported_asset_types"]:
        asset_table.add_row(asset_type, desc)
    console.print(asset_table)

    # ── 交易所支持 ──
    exch_table = Table(title="交易所支持情况")
    exch_table.add_column("交易所", style="cyan")
    exch_table.add_column("说明", style="white")
    exch_table.add_column("状态", justify="center")
    for name, desc, supported in meta["supported_exchanges"]:
        status = "[green]支持[/green]" if supported else "[red]不支持[/red]"
        exch_table.add_row(name, desc, status)
    console.print(exch_table)

    # ── 港股支持 ──
    hk_status = "[green]支持[/green]" if meta["hk_support"] else "[red]不支持[/red]"
    console.print(f"\n港股 (HKEX) 直接数据: {hk_status}")
    if not meta["hk_support"] and meta.get("hk_alternatives"):
        console.print("[dim]可使用以下 US ADR / 美股替代标的:[/dim]")
        alt_table = Table(show_header=True)
        alt_table.add_column("代码", style="cyan")
        alt_table.add_column("名称", style="white")
        for sym, name in meta["hk_alternatives"]:
            alt_table.add_row(sym, name)
        console.print(alt_table)

    # ── 数据间隔（需要实例化 provider） ──
    try:
        provider = _get_provider(source, config)
        console.print(_intervals_table(provider))

        # ── 限流信息 ──
        if source == "alphavantage":
            plan = provider.plan
            preset = AlphaVantageProvider.PLAN_PRESETS.get(plan, (0, 0, 0))
            cps, cpm, cpd = preset
            rate_table = Table(title="API 限流策略")
            rate_table.add_column("项目", style="cyan")
            rate_table.add_column("值", style="yellow")
            rate_table.add_row("当前 Plan", plan)
            if cps:
                rate_table.add_row("频率限制", f"{cps} 次/秒")
            elif cpm:
                rate_table.add_row("频率限制", f"{cpm} 次/分钟")
            rate_table.add_row("每日上限", str(cpd) if cpd else "无限制")
            console.print(rate_table)
    except SystemExit:
        console.print("[yellow]提示: 未配置 API Key，部分信息无法显示[/yellow]")

    console.print()


@app.command("hk-support")
def hk_support(
    source: str = typer.Option(DEFAULT_SOURCE, "--source", "-s", help="数据源名称"),
):
    """
    检查数据源是否支持港股，列出替代标的
    """
    meta = DATA_SOURCE_REGISTRY.get(source)
    if meta is None:
        console.print(f"[red]未知数据源: {source}[/red]")
        raise typer.Exit(1)

    console.print()
    supported = meta["hk_support"]
    status = "[bold green]支持[/bold green]" if supported else "[bold red]不支持[/bold red]"
    console.print(Panel(
        f"数据源: [cyan]{meta['display_name']}[/cyan]\n"
        f"港股 (HKEX) 直接数据: {status}\n\n"
        + ("可直接使用 .HK 后缀获取港股行情。" if supported
           else "Alpha Vantage 不支持 .HK / .HKG 后缀，\n"
                "获取港股相关数据请使用 US ADR 或 A 股替代标的。"),
        title="港股支持情况",
        border_style="yellow",
    ))

    alternatives = meta.get("hk_alternatives", [])
    if alternatives:
        table = Table(title="可用的港股替代标的 (US ADR / 美股)")
        table.add_column("#", style="dim", justify="right")
        table.add_column("代码", style="cyan")
        table.add_column("名称", style="white")
        for idx, (sym, name) in enumerate(alternatives, 1):
            table.add_row(str(idx), sym, name)
        console.print(table)

    console.print()


@app.command("search")
def search(
    keyword: str = typer.Argument(..., help="搜索关键词（公司名或代码）"),
    source: str = typer.Option(DEFAULT_SOURCE, "--source", "-s", help="数据源名称"),
    config_file: Optional[str] = typer.Option(None, "--config", help="配置文件路径"),
):
    """
    在数据源中搜索资产代码
    """
    setup_logger(level="WARNING")
    config = load_config(config_file)
    provider = _get_provider(source, config)

    async def _search():
        try:
            return await provider.search_symbols(keyword)
        finally:
            await provider.close()

    results = asyncio.run(_search())

    console.print()
    if not results:
        console.print(f"[yellow]未找到与 '{keyword}' 匹配的结果[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"搜索结果: '{keyword}'")
    table.add_column("代码", style="cyan")
    table.add_column("名称", style="white")
    table.add_column("类型", style="green")
    table.add_column("地区", style="magenta")
    table.add_column("货币", style="yellow")

    for r in results:
        table.add_row(
            r.get("symbol", ""),
            r.get("name", ""),
            r.get("type", ""),
            r.get("region", ""),
            r.get("currency", ""),
        )

    console.print(table)
    console.print()


@app.command("list-sources")
def list_sources():
    """
    列出所有已注册的数据源
    """
    console.print()
    tree = Tree("[bold]已注册数据源[/bold]")

    for key, meta in DATA_SOURCE_REGISTRY.items():
        branch = tree.add(
            f"[cyan]{key}[/cyan]"
            + (" [dim](默认)[/dim]" if key == DEFAULT_SOURCE else "")
        )
        branch.add(f"{meta['display_name']} — {meta['description']}")
        branch.add(f"港股支持: {'[green]是[/green]' if meta['hk_support'] else '[red]否[/red]'}")
        asset_types = ", ".join(t for t, _ in meta["supported_asset_types"])
        branch.add(f"资产类别: {asset_types}")

    console.print(tree)
    console.print()


def main():
    app()


if __name__ == "__main__":
    main()
