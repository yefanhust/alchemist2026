#!/usr/bin/env python3
"""
数据源信息查询脚本

查询各数据源的 high-level 信息，包括：
- 是否支持港股（HKEX）
- 支持的港股替代标的（US ADR 等）
- 支持的金融资产类别
- 支持的数据间隔
- 限流策略
- 各类型支持的全部标的列表
- 按行业（Sector/Industry）过滤股票

使用方法:
    python scripts/datasource_info.py info
    python scripts/datasource_info.py info --source alphavantage
    python scripts/datasource_info.py hk-support
    python scripts/datasource_info.py search --keyword Tencent
    python scripts/datasource_info.py list-sources
    python scripts/datasource_info.py list-assets stock
    python scripts/datasource_info.py list-assets etf --limit 100
    python scripts/datasource_info.py list-assets forex
    python scripts/datasource_info.py list-assets crypto --filter BTC
    python scripts/datasource_info.py list-assets stock --exchange NASDAQ
    python scripts/datasource_info.py list-assets stock --sector Technology
    python scripts/datasource_info.py list-assets stock --industry Software
    python scripts/datasource_info.py list-assets stock --export stocks.csv
    python scripts/datasource_info.py fetch-sector-info --exchange NASDAQ --limit 100
    python scripts/datasource_info.py list-sectors
"""

import asyncio
import csv
import io
import os
import sys
from typing import Optional, List, Dict, Any

import aiohttp
import aiosqlite
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from data.providers.alphavantage import AlphaVantageProvider
from data.providers import DataProvider, DataInterval
from data.cache.sqlite_cache import SQLiteCache
from utils.config import load_config, get_data_cache_path
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
        plan = getattr(config.alphavantage, 'plan', None)
        return AlphaVantageProvider(api_key=api_key, plan=plan)
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


# Alpha Vantage 资产列表 URL
AV_LISTING_STATUS_URL = "https://www.alphavantage.co/query"
AV_PHYSICAL_CURRENCY_URL = "https://www.alphavantage.co/physical_currency_list/"
AV_DIGITAL_CURRENCY_URL = "https://www.alphavantage.co/digital_currency_list/"

# 股票信息缓存数据库路径
STOCK_INFO_DB_PATH = get_data_cache_path("stock_info.db")


async def _fetch_stock_overview(
    session: aiohttp.ClientSession,
    api_key: str,
    symbol: str,
) -> Optional[Dict[str, Any]]:
    """
    获取单个股票的 OVERVIEW 信息（包含 Sector 和 Industry）

    Args:
        session: aiohttp 会话
        api_key: API 密钥
        symbol: 股票代码

    Returns:
        股票信息字典，包含 sector, industry 等字段
    """
    params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": api_key,
    }

    try:
        async with session.get(AV_LISTING_STATUS_URL, params=params) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()

            # 检查是否有错误或限流
            if "Error Message" in data or "Note" in data or "Information" in data:
                return None

            # 检查是否有数据
            if not data or "Symbol" not in data:
                return None

            # 解析字段
            def safe_float(val):
                try:
                    if val and val != "None" and val != "-":
                        return float(val)
                except (ValueError, TypeError):
                    pass
                return None

            return {
                "symbol": data.get("Symbol", symbol),
                "name": data.get("Name"),
                "exchange": data.get("Exchange"),
                "asset_type": data.get("AssetType"),
                "sector": data.get("Sector"),
                "industry": data.get("Industry"),
                "country": data.get("Country"),
                "currency": data.get("Currency"),
                "description": data.get("Description"),
                "market_cap": safe_float(data.get("MarketCapitalization")),
                "pe_ratio": safe_float(data.get("PERatio")),
                "dividend_yield": safe_float(data.get("DividendYield")),
                "eps": safe_float(data.get("EPS")),
                "beta": safe_float(data.get("Beta")),
                "high_52week": safe_float(data.get("52WeekHigh")),
                "low_52week": safe_float(data.get("52WeekLow")),
            }

    except Exception as e:
        console.print(f"[dim]获取 {symbol} 信息失败: {e}[/dim]")
        return None


async def _get_stock_info_with_cache(
    api_key: str,
    symbols: List[str],
    cache: SQLiteCache,
    force_refresh: bool = False,
    plan: str = "free",
) -> Dict[str, Dict[str, Any]]:
    """
    批量获取股票信息，优先使用缓存

    Args:
        api_key: API 密钥
        symbols: 股票代码列表
        cache: SQLite 缓存实例
        force_refresh: 是否强制刷新（忽略缓存）
        plan: API plan 类型 ("free" 或 "premium")

    Returns:
        {symbol: info_dict} 映射
    """
    results: Dict[str, Dict[str, Any]] = {}
    symbols_to_fetch: List[str] = []

    # 第一步：从缓存中查找
    if not force_refresh:
        cached = await cache.get_stock_info_batch(symbols)
        for symbol in symbols:
            symbol_upper = symbol.upper()
            if symbol_upper in cached and cached[symbol_upper].get("sector"):
                results[symbol_upper] = cached[symbol_upper]
            else:
                symbols_to_fetch.append(symbol_upper)
    else:
        symbols_to_fetch = [s.upper() for s in symbols]

    if not symbols_to_fetch:
        return results

    # 根据 plan 设置限流间隔
    if plan == "premium":
        # Premium: 75 次/分钟 → 60/75 = 0.8 秒
        rate_limit_interval = 0.8
        rate_limit_desc = "75 次/分钟"
    else:
        # Free: 5 次/分钟 → 60/5 = 12 秒
        rate_limit_interval = 12.0
        rate_limit_desc = "5 次/分钟"

    # 第二步：从 API 获取缺失的信息
    console.print(f"\n[dim]需要从 API 获取 {len(symbols_to_fetch)} 个股票的行业信息...[/dim]")
    console.print(f"[dim]API Plan: {plan} ({rate_limit_desc})[/dim]\n")

    async with aiohttp.ClientSession() as session:
        # 使用进度条显示
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]获取股票行业信息...",
                total=len(symbols_to_fetch)
            )

            for i, symbol in enumerate(symbols_to_fetch):
                info = await _fetch_stock_overview(session, api_key, symbol)

                if info and info.get("sector"):
                    results[symbol] = info
                    # 保存到缓存
                    await cache.save_stock_info(symbol, info)

                progress.update(task, advance=1, description=f"[cyan]获取 {symbol}...")

                # 根据 plan 限流
                if i < len(symbols_to_fetch) - 1:
                    await asyncio.sleep(rate_limit_interval)

    return results


async def _fetch_alphavantage_listing(api_key: str, asset_type: str) -> List[Dict[str, str]]:
    """
    获取 Alpha Vantage 支持的资产列表

    Args:
        api_key: API 密钥
        asset_type: 资产类型 (stock, etf, forex, crypto)

    Returns:
        资产列表，每项包含 symbol, name 等字段
    """
    results: List[Dict[str, str]] = []

    async with aiohttp.ClientSession() as session:
        if asset_type in ("stock", "etf"):
            # 使用 LISTING_STATUS 端点获取股票/ETF
            params = {
                "function": "LISTING_STATUS",
                "apikey": api_key,
            }
            async with session.get(AV_LISTING_STATUS_URL, params=params) as resp:
                if resp.status != 200:
                    raise Exception(f"API 请求失败: {resp.status}")
                text = await resp.text()

            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                row_type = row.get("assetType", "").lower()
                # 筛选 stock 或 etf
                if asset_type == "stock" and row_type == "stock":
                    results.append({
                        "symbol": row.get("symbol", ""),
                        "name": row.get("name", ""),
                        "exchange": row.get("exchange", ""),
                        "ipo_date": row.get("ipoDate", ""),
                        "status": row.get("status", ""),
                    })
                elif asset_type == "etf" and row_type == "etf":
                    results.append({
                        "symbol": row.get("symbol", ""),
                        "name": row.get("name", ""),
                        "exchange": row.get("exchange", ""),
                        "ipo_date": row.get("ipoDate", ""),
                        "status": row.get("status", ""),
                    })

        elif asset_type == "forex":
            # 获取物理货币列表
            async with session.get(AV_PHYSICAL_CURRENCY_URL) as resp:
                if resp.status != 200:
                    raise Exception(f"获取外汇列表失败: {resp.status}")
                text = await resp.text()

            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                results.append({
                    "symbol": row.get("currency code", ""),
                    "name": row.get("currency name", ""),
                })

        elif asset_type == "crypto":
            # 获取数字货币列表
            async with session.get(AV_DIGITAL_CURRENCY_URL) as resp:
                if resp.status != 200:
                    raise Exception(f"获取加密货币列表失败: {resp.status}")
                text = await resp.text()

            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                results.append({
                    "symbol": row.get("currency code", ""),
                    "name": row.get("currency name", ""),
                })

    return results


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


@app.command("list-assets")
def list_assets(
    asset_type: str = typer.Argument(
        ...,
        help="资产类型: stock, etf, forex, crypto",
    ),
    source: str = typer.Option(DEFAULT_SOURCE, "--source", "-s", help="数据源名称"),
    config_file: Optional[str] = typer.Option(None, "--config", help="配置文件路径"),
    limit: int = typer.Option(50, "--limit", "-n", help="显示数量限制"),
    offset: int = typer.Option(0, "--offset", help="跳过前 N 条记录"),
    filter_keyword: Optional[str] = typer.Option(None, "--filter", "-f", help="按关键词过滤（代码或名称）"),
    exchange_filter: Optional[str] = typer.Option(None, "--exchange", "-e", help="按交易所过滤（仅 stock/etf）"),
    sector_filter: Optional[str] = typer.Option(None, "--sector", help="按行业大类过滤（如 Technology, Healthcare）"),
    industry_filter: Optional[str] = typer.Option(None, "--industry", help="按细分行业过滤（如 Software, Biotechnology）"),
    export_csv: Optional[str] = typer.Option(None, "--export", help="导出到 CSV 文件路径"),
    force_refresh: bool = typer.Option(False, "--refresh", help="强制从 API 刷新行业信息（忽略缓存）"),
):
    """
    列出数据源支持的全部标的

    按资产类型列出所有支持的标的代码和名称。
    支持按行业（Sector/Industry）过滤，会自动缓存已查询的股票信息。

    示例:
        list-assets stock                          # 列出股票
        list-assets etf --limit 100                # 列出 ETF，显示 100 条
        list-assets forex                          # 列出外汇货币
        list-assets crypto -f BTC                  # 列出加密货币，过滤含 BTC
        list-assets stock -e NASDAQ                # 列出 NASDAQ 股票
        list-assets stock --sector Technology      # 列出科技行业股票
        list-assets stock --industry Software      # 列出软件行业股票
        list-assets stock --sector Healthcare --industry Biotechnology  # 组合过滤
        list-assets stock --export stocks.csv      # 导出到 CSV
        list-assets stock --sector Tech --refresh  # 强制刷新行业信息
    """
    setup_logger(level="WARNING")

    # 验证资产类型
    valid_types = ["stock", "etf", "forex", "crypto"]
    if asset_type.lower() not in valid_types:
        console.print(f"[red]错误: 无效的资产类型 '{asset_type}'[/red]")
        console.print(f"可选类型: {', '.join(valid_types)}")
        raise typer.Exit(1)

    asset_type = asset_type.lower()

    # 获取 API Key
    config = load_config(config_file)
    api_key = None
    if source == "alphavantage":
        api_key = getattr(config.alphavantage, 'api_key', None) or os.getenv("ALPHAVANTAGE_API_KEY")
        if not api_key:
            console.print("[red]错误: 未设置 Alpha Vantage API Key[/red]")
            raise typer.Exit(1)

    # 获取资产列表
    console.print(f"\n[dim]正在获取 {asset_type} 列表...[/dim]")

    async def _fetch():
        return await _fetch_alphavantage_listing(api_key, asset_type)

    try:
        assets = asyncio.run(_fetch())
    except Exception as e:
        console.print(f"[red]获取列表失败: {e}[/red]")
        raise typer.Exit(1)

    if not assets:
        console.print(f"[yellow]未找到 {asset_type} 类型的资产[/yellow]")
        raise typer.Exit(0)

    # 应用过滤
    if filter_keyword:
        kw = filter_keyword.upper()
        assets = [
            a for a in assets
            if kw in a.get("symbol", "").upper() or kw in a.get("name", "").upper()
        ]

    if exchange_filter and asset_type in ("stock", "etf"):
        ex = exchange_filter.upper()
        assets = [a for a in assets if ex in a.get("exchange", "").upper()]

    # 按行业过滤（仅 stock/etf）
    if (sector_filter or industry_filter) and asset_type in ("stock", "etf"):
        console.print(f"\n[cyan]正在按行业过滤...[/cyan]")

        async def _filter_by_sector():
            cache = SQLiteCache(db_path=STOCK_INFO_DB_PATH)
            await cache._ensure_initialized()

            # 策略：优先从缓存查询，避免遍历全部股票
            # 1. 先查缓存中匹配的股票
            cached_matches = await cache.search_stocks_by_sector(sector_filter, industry_filter)

            stats = await cache.get_stock_info_stats()
            console.print(f"[dim]股票信息缓存: {stats.get('total_stocks', 0)} 条记录, "
                         f"{stats.get('unique_sectors', 0)} 个行业大类, "
                         f"{stats.get('unique_industries', 0)} 个细分行业[/dim]")

            if cached_matches:
                console.print(f"[green]从缓存中找到 {len(cached_matches)} 只匹配的股票[/green]")
                # 转换为 assets 格式
                return [
                    {
                        "symbol": info["symbol"],
                        "name": info.get("name", ""),
                        "exchange": info.get("exchange", ""),
                        "sector": info.get("sector", ""),
                        "industry": info.get("industry", ""),
                    }
                    for info in cached_matches
                ]
            else:
                # 缓存中没有匹配结果
                console.print(f"[yellow]缓存中未找到匹配的股票[/yellow]")
                return []

        assets = asyncio.run(_filter_by_sector())

        if not assets:
            if sector_filter:
                console.print(f"[dim]Sector 过滤: {sector_filter}[/dim]")
            if industry_filter:
                console.print(f"[dim]Industry 过滤: {industry_filter}[/dim]")
            console.print(f"\n[dim]提示: 缓存为空或无匹配结果。使用以下命令积累缓存:[/dim]")
            console.print(f"[dim]  python scripts/datasource_info.py fetch-sector-info --exchange NASDAQ[/dim]")
            console.print(f"[dim]  python scripts/datasource_info.py list-sectors[/dim]")
            raise typer.Exit(0)

    total_count = len(assets)

    # 导出到 CSV（包含行业信息）
    if export_csv:
        with open(export_csv, "w", newline="", encoding="utf-8") as f:
            if assets:
                writer = csv.DictWriter(f, fieldnames=assets[0].keys())
                writer.writeheader()
                writer.writerows(assets)
        console.print(f"[green]已导出 {total_count} 条记录到 {export_csv}[/green]\n")
        return

    # 分页
    assets = assets[offset:offset + limit]

    if not assets:
        console.print(f"[yellow]在指定范围内未找到记录 (总数: {total_count})[/yellow]")
        raise typer.Exit(0)

    # 构建表格
    type_names = {
        "stock": "股票",
        "etf": "ETF",
        "forex": "外汇货币",
        "crypto": "加密货币",
    }

    # 检查是否有行业信息
    has_sector_info = sector_filter or industry_filter

    title_suffix = ""
    if sector_filter:
        title_suffix += f" | Sector: {sector_filter}"
    if industry_filter:
        title_suffix += f" | Industry: {industry_filter}"

    table = Table(
        title=f"{type_names[asset_type]}列表 (显示 {offset + 1}-{offset + len(assets)} / 共 {total_count} 条){title_suffix}"
    )
    table.add_column("#", style="dim", justify="right")
    table.add_column("代码", style="cyan")
    table.add_column("名称", style="white", max_width=40)

    if asset_type in ("stock", "etf"):
        table.add_column("交易所", style="magenta")
        if has_sector_info:
            table.add_column("行业", style="green", max_width=20)
            table.add_column("细分", style="yellow", max_width=25)
        else:
            table.add_column("IPO 日期", style="yellow")

    for idx, asset in enumerate(assets, offset + 1):
        if asset_type in ("stock", "etf"):
            if has_sector_info:
                table.add_row(
                    str(idx),
                    asset.get("symbol", ""),
                    asset.get("name", ""),
                    asset.get("exchange", ""),
                    asset.get("sector", ""),
                    asset.get("industry", ""),
                )
            else:
                table.add_row(
                    str(idx),
                    asset.get("symbol", ""),
                    asset.get("name", ""),
                    asset.get("exchange", ""),
                    asset.get("ipo_date", ""),
                )
        else:
            table.add_row(
                str(idx),
                asset.get("symbol", ""),
                asset.get("name", ""),
            )

    console.print(table)

    # 分页提示
    if offset + limit < total_count:
        next_offset = offset + limit
        console.print(
            f"\n[dim]提示: 使用 --offset {next_offset} 查看更多记录[/dim]"
        )
    console.print()


@app.command("fetch-sector-info")
def fetch_sector_info(
    source: str = typer.Option(DEFAULT_SOURCE, "--source", "-s", help="数据源名称"),
    config_file: Optional[str] = typer.Option(None, "--config", help="配置文件路径"),
    exchange_filter: Optional[str] = typer.Option(None, "--exchange", "-e", help="按交易所过滤（如 NASDAQ, NYSE）"),
    filter_keyword: Optional[str] = typer.Option(None, "--filter", "-f", help="按关键词过滤（代码或名称）"),
    limit: int = typer.Option(100, "--limit", "-n", help="获取数量限制"),
    offset: int = typer.Option(0, "--offset", help="跳过前 N 条记录"),
):
    """
    批量获取股票行业信息并缓存

    从 Alpha Vantage OVERVIEW API 获取股票的 Sector 和 Industry 信息，
    并保存到本地缓存。建议先用 --exchange 缩小范围。

    示例:
        fetch-sector-info --exchange NASDAQ --limit 100    # NASDAQ 前 100 只
        fetch-sector-info --exchange NYSE --limit 200      # NYSE 前 200 只
        fetch-sector-info --filter AAPL --limit 10         # 含 AAPL 的前 10 只
        fetch-sector-info --offset 100 --limit 100         # 跳过前 100，取接下来 100
    """
    setup_logger(level="WARNING")

    # 获取配置
    config = load_config(config_file)
    api_key = getattr(config.alphavantage, 'api_key', None) or os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        console.print("[red]错误: 未设置 Alpha Vantage API Key[/red]")
        raise typer.Exit(1)

    plan = getattr(config.alphavantage, 'plan', None) or os.getenv("ALPHAVANTAGE_PLAN", "free")

    # 获取股票列表
    console.print(f"\n[dim]正在获取股票列表...[/dim]")

    async def _fetch():
        return await _fetch_alphavantage_listing(api_key, "stock")

    try:
        assets = asyncio.run(_fetch())
    except Exception as e:
        console.print(f"[red]获取列表失败: {e}[/red]")
        raise typer.Exit(1)

    # 应用过滤
    if filter_keyword:
        kw = filter_keyword.upper()
        assets = [
            a for a in assets
            if kw in a.get("symbol", "").upper() or kw in a.get("name", "").upper()
        ]

    if exchange_filter:
        ex = exchange_filter.upper()
        assets = [a for a in assets if ex in a.get("exchange", "").upper()]

    # 分页
    total_available = len(assets)
    assets = assets[offset:offset + limit]

    if not assets:
        console.print(f"[yellow]未找到股票 (总数: {total_available})[/yellow]")
        raise typer.Exit(0)

    symbols = [a.get("symbol", "") for a in assets]
    console.print(f"[cyan]将获取 {len(symbols)} 只股票的行业信息 (共 {total_available} 只可用)[/cyan]")

    # 获取并缓存
    async def _fetch_and_cache():
        cache = SQLiteCache(db_path=STOCK_INFO_DB_PATH)
        await cache._ensure_initialized()

        # 获取股票信息（强制刷新）
        stock_info_map = await _get_stock_info_with_cache(
            api_key, symbols, cache, force_refresh=False, plan=plan
        )

        # 统计
        stats = await cache.get_stock_info_stats()
        return len(stock_info_map), stats

    fetched_count, stats = asyncio.run(_fetch_and_cache())

    console.print(f"\n[green]完成! 成功获取 {fetched_count} 只股票的行业信息[/green]")
    console.print(f"[dim]缓存统计: {stats.get('total_stocks', 0)} 条记录, "
                 f"{stats.get('unique_sectors', 0)} 个行业大类, "
                 f"{stats.get('unique_industries', 0)} 个细分行业[/dim]")

    if offset + limit < total_available:
        next_offset = offset + limit
        console.print(f"\n[dim]提示: 使用 --offset {next_offset} 继续获取更多[/dim]")

    console.print()


@app.command("list-sectors")
def list_sectors():
    """
    列出已缓存的所有行业分类

    显示本地缓存中存储的所有行业大类（Sector）和细分行业（Industry）。
    这些数据来自之前使用 --sector 或 --industry 过滤时查询的股票信息。
    """
    async def _list():
        cache = SQLiteCache(db_path=STOCK_INFO_DB_PATH)
        await cache._ensure_initialized()

        # 获取统计信息
        stats = await cache.get_stock_info_stats()

        # 获取所有 sector
        sectors = await cache.get_all_sectors()

        # 获取每个 sector 下的 industry
        async with aiosqlite.connect(cache.db_path) as db:
            db.row_factory = aiosqlite.Row

            sector_industries = {}
            for sector in sectors:
                cursor = await db.execute(
                    """
                    SELECT DISTINCT industry, COUNT(*) as count
                    FROM stock_info
                    WHERE sector = ? AND industry IS NOT NULL AND industry != ''
                    GROUP BY industry
                    ORDER BY count DESC
                    """,
                    (sector,)
                )
                rows = await cursor.fetchall()
                sector_industries[sector] = [(row["industry"], row["count"]) for row in rows]

        return stats, sectors, sector_industries

    stats, sectors, sector_industries = asyncio.run(_list())

    console.print()

    if not sectors:
        console.print("[yellow]本地缓存中暂无行业数据[/yellow]")
        console.print("\n[dim]提示: 使用 list-assets stock --sector <行业> 来获取并缓存行业信息[/dim]")
        console.print("[dim]例如: list-assets stock --sector Technology --limit 10[/dim]")
        return

    # 显示统计
    console.print(Panel(
        f"[cyan]已缓存股票数:[/cyan] {stats.get('total_stocks', 0)}\n"
        f"[cyan]行业大类数:[/cyan] {stats.get('unique_sectors', 0)}\n"
        f"[cyan]细分行业数:[/cyan] {stats.get('unique_industries', 0)}",
        title="股票信息缓存统计",
        border_style="blue",
    ))

    # 显示行业树
    tree = Tree("[bold]行业分类[/bold]")

    for sector in sectors:
        industries = sector_industries.get(sector, [])
        sector_branch = tree.add(f"[cyan]{sector}[/cyan] ({sum(c for _, c in industries)} 只股票)")

        for industry, count in industries[:10]:  # 最多显示 10 个细分行业
            sector_branch.add(f"[dim]{industry}[/dim] ({count})")

        if len(industries) > 10:
            sector_branch.add(f"[dim]... 还有 {len(industries) - 10} 个细分行业[/dim]")

    console.print(tree)
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
