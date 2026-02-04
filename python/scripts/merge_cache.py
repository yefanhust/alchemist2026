#!/usr/bin/env python3
"""
缓存合并脚本

合并 data/cache 中相同 Symbol 和 Interval 的缓存条目。
例如：
  alphavantage:AAPL:1d:20250203:20260203
  alphavantage:AAPL:1d:20250204:20260204
会被合并为一个条目，数据去重并按时间排序。

用法：
  python python/scripts/merge_cache.py              # 预览合并计划
  python python/scripts/merge_cache.py --execute    # 执行合并
  python python/scripts/merge_cache.py --dry-run    # 详细预览（显示数据统计）
"""

import asyncio
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import typer
from rich.console import Console
from rich.table import Table

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "alchemist"))

from data.cache.sqlite_cache import SQLiteCache
from data.models import MarketData, OHLCV

console = Console()
app = typer.Typer(help="缓存合并工具")


def parse_cache_key(key: str) -> Tuple[str, str, str, str, str]:
    """
    解析缓存键格式: provider:symbol:interval:start_date:end_date

    Returns:
        (provider, symbol, interval, start_date, end_date)
    """
    match = re.match(r"(\w+):(\w+):(\w+):(\d+):(\d+)", key)
    if match:
        return match.groups()
    return None


def format_date(date_str: str) -> str:
    """格式化日期字符串 20250203 -> 2025-02-03"""
    if len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str


async def analyze_cache(cache: SQLiteCache) -> Dict[str, List[Tuple[str, dict]]]:
    """
    分析缓存，找出可合并的条目

    Returns:
        {(provider, symbol, interval): [(key, info), ...]}
    """
    keys = await cache.keys("*")
    groups = defaultdict(list)

    for key in keys:
        parsed = parse_cache_key(key)
        if parsed:
            provider, symbol, interval, start, end = parsed
            group_key = (provider, symbol, interval)

            # 获取缓存数据统计
            data = await cache.get(key)
            if data and isinstance(data, MarketData):
                info = {
                    "key": key,
                    "start_date": start,
                    "end_date": end,
                    "count": len(data.data),
                    "data_start": data.start_date.strftime("%Y-%m-%d") if data.start_date else None,
                    "data_end": data.end_date.strftime("%Y-%m-%d") if data.end_date else None,
                }
                groups[group_key].append((key, info))

    # 只返回有多个条目的组
    return {k: v for k, v in groups.items() if len(v) > 1}


async def merge_cache_entries(
    cache: SQLiteCache,
    keys: List[str],
    provider: str,
    symbol: str,
    interval: str,
) -> Tuple[str, int, int]:
    """
    合并多个缓存条目

    Returns:
        (new_key, merged_count, original_total_count)
    """
    all_ohlcv = {}  # 使用 timestamp 作为键去重
    original_total = 0  # 原始数据总数

    for key in keys:
        data = await cache.get(key)
        if data and isinstance(data, MarketData):
            original_total += len(data.data)
            for ohlcv in data.data:
                # 使用 timestamp 去重，保留最新的数据
                all_ohlcv[ohlcv.timestamp] = ohlcv

    # 按时间排序
    sorted_ohlcv = sorted(all_ohlcv.values(), key=lambda x: x.timestamp)

    if not sorted_ohlcv:
        return None, 0, 0

    # 创建合并后的 MarketData
    merged_data = MarketData(
        symbol=symbol,
        data=sorted_ohlcv,
        metadata={
            "source": "cache_merged",
            "interval": interval,
            "merged_from": keys,
            "merged_at": datetime.now().isoformat(),
        },
    )

    # 生成新的缓存键
    start_date = sorted_ohlcv[0].timestamp.strftime("%Y%m%d")
    end_date = sorted_ohlcv[-1].timestamp.strftime("%Y%m%d")
    new_key = f"{provider}:{symbol}:{interval}:{start_date}:{end_date}"

    # 删除旧条目
    for key in keys:
        await cache.delete(key)

    # 保存合并后的数据
    await cache.set(new_key, merged_data)

    return new_key, len(sorted_ohlcv), original_total


@app.command()
def main(
    db_path: str = typer.Option(
        "./data/cache/market_data.db",
        "--db", "-d",
        help="缓存数据库路径"
    ),
    execute: bool = typer.Option(
        False,
        "--execute", "-e",
        help="执行合并操作（默认只预览）"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run", "-n",
        help="详细预览模式，显示数据统计"
    ),
):
    """合并相同 Symbol 的缓存条目"""

    async def run():
        cache = SQLiteCache(db_path=db_path)
        await cache._ensure_initialized()

        console.print(f"\n[bold]缓存数据库:[/bold] {db_path}\n")

        # 分析缓存
        mergeable = await analyze_cache(cache)

        if not mergeable:
            console.print("[green]✓ 没有需要合并的缓存条目[/green]")
            return

        # 显示合并计划
        table = Table(title="可合并的缓存条目")
        table.add_column("Symbol", style="cyan")
        table.add_column("Interval", style="magenta")
        table.add_column("条目数", justify="right")
        table.add_column("缓存键", style="dim")

        for (provider, symbol, interval), entries in mergeable.items():
            keys_display = "\n".join([
                f"  {info['key']} ({info['count']} 条)"
                for _, info in entries
            ])
            table.add_row(
                symbol,
                interval,
                str(len(entries)),
                keys_display,
            )

        console.print(table)

        if dry_run:
            # 详细预览
            console.print("\n[bold]详细数据统计:[/bold]")
            for (provider, symbol, interval), entries in mergeable.items():
                console.print(f"\n[cyan]{symbol}[/cyan] ({interval}):")
                for _, info in entries:
                    console.print(
                        f"  • {info['key']}\n"
                        f"    查询范围: {format_date(info['start_date'])} ~ {format_date(info['end_date'])}\n"
                        f"    实际数据: {info['data_start']} ~ {info['data_end']} ({info['count']} 条)"
                    )

        if not execute:
            console.print(
                "\n[yellow]提示:[/yellow] 使用 --execute 参数执行合并操作\n"
                "       使用 --dry-run 查看详细统计"
            )
            return

        # 执行合并
        console.print("\n[bold]执行合并...[/bold]\n")

        for (provider, symbol, interval), entries in mergeable.items():
            keys = [info["key"] for _, info in entries]
            original_count = sum(info["count"] for _, info in entries)

            new_key, merged_count, _ = await merge_cache_entries(
                cache, keys, provider, symbol, interval
            )

            if new_key:
                console.print(
                    f"[green]✓[/green] {symbol} ({interval}): "
                    f"{len(keys)} 条目合并为 1 条 "
                    f"({original_count} → {merged_count} 条数据，去重 {original_count - merged_count} 条)"
                )
                console.print(f"  新键: {new_key}")

        # 显示最终统计
        stats = await cache.stats()
        console.print(f"\n[bold]合并完成！[/bold]")
        console.print(f"  当前缓存条目: {stats['total_entries']}")
        console.print(f"  数据库大小: {stats['db_size_mb']} MB")

    asyncio.run(run())


if __name__ == "__main__":
    app()
