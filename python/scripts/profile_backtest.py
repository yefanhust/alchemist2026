#!/usr/bin/env python3
"""回测性能分析脚本"""

import asyncio
import time
import cProfile
import pstats
import io
from datetime import datetime
from functools import wraps

from core.portfolio import Portfolio
from data.providers.alphavantage import AlphaVantageProvider
from data.cache.sqlite_cache import SQLiteCache
from strategy.gold.strategy import GoldTradingStrategy
from strategy.builtin.weekly_dca import WeeklyDCAStrategy
from simulation.backtest import Backtester
from simulation.broker import BrokerConfig, VirtualBroker
from simulation.engine import SimulationEngine
from utils.config import load_config, get_data_cache_path
from utils.logger import setup_logger

setup_logger(level="WARNING")

GOLD_CROSS_MARKET_SYMBOLS = {
    "gold_miners": "GDX",
    "sp500": "SPY",
    "usd_index": "UUP",
    "inflation_expectations": "TIP",
    "treasury": "TLT",
    "vix": "VIXY",
}
GOLD_SPECIAL_DATA = {
    "treasury_yield": {"type": "treasury_yield", "maturity": "10year"},
    "eur_usd": {"type": "forex", "from": "EUR", "to": "USD"},
    "usd_jpy": {"type": "forex", "from": "USD", "to": "JPY"},
}
GOLD_PRIMARY_ETF = "GLD"
GOLD_ALL_SYMBOLS = [GOLD_PRIMARY_ETF] + list(GOLD_CROSS_MARKET_SYMBOLS.values())


async def main():
    config = load_config()
    start_date = datetime(1960, 1, 1)
    end_date = datetime(2025, 12, 31)

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

    print("=" * 60)
    print("黄金回测性能分析")
    print("=" * 60)

    # ---- Phase 1: 数据加载 ----
    print("\n[Phase 1] 数据加载")
    t0 = time.perf_counter()
    data = await backtester.fetch_data(GOLD_ALL_SYMBOLS, start_date, end_date)
    t_fetch = time.perf_counter() - t0
    print(f"  ETF 数据加载: {t_fetch:.3f}s")

    t0 = time.perf_counter()
    cross_market_symbols = dict(GOLD_CROSS_MARKET_SYMBOLS)
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
                cross_market_symbols[data_key] = special_data.symbol
        except Exception:
            pass
    t_special = time.perf_counter() - t0
    print(f"  特殊数据加载: {t_special:.3f}s")

    total_bars = sum(len(d) for d in data.values())
    print(f"  总数据量: {len(data)} 个标的, {total_bars} 条数据")
    for sym, md in data.items():
        if not md.is_empty:
            print(f"    {sym}: {len(md)} bars "
                  f"({md.start_date.strftime('%Y-%m-%d')} ~ "
                  f"{md.end_date.strftime('%Y-%m-%d')})")

    # ---- Phase 2: 引擎运行 ----
    print("\n[Phase 2] 引擎运行 - 黄金多因子策略")

    strat = GoldTradingStrategy(cross_market_symbols=cross_market_symbols, end_date=end_date)
    valid_symbols = [s for s, d in data.items() if not d.is_empty]

    actual_start_dates = [data[s].start_date for s in valid_symbols]
    latest_start = max(actual_start_dates)
    aligned_data = {s: data[s].slice(latest_start, end_date) for s in valid_symbols}

    portfolio = Portfolio(initial_capital=100000.0, name=strat.name)
    broker = VirtualBroker(config=broker_config)
    engine = SimulationEngine(
        portfolio=portfolio, broker=broker, strategies=[strat],
    )

    # monkey-patch 收集耗时
    timings = {
        "on_market_data": 0.0,
        "process_strategies": 0.0,
        "process_signals": 0.0,
        "on_day_end": 0.0,
        "step_count": 0,
        "day_count": 0,
    }

    orig_omd = engine.on_market_data
    orig_ps = engine.process_strategies
    orig_psig = engine.process_signals
    orig_ode = engine.on_day_end

    def timed_omd(symbol, ohlcv):
        t = time.perf_counter()
        r = orig_omd(symbol, ohlcv)
        timings["on_market_data"] += time.perf_counter() - t
        timings["step_count"] += 1
        return r

    def timed_ps():
        t = time.perf_counter()
        r = orig_ps()
        timings["process_strategies"] += time.perf_counter() - t
        return r

    def timed_psig(signals):
        t = time.perf_counter()
        r = orig_psig(signals)
        timings["process_signals"] += time.perf_counter() - t
        return r

    def timed_ode():
        t = time.perf_counter()
        r = orig_ode()
        timings["on_day_end"] += time.perf_counter() - t
        timings["day_count"] += 1
        return r

    engine.on_market_data = timed_omd
    engine.process_strategies = timed_ps
    engine.process_signals = timed_psig
    engine.on_day_end = timed_ode

    profiler = cProfile.Profile()
    t_engine_start = time.perf_counter()
    profiler.enable()
    engine.run(aligned_data)
    profiler.disable()
    t_engine = time.perf_counter() - t_engine_start

    print(f"\n  engine.run() 总耗时: {t_engine:.3f}s")
    print(f"  步骤数: {timings['step_count']}  天数: {timings['day_count']}")
    for key in ["on_market_data", "process_strategies", "process_signals", "on_day_end"]:
        v = timings[key]
        print(f"  {key:24s} {v:.3f}s ({v/t_engine*100:5.1f}%)")
    overhead = t_engine - sum(timings[k] for k in ["on_market_data", "process_strategies", "process_signals", "on_day_end"])
    print(f"  {'其他开销':24s} {overhead:.3f}s ({overhead/t_engine*100:5.1f}%)")

    # ---- Phase 3: 结果计算 ----
    print("\n[Phase 3] 结果计算")
    t0 = time.perf_counter()
    result = backtester._calculate_result(
        strategy=strat, portfolio=portfolio, broker=broker,
        start_date=start_date, end_date=end_date,
    )
    t_calc = time.perf_counter() - t0
    print(f"  _calculate_result: {t_calc:.3f}s")

    # ---- Phase 4: 基准策略 ----
    print("\n[Phase 4] 基准策略 (5个 WeeklyDCA)")
    gld_data = {GOLD_PRIMARY_ETF: aligned_data[GOLD_PRIMARY_ETF]}
    t_bench_total = 0
    for day in range(5):
        t0 = time.perf_counter()
        dca = WeeklyDCAStrategy(target_day=day, end_date=end_date)
        p = Portfolio(initial_capital=100000.0, name=dca.name)
        b = VirtualBroker(config=broker_config)
        e = SimulationEngine(portfolio=p, broker=b, strategies=[dca])
        e.run(gld_data)
        backtester._calculate_result(
            strategy=dca, portfolio=p, broker=b,
            start_date=start_date, end_date=end_date,
        )
        elapsed = time.perf_counter() - t0
        t_bench_total += elapsed
    print(f"  5个基准策略总耗时: {t_bench_total:.3f}s (平均 {t_bench_total/5:.3f}s)")

    # ---- cProfile ----
    print("\n" + "=" * 60)
    print("cProfile Top 25 (cumulative)")
    print("=" * 60)
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(25)
    print(stream.getvalue())

    print("=" * 60)
    print("cProfile Top 25 (tottime - 自身耗时)")
    print("=" * 60)
    stream2 = io.StringIO()
    pstats.Stats(profiler, stream=stream2).sort_stats("tottime").print_stats(25)
    print(stream2.getvalue())

    # ---- 汇总 ----
    print("=" * 60)
    print("耗时汇总")
    print("=" * 60)
    total = t_fetch + t_special + t_engine + t_calc + t_bench_total
    print(f"  数据加载:      {t_fetch + t_special:.3f}s ({(t_fetch + t_special)/total*100:.1f}%)")
    print(f"  黄金策略引擎:  {t_engine:.3f}s ({t_engine/total*100:.1f}%)")
    print(f"  结果计算:      {t_calc:.3f}s ({t_calc/total*100:.1f}%)")
    print(f"  基准策略(x5):  {t_bench_total:.3f}s ({t_bench_total/total*100:.1f}%)")
    print(f"  总计:          {total:.3f}s")

    # ---- 每步分析 ----
    print("\n" + "=" * 60)
    print("每步分析")
    print("=" * 60)
    if timings["step_count"] > 0:
        avg_step = t_engine / timings["step_count"] * 1000
        avg_strat = timings["process_strategies"] / timings["step_count"] * 1000
        print(f"  每步平均耗时: {avg_step:.4f}ms")
        print(f"  每步策略计算: {avg_strat:.4f}ms")

    n_gld = len(aligned_data.get("GLD", []))
    n_sym = len(aligned_data)
    print(f"\n  GLD 数据条数: {n_gld}")
    print(f"  跨市场标的数: {n_sym}")
    print(f"\n  on_data 每次调用的热点操作:")
    print(f"    1. 遍历 {n_sym-1} 个跨市场标的: np.array([d.close for d in data[ticker].data])")
    print(f"       → 随引擎 append 数据增长, 第 N 天需遍历 N 个 OHLCV 对象")
    print(f"    2. closes = np.array([d.close for d in data.data]) → 同上, GLD 主数据")
    print(f"    3. 4类因子各自的 numpy 计算 (均值/标准差/相关系数等)")
    print(f"\n  复杂度: O(days × symbols × avg_data_length)")
    print(f"  ≈ {n_gld} × {n_sym} × {n_gld//2} ≈ {n_gld * n_sym * n_gld // 2:,} 次元素访问")

    await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
