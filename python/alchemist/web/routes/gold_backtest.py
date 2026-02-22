"""
黄金策略回测 API 路由
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Query

from web.schemas.gold_backtest import (
    GoldBacktestResponse,
    DailySignalPoint,
    FactorScores,
    PricePoint,
    StrategyParams,
    ThresholdParams,
    PositionParams,
)
from data.models import MarketData

router = APIRouter()

# 跨市场 ETF 符号映射（与 assets.py 中的 CROSS_MARKET_ETF_SYMBOLS 一致）
CROSS_MARKET_SYMBOLS = {
    "gold_miners": "GDX",
    "sp500": "SPY",
    "usd_index": "UUP",
    "inflation_expectations": "TIP",
    "treasury": "TLT",
    "vix": "VIXY",
}

INDICATOR_LABELS = {
    "GDX": "GDX (矿业ETF)",
    "SPY": "SPY (标普500)",
    "UUP": "UUP (美元指数)",
    "TIP": "TIP (通胀预期)",
    "TLT": "TLT (长期国债)",
    "VIXY": "VIXY (波动率)",
}

AVAILABLE_INDICATORS = list(INDICATOR_LABELS.keys())

# 优化参数文件默认路径
OPTIMIZED_PARAMS_PATH = Path(__file__).resolve().parent.parent.parent.parent.parent / "data" / "output" / "gold_optimized_params.yaml"


def _build_strategy_params(strategy, source: str, source_file: Optional[str] = None) -> StrategyParams:
    """从策略对象提取参数构建 StrategyParams"""
    w = strategy.weights
    t = strategy.thresholds
    p = strategy.position_config
    return StrategyParams(
        source=source,
        source_file=source_file,
        weights=FactorScores(
            technical=round(w.technical, 4),
            cross_market=round(w.cross_market, 4),
            sentiment=round(w.sentiment, 4),
            macro=round(w.macro, 4),
        ),
        thresholds=ThresholdParams(
            boost_buy=round(t.boost_buy, 4),
            normal_buy=round(t.normal_buy, 4),
            reduce_buy=round(t.reduce_buy, 4),
            skip_buy=round(t.skip_buy, 4),
            partial_sell=round(t.partial_sell, 4),
        ),
        position=PositionParams(
            buy_day=p.buy_day,
            boost_multiplier=round(p.boost_multiplier, 4),
            reduce_multiplier=round(p.reduce_multiplier, 4),
            sell_fraction=round(p.sell_fraction, 4),
            force_sell_interval_days=p.force_sell_interval_days,
            force_sell_fraction=round(p.force_sell_fraction, 4) if p.force_sell_fraction else None,
            force_sell_profit_thresh=round(p.force_sell_profit_threshold, 4) if p.force_sell_profit_threshold else None,
        ),
    )


@router.get("/signals", response_model=GoldBacktestResponse)
async def get_gold_backtest_signals(
    request: Request,
    start_date: str = Query(description="开始日期 (YYYY-MM-DD)"),
    end_date: str = Query(description="结束日期 (YYYY-MM-DD)"),
    indicators: Optional[str] = Query(
        default=None,
        description="叠加指标，逗号分隔 (如 GDX,SPY,UUP)",
    ),
    use_optimized: bool = Query(
        default=True,
        description="是否使用优化参数（默认 true，自动查找优化参数文件）",
    ),
) -> GoldBacktestResponse:
    """
    计算黄金策略回测信号

    返回每日的因子得分、战术动作、信号标记以及价格数据。
    """
    # 解析日期
    try:
        start_dt = datetime.fromisoformat(start_date)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"无效的开始日期: {start_date}")
    try:
        end_dt = datetime.fromisoformat(end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"无效的结束日期: {end_date}")

    if start_dt >= end_dt:
        raise HTTPException(status_code=400, detail="开始日期必须早于结束日期")

    # 解析叠加指标
    requested_indicators = []
    if indicators:
        requested_indicators = [
            s.strip().upper() for s in indicators.split(",") if s.strip()
        ]

    service = request.app.state.cache_service

    # 策略需要预热数据（约 210 天历史），提前读取
    warmup_days = 300
    fetch_start = start_dt - timedelta(days=warmup_days)

    # 读取 GLD 数据
    gld_data = await service.get_ohlcv("GLD", "1d", fetch_start, end_dt)
    if gld_data is None or gld_data.is_empty:
        raise HTTPException(status_code=404, detail="未找到 GLD 缓存数据，请先运行回测获取数据")

    # 读取跨市场数据
    cross_market_data = {}
    for data_key, ticker in CROSS_MARKET_SYMBOLS.items():
        md = await service.get_ohlcv(ticker, "1d", fetch_start, end_dt)
        if md and not md.is_empty:
            cross_market_data[data_key] = md

    # 读取叠加指标数据（用于前端展示）
    indicator_series = {}
    for ticker in requested_indicators:
        if ticker not in AVAILABLE_INDICATORS:
            continue
        md = await service.get_ohlcv(ticker, "1d", start_dt, end_dt)
        if md and not md.is_empty:
            indicator_series[ticker] = [
                {"date": d.timestamp.strftime("%Y-%m-%d"), "close": d.close}
                for d in md.data
            ]

    # 初始化策略
    from strategy.gold import GoldTradingStrategy
    from core.asset import Asset
    from core.portfolio import Portfolio

    # 根据 use_optimized 决定是否加载优化参数
    param_source = "default"
    param_file = None
    if use_optimized and OPTIMIZED_PARAMS_PATH.is_file():
        strategy = GoldTradingStrategy.from_optimized_params(
            yaml_path=str(OPTIMIZED_PARAMS_PATH),
            cross_market_symbols=CROSS_MARKET_SYMBOLS,
        )
        param_source = "optimized"
        param_file = OPTIMIZED_PARAMS_PATH.name
    else:
        strategy = GoldTradingStrategy(cross_market_symbols=CROSS_MARKET_SYMBOLS)

    strategy_params = _build_strategy_params(strategy, param_source, param_file)

    asset = Asset(symbol="GLD")
    portfolio = Portfolio(initial_capital=100000)
    required = strategy.required_history

    # 找到 GLD 数据中落在用户请求的 [start_dt, end_dt] 区间内的索引
    all_dates = [d.timestamp for d in gld_data.data]

    # 逐日计算信号
    signals: List[DailySignalPoint] = []
    gold_prices: List[PricePoint] = []

    for i, ohlcv in enumerate(gld_data.data):
        ts = ohlcv.timestamp

        # 只输出用户请求区间内的数据点
        if ts < start_dt or ts > end_dt:
            continue

        # 价格点
        gold_prices.append(PricePoint(
            date=ts.strftime("%Y-%m-%d"),
            open=ohlcv.open,
            high=ohlcv.high,
            low=ohlcv.low,
            close=ohlcv.close,
            volume=ohlcv.volume,
        ))

        # 检查是否有足够的历史数据计算因子
        if i + 1 < required:
            signals.append(DailySignalPoint(
                date=ts.strftime("%Y-%m-%d"),
                composite_score=0.0,
                factor_scores=FactorScores(
                    technical=0.0, cross_market=0.0, sentiment=0.0, macro=0.0,
                ),
                tactical_action="hold",
            ))
            continue

        # 构造截至当天的数据切片
        slice_data = gld_data.data[: i + 1]
        closes = np.array([d.close for d in slice_data])
        volumes = np.array([d.volume for d in slice_data])

        market_data_dict = {
            "gold_etf": closes,
            "gold_etf_volume": volumes,
        }

        # 添加跨市场数据（截至当天）
        for data_key, md in cross_market_data.items():
            # 过滤到 <= ts 的数据
            filtered = [d.close for d in md.data if d.timestamp <= ts]
            if filtered:
                market_data_dict[data_key] = np.array(filtered)

        # 计算因子得分
        factors = strategy.calculate_factors(market_data_dict)
        composite = strategy._compute_composite_score(factors)

        # 判断战术动作
        weekday = ts.weekday()
        is_buy_day = weekday == strategy.position_config.buy_day

        if is_buy_day:
            action = strategy._describe_action(composite)
        else:
            action = "hold"

        # 生成实际信号（含冷却期、持仓期逻辑）
        slice_md = MarketData(symbol="GLD", data=list(slice_data))
        cross_np = {}
        for data_key, md in cross_market_data.items():
            filtered = [d.close for d in md.data if d.timestamp <= ts]
            if filtered:
                cross_np[data_key] = np.array(filtered)

        actual_signals = strategy._generate_tactical_signals(
            asset, slice_md, cross_np, portfolio,
        )

        signal_type = None
        signal_strength = None
        if actual_signals:
            sig = actual_signals[0]
            signal_type = sig.signal_type.value
            signal_strength = sig.strength
            # 如果有实际信号，用信号的 tactical_action
            if sig.metadata.get("tactical_action"):
                action = sig.metadata["tactical_action"]

        # 卖出信号也需要判断（非买入日也可能卖出）
        if not is_buy_day and composite < strategy.thresholds.partial_sell:
            has_pos = strategy._has_position.get(asset.symbol, False)
            if has_pos:
                action = "partial_sell"

        signals.append(DailySignalPoint(
            date=ts.strftime("%Y-%m-%d"),
            composite_score=round(composite, 4),
            factor_scores=FactorScores(
                technical=round(factors.get("technical", 0.0), 4),
                cross_market=round(factors.get("cross_market", 0.0), 4),
                sentiment=round(factors.get("sentiment", 0.0), 4),
                macro=round(factors.get("macro", 0.0), 4),
            ),
            tactical_action=action,
            signal_type=signal_type,
            signal_strength=round(signal_strength, 4) if signal_strength is not None else None,
        ))

    return GoldBacktestResponse(
        gold_prices=gold_prices,
        signals=signals,
        available_indicators=AVAILABLE_INDICATORS,
        indicator_series=indicator_series,
        factor_weights=FactorScores(
            technical=round(strategy.weights.technical, 4),
            cross_market=round(strategy.weights.cross_market, 4),
            sentiment=round(strategy.weights.sentiment, 4),
            macro=round(strategy.weights.macro, 4),
        ),
        strategy_params=strategy_params,
    )
