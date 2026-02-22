"""
黄金策略差分进化优化器

使用 scipy.optimize.differential_evolution 对黄金择时增强型定投策略的
全部权重、阈值和超参数进行全局优化，寻找远优于基准策略的参数组合。

防过拟合措施：
- Walk-forward 分割（训练集优化，验证集评估）
- 交易频率惩罚（过少或过多均扣分）
- 稳健性检查（训练集 vs 验证集 vs 全量对比）
"""

import multiprocessing
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import numpy as np
import yaml
from scipy.optimize import differential_evolution
from loguru import logger

from core.portfolio import Portfolio
from data.models import MarketData
from simulation.backtest import Backtester, BacktestResult
from simulation.broker import BrokerConfig
from strategy.gold.strategy import (
    GoldTradingStrategy,
    FactorWeights,
    TacticalThresholds,
    PositionConfig,
)


# 参数名称与索引映射
PARAM_NAMES = [
    # 因子权重（4个，需归一化）
    "w_technical",          # 0
    "w_cross_market",       # 1
    "w_sentiment",          # 2
    "w_macro",              # 3
    # 阈值（5个，需单调递减）
    "thresh_boost",         # 4
    "thresh_normal",        # 5
    "thresh_reduce",        # 6
    "thresh_skip",          # 7
    "thresh_sell",          # 8
    # 倍率（2个）
    "boost_multiplier",     # 9
    "reduce_multiplier",    # 10
    # 仓位管理（2个）
    "sell_fraction",        # 11
    "buy_day",              # 12
    # 强制止盈（3个）
    "force_sell_interval",  # 13
    "force_sell_fraction",  # 14
    "force_sell_profit_thresh",  # 15
]

# 参数边界
PARAM_BOUNDS = [
    (0.10, 0.60),   # w_technical
    (0.05, 0.40),   # w_cross_market
    (0.00, 0.30),   # w_sentiment
    (0.05, 0.40),   # w_macro
    (0.10, 0.60),   # thresh_boost
    (-0.10, 0.20),  # thresh_normal
    (-0.30, 0.00),  # thresh_reduce
    (-0.50, -0.10), # thresh_skip
    (-0.80, -0.30), # thresh_sell
    (1.50, 3.00),   # boost_multiplier
    (0.10, 0.80),   # reduce_multiplier
    (0.20, 0.80),   # sell_fraction
    (0, 4),          # buy_day (integer)
    (90, 365),       # force_sell_interval
    (0.10, 0.50),   # force_sell_fraction
    (0.02, 0.15),   # force_sell_profit_thresh
]


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, float]
    best_fitness: float
    train_result: Optional[BacktestResult]
    val_result: Optional[BacktestResult]
    full_result: Optional[BacktestResult]
    benchmark_result: Optional[BacktestResult]
    n_evaluations: int
    elapsed_seconds: float

    def save_params_yaml(self, path: str) -> None:
        """
        保存最优参数到 YAML 文件（人类可读，策略可加载）

        文件格式示例:
            optimized_at: "2026-02-22 10:30:00"
            fitness: 0.1234
            n_evaluations: 5000
            elapsed_seconds: 3600
            params:
              w_technical: 0.40
              ...
            performance:
              train: { total_return: 0.15, ... }
              validation: { ... }
              full: { ... }
              benchmark: { ... }
        """
        data: Dict[str, Any] = {
            "optimized_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fitness": float(round(self.best_fitness, 6)),
            "n_evaluations": int(self.n_evaluations),
            "elapsed_seconds": float(round(self.elapsed_seconds, 1)),
            "params": {k: float(round(v, 6)) if isinstance(v, (float, np.floating)) else int(v)
                       for k, v in self.best_params.items()},
        }

        def _summarize(r: Optional[BacktestResult]) -> Optional[Dict]:
            if r is None:
                return None
            return {
                "total_return": float(round(r.total_return, 6)),
                "annual_return": float(round(r.annual_return, 6)),
                "sharpe_ratio": float(round(r.sharpe_ratio, 4)),
                "sortino_ratio": float(round(r.sortino_ratio, 4)),
                "max_drawdown": float(round(r.max_drawdown, 6)),
                "volatility": float(round(r.volatility, 6)),
                "buy_count": int(r.buy_count),
                "sell_count": int(r.sell_count),
            }

        data["performance"] = {
            "train": _summarize(self.train_result),
            "validation": _summarize(self.val_result),
            "full": _summarize(self.full_result),
            "benchmark": _summarize(self.benchmark_result),
        }

        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info(f"优化结果已保存: {filepath}")

    @staticmethod
    def load_params_yaml(path: str) -> Dict[str, Any]:
        """
        从 YAML 文件加载优化参数

        Returns:
            params 字典，可直接传入 build_strategy()
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data["params"]


def decode_params(x: np.ndarray) -> Dict[str, Any]:
    """
    将优化器的连续向量解码为策略参数字典

    关键处理：
    1. 因子权重归一化为和为 1.0
    2. 阈值强制单调递减
    3. buy_day 四舍五入为整数
    4. force_sell_interval 四舍五入为整数
    """
    # 因子权重归一化
    raw_weights = x[0:4]
    w_sum = raw_weights.sum()
    if w_sum > 0:
        weights = raw_weights / w_sum
    else:
        weights = np.array([0.25, 0.25, 0.25, 0.25])

    # 阈值：强制单调递减 boost > normal > reduce > skip > sell
    thresholds = list(x[4:9])
    thresholds.sort(reverse=True)

    # 整数参数
    buy_day = int(round(np.clip(x[12], 0, 4)))
    force_sell_interval = int(round(x[13]))

    return {
        "w_technical": float(weights[0]),
        "w_cross_market": float(weights[1]),
        "w_sentiment": float(weights[2]),
        "w_macro": float(weights[3]),
        "thresh_boost": float(thresholds[0]),
        "thresh_normal": float(thresholds[1]),
        "thresh_reduce": float(thresholds[2]),
        "thresh_skip": float(thresholds[3]),
        "thresh_sell": float(thresholds[4]),
        "boost_multiplier": float(x[9]),
        "reduce_multiplier": float(x[10]),
        "sell_fraction": float(x[11]),
        "buy_day": buy_day,
        "force_sell_interval": force_sell_interval,
        "force_sell_fraction": float(x[14]),
        "force_sell_profit_thresh": float(x[15]),
    }


def build_strategy(
    params: Dict[str, Any],
    cross_market_symbols: Dict[str, str],
    end_date: datetime,
) -> GoldTradingStrategy:
    """从参数字典构建策略实例"""
    weights = FactorWeights(
        technical=params["w_technical"],
        cross_market=params["w_cross_market"],
        sentiment=params["w_sentiment"],
        macro=params["w_macro"],
    )
    thresholds = TacticalThresholds(
        boost_buy=params["thresh_boost"],
        normal_buy=params["thresh_normal"],
        reduce_buy=params["thresh_reduce"],
        skip_buy=params["thresh_skip"],
        partial_sell=params["thresh_sell"],
    )
    position_config = PositionConfig(
        buy_day=params["buy_day"],
        boost_multiplier=params["boost_multiplier"],
        reduce_multiplier=params["reduce_multiplier"],
        sell_fraction=params["sell_fraction"],
        force_sell_interval_days=params["force_sell_interval"],
        force_sell_fraction=params["force_sell_fraction"],
        force_sell_profit_threshold=params["force_sell_profit_thresh"],
    )
    return GoldTradingStrategy(
        weights=weights,
        thresholds=thresholds,
        position_config=position_config,
        cross_market_symbols=cross_market_symbols,
        end_date=end_date,
    )


def compute_fitness(result: BacktestResult) -> float:
    """
    计算复合适应度（最大化）

    fitness = total_return * 0.5 + sharpe_ratio * 0.3 - max_drawdown * 0.2

    加入交易频率惩罚：
    - 买入次数过少（<10次/年）：惩罚
    - 买入次数过多（>100次/年）：惩罚
    """
    total_return = result.total_return
    sharpe = result.sharpe_ratio
    drawdown = result.max_drawdown

    fitness = total_return * 0.5 + sharpe * 0.3 - drawdown * 0.2

    # 交易频率惩罚
    days = (result.end_date - result.start_date).days
    years = max(0.5, days / 365.25)
    buys_per_year = result.buy_count / years

    if buys_per_year < 10:
        fitness -= 0.05 * (10 - buys_per_year)
    elif buys_per_year > 100:
        fitness -= 0.01 * (buys_per_year - 100)

    return fitness


def run_backtest_sync(
    strategy: GoldTradingStrategy,
    data: Dict[str, MarketData],
    broker_config: BrokerConfig,
    initial_capital: float,
    start_date: datetime,
    end_date: datetime,
) -> BacktestResult:
    """同步运行一次回测（串行和并行模式共用）"""
    from simulation.engine import SimulationEngine
    from simulation.broker import VirtualBroker

    valid_data = {s: d for s, d in data.items() if not d.is_empty}
    if not valid_data:
        raise ValueError("无有效数据")

    portfolio = Portfolio(initial_capital=initial_capital, name=strategy.name)
    broker = VirtualBroker(config=broker_config)
    engine = SimulationEngine(
        portfolio=portfolio,
        broker=broker,
        strategies=[strategy],
    )
    engine.run(valid_data)

    backtester = Backtester.__new__(Backtester)
    backtester.broker_config = broker_config
    return backtester._calculate_result(
        strategy=strategy,
        portfolio=portfolio,
        broker=broker,
        start_date=start_date,
        end_date=end_date,
    )


# =====================================================================
# 并行 Worker：进程池初始化器模式
# 每个 worker 进程在启动时通过 _init_worker 接收数据，
# 存入进程局部的 _worker_ctx 字典，后续调用 _worker_objective 时直接读取，
# 避免每次评估都序列化/反序列化大量 MarketData。
# =====================================================================

_worker_ctx: Dict[str, Any] = {}


def _init_worker(
    train_data: Dict[str, MarketData],
    cross_market_symbols: Dict[str, str],
    broker_config: BrokerConfig,
    initial_capital: float,
    full_start: datetime,
    train_end: datetime,
) -> None:
    """初始化 worker 进程的共享数据（每个进程只调用一次）"""
    _worker_ctx["train_data"] = train_data
    _worker_ctx["cross_market_symbols"] = cross_market_symbols
    _worker_ctx["broker_config"] = broker_config
    _worker_ctx["initial_capital"] = initial_capital
    _worker_ctx["full_start"] = full_start
    _worker_ctx["train_end"] = train_end


def _worker_objective(x: np.ndarray) -> float:
    """并行 worker 的目标函数（模块级，可 pickle）"""
    try:
        params = decode_params(x)
        strategy = build_strategy(
            params,
            _worker_ctx["cross_market_symbols"],
            _worker_ctx["train_end"],
        )
        result = run_backtest_sync(
            strategy,
            _worker_ctx["train_data"],
            _worker_ctx["broker_config"],
            _worker_ctx["initial_capital"],
            _worker_ctx["full_start"],
            _worker_ctx["train_end"],
        )
        return -compute_fitness(result)
    except Exception:
        return 1e6


class GoldStrategyOptimizer:
    """
    黄金策略差分进化优化器

    使用 scipy.optimize.differential_evolution 在参数空间中
    搜索最优策略配置。支持多核 CPU 并行加速。
    """

    def __init__(
        self,
        data: Dict[str, MarketData],
        cross_market_symbols: Dict[str, str],
        broker_config: Optional[BrokerConfig] = None,
        initial_capital: float = 100000.0,
        train_ratio: float = 0.7,
        checkpoint_dir: Optional[str] = None,
    ):
        self.data = data
        self.cross_market_symbols = cross_market_symbols
        self.broker_config = broker_config or BrokerConfig()
        self.initial_capital = initial_capital
        self.train_ratio = train_ratio

        # Checkpoint 目录
        self._checkpoint_dir: Optional[Path] = None
        if checkpoint_dir:
            self._checkpoint_dir = Path(checkpoint_dir)
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 计算训练/验证分割点
        gld_data = data.get("GLD")
        if gld_data is None or gld_data.is_empty:
            raise ValueError("缺少 GLD 数据")

        all_dates = [bar.timestamp for bar in gld_data.data]
        all_dates.sort()
        self.full_start = all_dates[0]
        self.full_end = all_dates[-1]

        split_idx = int(len(all_dates) * train_ratio)
        self.train_end = all_dates[split_idx]
        self.val_start = all_dates[split_idx + 1] if split_idx + 1 < len(all_dates) else self.train_end

        logger.info(
            f"数据分割: 训练 {self.full_start.strftime('%Y-%m-%d')} ~ "
            f"{self.train_end.strftime('%Y-%m-%d')} | "
            f"验证 {self.val_start.strftime('%Y-%m-%d')} ~ "
            f"{self.full_end.strftime('%Y-%m-%d')}"
        )

        # 预切片训练数据
        self.train_data = {
            s: md.slice(self.full_start, self.train_end)
            for s, md in data.items()
        }

        # 评估计数
        self._eval_count = 0
        self._best_fitness = -np.inf
        self._best_params: Dict[str, Any] = {}
        self._start_time = 0.0
        self._generation = 0

    def _save_checkpoint(
        self,
        params: Dict[str, Any],
        fitness: float,
        generation: int = 0,
    ) -> None:
        """保存 checkpoint（每次发现新最优时调用）"""
        if self._checkpoint_dir is None:
            return
        try:
            ckpt_data = {
                "checkpoint_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "generation": generation,
                "elapsed_seconds": float(round(time.time() - self._start_time, 1)),
                "fitness": float(round(fitness, 6)),
                "params": {k: float(round(v, 6)) if isinstance(v, (float, np.floating)) else int(v)
                           for k, v in params.items()},
            }
            ckpt_path = self._checkpoint_dir / "checkpoint_best.yaml"
            with open(ckpt_path, "w") as f:
                yaml.dump(ckpt_data, f, default_flow_style=False,
                          allow_unicode=True, sort_keys=False)
        except Exception as e:
            logger.warning(f"保存 checkpoint 失败: {e}")

    def _objective(self, x: np.ndarray) -> float:
        """
        目标函数（最小化，所以取负适应度）

        每次调用：解码参数 → 构建策略 → 运行训练集回测 → 计算适应度
        """
        self._eval_count += 1

        try:
            params = decode_params(x)
            strategy = build_strategy(
                params, self.cross_market_symbols, self.train_end,
            )
            result = run_backtest_sync(
                strategy, self.train_data, self.broker_config,
                self.initial_capital, self.full_start, self.train_end,
            )
            fitness = compute_fitness(result)

            # 记录进度
            if fitness > self._best_fitness:
                self._best_fitness = fitness
                self._best_params = params
                elapsed = time.time() - self._start_time
                logger.info(
                    f"[{self._eval_count:4d}] 新最优 fitness={fitness:.4f} "
                    f"return={result.total_return:.2%} "
                    f"sharpe={result.sharpe_ratio:.2f} "
                    f"drawdown={result.max_drawdown:.2%} "
                    f"buys={result.buy_count} sells={result.sell_count} "
                    f"({elapsed:.0f}s)"
                )
                # 保存 checkpoint
                self._save_checkpoint(params, fitness)

            return -fitness  # scipy 最小化，取负

        except Exception as e:
            logger.warning(f"[{self._eval_count}] 评估失败: {e}")
            return 1e6  # 失败返回极大值

    def _generation_callback(self, xk, convergence=0):
        """
        每代结束的回调（并行模式下用于进度追踪和 checkpoint）

        Args:
            xk: 当前最优参数向量
            convergence: 收敛度指标
        """
        self._generation += 1
        params = decode_params(xk)
        fitness = -self._objective_value if hasattr(self, '_objective_value') else None

        # 在并行模式下，重新评估当前最优以获取 fitness
        # （DE callback 不直接提供 fitness，但 xk 是当前 best）
        try:
            strategy = build_strategy(
                params, self.cross_market_symbols, self.train_end,
            )
            result = run_backtest_sync(
                strategy, self.train_data, self.broker_config,
                self.initial_capital, self.full_start, self.train_end,
            )
            fitness = compute_fitness(result)
        except Exception:
            return

        elapsed = time.time() - self._start_time

        if fitness > self._best_fitness:
            self._best_fitness = fitness
            self._best_params = params
            self._save_checkpoint(params, fitness, self._generation)

        logger.info(
            f"[Gen {self._generation:3d}] best fitness={self._best_fitness:.4f} "
            f"convergence={convergence:.6f} ({elapsed:.0f}s)"
        )

    def optimize(
        self,
        popsize: int = 20,
        maxiter: int = 50,
        seed: Optional[int] = 42,
        tol: float = 1e-6,
        workers: int = 1,
    ) -> OptimizationResult:
        """
        运行差分进化优化

        Args:
            popsize: 种群大小（每代评估 popsize * len(params) 次）
            maxiter: 最大迭代代数
            seed: 随机种子（可复现）
            tol: 收敛容差
            workers: 并行 worker 数量（1=串行，-1=全部 CPU 核心）

        Returns:
            优化结果
        """
        self._eval_count = 0
        self._best_fitness = -np.inf
        self._best_params = {}
        self._generation = 0
        self._start_time = time.time()

        n_params = len(PARAM_BOUNDS)
        total_evals = popsize * n_params * maxiter

        # 解析 workers 数量
        n_cpus = os.cpu_count() or 1
        if workers == -1:
            n_workers = n_cpus
        elif workers <= 0:
            n_workers = max(1, n_cpus + workers)
        else:
            n_workers = workers
        parallel = n_workers > 1

        logger.info(
            f"开始差分进化优化: {n_params} 参数, "
            f"popsize={popsize}, maxiter={maxiter}, "
            f"预计最大 {total_evals} 次评估, "
            f"workers={n_workers} ({'并行' if parallel else '串行'})"
        )

        # 标记 buy_day 为整数参数
        integrality = np.zeros(n_params)
        integrality[12] = 1  # buy_day
        integrality[13] = 1  # force_sell_interval

        if parallel:
            # 并行模式：使用进程池 + 初始化器
            pool = multiprocessing.Pool(
                processes=n_workers,
                initializer=_init_worker,
                initargs=(
                    self.train_data,
                    self.cross_market_symbols,
                    self.broker_config,
                    self.initial_capital,
                    self.full_start,
                    self.train_end,
                ),
            )
            try:
                result = differential_evolution(
                    _worker_objective,
                    bounds=PARAM_BOUNDS,
                    popsize=popsize,
                    maxiter=maxiter,
                    seed=seed,
                    tol=tol,
                    mutation=(0.5, 1.5),
                    recombination=0.8,
                    strategy="best1bin",
                    integrality=integrality,
                    disp=False,
                    workers=pool.map,
                    updating="deferred",
                    callback=self._generation_callback,
                )
            finally:
                pool.close()
                pool.join()
        else:
            # 串行模式：保持原有逻辑（逐次评估日志）
            result = differential_evolution(
                self._objective,
                bounds=PARAM_BOUNDS,
                popsize=popsize,
                maxiter=maxiter,
                seed=seed,
                tol=tol,
                mutation=(0.5, 1.5),
                recombination=0.8,
                strategy="best1bin",
                integrality=integrality,
                disp=False,
            )

        elapsed = time.time() - self._start_time
        best_params = decode_params(result.x)
        best_fitness = -result.fun

        logger.info(
            f"优化完成: {elapsed:.0f}s, best fitness={best_fitness:.4f}"
        )

        # 在训练集上重新评估最优参数
        train_strategy = build_strategy(
            best_params, self.cross_market_symbols, self.train_end,
        )
        train_result = run_backtest_sync(
            train_strategy, self.train_data, self.broker_config,
            self.initial_capital, self.full_start, self.train_end,
        )

        # 在验证集上评估
        val_data = {
            s: md.slice(self.val_start, self.full_end)
            for s, md in self.data.items()
        }
        val_strategy = build_strategy(
            best_params, self.cross_market_symbols, self.full_end,
        )
        val_result = run_backtest_sync(
            val_strategy, val_data, self.broker_config,
            self.initial_capital, self.val_start, self.full_end,
        )

        # 在全量数据上评估
        full_strategy = build_strategy(
            best_params, self.cross_market_symbols, self.full_end,
        )
        full_result = run_backtest_sync(
            full_strategy, self.data, self.broker_config,
            self.initial_capital, self.full_start, self.full_end,
        )

        # 保存最终 checkpoint
        self._save_checkpoint(best_params, best_fitness, self._generation)

        return OptimizationResult(
            best_params=best_params,
            best_fitness=best_fitness,
            train_result=train_result,
            val_result=val_result,
            full_result=full_result,
            benchmark_result=None,  # 由调用方填充
            n_evaluations=result.nfev,
            elapsed_seconds=elapsed,
        )
