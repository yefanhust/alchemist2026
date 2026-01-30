# Alchemist2026 - 金融指标跟踪及智能交易系统

一个模块化的量化交易系统，支持模拟交易、策略回测和智能分析。

## 系统架构

```
alchemist2026/
├── docker/                     # Docker 配置
│   ├── Dockerfile
│   └── docker-compose.yml
├── src/                        # 源代码
│   ├── core/                   # 核心抽象层
│   │   ├── __init__.py
│   │   ├── asset.py           # 资产抽象
│   │   ├── portfolio.py       # 投资组合
│   │   ├── order.py           # 订单系统
│   │   └── position.py        # 持仓管理
│   ├── data/                   # 数据层
│   │   ├── __init__.py
│   │   ├── providers/         # 数据提供者
│   │   │   ├── __init__.py
│   │   │   ├── base.py        # 基础接口
│   │   │   └── alphavantage.py
│   │   ├── cache/             # 缓存系统
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   └── sqlite_cache.py
│   │   └── models.py          # 数据模型
│   ├── strategy/              # 策略模块
│   │   ├── __init__.py
│   │   ├── base.py            # 策略基类
│   │   ├── indicators/        # 技术指标 (GPU加速)
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── moving_average.py
│   │   │   ├── rsi.py
│   │   │   └── macd.py
│   │   └── builtin/           # 内置策略
│   │       ├── __init__.py
│   │       ├── sma_crossover.py
│   │       └── mean_reversion.py
│   ├── simulation/            # 模拟交易模块
│   │   ├── __init__.py
│   │   ├── engine.py          # 模拟引擎
│   │   ├── broker.py          # 模拟券商
│   │   └── backtest.py        # 回测系统
│   ├── analysis/              # 分析模块
│   │   ├── __init__.py
│   │   ├── performance.py     # 绩效分析
│   │   ├── risk.py            # 风险分析
│   │   └── visualization.py   # 可视化
│   ├── gpu/                   # GPU加速模块
│   │   ├── __init__.py
│   │   ├── utils.py           # GPU工具函数
│   │   └── kernels.py         # CUDA核函数
│   └── utils/                 # 工具模块
│       ├── __init__.py
│       ├── config.py          # 配置管理
│       ├── logger.py          # 日志系统
│       └── time_utils.py      # 时间工具
├── tests/                     # 测试
│   ├── __init__.py
│   ├── test_core/
│   ├── test_data/
│   ├── test_strategy/
│   └── test_simulation/
├── notebooks/                 # Jupyter notebooks
│   └── examples/
├── data/                      # 数据存储
│   ├── cache/                 # 缓存数据库
│   └── output/                # 输出结果
├── config/                    # 配置文件
│   ├── default.yaml
│   └── strategies/
├── scripts/                   # 脚本
│   └── run_backtest.py
├── requirements.txt
├── setup.py
└── .env.example
```

## 核心模块说明

### 1. 核心抽象层 (core/)
- **Asset**: 资产抽象，支持股票、ETF、加密货币等
- **Portfolio**: 投资组合管理，追踪持仓和资金
- **Order**: 订单系统，支持市价单、限价单等
- **Position**: 持仓管理，计算盈亏

### 2. 数据层 (data/)
- **DataProvider**: 数据提供者接口，可扩展支持多数据源
- **AlphaVantageProvider**: Alpha Vantage API集成
- **CacheSystem**: 智能缓存，避免重复API调用

### 3. 策略模块 (strategy/)
- **Strategy**: 策略基类，定义统一接口
- **Indicators**: GPU加速的技术指标计算
- **BuiltinStrategies**: 内置策略模板

### 4. 模拟交易模块 (simulation/)
- **SimulationEngine**: 事件驱动的模拟引擎
- **VirtualBroker**: 虚拟券商，模拟订单执行
- **Backtester**: 历史回测系统

### 5. GPU加速 (gpu/)
- 使用 CuPy/CUDA 加速大规模数据计算
- 技术指标并行计算
- 批量策略评估

## 快速开始

### 1. 构建开发环境
```bash
cd docker
docker-compose up -d --build
docker exec -it quant-dev bash
```

### 2. 配置 API Key
```bash
cp .env.example .env
# 编辑 .env 文件，填入 Alpha Vantage API Key
```

### 3. 运行示例回测
```bash
python scripts/run_backtest.py --strategy sma_crossover --symbol AAPL
```

## GPU 加速说明

系统利用双 GPU 进行以下加速：
- 技术指标批量计算
- 多策略并行回测
- 大规模历史数据处理
- 风险指标计算（VaR、蒙特卡洛模拟）

## 后续模块规划

1. **实时监控模块** - 实时行情监控和预警
2. **智能分析模块** - 机器学习驱动的市场分析
3. **交易执行模块** - 对接真实券商API
4. **Web界面模块** - 可视化管理界面
