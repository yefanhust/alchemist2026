# Alchemist2026 - 金融指标跟踪及智能交易系统

一个模块化的量化交易系统，支持模拟交易、策略回测和智能分析。

## 系统架构

```
alchemist2026/
├── docker/                          # Docker 配置
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── jupyter_config.py
├── python/                          # Python 源代码根目录
│   ├── alchemist2026/               # 主包
│   │   ├── __init__.py
│   │   ├── core/                    # 核心抽象层
│   │   │   ├── __init__.py
│   │   │   ├── asset.py            # 资产抽象
│   │   │   ├── portfolio.py        # 投资组合
│   │   │   ├── order.py            # 订单系统
│   │   │   └── position.py         # 持仓管理
│   │   ├── data/                    # 数据层
│   │   │   ├── __init__.py
│   │   │   ├── providers/          # 数据提供者
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py         # 基础接口
│   │   │   │   └── alphavantage.py
│   │   │   ├── cache/              # 缓存系统
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py
│   │   │   │   └── sqlite_cache.py
│   │   │   └── models.py           # 数据模型
│   │   ├── strategy/               # 策略模块
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # 策略基类
│   │   │   ├── indicators/         # 技术指标 (GPU加速)
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py
│   │   │   │   ├── moving_average.py
│   │   │   │   ├── rsi.py
│   │   │   │   └── macd.py
│   │   │   └── builtin/            # 内置策略
│   │   │       ├── __init__.py
│   │   │       ├── sma_crossover.py
│   │   │       └── mean_reversion.py
│   │   ├── simulation/             # 模拟交易模块
│   │   │   ├── __init__.py
│   │   │   ├── engine.py           # 模拟引擎
│   │   │   ├── broker.py           # 模拟券商
│   │   │   └── backtest.py         # 回测系统
│   │   ├── analysis/               # 分析模块
│   │   │   ├── __init__.py
│   │   │   ├── performance.py      # 绩效分析
│   │   │   ├── risk.py             # 风险分析
│   │   │   └── visualization.py    # 可视化
│   │   ├── gpu/                    # GPU加速模块
│   │   │   ├── __init__.py
│   │   │   ├── utils.py            # GPU工具函数
│   │   │   └── kernels.py          # CUDA核函数
│   │   ├── web/                    # Web服务模块
│   │   │   ├── __init__.py
│   │   │   ├── app.py              # FastAPI 应用
│   │   │   ├── config.py           # Web 配置
│   │   │   ├── routes/             # API 路由
│   │   │   ├── schemas/            # Pydantic 模型
│   │   │   ├── services/           # 业务逻辑
│   │   │   ├── static/             # 静态资源
│   │   │   ├── templates/          # Jinja2 模板
│   │   │   └── ssl/                # SSL 证书
│   │   └── utils/                  # 工具模块
│   │       ├── __init__.py
│   │       ├── config.py           # 配置管理
│   │       ├── logger.py           # 日志系统
│   │       └── time_utils.py       # 时间工具
│   ├── scripts/                     # 运行脚本
│   │   ├── run_backtest.py          # 策略回测
│   │   └── datasource_info.py       # 数据源信息查询
├── scripts/                         # Shell 脚本
│   ├── start_web.sh                 # Web 服务启动
│   └── generate_ssl.sh              # SSL 证书生成
│   ├── tests/                       # 测试
│   │   ├── __init__.py
│   │   ├── test_core/
│   │   ├── test_data/
│   │   ├── test_strategy/
│   │   └── test_simulation/
│   └── notebooks/                   # Jupyter notebooks
│       └── examples/
├── config/                          # 配置文件
│   ├── config.yaml                  # 用户配置（包含敏感信息，不提交到 Git）
│   ├── default.yaml.example         # 配置模板
│   └── strategies/
├── data/                            # 数据存储
│   ├── cache/                       # 缓存数据库
│   └── output/                      # 输出结果
├── logs/                            # 日志文件
├── requirements.txt
└── setup.py
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

### 6. Web服务 (web/)
- **FastAPI**: 高性能异步 Web 框架
- **数据可视化**: K线图、成交量图、多资产对比
- **REST API**: 缓存统计、市场数据查询
- **HTTPS**: 自签名证书，支持 Cloudflare 代理

## 数据库表结构

系统使用 SQLite 数据库（`data/cache/market_data.db`）持久化缓存数据，包含以下三张业务表：

### cache_entries — 通用缓存表

| 列 | 类型 | 说明 |
|----|------|------|
| `key` | TEXT (PK) | 缓存键，格式：`{provider}:{symbol}:{interval}:{start}:{end}` |
| `value` | BLOB | 序列化数据（MarketData 用 JSON，其他用 pickle） |
| `data_type` | TEXT | 值类型标记：`market_data` / `pickle` |
| `expires_at` | TIMESTAMP | 过期时间（NULL 表示永不过期） |
| `created_at` | TIMESTAMP | 创建时间 |

- **写入方**：`AlphaVantageProvider.get_historical_data()` 通过 `cache.set(key, data)` 写入
- **读取方**：Provider 缓存命中时读取；Web 仪表盘作为 fallback 数据源
- **特点**：粗粒度存储，一个 key 对应一整段时间范围的完整 `MarketData` 对象（JSON blob）
- **示例 key**：`alphavantage:GLD:1d:20240101:20251231`

### market_data — OHLCV 时序数据表

| 列 | 类型 | 说明 |
|----|------|------|
| `id` | INTEGER (PK) | 自增主键 |
| `symbol` | TEXT | 资产代码（如 GLD、SPY） |
| `interval` | TEXT | 数据间隔（如 1d、1h） |
| `timestamp` | TIMESTAMP | K 线时间戳 |
| `open` | REAL | 开盘价 |
| `high` | REAL | 最高价 |
| `low` | REAL | 最低价 |
| `close` | REAL | 收盘价 |
| `volume` | REAL | 成交量 |
| `adjusted_close` | REAL | 复权收盘价 |
| `created_at` | TIMESTAMP | 写入时间 |

- **写入方**：`AlphaVantageProvider.get_historical_data()` 通过 `cache.save_market_data()` 写入
- **读取方**：Web 仪表盘（K 线图、多资产对比、symbol 列表）优先从此表查询
- **特点**：细粒度存储，每行一根 K 线，支持按 symbol + 日期范围高效查询
- **唯一约束**：`(symbol, interval, timestamp)`，相同数据点自动覆盖更新

### stock_info — 股票元数据表

| 列 | 类型 | 说明 |
|----|------|------|
| `symbol` | TEXT (PK) | 股票代码 |
| `name` | TEXT | 公司名称 |
| `exchange` | TEXT | 交易所（NYSE、NASDAQ 等） |
| `sector` | TEXT | 行业大类（Technology、Healthcare 等） |
| `industry` | TEXT | 细分行业（Software、Biotechnology 等） |
| `market_cap` | REAL | 市值 |
| `pe_ratio` | REAL | 市盈率 |
| ... | | 其他基本面字段 |

- **写入方**：`datasource_info.py fetch-sector-info` 从 Alpha Vantage OVERVIEW API 获取并缓存
- **读取方**：`list-assets --sector/--industry` 按行业筛选、Web 仪表盘
- **特点**：按需积累，已缓存的 symbol 不会重复调用 API

### 表间关系与数据流

```
Alpha Vantage API
       │
       ▼
AlphaVantageProvider.get_historical_data()
       │
       ├──► cache.set(key, MarketData)      ──► cache_entries   (通用缓存，回测读取)
       │
       └──► cache.save_market_data(symbol)   ──► market_data     (时序表，Web 仪表盘读取)

Alpha Vantage OVERVIEW API
       │
       ▼
datasource_info.py fetch-sector-info
       │
       └──► cache.save_stock_info(symbol)    ──► stock_info      (元数据，行业筛选)
```

## 快速开始

### 0. 环境要求

- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA Docker Runtime（用于 GPU 支持）
- NVIDIA 驱动（支持 CUDA 12.2+）

#### 构建参数说明

- 镜像基于 `nvidia/cuda:12.2.0-devel-ubuntu22.04`
- 支持双 GPU 加速（CUDA 12.2）
- 预装 Python 3.11 及量化交易所需的全部依赖
- 包含 JupyterLab、NumPy、Pandas、CuPy、PyTorch 等库

### 1. 构建开发环境

#### 方法 1：使用Docker Compose 
```bash
# 在项目根目录下执行（无需 cd docker）
docker-compose -f docker/docker-compose.yml down          # 停止并移除旧容器
docker-compose -f docker/docker-compose.yml up -d --build  # 重新构建并启动容器
docker-compose -f docker/docker-compose.yml exec quant-dev bash  # 进入容器

# 查看所有运行的容器
docker-compose -f docker/docker-compose.yml ps

# 监控GPU使用
# 在容器内使用 nvidia-smi
docker-compose -f docker/docker-compose.yml exec quant-dev nvidia-smi
# 或使用 nvtop（更友好的界面）
docker-compose -f docker/docker-compose.yml exec quant-dev nvtop
```

#### 方法 2：使用 Docker 

```bash
# 进入 docker 目录
cd docker

# 构建镜像
docker build -t quant-trading-system:latest .

# 如果需要重新构建（不使用缓存）
docker build --no-cache -t quant-trading-system:latest .
```

验证 GPU 支持：

```bash
# 检查 NVIDIA Docker 是否正常工作
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### JupyterLab 使用

#### 方法 1：在主开发容器中启动 Jupyter

```bash
# 进入开发容器
docker-compose -f docker/docker-compose.yml exec quant-dev bash

# 在容器内启动 JupyterLab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

访问地址：`http://localhost:8888`

#### 方法 2：使用独立 Jupyter 服务（推荐）

```bash
# 使用 profile 启动独立的 Jupyter 容器
docker-compose -f docker/docker-compose.yml --profile jupyter up -d jupyter

# 查看 Jupyter 日志（获取访问 URL）
docker-compose -f docker/docker-compose.yml logs jupyter

# 停止 Jupyter 服务
docker-compose -f docker/docker-compose.yml --profile jupyter down
```

访问地址：`http://localhost:8889`

**注意**：独立 Jupyter 服务端口映射为 `8889:8888`，避免与主容器端口冲突。

#### JupyterLab 配置说明

- **工作目录**：`/workspace`
- **Notebook 目录**：`/workspace/notebooks`
- **无需 Token/密码**：开发环境已禁用身份验证
- **GPU 支持**：可使用所有 GPU（双卡）

#### 在 Jupyter 中验证 GPU

在 Notebook 中运行：

```python
import torch
import cupy as cp

# 检查 PyTorch GPU 支持
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# 检查 CuPy GPU 支持
print(f"CuPy CUDA available: {cp.cuda.is_available()}")
print(f"GPU devices: {cp.cuda.runtime.getDeviceCount()}")

# 显示 GPU 信息
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

### 2. 配置 API Key 及环境
#### 2.1 配置文件
```bash
# 复制配置模板
cp config/default.yaml.example config/config.yaml

# 编辑 config/config.yaml，填入 Alpha Vantage API Key 和其他敏感配置
# 注意：config/config.yaml 已加入 .gitignore，不会被提交到 Git
```

#### 2.2 环境变量

容器内默认环境变量：

```bash
NVIDIA_VISIBLE_DEVICES=all                # 使用所有 GPU
CUDA_VISIBLE_DEVICES=0,1                  # GPU 0 和 1
PYTHONPATH=/workspace/python/alchemist    # Python 模块搜索路径
TZ=Asia/Shanghai                          # 时区设置
```

可通过修改 `docker-compose.yml` 中的 `environment` 部分调整。


### 3. 运行示例回测

```bash
# 基础示例：SMA 交叉策略
python python/scripts/run_backtest.py backtest --strategy sma_crossover --symbol AAPL

# 均值回归策略，多个标的
python python/scripts/run_backtest.py backtest --strategy mean_reversion --symbol AAPL,MSFT,GOOGL

# 列出可用策略
python python/scripts/run_backtest.py list-strategies

# 检查 GPU 状态
python python/scripts/run_backtest.py check-gpu

# 查看帮助信息
python python/scripts/run_backtest.py --help
python python/scripts/run_backtest.py backtest --help
```

### 4. 查询数据源信息

使用 `datasource_info.py` 查看数据源能力概览（支持的交易所、资产类别、港股支持情况等）。

```bash
# 查看数据源全景信息（默认 alphavantage）
python python/scripts/datasource_info.py info

# 指定数据源
python python/scripts/datasource_info.py info --source alphavantage

# 检查港股 (HKEX) 支持情况及替代标的
python python/scripts/datasource_info.py hk-support

# 在线搜索资产代码
python python/scripts/datasource_info.py search 腾讯
python python/scripts/datasource_info.py search Tencent

# 列出所有已注册的数据源
python python/scripts/datasource_info.py list-sources

# 查看帮助信息
python python/scripts/datasource_info.py --help
```

`info` 命令输出包括：
- **金融资产类别** — stock、etf、forex、crypto
- **交易所支持** — NYSE/NASDAQ、上交所(.SHH)、深交所(.SHZ)、伦交所(.LON)、东证(.TYO)；HKEX 标红为不支持
- **港股替代标的** — TCEHY、BABA、JD 等常见港股 US ADR
- **数据间隔** — 1min ~ 月线
- **API 限流策略** — 当前 plan 及对应的频率/每日上限

#### 4.1 列出全部支持标的

使用 `list-assets` 命令按类型列出数据源支持的全部标的（股票、ETF、外汇、加密货币）。

```bash
# 列出股票（默认显示前 50 条）
python python/scripts/datasource_info.py list-assets stock

# 列出 ETF，显示 100 条
python python/scripts/datasource_info.py list-assets etf --limit 100

# 列出外汇货币
python python/scripts/datasource_info.py list-assets forex

# 列出加密货币，按关键词过滤
python python/scripts/datasource_info.py list-assets crypto --filter BTC

# 按交易所过滤（仅 stock/etf）
python python/scripts/datasource_info.py list-assets stock --exchange NASDAQ

# 分页查看（跳过前 50 条，显示接下来 50 条）
python python/scripts/datasource_info.py list-assets stock --offset 50 --limit 50

# 导出全部数据到 CSV 文件
python python/scripts/datasource_info.py list-assets stock --export stocks.csv
python python/scripts/datasource_info.py list-assets crypto --export crypto.csv
```

**支持的资产类型**：

| 类型 | 说明 | 数据来源 |
|-----|------|---------|
| `stock` | 股票（美股、A 股、多国市场） | Alpha Vantage LISTING_STATUS API |
| `etf` | 交易所交易基金 | Alpha Vantage LISTING_STATUS API |
| `forex` | 外汇货币（物理货币） | Alpha Vantage physical_currency_list |
| `crypto` | 加密货币（数字货币） | Alpha Vantage digital_currency_list |

**命令参数**：

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--limit, -n` | 显示数量限制 | 50 |
| `--offset` | 跳过前 N 条记录 | 0 |
| `--filter, -f` | 按关键词过滤（代码或名称） | - |
| `--exchange, -e` | 按交易所过滤（仅 stock/etf） | - |
| `--sector` | 按行业大类过滤（如 Technology, Healthcare） | - |
| `--industry` | 按细分行业过滤（如 Software, Biotechnology） | - |
| `--export` | 导出到 CSV 文件路径 | - |
| `--refresh` | 强制从 API 刷新行业信息（忽略缓存） | false |

#### 4.2 按行业过滤股票

使用 `--sector` 和 `--industry` 参数按行业筛选股票。首次查询会从 Alpha Vantage OVERVIEW API 获取行业信息，并自动缓存到本地 SQLite 数据库。

```bash
# 按行业大类过滤（如 Technology、Healthcare、Financial Services）
python python/scripts/datasource_info.py list-assets stock --sector Technology

# 按细分行业过滤（如 Software、Biotechnology、Banks）
python python/scripts/datasource_info.py list-assets stock --industry Software

# 组合过滤：科技行业中的软件公司
python python/scripts/datasource_info.py list-assets stock --sector Technology --industry Software

# 结合交易所过滤：NASDAQ 上的科技股
python python/scripts/datasource_info.py list-assets stock --exchange NASDAQ --sector Technology

# 强制刷新行业信息（忽略本地缓存）
python python/scripts/datasource_info.py list-assets stock --sector Technology --refresh

# 导出某行业的全部股票到 CSV
python python/scripts/datasource_info.py list-assets stock --sector Healthcare --export healthcare_stocks.csv
```

**常用行业分类（Sector）**：

| Sector | 说明 |
|--------|------|
| Technology | 科技 |
| Healthcare | 医疗健康 |
| Financial Services | 金融服务 |
| Consumer Cyclical | 可选消费 |
| Consumer Defensive | 必需消费 |
| Industrials | 工业 |
| Energy | 能源 |
| Basic Materials | 基础材料 |
| Real Estate | 房地产 |
| Utilities | 公用事业 |
| Communication Services | 通信服务 |

**批量获取行业信息并缓存**：

使用 `fetch-sector-info` 命令主动批量获取股票行业信息。建议先用 `--exchange` 缩小范围：

```bash
# 获取 NASDAQ 前 100 只股票的行业信息
python python/scripts/datasource_info.py fetch-sector-info --exchange NASDAQ --limit 100

# 获取 NYSE 前 200 只
python python/scripts/datasource_info.py fetch-sector-info --exchange NYSE --limit 200

# 继续获取（跳过已获取的）
python python/scripts/datasource_info.py fetch-sector-info --exchange NASDAQ --offset 100 --limit 100

# 按关键词过滤后获取
python python/scripts/datasource_info.py fetch-sector-info --filter AAPL --limit 10
```

**查看已缓存的行业列表**：

```bash
# 列出本地缓存中的所有行业分类
python python/scripts/datasource_info.py list-sectors
```

**缓存机制说明**：

- 行业信息缓存在 `./data/cache/stock_info.db`
- `list-assets --sector/--industry` 直接从缓存查询，秒级返回
- 使用 `fetch-sector-info` 命令主动积累缓存
- Premium 版本限流 75 次/分钟（0.8秒间隔），Free 版本 5 次/分钟（12秒间隔）
- 已缓存的股票不会重复调用 API

> **扩展数据源**：新增 DataProvider 实现后，只需在 `datasource_info.py` 的 `DATA_SOURCE_REGISTRY` 中注册即可被脚本自动识别。

### 5. 缓存管理

使用 `merge_cache.py` 合并相同 Symbol 和 Interval 的重复缓存条目。

**缓存键格式**：`{provider}:{symbol}:{interval}:{start_date}:{end_date}`

例如：
- `alphavantage:AAPL:1d:20250203:20260203`
- `alphavantage:AAPL:1d:20250204:20260204`

这两个条目会被合并为一个，数据去重并按时间排序。

```bash
# 预览合并计划（不执行）
python python/scripts/merge_cache.py

# 详细预览（显示数据统计）
python python/scripts/merge_cache.py --dry-run

# 执行合并
python python/scripts/merge_cache.py --execute

# 指定数据库路径
python python/scripts/merge_cache.py --db ./data/cache/market_data.db --execute
```

**输出示例**：

```
缓存数据库: ./data/cache/market_data.db

        可合并的缓存条目
┏━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Symbol ┃ Interval ┃ 条目数  ┃ 缓存键                                 ┃
┡━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ AAPL   │ 1d       │      2 │   alphavantage:AAPL:1d:... (251 条)   │
│        │          │        │   alphavantage:AAPL:1d:... (252 条)   │
└────────┴──────────┴────────┴───────────────────────────────────────┘

执行合并...

✓ AAPL (1d): 2 条目合并为 1 条 (503 → 252 条数据，去重 251 条)
  新键: alphavantage:AAPL:1d:20250203:20260204
```

### 6. 运行测试

测试框架会自动从 `config/config.yaml` 加载配置（包括 `alphavantage.api_key` 等），无需手动设置环境变量。

#### 6.1 Pytest 常用参数

| 参数 | 说明 | 示例 |
|-----|------|------|
| `-v` | 详细模式，显示每个测试用例的名称和结果 | `pytest -v` |
| `-s` | 显示 print 输出，不捕获 stdout/stderr | `pytest -s` |
| `-k` | 按名称过滤测试用例（支持表达式） | `pytest -k "commission"` |
| `--tb=short` | 简化 traceback 输出 | `pytest --tb=short` |
| `--tb=no` | 不显示 traceback | `pytest --tb=no` |
| `-x` | 遇到第一个失败立即停止 | `pytest -x` |
| `-q` | 简洁模式，只显示结果摘要 | `pytest -q` |
| `--lf` | 只运行上次失败的测试 | `pytest --lf` |
| `-n auto` | 并行运行测试（需安装 pytest-xdist） | `pytest -n auto` |

**常用组合**：

```bash
pytest -v -s              # 详细输出 + 显示 print
pytest -v --tb=short      # 详细输出 + 简化错误栈
pytest -v -x --tb=short   # 详细输出 + 首次失败停止 + 简化错误栈
pytest -v -k "pnl"        # 只运行名称含 "pnl" 的测试
```

#### 6.2 测试文件概览

| 测试文件 | 覆盖范围 | 依赖 |
|---------|---------|------|
| `test_simulation.py` | 模拟交易系统：手续费、滑点、订单成交、盈亏计算、收益/风险指标、交易统计 | 无外部依赖 |
| `test_sqlite_cache.py` | SQLite 缓存后端：CRUD、序列化、TTL、持久化、性能、并发 | 无外部依赖 |
| `test_alphavantage_hk.py` | Alpha Vantage 港股数据连通性测试 | 需要 API Key |

#### 6.3 模拟交易系统测试 (test_simulation.py)

验证交易执行、盈亏计算和绩效指标的准确性，共 70+ 测试用例。

**测试类覆盖范围**：

| 测试类 | 说明 |
|-------|------|
| `TestCommissionCalculation` | 手续费计算：百分比费率、最低手续费、大额订单、分数股 |
| `TestSlippageCalculation` | 滑点计算：百分比/固定/随机模式、买卖方向正确性 |
| `TestOrderFillPrice` | 订单成交价：市价单、限价单、止损单、止损限价单、跳空处理 |
| `TestPositionPnL` | 持仓盈亏：多头/空头、已实现/未实现、部分平仓、反手交易 |
| `TestPortfolio` | 投资组合：现金变动、持仓市值、总价值、多持仓管理 |
| `TestReturnMetrics` | 收益指标：总收益率、年化收益率（短期/长期/多年） |
| `TestRiskMetrics` | 风险指标：波动率、夏普比率、索提诺比率、最大回撤及持续时间 |
| `TestTradeStatistics` | 交易统计：胜率、平均盈亏、盈亏比（Profit Factor） |
| `TestOrderValidation` | 订单验证：资金检查、持仓检查、做空限制 |
| `TestTradingWorkflow` | 完整交易流程：买卖流程、累计盈亏、手续费/滑点影响 |
| `TestEdgeCases` | 边界条件：极小/极大订单、零价格、空数据 |
| `TestNumericalPrecision` | 数值精度：高精度计算验证 |

**运行示例**：

```bash
# 运行全部模拟交易测试
pytest python/tests/test_simulation.py -v

# 按测试类运行
pytest python/tests/test_simulation.py::TestCommissionCalculation -v
pytest python/tests/test_simulation.py::TestSlippageCalculation -v
pytest python/tests/test_simulation.py::TestPositionPnL -v
pytest python/tests/test_simulation.py::TestRiskMetrics -v

# 按关键字过滤
pytest python/tests/test_simulation.py -v -k "commission"      # 所有手续费相关
pytest python/tests/test_simulation.py -v -k "sharpe"          # 夏普比率相关
pytest python/tests/test_simulation.py -v -k "drawdown"        # 回撤相关
pytest python/tests/test_simulation.py -v -k "long or short"   # 多头或空头相关

# 运行单个测试用例
pytest python/tests/test_simulation.py::TestCommissionCalculation::test_commission_percentage_calculation -v
pytest python/tests/test_simulation.py::TestRiskMetrics::test_sharpe_ratio_calculation -v
```

#### 6.4 缓存系统测试 (test_sqlite_cache.py)

验证 SQLite 缓存后端的正确性和性能，共 70+ 测试用例。

**测试类覆盖范围**：

| 测试类 | 说明 |
|-------|------|
| `TestBasicCRUD` | 基本操作：set/get/delete/exists/clear |
| `TestSerialization` | 序列化：字符串、数字、字典、列表、字节、MarketData |
| `TestTTL` | 过期机制：TTL 设置、自动删除、默认 TTL、批量清理 |
| `TestBatchOperations` | 批量操作：set_many/get_many/delete_many |
| `TestKeyPatterns` | 键模式匹配：通配符 `*`、单字符 `?` |
| `TestMarketDataTable` | 市场数据专用表：save/get、日期过滤、多标的/多周期 |
| `TestPersistence` | 数据持久化：重启后数据保留 |
| `TestCacheIntegration` | 缓存集成：DataProvider + Cache 端到端测试 |
| `TestPerformance` | 性能测试：缓存 vs 远端响应时间对比 |
| `TestGetOrSet` | 缓存穿透保护：get_or_set 行为验证 |
| `TestStats` | 统计信息：缓存条目数、过期数、数据库大小 |
| `TestConcurrency` | 并发访问：并发读写、多实例初始化 |
| `TestContextManager` | 上下文管理器：async with 用法 |
| `TestEdgeCases` | 边界条件：超长键、空值、特殊字符、大数据量 |

**运行示例**：

```bash
# 运行全部缓存测试
pytest python/tests/test_sqlite_cache.py -v

# 按测试类运行
pytest python/tests/test_sqlite_cache.py::TestBasicCRUD -v
pytest python/tests/test_sqlite_cache.py::TestSerialization -v
pytest python/tests/test_sqlite_cache.py::TestTTL -v
pytest python/tests/test_sqlite_cache.py::TestPersistence -v
pytest python/tests/test_sqlite_cache.py::TestPerformance -v

# 按关键字过滤
pytest python/tests/test_sqlite_cache.py -v -k "market_data"   # MarketData 相关
pytest python/tests/test_sqlite_cache.py -v -k "ttl"           # TTL 相关
pytest python/tests/test_sqlite_cache.py -v -k "persist"       # 持久化相关
pytest python/tests/test_sqlite_cache.py -v -k "concurrent"    # 并发相关

# 运行单个测试用例
pytest python/tests/test_sqlite_cache.py::TestBasicCRUD::test_set_and_get -v
pytest python/tests/test_sqlite_cache.py::TestPerformance::test_cache_much_faster_than_remote -v
```

#### 6.5 Alpha Vantage 数据源测试 (test_alphavantage.py)

验证 Alpha Vantage API 获取数据。

**运行示例**：

```bash
# 需要 API Key
pytest python/tests/test_alphavantage.py -v -s
```

#### 6.6 运行所有测试

```bash
# 运行全部测试
pytest python/tests/ -v

# 运行全部测试（简化输出）
pytest python/tests/ -v --tb=short

# 运行全部测试（首次失败停止）
pytest python/tests/ -v -x
```

> **说明**：如果 `config/config.yaml` 中未配置 `alphavantage.api_key`，Alpha Vantage 相关测试会自动跳过。

### 端口映射

| 服务 | 容器端口 | 宿主机端口 | 说明 |
|-----|---------|-----------|------|
| 主容器 JupyterLab | 8888 | 8888 | 开发容器的 Jupyter |
| 独立 Jupyter | 8888 | 8889 | 独立 Jupyter 服务 |
| Web 服务 (HTTPS) | 8443 | 8443 | 数据可视化 Web 服务 |
| Redis | 6379 | 6379 | 缓存服务 |

### 7. Web 服务

Web 模块提供缓存数据可视化和 API 服务，基于 FastAPI + Uvicorn，支持 HTTPS。

#### 7.1 启动 Web 服务

```bash
# 进入项目目录
cd ~/workspace/alchemist2026

# 启动 Web 服务（前台运行，可看日志）
docker-compose -f docker/docker-compose.yml exec quant-dev /workspace/scripts/start_web.sh

# 或后台运行
docker-compose -f docker/docker-compose.yml exec -d quant-dev /workspace/scripts/start_web.sh
```

#### 7.2 访问地址

| 页面 | 地址 | 说明 |
|------|------|------|
| 首页仪表盘 | https://localhost:8443/ | 缓存数据概览 |
| K线图 | https://localhost:8443/chart/candlestick | OHLCV K线可视化 |
| 多资产对比 | https://localhost:8443/chart/comparison | 多 Symbol 收益率对比 |
| API 文档 | https://localhost:8443/docs | Swagger UI |
| 健康检查 | https://localhost:8443/health | 服务状态 |

#### 7.3 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/cache/stats` | GET | 缓存统计信息 |
| `/api/cache/keys` | GET | 所有缓存键 |
| `/api/market/symbols` | GET | 缓存的 Symbol 列表 |
| `/api/market/ohlcv/{symbol}` | GET | 获取 OHLCV 数据 |

**OHLCV 查询示例**：

```bash
# 获取 AAPL 全部数据
curl -sk https://localhost:8443/api/market/ohlcv/AAPL

# 指定日期范围
curl -sk "https://localhost:8443/api/market/ohlcv/AAPL?start_date=2026-01-01&end_date=2026-01-31"
```

#### 7.4 重启 Web 服务

```bash
# 找到运行中的 uvicorn 进程并停止
docker-compose -f docker/docker-compose.yml exec quant-dev pkill -f uvicorn

# 重新启动
docker-compose -f docker/docker-compose.yml exec quant-dev /workspace/scripts/start_web.sh
```

#### 7.5 SSL 证书

首次启动时会自动生成自签名证书，也可手动生成：

```bash
docker-compose -f docker/docker-compose.yml exec quant-dev /workspace/scripts/generate_ssl.sh
```

证书位置：`/workspace/python/alchemist/web/ssl/`

#### 7.6 域名访问（Cloudflare）

如需通过域名访问，配置 Cloudflare：

1. **DNS 记录**：添加 A 记录指向服务器公网 IP，开启代理（橙色云朵）
2. **SSL 模式**：设置为 `Full`（因为后端使用自签名证书）
3. **Origin Rules**：配置目标端口为 `8443`
   - Rules → Origin Rules → Create rule
   - When: `Hostname equals your.domain.com`
   - Then: Destination Port → Rewrite to `8443`

#### 7.7 Web 模块结构

```
python/alchemist/web/
├── __init__.py              # 模块入口
├── app.py                   # FastAPI 应用
├── config.py                # WebConfig 配置
├── routes/                  # API 路由
│   ├── health.py            # 健康检查
│   ├── cache.py             # 缓存 API
│   ├── market.py            # 市场数据 API
│   └── pages.py             # 前端页面
├── schemas/                 # Pydantic 模型
├── services/                # 业务逻辑
│   └── cache_service.py     # 缓存服务
├── static/                  # 静态资源
│   ├── css/main.css
│   └── js/charts.js
├── templates/               # Jinja2 模板
│   ├── base.html
│   ├── index.html
│   └── charts/
└── ssl/                     # SSL 证书
```

### 8. 黄金策略参数优化

使用差分进化算法（Differential Evolution）对黄金择时增强型定投策略的全部参数进行全局优化，包括因子权重、阈值、倍率和强制止盈等 16 个参数。默认使用全部 CPU 核心并行加速。

#### 8.1 运行优化

```bash
# 默认参数（3年数据，种群20，50代，全部CPU核心并行）
python python/scripts/run_backtest.py optimize-gold

# 自定义参数
python python/scripts/run_backtest.py optimize-gold \
    --start 2011-01-04 --end 2025-12-31 \
    --popsize 30 --maxiter 100 --seed 42

# 指定并行 worker 数量（默认 -1 = 全部核心）
python python/scripts/run_backtest.py optimize-gold --workers 48

# 串行模式（调试用，保留逐次评估日志）
python python/scripts/run_backtest.py optimize-gold --workers 1

# 建议使用 tee 保留终端输出，防止 SSH 断连丢失结果
python python/scripts/run_backtest.py optimize-gold \
    --start 2011-01-04 --end 2025-12-31 2>&1 | tee optimize.log
```

**优化参数说明**：

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--start` | 数据起始日期 | 当前日期 - 3年 |
| `--end` | 数据截止日期 | 当前日期 |
| `--popsize` | 差分进化种群大小 | 20 |
| `--maxiter` | 最大迭代代数 | 50 |
| `--train-ratio` | 训练集比例（Walk-forward 验证） | 0.7 |
| `--seed` | 随机种子（可复现） | 42 |
| `--workers` / `-w` | 并行 worker 数量（1=串行，-1=全部CPU核心） | -1 |

**并行加速说明**：

优化使用 `multiprocessing.Pool` 进行 CPU 多核并行，种群中各个体的回测评估在不同核心上同时运行。每个 worker 进程在启动时通过初始化器接收市场数据（只传输一次），避免每次评估的序列化开销。

- 串行模式（`--workers 1`）：逐次评估，每次发现新最优时打印日志
- 并行模式（默认）：按代追踪进度，每代结束后打印最优 fitness 和收敛度

#### 8.2 Checkpoint 与结果保存

优化过程中每次发现更优解都会自动保存 checkpoint，即使进程中断也不会丢失进度：

```
data/output/
├── checkpoints/
│   └── checkpoint_best.yaml    # 中间最优 checkpoint（实时更新）
└── gold_optimized_params.yaml  # 最终优化结果（优化完成后写入）
```

**checkpoint_best.yaml** 示例：

```yaml
checkpoint_at: "2026-02-22 10:30:00"
eval_count: 3200
elapsed_seconds: 5400.0
fitness: 0.1856
train_return: 0.2341
train_sharpe: 1.52
params:
  w_technical: 0.3512
  w_cross_market: 0.2891
  ...
```

**gold_optimized_params.yaml**（最终结果）示例：

```yaml
optimized_at: "2026-02-22 12:00:00"
fitness: 0.2034
n_evaluations: 16000
elapsed_seconds: 28800.0
params:
  w_technical: 0.3512
  w_cross_market: 0.2891
  w_sentiment: 0.0823
  w_macro: 0.2774
  thresh_boost: 0.2456
  thresh_normal: 0.0312
  thresh_reduce: -0.1089
  thresh_skip: -0.2567
  thresh_sell: -0.5234
  boost_multiplier: 2.15
  reduce_multiplier: 0.42
  sell_fraction: 0.55
  buy_day: 3
  force_sell_interval: 210
  force_sell_fraction: 0.25
  force_sell_profit_thresh: 0.08
performance:
  train:
    total_return: 0.2341
    annual_return: 0.1023
    sharpe_ratio: 1.52
    ...
  validation:
    total_return: 0.1856
    ...
  full:
    total_return: 0.2134
    ...
  benchmark:
    total_return: 0.1567
    ...
```

#### 8.3 使用优化参数运行回测

优化完成后，`gold-backtest` 命令会**自动加载**优化参数：

```bash
# 自动加载 data/output/gold_optimized_params.yaml（如果存在）
python python/scripts/run_backtest.py gold-backtest

# 手动指定参数文件
python python/scripts/run_backtest.py gold-backtest \
    --optimized-params data/output/gold_optimized_params.yaml

# 从 checkpoint 恢复使用（优化中断时）
python python/scripts/run_backtest.py gold-backtest \
    --optimized-params data/output/checkpoints/checkpoint_best.yaml

# 不使用优化参数，使用默认参数
python python/scripts/run_backtest.py gold-backtest --no-optimized
```

#### 8.4 防过拟合措施

- **Walk-forward 验证**：数据按 `train_ratio`（默认 70%）分割为训练集和验证集，仅在训练集上优化，验证集评估泛化能力
- **交易频率惩罚**：买入过少（<10次/年）或过多（>100次/年）均扣分
- **过拟合检测**：优化完成后自动对比训练集与验证集适应度比值，给出过拟合警告

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
4. ~~**Web界面模块**~~ - ✅ 已完成（数据可视化、K线图、多资产对比）
