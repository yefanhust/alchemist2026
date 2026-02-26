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
| `test_valuation.py` | 估值扫描系统：相对/绝对估值、情绪/宏观因子、综合打分、端到端流水线 | 无外部依赖 |
| `test_data_providers.py` | 新增数据源：yfinance 情绪数据、FRED 宏观时间序列 | yfinance（已安装）；FRED 实时测试需 `FRED_API_KEY` |
| `test_alphavantage.py` | Alpha Vantage 数据连通性测试 | 需要 API Key |

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

#### 6.6 估值扫描系统测试 (test_valuation.py)

验证估值评估流水线的正确性，共 41 个测试用例，**无需任何 API Key**，全部本地运行。

**测试类覆盖范围**：

| 测试类 | 测试数 | 说明 |
|-------|-------|------|
| `TestModels` | 7 | 数据模型：评级映射边界（A-F）、投资时间窗口（1M/3M/6M/1Y）、向后兼容别名、序列化 |
| `TestRelativeValuationFactors` | 8 | Z-Score 正负映射、百分位评分、PE 行业对比、缺失指标降级 |
| `TestAbsoluteValuation` | 5 | DCF 三情景计算、乐观≥中性≥悲观排序、剩余收益模型、空报表容错 |
| `TestSentimentFactors` | 5 | RSI 计算、数据不足降级、空头持仓/内部人交易数据集成 |
| `TestMacroFactors` | 5 | Fed 模型、高/低 VIX 方向、利率环境、空数据容错 |
| `TestValuationScorer` | 5 | 四维合成、自定义权重、A-F 评级一致性 |
| `TestHorizonWeights` | 10 | Horizon 动态权重：1M情绪主导/1Y基本面主导/自定义覆盖/权重和=1 |
| `TestIntegration` | 2 | 完整打分流水线端到端、行业汇总聚合计算 |

**运行示例**：

```bash
# 运行全部估值测试
pytest python/tests/test_valuation.py -v

# 按测试类运行
pytest python/tests/test_valuation.py::TestRelativeValuationFactors -v
pytest python/tests/test_valuation.py::TestAbsoluteValuation -v
pytest python/tests/test_valuation.py::TestMacroFactors -v
pytest python/tests/test_valuation.py::TestValuationScorer -v
pytest python/tests/test_valuation.py::TestIntegration -v

# 按关键字过滤
pytest python/tests/test_valuation.py -v -k "dcf"            # DCF 相关
pytest python/tests/test_valuation.py -v -k "grade"          # 评级映射相关
pytest python/tests/test_valuation.py -v -k "macro"          # 宏观因子相关
pytest python/tests/test_valuation.py -v -k "integration"    # 集成测试
```

#### 6.7 数据提供者测试 (test_data_providers.py)

验证新增数据源 yfinance 和 FRED 的数据获取能力，共 43 个测试用例（38 通过 + 5 跳过）。

- **单元测试（38个）**：全部使用 Mock，无需真实网络连接
- **实时数据测试（5个）**：FRED 实时接口测试，需设置环境变量 `FRED_API_KEY`（无 key 时自动跳过）

**测试类覆盖范围**：

| 测试类 | 测试数 | 依赖 | 说明 |
|-------|-------|------|------|
| `TestYFinanceProviderInit` | 5 | Mock | 初始化、is_available 属性、yfinance 缺失优雅降级 |
| `TestYFinanceShortInterest` | 4 | Mock | 空头持仓数据结构、symbol 大写规范化、异常容错 |
| `TestYFinanceInsiderTransactions` | 5 | Mock | 内部人交易结构、net_direction 方向性验证（+1/-1/0）、空 DataFrame 处理 |
| `TestYFinanceBatchSentiment` | 2 | Mock | 批量获取结构、空列表边界 |
| `TestYFinanceLiveData` | 3 | yfinance 网络 | AAPL/MSFT 真实数据结构验证（自动 skip 若不可用） |
| `TestFREDProviderInit` | 5 | Mock | API key 配置、环境变量读取、FRED_SERIES 常量 |
| `TestFREDGetSeriesMocked` | 4 | Mock | pd.Series 返回类型验证、DatetimeIndex、HTTP 错误、"." 占位符过滤 |
| `TestFREDGetLatestValue` | 2 | Mock | 最新值提取、空序列返回 None |
| `TestFREDMacroSnapshot` | 4 | Mock | 快照结构、yield_curve_spread 派生指标、部分缺失容错、无 key 空快照 |
| `TestFREDGetMultipleSeries` | 2 | Mock | 批量返回结构、跳过失败序列 |
| `TestFREDClose` | 2 | Mock | 会话关闭（有/无会话） |
| `TestFREDLiveData` | 5 | FRED API | DGS10/FEDFUNDS 时间序列、最新值范围验证、完整宏观快照（需 `FRED_API_KEY`） |

**运行示例**：

```bash
# 运行所有数据提供者测试（单元+实时，无 key 时实时测试自动跳过）
pytest python/tests/test_data_providers.py -v

# 仅运行 yfinance 相关测试
pytest python/tests/test_data_providers.py -v -k "yfinance or YFinance"

# 仅运行 FRED 相关测试
pytest python/tests/test_data_providers.py -v -k "fred or FRED"

# 仅运行单元测试（无网络依赖）
pytest python/tests/test_data_providers.py -v -k "not Live"

# 运行 FRED 实时测试（需先设置 API Key）
FRED_API_KEY=your_key pytest python/tests/test_data_providers.py::TestFREDLiveData -v -s

# 按测试类运行
pytest python/tests/test_data_providers.py::TestYFinanceShortInterest -v
pytest python/tests/test_data_providers.py::TestFREDMacroSnapshot -v
pytest python/tests/test_data_providers.py::TestFREDGetSeriesMocked -v
```

**FRED 实时测试配置**：

FRED API Key 可通过两种方式提供（二选一）：

```bash
# 方式1：环境变量（推荐用于 CI/CD）
export FRED_API_KEY=your_fred_api_key
pytest python/tests/test_data_providers.py::TestFREDLiveData -v

# 方式2：config.yaml
```

#### 6.8 运行所有测试

```bash
# 运行全部测试
pytest python/tests/ -v

# 运行全部测试（简化输出）
pytest python/tests/ -v --tb=short

# 运行全部测试（首次失败停止）
pytest python/tests/ -v -x

# 仅运行无需 API Key 的本地测试
pytest python/tests/ -v -k "not alphavantage and not Live"
```

> **说明**：如果 `config/config.yaml` 中未配置 `alphavantage.api_key`，Alpha Vantage 相关测试会自动跳过。`test_simulation.py`、`test_sqlite_cache.py`、`test_valuation.py` 以及 `test_data_providers.py` 的单元测试部分均无需 API Key，可直接运行。

### 端口映射

| 服务 | 容器端口 | 宿主机端口 | 说明 |
|-----|---------|-----------|------|
| 主容器 JupyterLab | 8888 | 8888 | 开发容器的 Jupyter |
| 独立 Jupyter | 8888 | 8889 | 独立 Jupyter 服务 |
| Web 服务 (HTTPS) | 8443 | 8443 | 数据可视化 Web 服务 |
| Redis | 6379 | 6379 | 缓存服务 |

### 7. Web 服务

Web 模块提供缓存数据可视化和 API 服务，基于 FastAPI + Uvicorn，支持 HTTPS。

#### 7.1 启动与重启 Web 服务

```bash
# 进入项目目录
cd ~/workspace/alchemist2026

# 启动 Web 服务（前台运行，可看日志）
# start_web.sh 会自动加载 docker/.env、检测并停止已运行的实例，可安全重复执行
docker-compose -f docker/docker-compose.yml exec quant-dev /workspace/scripts/start_web.sh

# 或后台运行
docker-compose -f docker/docker-compose.yml exec -d quant-dev /workspace/scripts/start_web.sh
```

**重启方式（三选一）**：

```bash
# 方式 1（推荐）：直接再次执行 start_web.sh
# 脚本内置"检测旧进程 → 停止 → 重新启动"逻辑，可重复运行无副作用
docker-compose -f docker/docker-compose.yml exec quant-dev /workspace/scripts/start_web.sh

# 方式 2：使用语义更明确的 restart_web.sh（内部调用 start_web.sh）
docker-compose -f docker/docker-compose.yml exec quant-dev /workspace/scripts/restart_web.sh

# 方式 3：手动两步操作（适合需要确认进程已停止的场景）
# 步骤1：停止现有进程
docker exec quant-dev pkill -f "uvicorn web.app" || true
# 步骤2：重新启动
docker-compose -f docker/docker-compose.yml exec quant-dev /workspace/scripts/start_web.sh
```

> **注意**：Web 服务依赖 `docker/.env` 中的 `WEB_AUTH_PASSWORD` 配置访问密码。
> `start_web.sh` 在启动时会自动 `source docker/.env`，同时 `web/config.py` 也内置了
> python-dotenv 兜底加载，确保无论通过何种方式启动 uvicorn，密码配置均能生效。

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

详见 **7.1 启动与重启 Web 服务** 中的三种重启方式。

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

### 9. 美股估值扫描

基于"筛选-剖析-验证-整合"四步框架，对全市场进行估值扫描，发掘在指定投资时间窗口（1M/3M/6M/1Y）内预期回归合理估值的美股。`--horizon` 指的是前瞻性时间窗口——预期标的在此期间内回归合理估值区间，而非向后回看的历史时段。

#### 9.1 系统架构

```
数据采集层 (data/)
├── AlphaVantageProvider     — OHLCV + OVERVIEW + 财务报表 (主数据源)
├── YFinanceSentimentProvider — 空头持仓 + 内部人交易 (情绪补充)
└── FREDProvider              — 宏观经济指标 (Fed 利率/CPI/VIX/GDP)

估值引擎层 (strategy/valuation/)
├── factors/
│   ├── relative.py          — 相对估值 (PE/PB/PS/PEG/EV-EBITDA/P-FCF/股东回报)
│   ├── absolute.py          — 绝对估值 (DCF 三情景 + 剩余收益模型)
│   ├── sentiment.py         — 情绪面 (RSI/量价异常/空头/内部人/动量)
│   └── macro.py             — 宏观面 (Fed 模型/20 法则/巴菲特指标/收益率曲线/VIX)
├── scorer.py                — 四维加权综合打分器 (A-F 字母评级)
├── scanner.py               — 扫描引擎 + 数据采集函数
├── universe.py              — 股票池管理 (S&P500/NASDAQ100/全市场/自定义)
└── models.py                — StockValuation / ScanResult 数据模型

输出层
├── CLI (run_backtest.py)    — 5 个子命令
└── Web (routes/valuation.py) — REST API + 交互式页面
```

#### 9.2 估值框架说明

| 步骤 | 维度 | 指标 |
|-----|------|------|
| Step 1 | 相对估值 | PE、PB、PS、PEG、EV/EBITDA、P/FCF、股东回报率、52周位置 |
| Step 2 | 绝对估值 | DCF（乐观/中性/悲观三情景）、剩余收益模型（RIM） |
| Step 3 | 情绪面 | RSI、成交量异常、波动率偏离、空头持仓比、内部人净方向、价格动量 |
| Step 4 | 宏观面 | Fed 模型（盈利收益率 vs 10Y 国债）、Rule of 20（PE+CPI）、巴菲特指标（总市值/GDP）、收益率曲线、VIX |

**因子权重随投资时间窗口 (`--horizon`) 动态调整**：

| Horizon | 相对估值 | 绝对估值(DCF) | 情绪面 | 宏观面 | 设计理由 |
|---------|---------|-------------|--------|--------|---------|
| **1M** | 30% | 15% | **35%** | 20% | 短期回归靠情绪催化（RSI超卖/轧空/内部人买入） |
| **3M** | 30% | 25% | 25% | 20% | 均衡（默认） |
| **6M** | 25% | **30%** | 20% | 25% | 中长期基本面开始主导 |
| **1Y** | 25% | **35%** | 15% | 25% | 长期回归靠 DCF 驱动，短期噪声已消化 |

> 用户可通过 Web 界面或配置文件自定义权重，自定义权重优先于 horizon 默认值。

综合分数范围 `[-1, +1]`，映射到字母评级：

| 评级 | 分数范围 | 含义 |
|-----|---------|------|
| **A** | < -0.50 | 强烈低估，关注买入 |
| **B** | -0.50 ~ -0.15 | 低估，值得研究 |
| **C** | -0.15 ~ +0.15 | 合理估值 |
| **D** | +0.15 ~ +0.50 | 高估，谨慎持有 |
| **F** | > +0.50 | 强烈高估，考虑减持 |

#### 9.3 配置

在 `config/config.yaml` 中追加估值配置节（参考 `config/default.yaml.example`）：

```yaml
valuation:
  weights:
    relative: 0.30
    absolute: 0.25
    sentiment: 0.25
    macro: 0.20
  dcf:
    projection_years: 5
    terminal_growth_rate: 0.025
    equity_risk_premium: 0.05
  default_horizon: "3M"     # 投资时间窗口 (1M/3M/6M/1Y)
  default_top_n: 50
  fred_api_key: ""        # 可选，宏观数据更完整；也可设置环境变量 FRED_API_KEY
```

FRED API Key 免费注册：https://fred.stlouisfed.org/docs/api/api_key.html

#### 9.4 典型工作流

`valuation-fetch` 的 `--data-type` 支持三种**互相独立**的数据类型，分别对应不同的 Alpha Vantage API 和估值用途：

| 数据类型 | API 端点 | 内容 | 估值用途 | API 消耗 |
|----------|---------|------|----------|---------|
| `overview` | `OVERVIEW` | PE、PB、PS、PEG、EV/EBITDA、MarketCap、行业等 | 相对估值因子 | 1 次/股票 |
| `ohlcv` | `TIME_SERIES_DAILY` | 日线行情（开高低收量） | 情绪因子（RSI、量价、波动率） | 1 次/股票 |
| `financials` | `CASH_FLOW` + `BALANCE_SHEET` + `INCOME_STATEMENT` | 年度/季度财务报表 | 绝对估值（DCF 模型） | 3 次/股票 |
| `all` | 以上全部 | 一次采集所有数据 | 覆盖全部估值维度 | 5 次/股票 |

> 三者无包含关系，可任意组合。`all` = `overview` + `ohlcv` + `financials`。默认仅采集 `overview`（性价比最高）。

`--batch-size` 默认为 **0（自动）**，根据 API plan 和 data-type 智能计算：

| Plan | 限制 | overview (1次) | financials (3次) | all (5次) |
|------|------|---------------|-----------------|-----------|
| **free** | 25次/天 | 自动 25 只/次 | 自动 8 只/次 | 自动 5 只/次 |
| **premium** | 75次/分钟 | 全部（rate limiter 控流） | 全部 | 全部 |

> 支持断点续传：自动跳过缓存有效期内的股票（overview 7天、financials 30天、ohlcv 3天），重复运行同一命令即可从上次中断处继续。

**第一步：增量采集基本面数据**（受 API 限流约束，分天运行）

```bash
# Free plan（25次/天）：每天运行同一命令，自动续传
python python/scripts/run_backtest.py valuation-fetch \
    --universe sp500 --data-type overview

# 采集财务报表（自动限制为 8 只/次，即 24 API 调用）
python python/scripts/run_backtest.py valuation-fetch \
    --universe sp500 --data-type financials

# Premium plan（75次/分钟）：一次全量采集，rate limiter 自动控速
python python/scripts/run_backtest.py valuation-fetch \
    --universe sp500 --data-type overview --data-type financials

# 显式覆盖批量大小
python python/scripts/run_backtest.py valuation-fetch \
    --universe sp500 --data-type overview --batch-size 10

# 采集自定义股票组合
python python/scripts/run_backtest.py valuation-fetch \
    --symbols AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA \
    --data-type overview --data-type financials
```

**第二步：查看数据覆盖率**

```bash
# 检查 S&P500 采集进度
python python/scripts/run_backtest.py valuation-status --universe sp500

# 示例输出：
# 股票池大小: 100
# ┌──────────────────┬─────┐
# │ 总股票数          │ 100 │
# │ 有 OVERVIEW      │  72 │
# │ 有财务报表        │  45 │
# │ OVERVIEW 覆盖率   │ 72.0% │
# │ 财务数据覆盖率    │ 45.0% │
# └──────────────────┴─────┘
```

**第三步：执行估值扫描**（纯本地计算，秒级完成）

```bash
# 扫描 S&P500，预期 3 个月内回归合理估值
python python/scripts/run_backtest.py valuation-scan \
    --horizon 3M --universe sp500 --top 30

# 扫描 NASDAQ100，1年窗口（DCF 权重 35%，情绪权重 15%）
python python/scripts/run_backtest.py valuation-scan \
    --horizon 1Y --universe nasdaq100 --top 50

# 扫描自定义组合并导出 CSV
python python/scripts/run_backtest.py valuation-scan \
    --horizon 6M --symbols AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,BRK.B \
    --top 10 --export results.csv
```

**第四步：单只股票深度分析**

```bash
# 分析苹果公司（默认 3 个月窗口）
python python/scripts/run_backtest.py valuation-stock AAPL

# 分析微软（6 个月窗口，DCF 权重提升到 30%）
python python/scripts/run_backtest.py valuation-stock MSFT --horizon 6M

# 示例输出：
# AAPL — Apple Inc.
# ┌────────┬──────────────────────────┐
# │ 行业   │ Technology / Consumer Electronics │
# │ 当前价 │ $213.49                  │
# │ 综合评级 │ B                       │
# │ 综合分数 │ -0.2341                 │
# │ 安全边际 │ +18.5%                  │
# └────────┴──────────────────────────┘
#
# 四维因子得分
# ┌──────────────────┬─────────┬──────┐
# │ 相对估值          │ -0.1823 │  30% │
# │ 绝对估值 (DCF)    │ -0.3456 │  25% │
# │ 市场情绪          │ -0.1901 │  25% │
# │ 宏观环境          │ -0.1234 │  20% │
# └──────────────────┴─────────┴──────┘
#
# DCF 三情景分析
# ┌────┬──────────┬──────────┐
# │ 乐观 │ $265.30  │ +24.3%  │
# │ 中性 │ $252.80  │ +18.5%  │
# │ 悲观 │ $198.20  │ -7.2%   │
# └────┴──────────┴──────────┘
```

**查看历史扫描记录**

```bash
python python/scripts/run_backtest.py valuation-history
python python/scripts/run_backtest.py valuation-history --last 10
```

#### 9.5 CLI 命令参数速览

| 命令 | 主要参数 | 说明 |
|-----|---------|------|
| `valuation-fetch` | `--universe` `--data-type` `--batch-size` | 增量采集基本面/财务数据 |
| `valuation-scan` | `--horizon` `--universe` `--top` `--export` | 执行全量扫描，输出排行 |
| `valuation-stock` | `SYMBOL` `--horizon` | 单只股票四维深度分析 |
| `valuation-status` | `--universe` | 查看数据采集覆盖率 |
| `valuation-history` | `--last` | 查看历史扫描记录 |

**`--universe` 可选值**：

| 值 | 说明 | 股票数 |
|----|------|-------|
| `sp500` | S&P 500 核心成分（默认） | ~100只预置 |
| `nasdaq100` | NASDAQ 100 核心成分 | ~65只预置 |
| `all` | 全市场（从 LISTING_STATUS 获取） | 数千只 |
| `custom` | 结合 `--symbols` 使用 | 自定义 |

**`--horizon`**（`valuation-scan` / `valuation-stock` 专用）：投资时间窗口（预期回归合理估值的期限），支持 `1M`（21个交易日）、`3M`（63个交易日，默认）、`6M`（126个交易日）、`1Y`（252个交易日）。不同窗口自动调整四维因子权重（见 9.2 表格）。向后兼容 `--period` 别名

**`--data-type` 可选值**（`valuation-fetch` 专用）：`overview`（OVERVIEW API，1次）、`financials`（三张财务报表，3次）、`ohlcv`（历史价格，1次）、`all`（以上全部，5次）

**`--batch-size`**（`valuation-fetch` 专用）：默认 `0`（自动根据 plan 和 data-type 计算），可手动指定覆盖

#### 9.6 Web 界面

启动 Web 服务后，访问 **https://localhost:8443/chart/valuation** 进入估值扫描页面：

- **股票池 & 投资时间窗口选择**：sp500 / nasdaq100 / 自定义，1M / 3M / 6M / 1Y（不同窗口自动调整因子权重）
- **权重调节滑块**：实时调整四维因子权重（相对/绝对/情绪/宏观），重新计算无需重新采集数据
- **低估/高估排行表**：可排序，含 A-F 评级徽章、四维分项、安全边际
- **行业热力图**：各行业平均估值分数 Plotly 柱状图
- **股票详情模态框**：四维雷达图 + DCF 三情景柱状图 + 关键指标表
- **宏观仪表盘**：10Y 国债收益率、VIX、CPI 等实时展示

**Web API 端点**：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/valuation/scan` | POST | 触发估值扫描（异步） |
| `/api/valuation/results` | GET | 获取最新扫描结果 |
| `/api/valuation/stock/{symbol}` | GET | 单只股票详细评估 |
| `/api/valuation/data-status` | GET | 数据覆盖率统计 |
| `/api/valuation/history` | GET | 历史扫描记录 |

#### 9.7 数据库新增表结构

估值模块在原有 SQLite 数据库中新增两张表：

**valuation_financials — 财务报表缓存**

| 列 | 类型 | 说明 |
|----|------|------|
| `symbol` | TEXT | 股票代码 |
| `fiscal_year` | TEXT | 财年（如 2024-09-30） |
| `report_type` | TEXT | 报表类型：income / balance / cashflow |
| `data_json` | TEXT | JSON 格式财务数据 |
| `created_at` | TIMESTAMP | 写入时间 |

**valuation_results — 扫描历史**

| 列 | 类型 | 说明 |
|----|------|------|
| `id` | INTEGER (PK) | 自增主键 |
| `scan_date` | TIMESTAMP | 扫描时间 |
| `lookback_period` | TEXT | 投资时间窗口（1M/3M/6M/1Y），API 中已更名为 `horizon` |
| `universe` | TEXT | 股票池名称 |
| `result_json` | TEXT | 完整扫描结果 JSON |
| `weights_json` | TEXT | 使用的因子权重 JSON |

stock_info 表同步扩展了 14 个基本面字段：`pb_ratio`、`ps_ratio`、`peg_ratio`、`forward_pe`、`ev_to_ebitda`、`book_value`、`revenue_per_share`、`profit_margin`、`operating_margin`、`return_on_equity`、`shares_outstanding`、`price_to_fcf`、`dividend_per_share`、`payout_ratio`。

#### 9.8 模块文件结构

```
python/alchemist/
├── data/providers/
│   ├── alphavantage.py         ← 扩展：get_stock_overview / get_income_statement
│   │                              get_balance_sheet / get_cash_flow / get_listing_status
│   ├── yfinance_provider.py    ← 新增：空头持仓 + 内部人交易
│   └── fred_provider.py        ← 新增：FRED 宏观指标
├── data/cache/
│   └── sqlite_cache.py         ← 扩展：valuation_financials / valuation_results 表
├── strategy/valuation/
│   ├── __init__.py
│   ├── models.py               ← StockValuation / ScanResult / score_to_grade
│   ├── scorer.py               ← ValuationScorer（四维加权合成）
│   ├── scanner.py              ← ValuationScanner + fetch_data_for_scan
│   ├── universe.py             ← StockUniverse（股票池管理）
│   └── factors/
│       ├── __init__.py
│       ├── relative.py         ← RelativeValuationFactors
│       ├── absolute.py         ← DCFModel / ResidualIncomeModel / AbsoluteValuationFactors
│       ├── sentiment.py        ← SentimentFactors
│       └── macro.py            ← MacroFactors
└── web/
    ├── routes/valuation.py     ← REST API 路由
    ├── schemas/valuation.py    ← Pydantic 请求/响应模型
    └── templates/charts/valuation.html  ← 交互式前端页面
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
4. ~~**Web界面模块**~~ - ✅ 已完成（数据可视化、K线图、多资产对比）
5. ~~**估值扫描模块**~~ - ✅ 已完成（四步估值框架、DCF三情景、A-F字母评级、CLI + Web双输出）
