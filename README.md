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
│   │   └── utils/                  # 工具模块
│   │       ├── __init__.py
│   │       ├── config.py           # 配置管理
│   │       ├── logger.py           # 日志系统
│   │       └── time_utils.py       # 时间工具
│   ├── scripts/                     # 运行脚本
│   │   ├── run_backtest.py          # 策略回测
│   │   └── datasource_info.py       # 数据源信息查询
│   ├── tests/                       # 测试
│   │   ├── __init__.py
│   │   ├── test_core/
│   │   ├── test_data/
│   │   ├── test_strategy/
│   │   └── test_simulation/
│   └── notebooks/                   # Jupyter notebooks
│       └── examples/
├── config/                          # 配置文件
│   ├── default.yaml
│   └── strategies/
├── data/                            # 数据存储
│   ├── cache/                       # 缓存数据库
│   └── output/                      # 输出结果
├── logs/                            # 日志文件
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
# 在项目根目录下执行（无需 cd docker）
docker-compose -f docker/docker-compose.yml down          # 停止并移除旧容器
docker-compose -f docker/docker-compose.yml up -d --build  # 构建并启动容器
docker-compose -f docker/docker-compose.yml exec quant-dev bash  # 进入容器
```

### 2. 配置 API Key
```bash
cp .env.example .env
# 编辑 .env 文件，填入 Alpha Vantage API Key
```

### 3. 安装项目（必需步骤）

**在容器内安装项目为可编辑模式（必须执行）：**

```bash
# 在容器内 /workspace 目录下执行
pip install -e .
```

**为什么必须执行此步骤？**
- 项目使用了 Python 包结构，模块之间有相对导入
- 不安装的话，会遇到 `ImportError: attempted relative import beyond top-level package` 错误
- 安装后代码修改仍会立即生效（可编辑模式）

**执行效果：**
- 解决所有模块导入问题
- 支持在任意目录运行脚本
- 注册 console_scripts 命令行工具

### 4. 运行示例回测

```bash
# 基础示例：SMA 交叉策略
python python/scripts/run_backtest.py --strategy sma_crossover --symbol AAPL

# 均值回归策略，多个标的
python python/scripts/run_backtest.py --strategy mean_reversion --symbol AAPL MSFT GOOGL

# 查看帮助信息
python python/scripts/run_backtest.py --help
```

### 5. 查询数据源信息

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

> **扩展数据源**：新增 DataProvider 实现后，只需在 `datasource_info.py` 的 `DATA_SOURCE_REGISTRY` 中注册即可被脚本自动识别。

### 6. 运行测试

测试框架会自动从项目根目录的 `.env` 文件加载 `ALPHAVANTAGE_API_KEY` 等环境变量，无需手动 export。

```bash
# 运行所有测试
pytest python/tests/ -v

# 仅运行港股连通性测试
pytest python/tests/test_alphavantage_hk.py -v -s

# 运行指定测试用例
pytest python/tests/test_alphavantage_hk.py -v -k test_get_hk_daily_data

# 查看详细输出（含 API 返回数据）
pytest python/tests/test_alphavantage_hk.py -v -s --tb=short
```

> **说明**：如果 `.env` 中未配置 `ALPHAVANTAGE_API_KEY`，Alpha Vantage 相关测试会自动跳过。

## Docker 环境详细指南

### 环境要求

- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA Docker Runtime（用于 GPU 支持）
- NVIDIA 驱动（支持 CUDA 12.2+）

验证 GPU 支持：

```bash
# 检查 NVIDIA Docker 是否正常工作
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### Docker 镜像构建

#### 方法 1：使用 docker-compose 构建（推荐）

```bash
# 在项目根目录或 docker 目录下执行
docker-compose -f docker/docker-compose.yml build
```

#### 方法 2：直接使用 Docker 构建

```bash
# 进入 docker 目录
cd docker

# 构建镜像
docker build -t quant-trading-system:latest .

# 如果需要重新构建（不使用缓存）
docker build --no-cache -t quant-trading-system:latest .
```

#### 构建参数说明

- 镜像基于 `nvidia/cuda:12.2.0-devel-ubuntu22.04`
- 支持双 GPU 加速（CUDA 12.2）
- 预装 Python 3.11 及量化交易所需的全部依赖
- 包含 JupyterLab、NumPy、Pandas、CuPy、PyTorch 等库

### Docker Compose 使用详解

#### 启动开发容器（主容器）

```bash
# 启动主开发容器 + Redis
docker-compose -f docker/docker-compose.yml up -d

# 进入容器
docker-compose -f docker/docker-compose.yml exec quant-dev bash

# 停止容器
docker-compose -f docker/docker-compose.yml down
```

#### 启动独立的 JupyterLab 服务

```bash
# 使用 profile 启动 Jupyter 服务
docker-compose -f docker/docker-compose.yml --profile jupyter up -d jupyter

# 查看 Jupyter 日志（获取访问 URL）
docker-compose -f docker/docker-compose.yml logs jupyter

# 停止 Jupyter 服务
docker-compose -f docker/docker-compose.yml --profile jupyter down
```

#### 启动所有服务

```bash
# 同时启动开发容器、Redis 和 Jupyter
docker-compose -f docker/docker-compose.yml --profile jupyter up -d

# 查看所有运行的容器
docker-compose -f docker/docker-compose.yml ps

# 停止所有服务
docker-compose -f docker/docker-compose.yml --profile jupyter down
```

#### 重新构建并启动

```bash
# 重新构建并启动
docker-compose -f docker/docker-compose.yml up -d --build
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

### 常用 Docker 操作

#### 查看日志

```bash
# 查看主容器日志
docker-compose -f docker/docker-compose.yml logs -f quant-dev

# 查看 Jupyter 日志
docker-compose -f docker/docker-compose.yml logs -f jupyter

# 查看 Redis 日志
docker-compose -f docker/docker-compose.yml logs -f redis
```

#### 进入容器

```bash
# 进入主开发容器
docker-compose -f docker/docker-compose.yml exec quant-dev bash

# 进入 Jupyter 容器
docker-compose -f docker/docker-compose.yml exec jupyter bash

# 以 root 用户进入
docker-compose -f docker/docker-compose.yml exec -u root quant-dev bash
```

#### 容器管理

```bash
# 重启容器
docker-compose -f docker/docker-compose.yml restart quant-dev

# 停止容器（保留数据）
docker-compose -f docker/docker-compose.yml stop

# 启动已停止的容器
docker-compose -f docker/docker-compose.yml start

# 删除容器和卷（清理所有数据）
docker-compose -f docker/docker-compose.yml down -v
```

#### 监控 GPU 使用

```bash
# 在容器内使用 nvidia-smi
docker-compose -f docker/docker-compose.yml exec quant-dev nvidia-smi

# 或使用 nvtop（更友好的界面）
docker-compose -f docker/docker-compose.yml exec quant-dev nvtop
```

### 目录映射说明

容器内 `/workspace` 目录映射关系：

| 容器路径 | 宿主机路径 | 说明 |
|---------|-----------|------|
| `/workspace/python` | `../python` | Python 源代码根目录（含包、脚本、测试、notebooks） |
| `/workspace/config` | `../config` | 配置文件 |
| `/workspace/data` | `../data` | 数据文件（持久化） |
| `/workspace/logs` | `../logs` | 日志文件 |
| `/workspace/.env` | `../.env` | 环境变量（只读） |

所有修改会实时同步到宿主机。

### 端口映射

| 服务 | 容器端口 | 宿主机端口 | 说明 |
|-----|---------|-----------|------|
| 主容器 JupyterLab | 8888 | 8888 | 开发容器的 Jupyter |
| 独立 Jupyter | 8888 | 8889 | 独立 Jupyter 服务 |
| Web 服务 | 8000 | 8000 | API 或 Web 应用 |
| Redis | 6379 | 6379 | 缓存服务 |

### 环境变量

容器内默认环境变量：

```bash
NVIDIA_VISIBLE_DEVICES=all      # 使用所有 GPU
CUDA_VISIBLE_DEVICES=0,1        # GPU 0 和 1
PYTHONPATH=/workspace/python    # Python 模块搜索路径
TZ=Asia/Shanghai                # 时区设置
```

可通过修改 `docker-compose.yml` 中的 `environment` 部分调整。

### 常见问题

**Q: 容器无法访问 GPU**

```bash
# 确保安装了 NVIDIA Docker Runtime
nvidia-smi

# 检查 Docker 是否支持 GPU
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

**Q: JupyterLab 无法访问**

- 检查容器是否运行：`docker-compose ps`
- 检查端口是否被占用：`netstat -tuln | grep 8888`
- 查看日志：`docker-compose logs jupyter`

**Q: 权限问题**

```bash
# 容器内使用 developer 用户（UID 1000）
# 如需 root 权限
docker-compose exec -u root quant-dev bash
```

**Q: 修改代码后需要重启容器吗？**

不需要。代码目录已挂载为数据卷，修改会实时同步。只有修改 Dockerfile 或安装新依赖时才需要重新构建。

**Q: 如何安装额外的 Python 包？**

```bash
# 方法 1：进入容器临时安装（重启后失效）
docker-compose exec quant-dev pip install package-name

# 方法 2：修改 Dockerfile 并重新构建（永久）
# 编辑 Dockerfile，添加包后：
docker-compose build
docker-compose up -d
```

### 性能优化建议

1. **使用 BuildKit 加速构建**：
   ```bash
   DOCKER_BUILDKIT=1 docker-compose build
   ```

2. **清理无用镜像和容器**：
   ```bash
   docker system prune -a
   ```

3. **GPU 内存管理**：
   - 在代码中使用 `torch.cuda.empty_cache()` 释放显存
   - 合理设置 batch size

4. **挂载数据卷性能**：
   - 大文件读写建议使用 `/workspace/data` 目录
   - 避免频繁读写小文件

### Docker 快速命令速查

```bash
# 1. 构建镜像
docker-compose -f docker/docker-compose.yml build

# 2. 启动开发环境 + Redis
docker-compose -f docker/docker-compose.yml up -d

# 3. 启动 Jupyter（独立服务）
docker-compose -f docker/docker-compose.yml --profile jupyter up -d jupyter

# 4. 进入开发容器
docker-compose -f docker/docker-compose.yml exec quant-dev bash

# 5. 查看日志
docker-compose -f docker/docker-compose.yml logs -f

# 6. 停止所有服务
docker-compose -f docker/docker-compose.yml --profile jupyter down
```

### 开发工作流示例

```bash
# 1. 首次启动
cd /path/to/alchemist2026
docker-compose -f docker/docker-compose.yml up -d --build

# 2. 进入容器
docker-compose -f docker/docker-compose.yml exec quant-dev bash

# 3. 首次使用：安装项目为可编辑包（重要！）
cd /workspace
pip install -e .

# 4. 配置 API Key
cp .env.example .env
# 编辑 .env 文件，填入必要的 API Key

# 5. 查看数据源支持情况
python python/scripts/datasource_info.py info
python python/scripts/datasource_info.py hk-support

# 6. 运行回测脚本（需先执行步骤 3 安装项目）
python python/scripts/run_backtest.py --strategy sma_crossover --symbol AAPL

# 7. 运行测试（自动从 .env 加载 API Key）
pytest python/tests/ -v

# 8. 启动 Jupyter 进行数据分析
docker-compose -f docker/docker-compose.yml --profile jupyter up -d jupyter
# 浏览器打开 http://localhost:8889

# 9. 完成工作后停止
docker-compose -f docker/docker-compose.yml down
```

#### 关于 `pip install -e .`

在容器内首次使用前，**强烈建议**运行 `pip install -e .` 将项目安装为可编辑模式：

**优点：**
- 解决 Python 模块相对导入问题
- 代码修改立即生效，无需重新安装
- 支持 `console_scripts` 入口点
- 符合 Python 项目开发最佳实践

**注意：** 由于项目目录通过 Docker 卷映射，容器重启后安装仍然有效，无需重复执行。

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
