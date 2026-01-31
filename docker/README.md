# Docker 环境使用指南

本文档说明如何构建和使用量化交易系统的 Docker 开发环境。

## 环境要求

- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA Docker Runtime（用于 GPU 支持）
- NVIDIA 驱动（支持 CUDA 12.2+）

### 验证 GPU 支持

```bash
# 检查 NVIDIA Docker 是否正常工作
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

## 一、构建 Docker 镜像

### 方法 1：使用 docker-compose 构建（推荐）

```bash
# 在项目根目录或 docker 目录下执行
docker-compose -f docker/docker-compose.yml build
```

### 方法 2：直接使用 Docker 构建

```bash
# 进入 docker 目录
cd docker

# 构建镜像
docker build -t quant-trading-system:latest .

# 如果需要重新构建（不使用缓存）
docker build --no-cache -t quant-trading-system:latest .
```

### 构建参数说明

- 镜像基于 `nvidia/cuda:12.2.0-devel-ubuntu22.04`
- 支持双 GPU 加速（CUDA 12.2）
- 预装 Python 3.11 及量化交易所需的全部依赖
- 包含 JupyterLab、NumPy、Pandas、CuPy、PyTorch 等库

## 二、使用 Docker Compose

### 2.1 启动开发容器（主容器）

```bash
# 启动主开发容器 + Redis
docker-compose -f docker/docker-compose.yml up -d

# 进入容器
docker-compose -f docker/docker-compose.yml exec quant-dev bash

# 停止容器
docker-compose -f docker/docker-compose.yml down
```

### 2.2 启动独立的 JupyterLab 服务

```bash
# 使用 profile 启动 Jupyter 服务
docker-compose -f docker/docker-compose.yml --profile jupyter up -d jupyter

# 查看 Jupyter 日志（获取访问 URL）
docker-compose -f docker/docker-compose.yml logs jupyter

# 停止 Jupyter 服务
docker-compose -f docker/docker-compose.yml --profile jupyter down
```

### 2.3 启动所有服务

```bash
# 同时启动开发容器、Redis 和 Jupyter
docker-compose -f docker/docker-compose.yml --profile jupyter up -d

# 查看所有运行的容器
docker-compose -f docker/docker-compose.yml ps

# 停止所有服务
docker-compose -f docker/docker-compose.yml --profile jupyter down
```

### 2.4 重新构建并启动

```bash
# 重新构建并启动
docker-compose -f docker/docker-compose.yml up -d --build
```

## 三、启动和使用 JupyterLab

### 方法 1：在主开发容器中启动 Jupyter

```bash
# 进入开发容器
docker-compose -f docker/docker-compose.yml exec quant-dev bash

# 在容器内启动 JupyterLab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

访问地址：`http://localhost:8888`

### 方法 2：使用独立 Jupyter 服务（推荐）

```bash
# 使用 profile 启动独立的 Jupyter 容器
docker-compose -f docker/docker-compose.yml --profile jupyter up -d jupyter
```

访问地址：`http://localhost:8889`

**注意**：独立 Jupyter 服务端口映射为 `8889:8888`，避免与主容器端口冲突。

### JupyterLab 配置说明

- **工作目录**：`/workspace`
- **Notebook 目录**：`/workspace/notebooks`
- **无需 Token/密码**：开发环境已禁用身份验证
- **GPU 支持**：可使用所有 GPU（双卡）

### 在 Jupyter 中验证 GPU

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

## 四、常用操作

### 4.1 查看日志

```bash
# 查看主容器日志
docker-compose -f docker/docker-compose.yml logs -f quant-dev

# 查看 Jupyter 日志
docker-compose -f docker/docker-compose.yml logs -f jupyter

# 查看 Redis 日志
docker-compose -f docker/docker-compose.yml logs -f redis
```

### 4.2 进入容器

```bash
# 进入主开发容器
docker-compose -f docker/docker-compose.yml exec quant-dev bash

# 进入 Jupyter 容器
docker-compose -f docker/docker-compose.yml exec jupyter bash

# 以 root 用户进入
docker-compose -f docker/docker-compose.yml exec -u root quant-dev bash
```

### 4.3 容器管理

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

### 4.4 监控 GPU 使用

```bash
# 在容器内使用 nvidia-smi
docker-compose -f docker/docker-compose.yml exec quant-dev nvidia-smi

# 或使用 nvtop（更友好的界面）
docker-compose -f docker/docker-compose.yml exec quant-dev nvtop
```

## 五、目录映射说明

容器内 `/workspace` 目录映射关系：

| 容器路径 | 宿主机路径 | 说明 |
|---------|-----------|------|
| `/workspace/python` | `../python` | Python 源代码根目录（含包、脚本、测试、notebooks） |
| `/workspace/config` | `../config` | 配置文件 |
| `/workspace/data` | `../data` | 数据文件（持久化） |
| `/workspace/logs` | `../logs` | 日志文件 |
| `/workspace/.env` | `../.env` | 环境变量（只读） |

所有修改会实时同步到宿主机。

## 六、端口映射

| 服务 | 容器端口 | 宿主机端口 | 说明 |
|-----|---------|-----------|------|
| 主容器 JupyterLab | 8888 | 8888 | 开发容器的 Jupyter |
| 独立 Jupyter | 8888 | 8889 | 独立 Jupyter 服务 |
| Web 服务 | 8000 | 8000 | API 或 Web 应用 |
| Redis | 6379 | 6379 | 缓存服务 |

## 七、环境变量

容器内默认环境变量：

```bash
NVIDIA_VISIBLE_DEVICES=all      # 使用所有 GPU
CUDA_VISIBLE_DEVICES=0,1        # GPU 0 和 1
PYTHONPATH=/workspace/python    # Python 模块搜索路径
TZ=Asia/Shanghai                # 时区设置
```

可通过修改 `docker-compose.yml` 中的 `environment` 部分调整。

## 八、常见问题

### Q1: 容器无法访问 GPU

**解决方案**：
```bash
# 确保安装了 NVIDIA Docker Runtime
nvidia-smi

# 检查 Docker 是否支持 GPU
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### Q2: JupyterLab 无法访问

**解决方案**：
- 检查容器是否运行：`docker-compose ps`
- 检查端口是否被占用：`netstat -tuln | grep 8888`
- 查看日志：`docker-compose logs jupyter`

### Q3: 权限问题

**解决方案**：
```bash
# 容器内使用 developer 用户（UID 1000）
# 如需 root 权限
docker-compose exec -u root quant-dev bash
```

### Q4: 修改代码后需要重启容器吗？

**不需要**。代码目录已挂载为数据卷，修改会实时同步。只有修改 Dockerfile 或安装新依赖时才需要重新构建。

### Q5: 如何安装额外的 Python 包？

```bash
# 方法 1：进入容器临时安装（重启后失效）
docker-compose exec quant-dev pip install package-name

# 方法 2：修改 Dockerfile 并重新构建（永久）
# 编辑 Dockerfile，添加包后：
docker-compose build
docker-compose up -d
```

## 九、性能优化建议

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

## 十、快速启动命令速查

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

## 十一、开发工作流示例

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

# 5. 运行回测脚本（需先执行步骤 3 安装项目）
python python/scripts/run_backtest.py --strategy sma_crossover --symbol AAPL

# 6. 运行测试
pytest python/tests/

# 7. 启动 Jupyter 进行数据分析
docker-compose -f docker/docker-compose.yml --profile jupyter up -d jupyter
# 浏览器打开 http://localhost:8889

# 8. 完成工作后停止
docker-compose -f docker/docker-compose.yml down
```

### 关于 `pip install -e .`

在容器内首次使用前，**强烈建议**运行 `pip install -e .` 将项目安装为可编辑模式：

**优点：**
- 解决 Python 模块相对导入问题
- 代码修改立即生效，无需重新安装
- 支持 `console_scripts` 入口点
- 符合 Python 项目开发最佳实践

**注意：** 由于项目目录通过 Docker 卷映射，容器重启后安装仍然有效，无需重复执行。

---

**维护者**: Alchemist Team
**最后更新**: 2026-01-30
