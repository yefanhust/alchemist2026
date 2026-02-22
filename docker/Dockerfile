# 金融交易系统开发容器
# 基于 NVIDIA CUDA 镜像，支持双 GPU 加速

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    htop \
    nvtop \
    sqlite3 \
    libsqlite3-dev \
    libpq-dev \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 设置时区为 Asia/Shanghai
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" > /etc/timezone

# 设置 Python 3.11 为默认
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# 升级 pip
RUN python -m pip install --upgrade pip setuptools wheel

# 安装 Python 依赖 - 分层安装以利用缓存
# 1. 核心科学计算库
RUN pip install \
    numpy==1.26.2 \
    pandas==2.1.3 \
    scipy==1.11.4

# 2. GPU 加速库
RUN pip install \
    cupy-cuda12x==12.3.0 \
    numba==0.58.1

# 3. 金融数据和分析库
RUN pip install \
    yfinance==0.2.33 \
    alpha-vantage==2.3.1 \
    requests==2.31.0 \
    aiohttp==3.9.1

# 4. 技术分析库
RUN pip install \
    ta-lib-bin==0.4.26 \
    ta

# 5. 可视化库
RUN pip install \
    matplotlib==3.8.2 \
    plotly==5.18.0 \
    seaborn==0.13.0

# 6. 数据库支持
RUN pip install \
    sqlalchemy==2.0.23 \
    alembic==1.12.1 \
    redis==5.0.1 \
    h5py==3.10.0

# 7. 开发工具
RUN pip install \
    jupyterlab==4.0.9 \
    ipywidgets==8.1.1 \
    pytest==7.4.3 \
    pytest-asyncio==0.21.1 \
    black==23.11.0 \
    isort==5.12.0 \
    mypy==1.7.1 \
    ruff==0.1.6

# 8. 配置和日志
RUN pip install \
    pyyaml==6.0.1 \
    python-dotenv==1.0.0 \
    loguru==0.7.2 \
    rich==13.7.0 \
    click==8.1.7 \
    typer==0.9.0

# 9. 异步和并发
RUN pip install \
    asyncio==3.4.3 \
    aiosqlite==0.19.0 \
    joblib==1.3.2

# 10. 机器学习（为后续模块预留）
RUN pip install scikit-learn==1.3.2
RUN pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121

# 11. Web 框架
RUN pip install \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    python-multipart==0.0.6 \
    bcrypt==4.2.1 \
    itsdangerous==2.2.0

# 创建非 root 用户
RUN useradd -m -s /bin/bash developer \
    && chown -R developer:developer /workspace

# 创建必要的目录
RUN mkdir -p /workspace/data/cache \
    && mkdir -p /workspace/data/output \
    && mkdir -p /workspace/logs \
    && chown -R developer:developer /workspace

# 切换到开发者用户
USER developer

# 配置 JupyterLab
RUN mkdir -p /home/developer/.jupyter
COPY --chown=developer:developer jupyter_config.py /home/developer/.jupyter/jupyter_lab_config.py

# 暴露端口
# JupyterLab
EXPOSE 8888
# 可能的 Web 服务
EXPOSE 8000

# 默认启动命令
CMD ["bash"]
