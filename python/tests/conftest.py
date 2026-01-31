"""
测试全局配置

自动加载项目根目录的 .env 文件，使测试可以读取 API Key 等环境变量。
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv


def pytest_configure(config):
    """pytest 启动时自动加载 .env 文件"""
    # 按优先级查找 .env 文件
    search_paths = [
        Path.cwd() / ".env",                          # 当前工作目录
        Path(__file__).resolve().parent.parent.parent / ".env",  # 项目根目录 (python/tests/../../.env)
        Path("/workspace/.env"),                       # 容器内路径
    ]

    for env_path in search_paths:
        if env_path.is_file():
            load_dotenv(env_path, override=True)
            print(f"\n[conftest] 已加载环境变量: {env_path}")
            return

    print("\n[conftest] 未找到 .env 文件，跳过环境变量加载")


def pytest_collection_modifyitems(config, items):
    """为需要 API Key 的测试自动添加 skip 标记"""
    skip_no_api_key = pytest.mark.skip(reason="未设置 ALPHAVANTAGE_API_KEY 环境变量")

    for item in items:
        if "alphavantage" in item.nodeid and not os.getenv("ALPHAVANTAGE_API_KEY"):
            item.add_marker(skip_no_api_key)
