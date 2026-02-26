"""
测试全局配置

自动加载项目配置文件，使测试可以读取 API Key 等配置项。
"""

import os
from pathlib import Path

import pytest

from alchemist.utils.config import load_config, get_config, reset_config


def pytest_configure(config):
    """pytest 启动时自动加载配置文件"""
    # 重置全局配置，确保每次测试使用最新配置
    reset_config()

    # 按优先级查找配置文件
    search_paths = [
        Path.cwd() / "config" / "config.yaml",
        Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml",
        Path("/workspace/config/config.yaml"),
    ]

    for config_path in search_paths:
        if config_path.is_file():
            app_config = load_config(str(config_path))
            print(f"\n[conftest] 已加载配置文件: {config_path}")

            # 将 fred_api_key 注入环境变量（若未通过 env 显式设置）
            fred_key = app_config.fred_api_key
            if fred_key and not os.environ.get("FRED_API_KEY"):
                os.environ["FRED_API_KEY"] = fred_key

            return

    print("\n[conftest] 未找到 config/config.yaml，使用默认配置")


def pytest_collection_modifyitems(config, items):
    """为需要 API Key 的测试自动添加 skip 标记"""
    app_config = get_config()
    skip_no_api_key = pytest.mark.skip(reason="未设置 alphavantage.api_key 配置")

    for item in items:
        if "alphavantage" in item.nodeid and not app_config.alphavantage.api_key:
            item.add_marker(skip_no_api_key)
