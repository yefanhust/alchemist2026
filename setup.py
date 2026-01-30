"""
金融指标跟踪及智能交易系统
"""

from setuptools import setup, find_packages

setup(
    name="alchemist2026",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Alchemist2026 - 模块化量化交易系统，支持模拟交易、策略回测和智能分析",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/alchemist2026",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "scipy>=1.11.0",
        "sqlalchemy>=2.0.0",
        "requests>=2.31.0",
        "aiohttp>=3.9.0",
        "pyyaml>=6.0.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
        "pydantic>=2.5.0",
    ],
    extras_require={
        "gpu": [
            "cupy-cuda12x>=12.3.0",
            "numba>=0.58.0",
            "torch>=2.1.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "mypy>=1.7.0",
            "ruff>=0.1.0",
        ],
        "notebook": [
            "jupyterlab>=4.0.0",
            "ipywidgets>=8.1.0",
            "matplotlib>=3.8.0",
            "plotly>=5.18.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    entry_points={
        "console_scripts": [
            "quant-backtest=scripts.run_backtest:main",
        ],
    },
)
