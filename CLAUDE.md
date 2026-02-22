# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Alchemist2026 is a modular quantitative trading system supporting simulated trading, strategy backtesting, and intelligent analysis. It features a gold multi-factor tactical DCA strategy as its primary trading strategy.

## Development Environment

All development happens inside the `quant-dev` Docker container (NVIDIA CUDA base image, Python 3.11).

**Critical import pattern:** The PYTHONPATH is set to `/workspace/python/alchemist`, so imports use:
```python
from strategy.gold import GoldTradingStrategy  # NOT from alchemist.strategy.gold
from data.cache.sqlite_cache import SQLiteCache
from web.routes import pages
```

## Common Commands

All commands run inside the Docker container:
```bash
docker exec -it quant-dev bash
```

### Tests
```bash
# Run all tests
docker exec -it quant-dev bash -c "cd /workspace && PYTHONPATH=/workspace/python/alchemist pytest python/tests/ -v"

# Run a single test file
docker exec -it quant-dev bash -c "cd /workspace && PYTHONPATH=/workspace/python/alchemist pytest python/tests/test_gold_signals.py -v"

# Run tests matching a pattern
docker exec -it quant-dev bash -c "cd /workspace && PYTHONPATH=/workspace/python/alchemist pytest python/tests/ -k 'test_name_pattern' -v"
```

Tests use pytest + pytest-asyncio. The conftest.py auto-loads `config/config.yaml` and auto-skips tests requiring API keys if not configured.

### Web Server
```bash
# Start web service (HTTPS on port 8443)
docker exec -it quant-dev bash -c "cd /workspace && bash scripts/start_web.sh"

# Or directly with uvicorn
docker exec -it quant-dev bash -c "cd /workspace && PYTHONPATH=/workspace/python/alchemist uvicorn web.app:app --host 0.0.0.0 --port 8443 --ssl-keyfile=certs/key.pem --ssl-certfile=certs/cert.pem --reload"
```

### Docker
```bash
cd docker && docker-compose up quant-dev    # Main dev container
cd docker && docker-compose up -d           # All services (background)
```

## Architecture

Source code root: `python/alchemist/`

| Package | Purpose |
|---------|---------|
| `core/` | Domain abstractions: Asset, Portfolio, Order, Position |
| `data/` | Data providers (AlphaVantage) and caching (SQLiteCache) |
| `strategy/` | Trading strategies, technical indicators, signal modules |
| `simulation/` | Backtesting engine |
| `web/` | FastAPI web service (routes, schemas, services, templates) |
| `utils/` | Configuration loader, logging, time utilities |
| `gpu/` | GPU acceleration utilities (CUDA/CuPy) |

### Gold Strategy (`strategy/gold/`)
- `strategy.py` — `GoldTradingStrategy`: tactical DCA with multi-factor scoring
- `signals/technical.py` — Technical analysis factors
- `signals/cross_market.py` — Cross-market correlation factors (GDX, SPY, UUP, TIP, TLT, VIXY)
- `signals/sentiment.py` — Market sentiment factors
- `signals/macro.py` — Macroeconomic factors
- Key methods: `calculate_factors()`, `_compute_composite_score()`, `_describe_action()`, `_generate_tactical_signals()`

### Web (`web/`)
- Framework: FastAPI + Jinja2 + Plotly.js + Bootstrap 5
- Routes: `health`, `cache`, `market`, `pages`, `gold_backtest`
- API prefix: `/api/` (e.g., `/api/gold-backtest/signals`, `/api/market/ohlcv`)
- Pages: dashboard (`/`), candlestick (`/chart/candlestick`), comparison (`/chart/comparison`), gold backtest (`/chart/gold-backtest`)
- Templates extend `base.html`, charts use Plotly.js with shared x-axis subplots

### Data Layer
- `data/providers/alphavantage.py` — AlphaVantageProvider with methods for OHLCV, treasury yields, forex, economic indicators
- `data/cache/sqlite_cache.py` — SQLiteCache for persistent OHLCV caching
- `data/models.py` — MarketData, OHLCV dataclasses
- Config in `config/config.yaml` (API keys, rate limits, database settings)

## Key Conventions

- Pydantic v2 models for API schemas (`web/schemas/`)
- Async routes and cache operations (asyncio)
- Strategy requires ~210 days of warmup history data
- Cross-market symbols: GDX, SPY, UUP, TIP, TLT, VIXY (mapped to factor data keys)
- Web service runs on HTTPS port 8443 (domain: quant.enrichir.top)
