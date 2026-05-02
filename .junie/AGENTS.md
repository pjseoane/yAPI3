# Developer Guide for yfinance_api3

This document provides essential information for developers working on the `yfinance_api3` project.

## 1. Build and Configuration Instructions

### Prerequisites
- **Python Version**: `>= 3.14` is required.
- **Build System**: Uses `setuptools` and `wheel`.

### Installation
To install the package in editable mode with development dependencies:
```bash
pip install -e ".[dev]"
```

### Optional Dependencies
- `dashboard-dash`: For Dash-based dashboards.
- `dashboard-streamlit`: For Streamlit-based dashboards.

Install them using:
```bash
pip install -e ".[dashboard-dash,dashboard-streamlit]"
```

## 2. Testing Information

### Configuration
Tests are located in the `tests/` directory and managed by `pytest`. Configuration is defined in `pyproject.toml`.

### Running Tests
To run all tests:
```bash
pytest
```

To run a specific test file:
```bash
pytest tests/test_package.py
```

### Adding New Tests
1. Create a new file in the `tests/` directory with the prefix `test_` (e.g., `tests/test_feature.py`).
2. Import `pytest` and the necessary components from `yfinance_api3`.
3. Define functions starting with `test_`.

### Demonstration Test
Here is a simple test that verifies the `StockClient` initialization and cache functionality:

```python
import pytest
from yfinance_api3.classes.stock_client import StockClient

def test_stock_client_initialization():
    """Verify StockClient can be initialized with custom TTL values."""
    client = StockClient(ttl_price=120, ttl_info=1800)
    assert client.ttl_price == 120
    assert client.ttl_info == 1800

def test_stock_client_cache_functionality():
    """Verify the internal TTLCache stores and retrieves values."""
    client = StockClient()
    client._cache.set("dev_key", "dev_value", ttl=60)
    hit, value = client._cache.get("dev_key")
    assert hit is True
    assert value == "dev_value"
```

## 3. Additional Development Information

### Code Style
- **Linting & Formatting**: The project uses `ruff` for linting and `black` (via `ruff`) for formatting.
- **Type Checking**: `mypy` is used for static type checking.
- **Rules**: Follow the existing patterns in `yfinance_api3/classes` for core logic and `yfinance_api3/modules` for functional extensions.

### Architecture Overview
- `yfinance_api3.classes.stock_client.StockClient`: Core data fetcher with built-in TTL caching.
- `yfinance_api3.classes.quant_analytics.QuantAnalytics`: Main engine for quantitative metrics (returns, volatility, seasonality, etc.).
- `yfinance_api3.modules.plots`: Comprehensive plotting library using Plotly.

### Debugging Tips
- Check cache statistics using `client.cache_stats()`.
- Use `client.invalidate(symbol)` to clear cached data for a specific ticker during interactive debugging.
