import pandas as pd
import pytest

import yfinance_api3
from yfinance_api3.classes.quant_analytics import QuantAnalytics
from yfinance_api3.classes.stock_client import StockClient


def test_public_api_exports() -> None:
    assert "QuantAnalytics" in yfinance_api3.__all__
    assert "StockClient" in yfinance_api3.__all__
    assert isinstance(yfinance_api3.__version__, str)


def test_stock_client_cache_round_trip() -> None:
    cache = StockClient()._cache
    cache.set("demo", {"ok": True}, ttl=60)
    hit, value = cache.get("demo")
    assert hit is True
    assert value == {"ok": True}


def test_quant_analytics_simple_helpers() -> None:
    quant = QuantAnalytics(StockClient())
    prices = pd.Series([100.0, 110.0, 121.0])

    simple = quant._simple_returns(prices)
    log = quant._log_returns(prices)

    assert simple.tolist() == pytest.approx([0.10, 0.10])
    assert log.tolist() == pytest.approx([0.09531018, 0.09531018])


def test_period_slice_returns_recent_rows() -> None:
    rows = [
        {"date": "2022-01-01", "adj_close": 100.0},
        {"date": "2023-01-01", "adj_close": 110.0},
        {"date": "2024-01-01", "adj_close": 120.0},
    ]

    sliced = StockClient._slice_to_period(rows, "1y")

    assert len(sliced) == 2
    assert sliced[0]["date"] == "2023-01-01"
    assert sliced[1]["date"] == "2024-01-01"
