import pandas as pd
import pytest

import yfinance_api3
from yfinance_api3.classes.quant_analytics import QuantAnalytics
from yfinance_api3.classes.stock_client import StockClient
from yfinance_api3.modules.alerts import Alert, AlertEngine
import yfinance_api3.modules.portfolio as portfolio


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


def test_stock_client_invalidate_symbol_clears_matching_entries() -> None:
    client = StockClient()
    aapl_key = client._cache._make_key("history", "AAPL", "1y", "1d", True)
    msft_key = client._cache._make_key("history", "MSFT", "1y", "1d", True)

    client._cache.set(aapl_key, {"symbol": "AAPL"}, ttl=60)
    client._cache.set(msft_key, {"symbol": "MSFT"}, ttl=60)

    removed = client.invalidate("AAPL")

    assert removed == 1
    assert client._cache.get(aapl_key) == (False, None)
    assert client._cache.get(msft_key) == (True, {"symbol": "MSFT"})


def test_stock_client_price_change_preserves_previous_close(monkeypatch: pytest.MonkeyPatch) -> None:
    client = StockClient()
    monkeypatch.setattr(
        client,
        "get_price",
        lambda symbol: {"symbol": symbol, "price": 110.0, "previous_close": 100.0},
    )

    change = client.get_price_change("AAPL")

    assert change["previous_close"] == 100.0
    assert change["change"] == 10.0
    assert change["change_pct"] == 10.0


def test_alert_engine_list_alerts_uses_private_last_trigger() -> None:
    engine = AlertEngine(QuantAnalytics(StockClient()), StockClient())
    alert = Alert(name="demo", symbol="AAPL", check_fn=lambda q, c: None)
    alert.mark_triggered()
    engine.add(alert)

    rows = engine.list_alerts()

    assert rows[0]["name"] == "demo"
    assert isinstance(rows[0]["last_trigger"], str)


def test_max_sharpe_uses_risk_free_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        portfolio,
        "_get_inputs",
        lambda quant, symbols, period, risk_free_rate: (
            portfolio.np.array([0.10, 0.20]),
            portfolio.np.array([[0.0025, 0.0], [0.0, 0.0144]]),
        ),
    )

    result = portfolio.max_sharpe(
        quant=None,
        symbols=["LOW_VOL", "HIGH_RET"],
        risk_free_rate=0.08,
    )

    assert result.weights[1] > result.weights[0]


def test_calmar_ratio_uses_requested_period_in_reports() -> None:
    class DummyQuant:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def historical_volatility(self, symbol: str, period: str) -> float:
            return 0.2

        def max_drawdown(self, symbol: str, period: str) -> float:
            return -0.1

        def sharpe_ratio(self, symbol: str, period: str, risk_free_rate: float) -> float:
            return 1.0

        def sortino_ratio(self, symbol: str, period: str, risk_free_rate: float) -> float:
            return 1.2

        def beta(self, symbol: str, benchmark: str, period: str) -> float:
            return 0.9

        def calmar_ratio(self, symbol: str, period: str) -> float:
            self.calls.append((symbol, period))
            return 1.5

        def var(self, symbol: str, period: str, confidence: float = 0.95) -> float:
            return 0.03

        def cvar(self, symbol: str, period: str, confidence: float = 0.95) -> float:
            return 0.04

    dummy = DummyQuant()

    report = QuantAnalytics.stock_report(dummy, "AAPL", period="5y")
    table = QuantAnalytics.metrics_df(dummy, ["AAPL"], period="5y")

    assert report["calmar_ratio"] == 1.5
    assert table.loc["calmar_ratio", "AAPL"] == 1.5
    assert dummy.calls == [("AAPL", "5y"), ("AAPL", "5y")]
