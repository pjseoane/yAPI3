"""yfinance_api3 — reusable quantitative finance engine."""

from yfinance_api3.classes.stock_client import StockClient
from yfinance_api3.classes.quant_analytics import QuantAnalytics

__version__ = "0.1.0"
__all__ = ["StockClient", "QuantAnalytics", "__version__"]
