from importlib.metadata import PackageNotFoundError, version

from .classes.quant_analytics import QuantAnalytics
from .classes.stock_client import StockClient

try:
    __version__ = version("yfinance_api3")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["QuantAnalytics", "StockClient", "__version__"]
