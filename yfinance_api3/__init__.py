"""
yfinance_api3 -- Quantitative finance engine.

Quick start
-----------
    from yfinance_api3 import StockClient, QuantAnalytics
    import yfinance_api3.modules.plots as plots

    client = StockClient()
    quant  = QuantAnalytics(client)

    plots.cumulative_returns(quant, ["AAPL","MSFT"], period="3y").show()

Classes (yfinance_api3.classes)
-------------------------------
    StockClient         -- yfinance wrapper with TTL cache
    QuantAnalytics      -- full metrics, returns, seasonality, Kelly...
    OptionsAnalyzer     -- options chain, Greeks, GEX, PCR, max pain
    OptionsStrategy     -- multi-leg strategy builder + B-S pricing
    pricing             -- BlackScholesModel, Binomial, Engine, PricingStrategy
    PositionsBook       -- live position tracker (options + underlying)
    PortfolioBook       -- multi-ticker portfolio aggregator
    WatchList           -- saved tickers with notes

Modules (yfinance_api3.modules)
-------------------------------
    plots               -- 40 Plotly charts (import as: import plots as plots)
    portfolio           -- efficient frontier, max Sharpe, risk parity
    backtest            -- strategy backtesting with transaction costs
    montecarlo          -- historical bootstrap, normal, Student-t simulation
    factors             -- Fama-French FF3/FF5/MOM/FF6 factor regression
    etf                 -- ETF concentration, holdings, sector weights
    alerts              -- AlertEngine + built-in alert factories
    report              -- HTML report builder (auto_report)
"""

from yfinance_api3.classes.stock_client import StockClient
from yfinance_api3.classes.quant_analytics import QuantAnalytics

__version__ = "0.1.0"
__author__  = "Paulino"
__all__     = ["StockClient", "QuantAnalytics", "__version__"]
