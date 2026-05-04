"""
QuantAnalytics — quantitative metrics built on top of StockClient.

Requires: numpy, pandas, scipy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from yfinance_api3.classes.stock_client import StockClient


class QuantAnalytics:
    """
    Compute quantitative metrics for one or many symbols.

    All historical data is fetched through a StockClient instance,
    so the cache layer is automatically reused.

    Parameters
    ----------
    client : StockClient
        A configured StockClient (handles fetching + caching).
    trading_days : int
        Number of trading days per year used for annualisation (default 252).
    """

    def __init__(self, client: StockClient, trading_days: int = 252) -> None:
        self.client = client
        self.trading_days = trading_days

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prices(self, symbol: str, period: str, interval: str = "1d") -> pd.Series:
        """
        Return an Adj-Close price Series indexed by date.

        Always uses adjusted=True so prices are split- and dividend-corrected.
        This is the correct input for return, volatility, beta, and Sharpe
        calculations — raw closes would distort every metric around ex-div dates
        and stock splits.

        Period handling
        ---------------
        yfinance natively supports up to "10y". Any "Ny" period where N > 10
        (e.g. "20y", "15y") is transparently handled by fetching "max" data
        and slicing to the last N years — so every method that calls _prices()
        automatically supports extended periods without any extra logic.
        """
        # translate "Ny" periods > 10y to "max" + date slice
        if period.endswith("y") and period[:-1].isdigit():
            n_years = int(period[:-1])
            fetch_period = "max" if n_years > 10 else period
        else:
            n_years = None
            fetch_period = period

        bars = self.client.get_history(symbol, period=fetch_period, interval=interval, adjusted=True)
        if not bars:
            raise ValueError(f"No history returned for '{symbol}' (period={period})")
        df = pd.DataFrame(bars)
        dates = pd.to_datetime(df["date"])
        # strip timezone so all downstream code (plots, groupby, joins) gets
        # clean tz-naive date-only timestamps — yfinance returns tz-aware index
        df["date"] = dates.dt.tz_convert(None) if dates.dt.tz is not None else dates
        df["date"] = df["date"].dt.normalize()   # drop time component (00:00:00)
        prices = df.set_index("date")["adj_close"].sort_index().astype(float)

        # slice to requested period if > 10y
        if n_years is not None:
            cutoff = prices.index[-1] - pd.DateOffset(years=n_years)
            prices = prices[prices.index >= cutoff]

        return prices

    def _prices_bulk(
        self, symbols: list[str], period: str, interval: str = "1d"
    ) -> pd.DataFrame:
        """Return a DataFrame of closing prices, one column per symbol."""
        series = {sym: self._prices(sym, period, interval) for sym in symbols}
        return pd.DataFrame(series).sort_index()

    # ------------------------------------------------------------------
    # ARCHITECTURE RULE — always use _prices() for historical data
    # ------------------------------------------------------------------
    #
    # _prices() is the ONLY correct way to fetch historical price data
    # inside QuantAnalytics. It handles:
    #
    #   1. Period translation  — any "Ny" period (e.g. "20y", "15y") is
    #                            transparently converted to "max" + date slice,
    #                            since yfinance only supports up to "10y" natively.
    #
    #   2. Timezone stripping  — yfinance returns tz-aware DatetimeIndex.
    #                            _prices() converts to tz-naive, date-only index
    #                            so downstream code (plots, groupby, joins) works
    #                            correctly across all pandas versions.
    #
    #   3. Single cache gate   — all fetches flow through StockClient.get_history()
    #                            which applies TTL caching and period-aware slicing.
    #
    # NEVER call self.client.get_history() directly from a public method.
    # NEVER call yfinance directly from inside QuantAnalytics.
    #
    # Every new method that needs price history should follow this pattern:
    #
    #     prices = self._prices(symbol, period)          # single symbol
    #     prices = self._prices_bulk(symbols, period)    # multiple symbols
    #
    # ------------------------------------------------------------------

    @staticmethod
    def _log_returns(prices: pd.Series) -> pd.Series:
        return np.log(prices / prices.shift(1)).dropna()

    @staticmethod
    def _simple_returns(prices: pd.Series) -> pd.Series:
        return prices.pct_change().dropna()

    # ------------------------------------------------------------------
    # Returns
    # ------------------------------------------------------------------

    def returns(
        self,
        symbol: str,
        period: str = "1y",
        method: str = "log",      # "log" | "simple"
        interval: str = "1d",
    ) -> pd.Series:
        """Daily log or simple returns for a single symbol."""
        prices = self._prices(symbol, period, interval)
        fn = self._log_returns if method == "log" else self._simple_returns
        series = fn(prices)
        series.name = symbol
        return series

    def returns_bulk(
        self,
        symbols: list[str],
        period: str = "1y",
        method: str = "log",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Daily returns for multiple symbols aligned on the same dates."""
        prices = self._prices_bulk(symbols, period, interval)
        fn = (lambda p: np.log(p / p.shift(1))) if method == "log" else (lambda p: p.pct_change())
        return fn(prices).dropna()

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------

    def historical_volatility(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        annualise: bool = True,
    ) -> float:
        """
        Annualised historical volatility (std of log returns).

        Returns a float, e.g. 0.28 means 28 % annualised vol.
        """
        rets = self._log_returns(self._prices(symbol, period, interval))
        vol = rets.std()
        return float(vol * np.sqrt(self.trading_days)) if annualise else float(vol)

    def rolling_volatility(
        self,
        symbol: str,
        period: str = "1y",
        window: int = 21,          # ~1 trading month
        interval: str = "1d",
        annualise: bool = True,
    ) -> pd.Series:
        """Rolling volatility over a *window*-day window."""
        rets = self._log_returns(self._prices(symbol, period, interval))
        rv = rets.rolling(window).std()
        if annualise:
            rv = rv * np.sqrt(self.trading_days)
        rv.name = f"{symbol}_vol_{window}d"
        return rv.dropna()

    def volatility_bulk(
        self,
        symbols: list[str],
        period: str = "1y",
        interval: str = "1d",
    ) -> dict[str, float]:
        """Annualised historical vol for every symbol in one call."""
        return {
            sym: self.historical_volatility(sym, period, interval)
            for sym in symbols
        }

    # ------------------------------------------------------------------
    # Correlation & covariance
    # ------------------------------------------------------------------

    def correlation_matrix(
        self,
        symbols: list[str],
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Pearson correlation matrix of daily log returns."""
        return self.returns_bulk(symbols, period, interval=interval).corr()

    def covariance_matrix(
        self,
        symbols: list[str],
        period: str = "1y",
        interval: str = "1d",
        annualise: bool = True,
    ) -> pd.DataFrame:
        """Annualised covariance matrix of daily log returns."""
        cov = self.returns_bulk(symbols, period, interval=interval).cov()
        return cov * (self.trading_days if annualise else 1)

    # ------------------------------------------------------------------
    # Beta
    # ------------------------------------------------------------------

    def beta(
        self,
        symbol: str,
        benchmark: str = "SPY",
        period: str = "1y",
        interval: str = "1d",
    ) -> float:
        """
        Beta of *symbol* relative to *benchmark* (OLS slope).

        A beta > 1 means the stock amplifies benchmark moves.
        """
        rets = self.returns_bulk([symbol, benchmark], period, interval=interval).dropna()
        slope, *_ = np.polyfit(rets[benchmark], rets[symbol], deg=1)
        return float(slope)

    def beta_bulk(
        self,
        symbols: list[str],
        benchmark: str = "SPY",
        period: str = "1y",
    ) -> dict[str, float]:
        """Beta for multiple symbols against the same benchmark."""
        all_syms = list(dict.fromkeys([benchmark] + symbols))  # benchmark first, deduped
        rets = self.returns_bulk(all_syms, period).dropna()
        bench_rets = rets[benchmark]
        results = {}
        for sym in symbols:
            slope, *_ = np.polyfit(bench_rets, rets[sym], deg=1)
            results[sym] = float(slope)
        return results

    # ------------------------------------------------------------------
    # Risk-adjusted returns
    # ------------------------------------------------------------------

    def sharpe_ratio(
        self,
        symbol: str,
        period: str = "1y",
        risk_free_rate: float = 0.05,   # annual, e.g. 0.05 = 5 %
        interval: str = "1d",
    ) -> float:
        """
        Annualised Sharpe ratio.

        sharpe = (mean_return - rf) / std  ×  √trading_days
        """
        rets = self._simple_returns(self._prices(symbol, period, interval))
        daily_rf = risk_free_rate / self.trading_days
        excess = rets - daily_rf
        return float(excess.mean() / excess.std() * np.sqrt(self.trading_days))

    def sortino_ratio(
        self,
        symbol: str,
        period: str = "1y",
        risk_free_rate: float = 0.05,
        interval: str = "1d",
    ) -> float:
        """
        Sortino ratio — like Sharpe but penalises only downside deviation.
        """
        rets = self._simple_returns(self._prices(symbol, period, interval))
        daily_rf = risk_free_rate / self.trading_days
        excess = rets - daily_rf
        downside = excess[excess < 0].std()
        return float(excess.mean() / downside * np.sqrt(self.trading_days))

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------

    def drawdown_series(self, symbol: str, period: str = "1y") -> pd.Series:
        """Return the full drawdown time-series (0 to -1 scale)."""
        prices = self._prices(symbol, period)
        roll_max = prices.cummax()
        dd = (prices - roll_max) / roll_max
        dd.name = f"{symbol}_drawdown"
        return dd

    def max_drawdown(self, symbol: str, period: str = "1y") -> float:
        """Maximum drawdown over the period (negative float, e.g. -0.35 = -35 %)."""
        return float(self.drawdown_series(symbol, period).min())

    def calmar_ratio(
        self,
        symbol: str,
        period: str = "3y",
    ) -> float:
        """Calmar ratio = annualised return / |max drawdown|."""
        prices = self._prices(symbol, period)
        n_years = len(prices) / self.trading_days
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        annual_return = (1 + total_return) ** (1 / n_years) - 1
        mdd = abs(self.max_drawdown(symbol, period))
        return float(annual_return / mdd) if mdd else float("inf")

    # ------------------------------------------------------------------
    # Value at Risk
    # ------------------------------------------------------------------

    def var(
        self,
        symbol: str,
        period: str = "1y",
        confidence: float = 0.95,
        method: str = "historical",   # "historical" | "parametric"
        horizon: int = 1,             # days
    ) -> float:
        """
        Value at Risk (VaR) at the given confidence level.

        Returns a positive float representing the potential loss,
        e.g. 0.025 means 2.5 % of portfolio value at risk.

        method="historical"  : empirical quantile of past returns
        method="parametric"  : assumes normally distributed returns
        """
        rets = self._simple_returns(self._prices(symbol, period))
        if method == "historical":
            var_1d = float(-np.percentile(rets, (1 - confidence) * 100))
        else:  # parametric
            mu, sigma = rets.mean(), rets.std()
            var_1d = float(-(mu + stats.norm.ppf(1 - confidence) * sigma))
        return var_1d * np.sqrt(horizon)

    def cvar(
        self,
        symbol: str,
        period: str = "1y",
        confidence: float = 0.95,
    ) -> float:
        """
        Conditional VaR (Expected Shortfall) — average loss beyond VaR.
        """
        rets = self._simple_returns(self._prices(symbol, period))
        threshold = np.percentile(rets, (1 - confidence) * 100)
        tail = rets[rets <= threshold]
        return float(-tail.mean())

    # ------------------------------------------------------------------
    # Portfolio-level
    # ------------------------------------------------------------------

    def portfolio_return(
        self,
        symbols: list[str],
        weights: list[float],
        period: str = "1y",
    ) -> float:
        """
        Expected annualised portfolio return given weights.

        weights must sum to 1.0.
        """
        weights = np.array(weights)
        assert abs(weights.sum() - 1.0) < 1e-6, "Weights must sum to 1"
        rets = self.returns_bulk(symbols, period).mean()
        return float((rets @ weights) * self.trading_days)

    def portfolio_volatility(
        self,
        symbols: list[str],
        weights: list[float],
        period: str = "1y",
    ) -> float:
        """Annualised portfolio volatility given weights."""
        weights = np.array(weights)
        cov = self.covariance_matrix(symbols, period)
        variance = weights @ cov.values @ weights
        return float(np.sqrt(variance))

    def portfolio_sharpe(
        self,
        symbols: list[str],
        weights: list[float],
        period: str = "1y",
        risk_free_rate: float = 0.05,
    ) -> float:
        """Sharpe ratio for the weighted portfolio."""
        ret = self.portfolio_return(symbols, weights, period)
        vol = self.portfolio_volatility(symbols, weights, period)
        return float((ret - risk_free_rate) / vol)

    def portfolio_summary(
        self,
        symbols: list[str],
        weights: list[float],
        period: str = "1y",
        risk_free_rate: float = 0.05,
    ) -> dict:
        """One-call summary of all portfolio-level metrics."""
        return {
            "symbols": symbols,
            "weights": weights,
            "period": period,
            "annualised_return": self.portfolio_return(symbols, weights, period),
            "annualised_volatility": self.portfolio_volatility(symbols, weights, period),
            "sharpe_ratio": self.portfolio_sharpe(symbols, weights, period, risk_free_rate),
            "correlation_matrix": self.correlation_matrix(symbols, period).to_dict(),
        }

    # ------------------------------------------------------------------
    # DataFrame outputs  (dates as rows, symbols as columns)
    # ------------------------------------------------------------------

    def prices_df(
        self,
        symbols: list[str],
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Closing prices aligned on the same date index.

              date       AAPL    MSFT    GOOGL
        2024-01-02  185.20  374.02  140.93
        ...
        """
        return self._prices_bulk(symbols, period, interval)

    def returns_df(
        self,
        symbols: list[str],
        period: str = "1y",
        method: str = "log",        # "log" | "simple"
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Daily returns, one column per symbol.

              date       AAPL    MSFT    GOOGL
        2024-01-03   0.0023  -0.0011   0.0041
        ...
        """
        return self.returns_bulk(symbols, period, method, interval)

    def cumulative_returns_df(
        self,
        symbols: list[str],
        period: str = "1y",
        interval: str = "1d",
        base: float = 100.0,        # rebase starting value (100 = percentage chart)
    ) -> pd.DataFrame:
        """
        Cumulative simple returns rebased to *base* (default 100).

        Useful for comparing growth of $100 invested across stocks.

              date       AAPL    MSFT    GOOGL
        2024-01-02  100.00  100.00  100.00
        2024-01-03  102.30   99.89  100.41
        ...
        """
        prices = self._prices_bulk(symbols, period, interval)
        return base * prices / prices.iloc[0]

    def rolling_volatility_df(
        self,
        symbols: list[str],
        period: str = "1y",
        window: int = 21,
        interval: str = "1d",
        annualise: bool = True,
    ) -> pd.DataFrame:
        """
        Rolling annualised volatility for each symbol.

              date       AAPL    MSFT    GOOGL
        2024-02-01   0.2341  0.1987  0.2103
        ...
        """
        series = {}
        for sym in symbols:
            rets = self._log_returns(self._prices(sym, period, interval))
            rv = rets.rolling(window).std()
            if annualise:
                rv = rv * np.sqrt(self.trading_days)
            series[sym] = rv
        return pd.DataFrame(series).dropna()

    def drawdown_df(
        self,
        symbols: list[str],
        period: str = "1y",
    ) -> pd.DataFrame:
        """
        Drawdown series for each symbol (0 to -1 scale).

              date       AAPL    MSFT    GOOGL
        2024-01-02   0.0000   0.0000   0.0000
        2024-03-15  -0.1230  -0.0891  -0.1502
        ...
        """
        series = {}
        for sym in symbols:
            prices = self._prices(sym, period)
            roll_max = prices.cummax()
            series[sym] = (prices - roll_max) / roll_max
        return pd.DataFrame(series)

    def rolling_sharpe_df(
        self,
        symbols: list[str],
        period: str = "1y",
        window: int = 63,           # ~1 trading quarter
        risk_free_rate: float = 0.05,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Rolling Sharpe ratio over a *window*-day window, one column per symbol.
        """
        daily_rf = risk_free_rate / self.trading_days
        series = {}
        for sym in symbols:
            rets = self._simple_returns(self._prices(sym, period, interval))
            excess = rets - daily_rf
            rolling_sharpe = (
                excess.rolling(window).mean()
                / excess.rolling(window).std()
                * np.sqrt(self.trading_days)
            )
            series[sym] = rolling_sharpe
        return pd.DataFrame(series).dropna()

    def rolling_beta_df(
        self,
        symbols: list[str],
        benchmark: str = "SPY",
        period: str = "1y",
        window: int = 63,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Rolling beta versus *benchmark*, one column per symbol.
        """
        all_syms = list(dict.fromkeys([benchmark] + symbols))
        rets = self.returns_bulk(all_syms, period, interval=interval).dropna()
        bench = rets[benchmark]
        series = {}
        for sym in symbols:
            stock = rets[sym]
            cov = stock.rolling(window).cov(bench)
            var = bench.rolling(window).var()
            series[sym] = cov / var
        return pd.DataFrame(series).dropna()

    def metrics_df(
        self,
        symbols: list[str],
        benchmark: str = "SPY",
        period: str = "1y",
        risk_free_rate: float = 0.05,
    ) -> pd.DataFrame:
        """
        Cross-sectional summary table — one row per metric, one column per symbol.

                                AAPL    MSFT    GOOGL
        annualised_volatility  0.281   0.199   0.231
        max_drawdown          -0.143  -0.098  -0.172
        sharpe_ratio           1.320   1.540   1.102
        sortino_ratio          1.891   2.210   1.543
        beta                   1.142   0.921   1.034
        var_95_1d              0.022   0.017   0.021
        cvar_95_1d             0.032   0.025   0.030
        """
        records = {}
        for sym in symbols:
            records[sym] = {
                "annualised_volatility": self.historical_volatility(sym, period),
                "max_drawdown":          self.max_drawdown(sym, period),
                "sharpe_ratio":          self.sharpe_ratio(sym, period, risk_free_rate),
                "sortino_ratio":         self.sortino_ratio(sym, period, risk_free_rate),
                "beta":                  self.beta(sym, benchmark, period),
                "calmar_ratio":          self.calmar_ratio(sym, period),
                "var_95_1d":             self.var(sym, period, confidence=0.95),
                "cvar_95_1d":            self.cvar(sym, period, confidence=0.95),
            }
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Convenience: full single-stock report
    # ------------------------------------------------------------------

    def stock_report(
        self,
        symbol: str,
        benchmark: str = "SPY",
        period: str = "1y",
        risk_free_rate: float = 0.05,
    ) -> dict:
        """All key metrics for a single stock in one dict."""
        return {
            "symbol": symbol,
            "period": period,
            "annualised_volatility": self.historical_volatility(symbol, period),
            "max_drawdown": self.max_drawdown(symbol, period),
            "sharpe_ratio": self.sharpe_ratio(symbol, period, risk_free_rate),
            "sortino_ratio": self.sortino_ratio(symbol, period, risk_free_rate),
            "beta": self.beta(symbol, benchmark, period),
            "calmar_ratio": self.calmar_ratio(symbol, period),
            "var_95_1d": self.var(symbol, period, confidence=0.95),
            "cvar_95_1d": self.cvar(symbol, period, confidence=0.95),
        }

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------

    def monte_carlo(
        self,
        symbol: str,
        horizon_years: float = 2.0,
        n_sims: int = 10_000,
        target_gain: float = 0.30,
        max_drawdown_limit: float = 0.25,
        history_period: str = "2y",
        drift_override: float | None = None,
        volatility_override: float | None = None,
        seed: int | None = None,
        return_paths: bool = False,
        n_sample_paths: int = 200,
    ) -> dict:
        """
        Geometric Brownian Motion Monte Carlo for a single symbol.

        Drift (μ) and volatility (σ) are estimated from *history_period*
        of real price data fetched through the StockClient cache.  Both
        can be overridden for scenario analysis.

        Parameters
        ----------
        symbol               : ticker to simulate
        horizon_years        : simulation horizon in years (default 2)
        n_sims               : number of Monte Carlo paths (default 10 000)
        target_gain          : minimum total return to count as a win,
                               e.g. 0.30 = +30 % (default 0.30)
        max_drawdown_limit   : maximum tolerated peak-to-trough drawdown,
                               e.g. 0.25 = 25 % (default 0.25)
        history_period       : yfinance period used to calibrate μ and σ
                               (default "2y")
        drift_override       : annualised drift to use instead of historical
                               (e.g. 0.15 for +15 %)
        volatility_override  : annualised vol to use instead of historical
                               (e.g. 0.35 for 35 %)
        seed                 : random seed for reproducibility (default None)
        return_paths         : if True, include *n_sample_paths* simulated
                               price paths in the result (memory-intensive
                               for large n_sims — keep n_sample_paths small)
        n_sample_paths       : number of paths to store when return_paths=True

        Returns
        -------
        dict with keys:
          symbol                : ticker
          entry_price           : last closing price used as starting point
          horizon_years         : as supplied
          n_sims                : as supplied
          mu_annual             : annualised drift used in simulation
          sigma_annual          : annualised volatility used in simulation
          target_gain           : as supplied (fraction)
          max_drawdown_limit    : as supplied (fraction)
          prob_gain             : P(final return >= target_gain)
          prob_drawdown_ok      : P(max drawdown <= max_drawdown_limit)
          prob_both             : P(gain AND drawdown constraint both met) ← key metric
          prob_loss             : P(final return < 0)
          median_return         : 50th-percentile final total return
          percentile_10         : 10th-percentile final total return
          percentile_90         : 90th-percentile final total return
          expected_return       : mean final total return across all paths
          paths                 : list of lists (only present if return_paths=True)
                                  each sub-list is a normalised price path (1.0 = entry)

        Examples
        --------
        >>> qa = QuantAnalytics(client)
        >>> result = qa.monte_carlo("NFLX")
        >>> print(f"Odds of both targets: {result['prob_both']:.1%}")
        Odds of both targets: 23.4%

        >>> # Scenario: lower vol assumption
        >>> result = qa.monte_carlo("NFLX", volatility_override=0.25)

        >>> # Get paths for plotting
        >>> result = qa.monte_carlo("NFLX", return_paths=True, n_sample_paths=500)
        >>> paths_df = pd.DataFrame(result["paths"]).T   # shape: steps × n_sample_paths
        """
        rng = np.random.default_rng(seed)

        # ── calibrate μ and σ from real data ──────────────────────────────
        prices = self._prices(symbol, period=history_period)
        entry_price = float(prices.iloc[-1])

        log_rets = self._log_returns(prices)

        if volatility_override is not None:
            sigma = float(volatility_override)
        else:
            sigma = float(log_rets.std() * np.sqrt(self.trading_days))

        if drift_override is not None:
            mu = float(drift_override)
        else:
            # annualise the mean log return; note: E[log r] ≠ E[r], so this is
            # the log-space drift — appropriate for GBM
            mu = float(log_rets.mean() * self.trading_days)

        # ── GBM parameters ────────────────────────────────────────────────
        n_steps = int(round(horizon_years * self.trading_days))
        dt = 1.0 / self.trading_days
        drift = (mu - 0.5 * sigma ** 2) * dt        # per-step drift
        vol   = sigma * np.sqrt(dt)                  # per-step vol

        # ── simulate all paths at once (vectorised) ───────────────────────
        # shape: (n_steps, n_sims)
        z = rng.standard_normal((n_steps, n_sims))
        log_returns_matrix = drift + vol * z         # daily log increments
        # cumulative log return up to each step → shape (n_steps, n_sims)
        cum_log = np.cumsum(log_returns_matrix, axis=0)
        # prepend row of zeros (t=0, price = entry)
        cum_log = np.vstack([np.zeros((1, n_sims)), cum_log])
        price_paths = entry_price * np.exp(cum_log)  # shape: (n_steps+1, n_sims)

        # ── compute final returns ─────────────────────────────────────────
        final_prices  = price_paths[-1, :]
        final_returns = (final_prices - entry_price) / entry_price

        # ── compute max drawdown per path ─────────────────────────────────
        # running peak for each path across time steps
        running_peak = np.maximum.accumulate(price_paths, axis=0)
        drawdowns    = (price_paths - running_peak) / running_peak  # ≤ 0
        max_drawdown_per_path = np.abs(drawdowns.min(axis=0))       # positive

        # ── probability calculations ──────────────────────────────────────
        gain_ok = final_returns >= target_gain
        dd_ok   = max_drawdown_per_path <= max_drawdown_limit
        both_ok = gain_ok & dd_ok

        prob_gain        = float(gain_ok.mean())
        prob_drawdown_ok = float(dd_ok.mean())
        prob_both        = float(both_ok.mean())
        prob_loss        = float((final_returns < 0).mean())

        # ── distribution stats ────────────────────────────────────────────
        median_return   = float(np.median(final_returns))
        pct_10          = float(np.percentile(final_returns, 10))
        pct_90          = float(np.percentile(final_returns, 90))
        expected_return = float(final_returns.mean())

        result = {
            "symbol":             symbol,
            "entry_price":        entry_price,
            "horizon_years":      horizon_years,
            "n_sims":             n_sims,
            "mu_annual":          mu,
            "sigma_annual":       sigma,
            "target_gain":        target_gain,
            "max_drawdown_limit": max_drawdown_limit,
            "prob_gain":          prob_gain,
            "prob_drawdown_ok":   prob_drawdown_ok,
            "prob_both":          prob_both,
            "prob_loss":          prob_loss,
            "median_return":      median_return,
            "percentile_10":      pct_10,
            "percentile_90":      pct_90,
            "expected_return":    expected_return,
        }

        if return_paths:
            # normalise paths to 1.0 = entry price; store as list of lists
            # each element is one path (length = n_steps + 1)
            idx = rng.choice(n_sims, size=min(n_sample_paths, n_sims), replace=False)
            sample = price_paths[:, idx] / entry_price   # shape: (steps+1, n_sample_paths)
            result["paths"] = sample.T.tolist()           # list of n_sample_paths lists

        return result

    def monte_carlo_bulk(
        self,
        symbols: list[str],
        horizon_years: float = 2.0,
        n_sims: int = 10_000,
        target_gain: float = 0.30,
        max_drawdown_limit: float = 0.25,
        history_period: str = "2y",
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Run monte_carlo() for multiple symbols and return a comparison DataFrame.

        Returns a DataFrame with one column per symbol and these index rows:
          entry_price, mu_annual, sigma_annual, prob_gain,
          prob_drawdown_ok, prob_both, prob_loss,
          median_return, percentile_10, percentile_90, expected_return

        Example
        -------
        >>> df = qa.monte_carlo_bulk(["NFLX", "AAPL", "MSFT"])
        >>> print(df.loc["prob_both"])
        NFLX    0.234
        AAPL    0.318
        MSFT    0.291
        """
        records = {}
        for i, sym in enumerate(symbols):
            # increment seed per symbol so runs are independent but reproducible
            sym_seed = None if seed is None else seed + i
            r = self.monte_carlo(
                sym,
                horizon_years=horizon_years,
                n_sims=n_sims,
                target_gain=target_gain,
                max_drawdown_limit=max_drawdown_limit,
                history_period=history_period,
                seed=sym_seed,
                return_paths=False,
            )
            records[sym] = {k: v for k, v in r.items() if k not in ("symbol", "paths")}
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Seasonality
    # ------------------------------------------------------------------

    def seasonality_holding_sharpe(
        self,
        symbol:         str,
        period:         str       = "15y",
        holding_years:  list[int] = None,
        risk_free_rate: float     = 0.0,
    ) -> pd.DataFrame:
        """
        Buy-and-hold Sharpe for each calendar month across multiple holding periods.

        For each month M and holding period H:
          1. Find every historical entry point in month M
          2. Compute total return H years later
          3. Sharpe = mean(returns) / std(returns) — annualised

        This answers: "If I buy every January and hold 2 years,
        what is my expected risk-adjusted return?"

        Comparing 2y vs 5y holding periods reveals:
          2y Sharpe > 5y : better short-term seasonal momentum
          5y Sharpe > 2y : patience rewarded, noise washes out
          Both high       : strong entry point regardless of horizon
          Both low        : no persistent edge from that month

        Parameters
        ----------
        symbol        : ticker
        period        : history window to fetch (default "15y")
        holding_years : list of holding periods in years (default [2, 5])
        risk_free_rate: annual risk-free rate (default 0.0)

        Returns
        -------
        DataFrame — one row per month (Jan-Dec), columns:
          sharpe_{H}y   : annualised Sharpe of H-year holds entered this month
          mean_{H}y     : mean total return over H years
          std_{H}y      : std of total returns
          win_rate_{H}y : % of entries profitable after H years
          n_{H}y        : number of entry points with full H-year history
          trend         : "patient" | "momentum" | "consistent" | "weak" | "n/a"
                          compares short vs long holding Sharpe

        Example
        -------
        df = qa.seasonality_holding_sharpe("SPY", period="15y", holding_years=[2, 5])
        print(df.sort_values("sharpe_2y", ascending=False))
        # → best month to buy for a 2-year hold
        """
        import numpy as np

        if holding_years is None:
            holding_years = [2, 5]

        prices = self._prices(symbol, period)
        # use month-end prices
        monthly = prices.resample("ME").last().dropna()

        month_names = {
            1:"Jan", 2:"Feb", 3:"Mar",  4:"Apr",  5:"May",  6:"Jun",
            7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"
        }

        rows = {}
        for month_num in range(1, 13):
            month_label = month_names[month_num]
            # all month-end dates in this calendar month
            entry_dates = monthly[monthly.index.month == month_num].index
            row = {}
            sharpes = []

            for H in holding_years:
                H_months = H * 12
                returns  = []

                for entry in entry_dates:
                    # exit date: H years later (same month)
                    exit_idx = monthly.index.get_loc(entry) + H_months
                    if exit_idx >= len(monthly):
                        continue   # no full history yet
                    p_entry = float(monthly.iloc[monthly.index.get_loc(entry)])
                    p_exit  = float(monthly.iloc[exit_idx])
                    total_return = (p_exit / p_entry) - 1.0
                    # annualise
                    ann_return = (1 + total_return) ** (1/H) - 1
                    returns.append(ann_return)

                n = len(returns)
                if n < 2:
                    row[f"sharpe_{H}y"]   = float("nan")
                    row[f"mean_{H}y"]     = float("nan")
                    row[f"std_{H}y"]      = float("nan")
                    row[f"win_{H}y"]      = float("nan")
                    row[f"n_{H}y"]        = n
                    sharpes.append(float("nan"))
                    continue

                arr    = np.array(returns)
                rf_ann = risk_free_rate
                excess = arr - rf_ann
                mean_e = float(excess.mean())
                std_e  = float(arr.std(ddof=1))
                sharpe = round(mean_e / std_e, 3) if std_e > 0 else 0.0

                row[f"sharpe_{H}y"] = sharpe
                row[f"mean_{H}y"]   = round(float(arr.mean()), 4)
                row[f"std_{H}y"]    = round(float(arr.std(ddof=1)), 4)
                row[f"win_{H}y"]    = round(float((arr > 0).mean()), 3)
                row[f"n_{H}y"]      = n
                sharpes.append(sharpe)

            # trend: short hold vs long hold
            valid = [s for s in sharpes if s == s]  # remove nan
            if len(valid) >= 2:
                short_sharpe = sharpes[0]
                long_sharpe  = sharpes[-1]
                if short_sharpe > long_sharpe + 0.3:
                    row["edge"] = "momentum"      # better short-term
                elif long_sharpe > short_sharpe + 0.3:
                    row["edge"] = "patient"       # better long-term
                elif valid[0] > 0.5 and valid[-1] > 0.5:
                    row["edge"] = "consistent"    # both strong
                else:
                    row["edge"] = "weak"
            else:
                row["edge"] = "n/a"

            rows[month_label] = row

        df = pd.DataFrame(rows).T
        df.index.name = "entry_month"

        # order columns
        cols = []
        for H in holding_years:
            cols += [f"sharpe_{H}y", f"mean_{H}y", f"std_{H}y",
                     f"win_{H}y", f"n_{H}y"]
        cols.append("edge")
        df = df[[c for c in cols if c in df.columns]]

        return df


    def seasonality_holding_drawdown(
        self,
        symbol:        str,
        period:        str       = "15y",
        holding_years: list[int] = None,
    ) -> pd.DataFrame:
        """
        Maximum drawdown during hold period for each entry month.

        For each calendar month and holding period, measures the worst
        peak-to-trough decline experienced DURING the hold — not just
        the final return. This is the "stomach test": can you hold
        through the worst drawdown to reach the final return?

        Returns
        -------
        DataFrame — one row per month, columns:
          max_dd_{H}y   : average max drawdown during H-year hold
          worst_dd_{H}y : worst single max drawdown experienced
          recovery_{H}y : % of entries that recovered to positive by exit
          pain_ratio_{H}y: mean_return / abs(mean_max_dd) — reward/pain
        """
        import numpy as np

        if holding_years is None:
            holding_years = [2, 5]

        prices  = self._prices(symbol, period)
        monthly = prices.resample("ME").last().dropna()

        month_names = {
            1:"Jan", 2:"Feb", 3:"Mar",  4:"Apr",  5:"May",  6:"Jun",
            7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"
        }

        rows = {}
        for month_num in range(1, 13):
            month_label  = month_names[month_num]
            entry_dates  = monthly[monthly.index.month == month_num].index
            row = {}

            for H in holding_years:
                H_months    = H * 12
                max_dds     = []
                final_rets  = []

                for entry in entry_dates:
                    i_entry = monthly.index.get_loc(entry)
                    i_exit  = i_entry + H_months
                    if i_exit >= len(monthly):
                        continue

                    hold_prices = monthly.iloc[i_entry:i_exit + 1].values
                    p0          = hold_prices[0]
                    # running max and drawdown
                    running_max = np.maximum.accumulate(hold_prices)
                    dd          = (hold_prices - running_max) / running_max
                    max_dd      = float(dd.min())
                    final_ret   = (hold_prices[-1] / p0) - 1.0

                    max_dds.append(max_dd)
                    final_rets.append(final_ret)

                n = len(max_dds)
                if n == 0:
                    for k in [f"max_dd_{H}y", f"worst_dd_{H}y",
                               f"recovery_{H}y", f"pain_ratio_{H}y"]:
                        row[k] = float("nan")
                    continue

                arr_dd   = np.array(max_dds)
                arr_ret  = np.array(final_rets)
                mean_dd  = float(arr_dd.mean())
                pain     = round(float(arr_ret.mean()) / abs(mean_dd), 3)                            if mean_dd != 0 else float("nan")

                row[f"max_dd_{H}y"]     = round(mean_dd, 4)
                row[f"worst_dd_{H}y"]   = round(float(arr_dd.min()), 4)
                row[f"recovery_{H}y"]   = round(float((arr_ret > 0).mean()), 3)
                row[f"pain_ratio_{H}y"] = pain

            rows[month_label] = row

        df = pd.DataFrame(rows).T
        df.index.name = "entry_month"
        cols = []
        for H in holding_years:
            cols += [f"max_dd_{H}y", f"worst_dd_{H}y",
                     f"recovery_{H}y", f"pain_ratio_{H}y"]
        return df[[c for c in cols if c in df.columns]]

    def seasonality_decade_analysis(
        self,
        symbol:        str,
        period:        str = "30y",
        holding_years: int = 1,
    ) -> pd.DataFrame:
        """
        Hit rate and mean return per entry month broken down by decade.

        Answers: "Was January strong in 2000-2010? What about 2010-2020?
        Is the pattern consistent across decades or regime-dependent?"

        Returns
        -------
        DataFrame — MultiIndex (decade, month), columns:
          mean_return, win_rate, n_obs, sharpe
        """
        import numpy as np

        prices  = self._prices(symbol, period)
        monthly = prices.resample("ME").last().dropna()

        month_names = {
            1:"Jan", 2:"Feb", 3:"Mar",  4:"Apr",  5:"May",  6:"Jun",
            7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"
        }

        H_months = holding_years * 12
        rows = []

        for month_num in range(1, 13):
            month_label = month_names[month_num]
            entry_dates = monthly[monthly.index.month == month_num].index

            for entry in entry_dates:
                i_entry = monthly.index.get_loc(entry)
                i_exit  = i_entry + H_months
                if i_exit >= len(monthly):
                    continue
                p_entry = float(monthly.iloc[i_entry])
                p_exit  = float(monthly.iloc[i_exit])
                ret     = (p_exit / p_entry) ** (1/holding_years) - 1
                year    = entry.year
                decade  = f"{(year // 10) * 10}s"
                rows.append({"decade": decade, "month": month_label,
                             "return": ret, "year": year})

        df_raw = pd.DataFrame(rows)
        if df_raw.empty:
            return pd.DataFrame()

        result = (df_raw.groupby(["decade","month"])["return"]
                  .agg(
                      mean_return = "mean",
                      std_return  = "std",
                      win_rate    = lambda x: (x > 0).mean(),
                      n_obs       = "count",
                  )
                  .round(4))

        result["sharpe"] = (result["mean_return"] /
                            result["std_return"].replace(0, float("nan"))).round(3)

        # reindex to natural month order
        month_order  = list(month_names.values())
        decade_order = sorted(df_raw["decade"].unique())
        result = result.reindex(
            pd.MultiIndex.from_product([decade_order, month_order],
                                       names=["decade","month"])
        ).dropna(how="all")

        return result

    def seasonality_cross_asset(
        self,
        symbols:       list[str],
        period:        str       = "10y",
        holding_years: list[int] = None,
    ) -> pd.DataFrame:
        """
        Compare holding-period Sharpe across multiple assets for each month.

        Ranks assets by their seasonal entry-point quality, making it easy
        to identify which asset has the strongest seasonal signal for each
        calendar month.

        Returns
        -------
        DataFrame — one row per month, one column group per asset:
          {symbol}_sharpe_{H}y, {symbol}_win_{H}y
        Plus a "best_asset_{H}y" column showing the top-ranked asset.
        """
        if holding_years is None:
            holding_years = [2, 5]

        results = {}
        for sym in symbols:
            try:
                df = self.seasonality_holding_sharpe(
                    sym, period=period, holding_years=holding_years
                )
                results[sym] = df
            except Exception as e:
                print(f"Skipping {sym}: {e}")

        if not results:
            return pd.DataFrame()

        # combine into one wide DataFrame
        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        combined = pd.DataFrame(index=month_names)
        combined.index.name = "entry_month"

        for sym, df in results.items():
            for H in holding_years:
                col_s = f"sharpe_{H}y"
                col_w = f"win_{H}y"
                if col_s in df.columns:
                    combined[f"{sym}_{col_s}"] = df.reindex(month_names)[col_s]
                if col_w in df.columns:
                    combined[f"{sym}_{col_w}"] = df.reindex(month_names)[col_w]

        # best asset per month per holding period
        for H in holding_years:
            sharpe_cols = [f"{sym}_sharpe_{H}y" for sym in results]
            avail       = [c for c in sharpe_cols if c in combined.columns]
            if avail:
                combined[f"best_{H}y"] = (
                    combined[avail]
                    .idxmax(axis=1)
                    .str.replace(f"_sharpe_{H}y", "", regex=False)
                )

        return combined.round(4)

    def seasonality_combined_score(
        self,
        symbol:        str,
        period:        str       = "15y",
        holding_years: list[int] = None,
    ) -> pd.DataFrame:
        """
        Combined actionable score per entry month.

        Aggregates all seasonal signals into one ranked score:
          score = (sharpe_2y × 0.3 + sharpe_5y × 0.3 +
                   win_rate_2y × 0.2 + win_rate_5y × 0.2) ×
                  reliability_score

        Multiplying by reliability_score penalises months with
        few observations — a great Sharpe on 3 data points is
        less trustworthy than a moderate Sharpe on 15 points.

        Returns
        -------
        DataFrame sorted by combined_score descending, columns:
          sharpe_2y, sharpe_5y, win_2y, win_5y,
          reliability_score, combined_score, rank, signal
        """
        if holding_years is None:
            holding_years = [2, 5]

        # holding Sharpe
        hs = self.seasonality_holding_sharpe(
            symbol, period=period, holding_years=holding_years
        )
        # reliability from monthly seasonality
        try:
            pivot = self.monthly_seasonality(symbol, period=period)
            stats = self.seasonality_stats(pivot)
            hs["reliability_score"] = stats.reindex(hs.index)["reliability_score"]
        except Exception:
            hs["reliability_score"] = 0.5

        import numpy as np

        H0, H1 = holding_years[0], holding_years[-1]
        s0 = hs.get(f"sharpe_{H0}y", 0).fillna(0)
        s1 = hs.get(f"sharpe_{H1}y", 0).fillna(0)
        w0 = hs.get(f"win_{H0}y",    0).fillna(0)
        w1 = hs.get(f"win_{H1}y",    0).fillna(0)
        rs = hs.get("reliability_score", 0.5).fillna(0.5)

        raw_score = (s0 * 0.30 + s1 * 0.30 +
                     w0 * 0.20 + w1 * 0.20)
        combined  = (raw_score * rs).round(4)

        out = pd.DataFrame({
            f"sharpe_{H0}y": s0,
            f"sharpe_{H1}y": s1,
            f"win_{H0}y":    w0,
            f"win_{H1}y":    w1,
            "reliability":   rs,
            "combined_score": combined,
        })
        out["rank"]   = out["combined_score"].rank(ascending=False).astype(int)
        out["signal"] = out["combined_score"].apply(
            lambda x: "strong buy" if x > 0.6  else
                      "buy"        if x > 0.3  else
                      "neutral"    if x > -0.1 else
                      "avoid"
        )
        out.index.name = "entry_month"
        return out.sort_values("combined_score", ascending=False)


    def weekly_seasonality(
        self,
        symbol: str,
        period: str = "10y",
    ) -> pd.DataFrame:
        """
        Weekly seasonality study.

        Returns a pivot table:
          rows    : ISO week number (1–52/53)
          columns : calendar year
          values  : mean weekly return for that year/week (NaN if no data)

        Summing or averaging across columns gives the multi-year seasonal pattern.
        Each column is one year so you can also compare individual years directly.

        Example
        -------
        df = qa.weekly_seasonality("AAPL", period="10y")
        df["mean"] = df.mean(axis=1)   # average across all years
        df["win_rate"] = (df > 0).mean(axis=1)
        """
        prices = self._prices(symbol, period=period, interval="1d")
        weekly = prices.resample("W-FRI").last().dropna()
        rets   = weekly.pct_change().dropna()

        # extract values as plain numpy — avoids ALL tz-aware index issues
        df = pd.DataFrame({
            "ret":  rets.values,
            "week": rets.index.strftime("%V").astype(int).values,
            "year": rets.index.year.values,
        })

        pivot = df.pivot_table(
            index="week", columns="year", values="ret", aggfunc="mean"
        )
        pivot.index.name   = "week"
        pivot.columns.name = "year"
        return pivot

    def monthly_seasonality(
        self,
        symbol: str,
        period: str = "10y",
    ) -> pd.DataFrame:
        """
        Monthly seasonality study.

        Returns a pivot table:
          rows    : month name (Jan–Dec)
          columns : calendar year
          values  : monthly return for that year/month (NaN if no data)

        Same design as weekly_seasonality() — derive stats across columns:
          pivot.mean(axis=1)          # average return per month
          (pivot > 0).mean(axis=1)    # win rate per month
          pivot.std(axis=1)           # volatility per month

        Supports any "Ny" period string (e.g. "20y", "15y") — fetches
        "max" data from yfinance and filters to the last N calendar years,
        since yfinance only supports up to "10y" as a native period.
        """
        # parse period → fetch "max" for anything beyond 10y
        if period.endswith("y") and period[:-1].isdigit():
            n_years      = int(period[:-1])
            fetch_period = "max" if n_years > 10 else period
        else:
            n_years      = None
            fetch_period = period

        prices  = self._prices(symbol, period=fetch_period, interval="1d")
        monthly = prices.resample("ME").last().dropna()
        rets    = monthly.pct_change().dropna()

        _month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]

        df = pd.DataFrame({
            "ret":   rets.values,
            "month": rets.index.month.values,
            "year":  rets.index.year.values,
        })

        pivot = df.pivot_table(
            index="month", columns="year", values="ret", aggfunc="mean"
        )
        pivot.index        = [_month_names[m - 1] for m in pivot.index]
        pivot.index.name   = "month"
        pivot.columns.name = "year"

        # filter to last n_years if an explicit year count was requested
        if n_years is not None:
            all_years = sorted(pivot.columns.tolist())
            pivot     = pivot[all_years[-n_years:]]

        return pivot

    def seasonality_stats(
        self,
        pivot: "pd.DataFrame",
    ) -> pd.DataFrame:
        """
        Compute summary statistics from a seasonality pivot table.

        Works with both weekly_seasonality() and monthly_seasonality() output.

        Parameters
        ----------
        pivot : DataFrame returned by weekly_seasonality() or monthly_seasonality()
                rows = period (week/month), columns = year

        Returns
        -------
        DataFrame with one row per period and columns:
          mean        : average return across years
          median      : median return
          std         : standard deviation (dispersion of outcomes)
          min         : worst year
          max         : best year
          win_rate    : fraction of years with positive return
          n_obs       : number of years with data
          skew        : return distribution skew (+ = more upside outliers)
          best_year   : year of the best return
          worst_year  : year of the worst return
          sharpe      : mean / std (risk-adjusted seasonal edge)
          reliability_score : composite score 0-1 (sortable)
          reliability       : "high" (>=0.65), "medium" (>=0.35), "low" (<0.35)
                              Combines: sample size (40%) + directional
                              consistency (40%) + Sharpe (20%) - skew penalty

        Example
        -------
        pivot = qa.monthly_seasonality("SPY", period="10y")
        stats = qa.seasonality_stats(pivot)
        print(stats.sort_values("mean", ascending=False))
        """
        rows = {}
        for period_label in pivot.index:
            row_data = pivot.loc[period_label].dropna()
            if row_data.empty:
                continue
            n = len(row_data)
            mean   = float(row_data.mean())
            std    = float(row_data.std()) if n > 1 else 0.0
            win_rate     = float((row_data > 0).mean())
            skew         = float(row_data.skew()) if n > 2 else 0.0
            sharpe       = round(mean / std, 3) if std > 0 else 0.0

            # ── reliability score (0-1) ───────────────────────────────────
            # Combines three independent dimensions:
            #
            # 1. size_score (40%) — sample size confidence
            #    More years = tighter confidence interval on the mean.
            #    Saturates at 10 years (diminishing returns beyond that).
            #
            # 2. consistency_score (40%) — directional consistency
            #    win_rate=1.0 or 0.0 → score=1.0 (always same direction)
            #    win_rate=0.5       → score=0.0 (random, no edge)
            #    Formula: |win_rate - 0.5| × 2 maps [0.5,1] → [0,1]
            #
            # 3. sharpe_score (20%) — risk-adjusted magnitude
            #    High Sharpe = consistent edge relative to noise.
            #    Saturates at Sharpe=2 (very strong seasonal signal).
            #
            # Skew penalty: negative skew (fat left tail) reduces score
            # because even a consistent positive return can be wiped out
            # by a single bad year with a large negative outlier.
            #
            # Final labels:
            #   >= 0.65 → "high"    (strong, consistent, well-sampled)
            #   >= 0.35 → "medium"  (some edge but uncertainty remains)
            #   <  0.35 → "low"     (unreliable, small sample or random)

            size_score        = min(n / 10.0, 1.0)
            consistency_score = abs(win_rate - 0.5) * 2.0
            sharpe_score      = min(abs(sharpe) / 2.0, 1.0)

            # skew penalty: negative skew up to -0.15 reduction
            skew_penalty = max(min(-skew * 0.05, 0.15), 0.0) if n > 2 else 0.0

            rel_score = (
                size_score        * 0.40 +
                consistency_score * 0.40 +
                sharpe_score      * 0.20
            ) - skew_penalty

            rel_score = round(max(min(rel_score, 1.0), 0.0), 3)

            if rel_score >= 0.65:
                rel_label = "high"
            elif rel_score >= 0.35:
                rel_label = "medium"
            else:
                rel_label = "low"

            rows[period_label] = {
                "mean":              mean,
                "median":            float(row_data.median()),
                "std":               std,
                "min":               float(row_data.min()),
                "max":               float(row_data.max()),
                "win_rate":          win_rate,
                "n_obs":             n,
                "skew":              skew,
                "best_year":         int(row_data.idxmax()),
                "worst_year":        int(row_data.idxmin()),
                "sharpe":            sharpe,
                "reliability_score": rel_score,
                "reliability":       rel_label,
            }
        df = pd.DataFrame(rows).T
        df.index.name = pivot.index.name
        return df


    def seasonality_heatmap_data(
        self,
        symbol: str,
        period: str = "10y",
    ) -> pd.DataFrame:
        """
        Return a year × week matrix of weekly returns — ready for a heatmap.

        Rows   : calendar year
        Columns: ISO week number (1–52)
        Values : weekly return (NaN where data is unavailable)

        Delegates to weekly_seasonality() which handles any "Ny" period
        (including "20y") via "max" fetch + year filtering.
        Simply transposes the (week × year) pivot to (year × week).
        """
        # weekly_seasonality returns (week × year) — transpose to (year × week)
        pivot_wk_yr = self.weekly_seasonality(symbol, period=period)
        return pivot_wk_yr.T

    # ------------------------------------------------------------------
    # Rolling return analysis
    # ------------------------------------------------------------------

    def rolling_returns(
        self,
        symbol: str,
        hold_years: float = 5.0,
        period: str = "20y",
    ) -> pd.Series:
        """
        Rolling N-year holding period returns.

        For every trading day in the last *period* of history, computes
        the total return of buying on that day and holding for exactly
        *hold_years* years. Returns a Series indexed by entry date.

        This answers: "if I had bought on any random day in the last 20
        years and held for N years, what return would I have gotten?"

        Parameters
        ----------
        symbol     : ticker
        hold_years : holding period in years (e.g. 1, 3, 5, 10)
        period     : history window — should be longer than hold_years
                     so there are meaningful entry dates (default "20y")

        Returns
        -------
        pd.Series  index = entry date, value = total return over hold_years
                   e.g. 0.45 means +45% over the holding period
                   NaN entries are dropped (insufficient forward data)

        Example
        -------
        rets = qa.rolling_returns("SPY", hold_years=5, period="20y")
        print(f"Avg 5y return: {rets.mean():.1%}")
        print(f"Win rate:      {(rets > 0).mean():.1%}")
        """
        # _prices() handles "Ny" periods > 10y transparently via "max" fetch
        prices    = self._prices(symbol, period=period)
        hold_days = int(hold_years * self.trading_days)

        # total return from each entry date to hold_days later
        future_price = prices.shift(-hold_days)
        total_return = (future_price - prices) / prices

        # drop entries where we don't have hold_days of forward data
        return total_return.dropna().rename(f"{symbol}_{hold_years}y_return")

    def rolling_returns_stats(
        self,
        symbol: str,
        hold_years: float = 5.0,
        period: str = "20y",
    ) -> dict:
        """
        Summary statistics for rolling N-year holding period returns.

        Answers the full picture of: "if I bought on any day in the
        last *period* and held for *hold_years*, what happened?"

        Returns
        -------
        dict with keys:
          symbol          : ticker
          hold_years      : holding period used
          n_entries       : number of valid entry dates
          mean_return     : average total return across all entry dates
          median_return   : median total return
          std_return      : standard deviation of outcomes
          min_return      : worst possible entry (bad timing)
          max_return      : best possible entry (perfect timing)
          win_rate        : fraction of entry dates with positive return
          pct_10          : 10th percentile (bad but not worst)
          pct_25          : 25th percentile
          pct_75          : 75th percentile
          pct_90          : 90th percentile (good but not best)
          best_entry      : date of best entry
          worst_entry     : date of worst entry
          cagr_mean       : mean return annualised (mean_return / hold_years approx)
          prob_double     : probability of doubling your money (+100%)
          prob_halve      : probability of losing half your money (-50%)
        """
        rets = self.rolling_returns(symbol, hold_years, period)
        if rets.empty:
            raise ValueError(f"No valid rolling returns for {symbol} "
                             f"with {hold_years}y hold in {period} window")

        cagr_mean = (1 + rets.mean()) ** (1 / hold_years) - 1

        return {
            "symbol":       symbol,
            "hold_years":   hold_years,
            "n_entries":    len(rets),
            "mean_return":  float(rets.mean()),
            "median_return":float(rets.median()),
            "std_return":   float(rets.std()),
            "min_return":   float(rets.min()),
            "max_return":   float(rets.max()),
            "win_rate":     float((rets > 0).mean()),
            "pct_10":       float(rets.quantile(0.10)),
            "pct_25":       float(rets.quantile(0.25)),
            "pct_75":       float(rets.quantile(0.75)),
            "pct_90":       float(rets.quantile(0.90)),
            "best_entry":   rets.idxmax().date(),
            "worst_entry":  rets.idxmin().date(),
            "cagr_mean":    float(cagr_mean),
            "prob_double":  float((rets >= 1.0).mean()),
            "prob_halve":   float((rets <= -0.5).mean()),
        }

    def rolling_returns_bulk(
        self,
        symbols: list[str],
        hold_years: float = 5.0,
        period: str = "20y",
    ) -> pd.DataFrame:
        """
        Compare rolling N-year returns across multiple symbols.

        Returns a DataFrame (stats × symbols) sorted by mean_return
        descending — useful for comparing which stocks rewarded patience
        most consistently.

        Example
        -------
        df = qa.rolling_returns_bulk(["AAPL","MSFT","NVDA","SPY"], hold_years=5)
        print(df.loc[["mean_return","win_rate","worst_entry"]].T)
        """
        rows = {}
        for sym in symbols:
            try:
                rows[sym] = self.rolling_returns_stats(sym, hold_years, period)
            except Exception as e:
                print(f"Skipping {sym}: {e}")

        # rows is {symbol: stats_dict} — build with symbols as rows directly
        df = pd.DataFrame(rows).T        # symbols as rows, stats as columns
        df.index.name = "symbol"
        return df.sort_values("mean_return", ascending=False)


    # ------------------------------------------------------------------
    # Kelly Criterion
    # ------------------------------------------------------------------

    def kelly(
        self,
        symbol: str,
        period: str = "2y",
        fractional: float = 1.0,
        risk_free_rate: float = 0.05,
    ) -> dict:
        """
        Kelly Criterion position sizing for a single stock.

        ── What Kelly tells you ────────────────────────────────────────
        Kelly answers: given the statistical edge and risk of this asset,
        what fraction of capital should be allocated to maximise long-run
        geometric growth?

        For a continuous return distribution the formula simplifies to:

            f* = μ / σ²

        where μ is the expected excess return (above risk-free) and σ²
        is the variance of returns. Both are annualised.

        ── Interpreting the result ─────────────────────────────────────
        f* = 0.30  → full Kelly says risk 30% of capital on this stock
        f* < 0     → no edge over the risk-free rate — do not hold
        f* > 1     → extreme edge signal (rare; usually means overfitting
                     to a short or lucky history — treat with scepticism)

        ── Why full Kelly is almost never used ─────────────────────────
        Full Kelly assumes:
          • The return distribution is perfectly known (it never is)
          • You can tolerate the resulting drawdowns (psychologically hard)
          • Returns are i.i.d. (autocorrelation and regimes exist)

        In practice, fractional Kelly is standard:
          • Half Kelly  (fractional=0.5) — halves volatility, keeps ~75%
            of the growth rate. Most common real-world choice.
          • Quarter Kelly (fractional=0.25) — very conservative, smooth
            equity curve, good for uncertain or short histories.

        ── Best use: relative ranking, not absolute sizing ─────────────
        The raw fraction is sensitive to estimation error in μ and σ.
        The most robust use of Kelly is as a ranking tool:
          kelly_bulk(symbols) sorted highest → lowest shows which stocks
          had the strongest risk-adjusted edge in the history window.
        Use the absolute fraction only as a rough upper bound on sizing.

        Parameters
        ----------
        symbol        : ticker
        period        : history window for μ and σ estimation (default "2y")
        fractional    : Kelly multiplier — 1.0 = full Kelly, 0.5 = half
                        Kelly, 0.25 = quarter Kelly (default 1.0)
        risk_free_rate: annual risk-free rate (default 0.05)

        Returns
        -------
        dict with keys:
          symbol           : ticker
          period           : history window used
          full_kelly       : f* = μ / σ²  (uncapped)
          fractional_kelly : full_kelly × fractional
          fraction_label   : human-readable label (e.g. "Half Kelly")
          mu_annual        : annualised expected excess return
          sigma_annual     : annualised volatility
          sharpe_ratio     : μ / σ  (related: Kelly = Sharpe / σ)
          edge             : μ  (positive = edge over risk-free)
          has_edge         : True when full_kelly > 0
          suggested_max    : min(fractional_kelly, 0.25) — a conservative
                             cap that accounts for estimation uncertainty
        """
        rets     = self._simple_returns(self._prices(symbol, period))
        daily_rf = risk_free_rate / self.trading_days
        excess   = rets - daily_rf

        mu    = float(excess.mean() * self.trading_days)      # annualised
        sigma = float(rets.std() * np.sqrt(self.trading_days))
        var   = sigma ** 2

        full_kelly = mu / var if var > 0 else 0.0
        frac_kelly = full_kelly * fractional

        # fraction label
        labels = {1.0: "Full Kelly", 0.5: "Half Kelly",
                  0.25: "Quarter Kelly", 0.33: "Third Kelly"}
        label = labels.get(fractional, f"{fractional:.0%} Kelly")

        return {
            "symbol":           symbol,
            "period":           period,
            "full_kelly":       round(full_kelly, 4),
            "fractional_kelly": round(frac_kelly, 4),
            "fraction_label":   label,
            "mu_annual":        round(mu, 4),
            "sigma_annual":     round(sigma, 4),
            "sharpe_ratio":     round(mu / sigma, 4) if sigma > 0 else 0.0,
            "edge":             round(mu, 4),
            "has_edge":         full_kelly > 0,
            "suggested_max":    round(min(max(frac_kelly, 0.0), 0.25), 4),
        }

    def kelly_bulk(
        self,
        symbols: list[str],
        period: str = "2y",
        fractional: float = 0.5,
        risk_free_rate: float = 0.05,
    ) -> pd.DataFrame:
        """
        Kelly Criterion for multiple symbols — returns a ranked comparison table.

        ── Best use ────────────────────────────────────────────────────
        Use kelly_bulk() as a relative ranking tool rather than relying
        on the absolute fraction numbers. The ranking is more stable than
        the raw fractions across different history windows because ranking
        is less sensitive to small changes in μ and σ estimates.

        Sorting by full_kelly descending answers:
          "Which stocks had the strongest risk-adjusted edge historically?"

        Parameters
        ----------
        symbols       : list of tickers
        period        : history window (default "2y" — long enough for
                        reliable σ, short enough to reflect recent regime)
        fractional    : Kelly multiplier applied to all symbols (default 0.5
                        = half Kelly, the most common practical choice)
        risk_free_rate: annual risk-free rate (default 0.05)

        Returns
        -------
        DataFrame sorted by full_kelly descending, columns:
          full_kelly, fractional_kelly, mu_annual, sigma_annual,
          sharpe_ratio, has_edge, suggested_max
        """
        rows = {}
        for sym in symbols:
            try:
                r = self.kelly(sym, period, fractional, risk_free_rate)
                rows[sym] = {k: v for k, v in r.items()
                             if k not in ("symbol", "period", "fraction_label", "edge")}
            except Exception as e:
                print(f"Skipping {sym}: {e}")

        df = pd.DataFrame(rows).T
        df.index.name = "symbol"
        return df.sort_values("full_kelly", ascending=False)

    def kelly_portfolio(
        self,
        symbols: list[str],
        period: str = "2y",
        fractional: float = 0.5,
        risk_free_rate: float = 0.05,
        normalise: bool = True,
    ) -> pd.Series:
        """
        Kelly-optimal portfolio weights across multiple symbols.

        ── How it works ────────────────────────────────────────────────
        Each symbol gets a weight proportional to its fractional Kelly
        fraction. Symbols with no edge (Kelly ≤ 0) get zero weight.

        This is a simplified single-asset Kelly applied independently —
        not the full multi-asset Kelly (which requires inverting the
        covariance matrix and is very sensitive to estimation error).
        The simplified version is more robust in practice.

        ── Comparison with other optimisers ────────────────────────────
        Max Sharpe  → maximises return per unit of total risk
        Min Variance → minimises total portfolio vol
        Risk Parity  → equalises risk contribution per asset
        Kelly        → maximises long-run geometric growth rate
                       (implicitly penalises variance, rewards edge)

        Parameters
        ----------
        symbols    : list of tickers
        period     : history window (default "2y")
        fractional : Kelly multiplier (default 0.5 = half Kelly)
        normalise  : if True, weights sum to 1.0 (default True)
                     if False, weights are raw Kelly fractions (may not sum to 1)

        Returns
        -------
        pd.Series  symbol → weight, sorted descending
                   (zero-weight symbols included for transparency)
        """
        bulk = self.kelly_bulk(symbols, period, fractional, risk_free_rate)

        weights = bulk["fractional_kelly"].astype(float).clip(lower=0.0)

        if normalise and weights.sum() > 0:
            weights = weights / weights.sum()

        return weights.sort_values(ascending=False).rename("kelly_weight")

    # ------------------------------------------------------------------
    # Missing best/worst days analysis
    # ------------------------------------------------------------------

    def best_worst_days_impact(
        self,
        symbol: str,
        period: str = "20y",
        miss_scenarios: list[int] | None = None,
        initial_value: float = 10_000.0,
    ) -> pd.DataFrame:
        """
        "Cost of missing the best days" study.

        Computes what happens to a buy-and-hold investment if you were
        out of the market on the N best (or worst) trading days.

        This is one of the most compelling arguments for staying invested —
        the best days tend to cluster around the worst days (crashes and
        recoveries happen close together), so trying to time the market
        and missing just a handful of days can dramatically reduce returns.

        Parameters
        ----------
        symbol         : ticker (typically a broad index ETF like "SPY")
        period         : history window (default "20y")
        miss_scenarios : list of N values to test (default [5, 10, 20, 30, 40, 50])
        initial_value  : starting portfolio value (default 10,000)

        Returns
        -------
        DataFrame with one row per scenario:
          scenario          : description (e.g. "Miss best 10 days")
          final_value       : ending portfolio value
          total_return      : total return over the period
          cagr              : annualised compound return
          vs_buy_hold       : difference in final value vs buy-and-hold
          days_missed       : number of best days missed
          type              : "buy_and_hold" | "miss_best" | "miss_worst"

        The DataFrame also includes:
          - "Buy & Hold" baseline row
          - Miss best N days (reduced returns — cost of being out)
          - Miss worst N days (improved returns — lucky avoidance)

        Example
        -------
        df = qa.best_worst_days_impact("SPY", period="20y")
        print(df[["scenario","final_value","total_return","cagr"]])
        """
        miss_scenarios = miss_scenarios or [5, 10, 20, 30, 40, 50]

        prices     = self._prices(symbol, period=period)
        rets       = prices.pct_change().dropna()
        n_days     = len(rets)
        n_years    = n_days / self.trading_days

        def _terminal(returns_series) -> tuple[float, float, float]:
            final   = initial_value * (1 + returns_series).cumprod().iloc[-1]
            total_r = final / initial_value - 1
            cagr    = (1 + total_r) ** (1 / n_years) - 1 if n_years > 0 else 0.0
            return float(final), float(total_r), float(cagr)

        rows = []

        # buy & hold baseline
        bh_final, bh_ret, bh_cagr = _terminal(rets)
        rows.append({
            "scenario":    "Buy & Hold",
            "days_missed": 0,
            "final_value": bh_final,
            "total_return":bh_ret,
            "cagr":        bh_cagr,
            "vs_buy_hold": 0.0,
            "type":        "buy_and_hold",
        })

        # miss best N days
        sorted_best = rets.sort_values(ascending=False)
        for n in miss_scenarios:
            best_days   = sorted_best.head(n).index
            adj_rets    = rets.copy()
            adj_rets[best_days] = 0.0
            final, total_r, cagr = _terminal(adj_rets)
            rows.append({
                "scenario":    f"Miss best {n} days",
                "days_missed": n,
                "final_value": final,
                "total_return":total_r,
                "cagr":        cagr,
                "vs_buy_hold": final - bh_final,
                "type":        "miss_best",
            })

        # miss worst N days
        sorted_worst = rets.sort_values(ascending=True)
        for n in miss_scenarios:
            worst_days  = sorted_worst.head(n).index
            adj_rets    = rets.copy()
            adj_rets[worst_days] = 0.0
            final, total_r, cagr = _terminal(adj_rets)
            rows.append({
                "scenario":    f"Miss worst {n} days",
                "days_missed": n,
                "final_value": final,
                "total_return":total_r,
                "cagr":        cagr,
                "vs_buy_hold": final - bh_final,
                "type":        "miss_worst",
            })

        df = pd.DataFrame(rows)

        # add metadata
        df["symbol"]        = symbol
        df["period"]        = period
        df["n_trading_days"]= n_days
        df["initial_value"] = initial_value

        return df

    def best_worst_days_detail(
        self,
        symbol: str,
        period: str = "20y",
        n: int = 20,
    ) -> pd.DataFrame:
        """
        Return the N best and N worst individual trading days.

        Useful for seeing WHEN the extreme days occurred —
        they almost always cluster around crises (2008, 2020, 2022),
        which is why timing the market is so hard.

        Returns
        -------
        DataFrame sorted by return descending:
          date      : trading date
          return    : daily return
          rank      : 1 = best day, -1 = worst day
          type      : "best" | "worst"
        """
        prices = self._prices(symbol, period=period)
        rets   = prices.pct_change().dropna()

        # handle Series with name
        def _to_df(s, kind):
            df = s.reset_index()
            df.columns = ["date", "return"]
            df["type"] = kind
            return df

        best_df  = _to_df(rets.nlargest(n),  "best")
        worst_df = _to_df(rets.nsmallest(n), "worst")

        combined = pd.concat([best_df, worst_df]).sort_values(
            "return", ascending=False
        ).reset_index(drop=True)
        combined.index = combined.index + 1
        combined.index.name = "rank"
        return combined

