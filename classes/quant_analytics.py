"""
QuantAnalytics — quantitative metrics built on top of StockClient.

Requires: numpy, pandas, scipy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from classes.stock_client import StockClient


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
        """
        bars = self.client.get_history(symbol, period=period, interval=interval, adjusted=True)
        if not bars:
            raise ValueError(f"No history returned for '{symbol}' (period={period})")
        df = pd.DataFrame(bars)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["adj_close"].sort_index().astype(float)

    def _prices_bulk(
        self, symbols: list[str], period: str, interval: str = "1d"
    ) -> pd.DataFrame:
        """Return a DataFrame of closing prices, one column per symbol."""
        series = {sym: self._prices(sym, period, interval) for sym in symbols}
        return pd.DataFrame(series).sort_index()

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
                "calmar_ratio":          self.calmar_ratio(sym, "3y"),
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
            "calmar_ratio": self.calmar_ratio(symbol, "3y"),
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

    def weekly_seasonality(
        self,
        symbol: str,
        period: str = "10y",
        min_observations: int = 3,
    ) -> pd.DataFrame:
        """
        Weekly seasonality study — mean return for each ISO week of the year.

        For each of the 52 (or 53) weeks, aggregates all occurrences across
        the available history and returns statistics per week.

        Parameters
        ----------
        symbol           : ticker
        period           : history window — "5y", "10y", "max" recommended
        min_observations : weeks with fewer years of data are flagged (column
                           "reliable" = False)

        Returns
        -------
        DataFrame indexed by week number (1–52) with columns:
          mean_return   : average weekly return across all years
          median_return : median weekly return
          std_return    : standard deviation of weekly returns
          win_rate      : fraction of years where that week was positive
          n_obs         : number of yearly observations for that week
          reliable      : True when n_obs >= min_observations
          cumulative    : cumulative sum of mean_return (shows seasonal drift)
        """
        # fetch daily adj-close and resample to weekly (Friday close)
        prices = self._prices(symbol, period=period, interval="1d")
        weekly = prices.resample("W-FRI").last().dropna()
        weekly_returns = weekly.pct_change().dropna()

        # tag each row with its ISO week number
        df = weekly_returns.to_frame(name="return")
        df["week"] = df.index.isocalendar().week.astype(int)
        df["year"] = df.index.year

        # aggregate per week number
        grouped = df.groupby("week")["return"]
        stats = pd.DataFrame({
            "mean_return":   grouped.mean(),
            "median_return": grouped.median(),
            "std_return":    grouped.std(),
            "win_rate":      grouped.apply(lambda x: (x > 0).mean()),
            "n_obs":         grouped.count(),
        })

        stats["reliable"]   = stats["n_obs"] >= min_observations
        stats["cumulative"] = stats["mean_return"].cumsum()
        return stats

    def monthly_seasonality(
        self,
        symbol: str,
        period: str = "10y",
        min_observations: int = 3,
    ) -> pd.DataFrame:
        """
        Monthly seasonality — same as weekly_seasonality but at month granularity.

        Returns a DataFrame indexed 1–12 with the same columns.
        Useful as a higher-level complement to the weekly view.
        """
        prices = self._prices(symbol, period=period, interval="1d")
        monthly = prices.resample("BME").last().dropna()
        monthly_returns = monthly.pct_change().dropna()

        df = monthly_returns.to_frame(name="return")
        df["month"] = df.index.month

        grouped = df.groupby("month")["return"]
        stats = pd.DataFrame({
            "mean_return":   grouped.mean(),
            "median_return": grouped.median(),
            "std_return":    grouped.std(),
            "win_rate":      grouped.apply(lambda x: (x > 0).mean()),
            "n_obs":         grouped.count(),
        })
        stats.index = pd.Index(
            ["Jan","Feb","Mar","Apr","May","Jun",
             "Jul","Aug","Sep","Oct","Nov","Dec"],
            name="month"
        )
        stats["reliable"]   = stats["n_obs"] >= min_observations
        stats["cumulative"] = stats["mean_return"].cumsum()
        return stats

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
        """
        prices = self._prices(symbol, period=period, interval="1d")
        weekly = prices.resample("W-FRI").last().dropna()
        weekly_returns = weekly.pct_change().dropna()

        df = weekly_returns.to_frame(name="return")
        df["week"] = df.index.isocalendar().week.astype(int)
        df["year"] = df.index.year

        # pivot: rows=year, cols=week
        pivot = df.pivot_table(
            index="year", columns="week", values="return", aggfunc="first"
        )
        # keep only weeks 1–52
        pivot = pivot[[c for c in range(1, 53) if c in pivot.columns]]
        return pivot
