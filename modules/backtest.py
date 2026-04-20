"""
backtest.py — event-driven backtesting built on QuantAnalytics.

A strategy is any callable:
    strategy(prices: pd.DataFrame) -> pd.DataFrame
    - prices  : adj-close prices, dates × symbols
    - returns : target weights DataFrame, same index, same columns, rows sum to 1

Built-in strategies (return a configured callable)
--------------------------------------------------
buy_and_hold(weights=None)          equal-weight, never rebalances
ma_crossover(fast, slow)            long when fast MA > slow MA, else cash
momentum(lookback, top_n)           long the top_n symbols by past return
mean_reversion(lookback, top_n)     long the bottom_n symbols (contrarian)
target_weights(weights)             fixed allocation, rebalances periodically

Usage
-----
    from classes.stock_client import StockClient
    from classes.quant_analytics import QuantAnalytics
    from modules.backtest import run, ma_crossover, momentum, buy_and_hold

    client  = StockClient()
    quant   = QuantAnalytics(client)
    symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "JPM"]

    result = run(quant, symbols,
                 strategy=ma_crossover(fast=20, slow=50),
                 period="3y",
                 rebalance="weekly",
                 transaction_cost_bps=10,
                 benchmark="SPY")

    print(result)
    result.metrics_df()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import pandas as pd

from classes.quant_analytics import QuantAnalytics


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Output of backtest.run().

    Attributes
    ----------
    equity_curve    : portfolio value over time (starts at initial_capital)
    returns         : daily portfolio returns
    positions       : target weights at each rebalance date (dates × symbols)
    trades          : weight changes at each rebalance (non-zero = traded)
    metrics         : dict of performance metrics
    strategy_name   : display name of the strategy
    symbols         : tickers used
    period          : historical window
    benchmark_result: optional BacktestResult for the benchmark
    """
    equity_curve:     pd.Series
    returns:          pd.Series
    positions:        pd.DataFrame
    trades:           pd.DataFrame
    metrics:          dict
    strategy_name:    str
    symbols:          list[str]
    period:           str
    initial_capital:  float
    benchmark_result: "BacktestResult | None" = field(default=None, repr=False)

    # ------------------------------------------------------------------
    def metrics_df(self) -> pd.DataFrame:
        """Return metrics as a formatted single-column DataFrame."""
        rows = {k: [f"{v:.2%}" if isinstance(v, float) and abs(v) < 100
                    else f"{v:.2f}" if isinstance(v, float)
                    else v]
                for k, v in self.metrics.items()}
        df = pd.DataFrame(rows, index=[self.strategy_name]).T
        if self.benchmark_result:
            bm = self.benchmark_result
            brows = {k: [f"{v:.2%}" if isinstance(v, float) and abs(v) < 100
                         else f"{v:.2f}" if isinstance(v, float)
                         else v]
                     for k, v in bm.metrics.items()}
            df[bm.strategy_name] = pd.DataFrame(brows, index=[bm.strategy_name]).T
        return df

    def __repr__(self) -> str:
        m = self.metrics
        lines = [
            f"BacktestResult — {self.strategy_name}",
            f"  Period       : {self.period}",
            f"  CAGR         : {m.get('cagr', 0):.2%}",
            f"  Volatility   : {m.get('volatility', 0):.2%}",
            f"  Sharpe       : {m.get('sharpe_ratio', 0):.3f}",
            f"  Max Drawdown : {m.get('max_drawdown', 0):.2%}",
            f"  Calmar       : {m.get('calmar_ratio', 0):.3f}",
            f"  # Trades     : {m.get('n_trades', 0)}",
            f"  Total costs  : {m.get('total_cost', 0):.2%}",
        ]
        if self.benchmark_result:
            bm = self.benchmark_result.metrics
            lines += [
                f"\n  Benchmark ({self.benchmark_result.strategy_name})",
                f"  CAGR         : {bm.get('cagr', 0):.2%}",
                f"  Sharpe       : {bm.get('sharpe_ratio', 0):.3f}",
                f"  Max Drawdown : {bm.get('max_drawdown', 0):.2%}",
            ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metrics calculation
# ---------------------------------------------------------------------------

def _calc_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    cost_series: pd.Series,
    trading_days: int = 252,
    risk_free_rate: float = 0.05,
) -> dict:
    n_years  = len(returns) / trading_days
    total_ret = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    cagr      = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    vol    = returns.std() * np.sqrt(trading_days)
    daily_rf = risk_free_rate / trading_days
    excess = returns - daily_rf
    sharpe = excess.mean() / excess.std() * np.sqrt(trading_days) if excess.std() > 0 else 0.0

    downside = excess[excess < 0].std()
    sortino  = excess.mean() / downside * np.sqrt(trading_days) if downside > 0 else 0.0

    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    max_dd   = drawdown.min()
    calmar   = cagr / abs(max_dd) if max_dd != 0 else float("inf")

    n_trades    = int((trades.abs() > 1e-6).any(axis=1).sum())
    total_cost  = float(cost_series.sum())

    return {
        "cagr":             cagr,
        "total_return":     total_ret,
        "volatility":       vol,
        "sharpe_ratio":     sharpe,
        "sortino_ratio":    sortino,
        "max_drawdown":     max_dd,
        "calmar_ratio":     calmar,
        "n_trades":         n_trades,
        "total_cost":       total_cost,
        "best_day":         returns.max(),
        "worst_day":        returns.min(),
        "win_rate":         (returns > 0).mean(),
    }


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run(
    quant: QuantAnalytics,
    symbols: list[str],
    strategy: Callable,
    period: str = "3y",
    rebalance: str = "monthly",     # "daily" | "weekly" | "monthly" | "quarterly"
    transaction_cost_bps: float = 10.0,
    initial_capital: float = 100_000.0,
    risk_free_rate: float = 0.05,
    benchmark: str | None = "SPY",
    strategy_name: str | None = None,
) -> BacktestResult:
    """
    Run a strategy over historical data and return a BacktestResult.

    Parameters
    ----------
    quant                : QuantAnalytics instance
    symbols              : list of tickers to trade
    strategy             : callable(prices) -> weights — see built-in strategies
    period               : historical window (e.g. "3y", "5y")
    rebalance            : how often to rebalance — "daily", "weekly",
                           "monthly", "quarterly"
    transaction_cost_bps : round-trip cost per trade in basis points (default 10)
    initial_capital      : starting portfolio value (default 100,000)
    risk_free_rate       : annual risk-free rate for Sharpe/Sortino
    benchmark            : ticker to run as a buy-and-hold comparison (None to skip)
    strategy_name        : display name (inferred from strategy if not set)
    """
    cost_rate = transaction_cost_bps / 10_000.0
    name      = strategy_name or getattr(strategy, "__name__", "Strategy")

    # --- fetch prices ----------------------------------------------------
    prices = quant._prices_bulk(symbols, period=period)
    prices = prices.dropna(how="all").ffill()
    # force a clean tz-naive date-only index regardless of yfinance version
    prices.index = pd.to_datetime(
        prices.index.strftime("%Y-%m-%d") if hasattr(prices.index, "strftime")
        else prices.index
    )

    # --- generate raw weights from strategy (full daily signal) ----------
    raw_weights = strategy(prices)

    # ensure valid weights: fill NaN with 0, clip negatives, normalise rows
    raw_weights = raw_weights.reindex(prices.index).fillna(0.0)
    raw_weights = raw_weights.clip(lower=0)
    row_sums    = raw_weights.sum(axis=1).replace(0, np.nan)
    raw_weights = raw_weights.div(row_sums, axis=0).fillna(0.0)

    # --- apply rebalance frequency ---------------------------------------
    freq_map = {"daily": "B", "weekly": "W-FRI", "monthly": "BME", "quarterly": "BQE"}
    freq     = freq_map.get(rebalance, "BME")
    rebal_dates = pd.date_range(
        start=prices.index[0], end=prices.index[-1], freq=freq
    )
    rebal_dates = prices.index[prices.index.isin(rebal_dates) |
                               (prices.index == prices.index[0])]

    # build held weights: hold last rebalance weight until next rebalance
    held_weights = pd.DataFrame(0.0, index=prices.index, columns=symbols)
    last_w = pd.Series(0.0, index=symbols)
    for date in prices.index:
        if date in rebal_dates:
            last_w = raw_weights.loc[date]
        held_weights.loc[date] = last_w

    # --- simulate trades and P&L -----------------------------------------
    daily_returns  = prices.pct_change().fillna(0.0)
    weight_changes = held_weights.diff().fillna(held_weights.iloc[[0]])

    # transaction cost: cost_rate × |Δweight| per position
    cost_per_day = (weight_changes.abs() * cost_rate).sum(axis=1)

    # portfolio daily return = Σ(weight_i × asset_return_i) − cost
    port_returns = (held_weights.shift(1).fillna(0) * daily_returns).sum(axis=1)
    port_returns -= cost_per_day

    equity_curve = initial_capital * (1 + port_returns).cumprod()
    equity_curve.iloc[0] = initial_capital

    metrics = _calc_metrics(
        port_returns, equity_curve, weight_changes,
        cost_per_day, quant.trading_days, risk_free_rate,
    )

    result = BacktestResult(
        equity_curve=equity_curve,
        returns=port_returns,
        positions=held_weights,
        trades=weight_changes,
        metrics=metrics,
        strategy_name=name,
        symbols=symbols,
        period=period,
        initial_capital=initial_capital,
    )

    # --- benchmark -------------------------------------------------------
    if benchmark:
        bm_result = run(
            quant, [benchmark],
            strategy=buy_and_hold(),
            period=period,
            rebalance="monthly",
            transaction_cost_bps=0,
            initial_capital=initial_capital,
            risk_free_rate=risk_free_rate,
            benchmark=None,
            strategy_name=benchmark,
        )
        result.benchmark_result = bm_result

    return result


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------

def buy_and_hold(weights: dict | None = None):
    """
    Equal-weight (or custom fixed weights) buy-and-hold.

    weights : optional dict {symbol: weight} — defaults to equal weight.
              Weights are normalised to sum to 1.

    Example:
        buy_and_hold()                          # equal weight
        buy_and_hold({"AAPL": 0.6, "MSFT": 0.4})
    """
    def strategy(prices: pd.DataFrame) -> pd.DataFrame:
        n = len(prices.columns)
        if weights is None:
            w = {sym: 1.0 / n for sym in prices.columns}
        else:
            total = sum(weights.values())
            w = {sym: weights.get(sym, 0.0) / total for sym in prices.columns}
        return pd.DataFrame(
            [w] * len(prices),
            index=prices.index,
            columns=prices.columns,
        )
    strategy.__name__ = "Buy & Hold"
    return strategy


def ma_crossover(fast: int = 20, slow: int = 50):
    """
    Moving average crossover — long equal-weight when fast MA > slow MA, else cash.

    fast : short-window MA period in days (default 20)
    slow : long-window MA period in days (default 50)

    Each symbol is independently signalled. Symbols with fast > slow get equal
    share of the portfolio; all others go to cash (weight = 0).

    Example:
        ma_crossover(fast=10, slow=30)
        ma_crossover(fast=50, slow=200)   # golden cross
    """
    def strategy(prices: pd.DataFrame) -> pd.DataFrame:
        fast_ma = prices.rolling(fast).mean()
        slow_ma = prices.rolling(slow).mean()
        signal  = (fast_ma > slow_ma).astype(float)
        row_sum = signal.sum(axis=1).replace(0, np.nan)
        return signal.div(row_sum, axis=0).fillna(0.0)
    strategy.__name__ = f"MA Crossover ({fast}/{slow})"
    return strategy


def momentum(lookback: int = 63, top_n: int | None = None):
    """
    Momentum — long the top_n symbols by past return over *lookback* days.

    lookback : return calculation window in days (default 63 = ~1 quarter)
    top_n    : number of top symbols to hold (default = half the universe)

    Equal weight among selected symbols. Remaining symbols get 0.

    Example:
        momentum(lookback=21, top_n=3)    # top 3 over past month
        momentum(lookback=252)            # top half over past year
    """
    def strategy(prices: pd.DataFrame) -> pd.DataFrame:
        n = len(prices.columns)
        k = top_n or max(1, n // 2)
        past_returns = prices.pct_change(lookback)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for date, row in past_returns.iterrows():
            valid = row.dropna()
            if len(valid) == 0:
                continue
            top = valid.nlargest(min(k, len(valid))).index
            weights.loc[date, top] = 1.0 / len(top)
        return weights
    strategy.__name__ = f"Momentum ({lookback}d top-{top_n or 'half'})"
    return strategy


def mean_reversion(lookback: int = 20, top_n: int | None = None):
    """
    Mean reversion — long the top_n worst recent performers (contrarian).

    lookback : return calculation window in days (default 20 = ~1 month)
    top_n    : number of symbols to hold (default = half the universe)

    Equal weight among selected symbols.

    Example:
        mean_reversion(lookback=5, top_n=2)   # fade last week's losers
        mean_reversion(lookback=20)            # monthly contrarian
    """
    def strategy(prices: pd.DataFrame) -> pd.DataFrame:
        n = len(prices.columns)
        k = top_n or max(1, n // 2)
        past_returns = prices.pct_change(lookback)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for date, row in past_returns.iterrows():
            valid = row.dropna()
            if len(valid) == 0:
                continue
            bottom = valid.nsmallest(min(k, len(valid))).index
            weights.loc[date, bottom] = 1.0 / len(bottom)
        return weights
    strategy.__name__ = f"Mean Reversion ({lookback}d top-{top_n or 'half'})"
    return strategy


def target_weights(weights: dict):
    """
    Rebalance to fixed target weights periodically.

    weights : dict {symbol: weight} — normalised to sum to 1.

    Identical to buy_and_hold() but semantically clearer when used
    with a non-trivial allocation (e.g. from portfolio.max_sharpe).

    Example:
        w = portfolio.max_sharpe(quant, symbols).weights_df.to_dict()
        run(quant, symbols, strategy=target_weights(w), rebalance="monthly")
    """
    def strategy(prices: pd.DataFrame) -> pd.DataFrame:
        total = sum(weights.values())
        w = {sym: weights.get(sym, 0.0) / total for sym in prices.columns}
        return pd.DataFrame(
            [w] * len(prices),
            index=prices.index,
            columns=prices.columns,
        )
    strategy.__name__ = "Target Weights"
    return strategy
