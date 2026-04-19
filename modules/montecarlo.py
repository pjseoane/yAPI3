"""
montecarlo.py — Monte Carlo portfolio simulation built on QuantAnalytics.

Three simulation methods:
  "historical"  : block bootstrap of actual historical returns
                  Non-parametric — preserves fat tails, autocorrelation,
                  and cross-asset correlation structure.
  "normal"      : draws from a multivariate normal distribution fitted to
                  historical mean and covariance. Fastest but underestimates
                  tail risk (thin tails assumption).
  "t_dist"      : multivariate Student-t — heavier tails than normal,
                  better for risk/VaR estimation.

Usage
-----
    from modules.montecarlo import simulate
    import modules.plots as plots

    result = simulate(
        quant, symbols, weights,
        horizon=252,          # 1 trading year forward
        n_sims=1000,
        method="historical",
        initial_value=100_000,
        period="3y",          # history window to fit on
    )

    print(result)
    plots.monte_carlo(result).show()
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    """
    Output of montecarlo.simulate().

    Attributes
    ----------
    paths           : simulated portfolio value paths (horizon × n_sims)
    percentiles     : DataFrame with [5, 25, 50, 75, 95] percentile paths
    final_values    : terminal portfolio values across all simulations (n_sims,)
    metrics         : dict of risk/return statistics
    symbols         : tickers used
    weights         : portfolio weights
    initial_value   : starting portfolio value
    horizon         : simulation horizon in trading days
    n_sims          : number of simulated paths
    method          : simulation method used
    period          : historical window used to fit the model
    """
    paths:         pd.DataFrame          # index=day 0..horizon, cols=sim_0..sim_N
    percentiles:   pd.DataFrame          # index=day 0..horizon, cols=[5,25,50,75,95]
    final_values:  np.ndarray            # shape (n_sims,)
    metrics:       dict
    symbols:       list[str]
    weights:       np.ndarray
    initial_value: float
    horizon:       int
    n_sims:        int
    method:        str
    period:        str

    def __repr__(self) -> str:
        m = self.metrics
        lines = [
            f"MonteCarloResult — {self.method}  ({self.n_sims:,} sims, {self.horizon}d horizon)",
            f"  Median final value : ${m['median_final']:,.0f}  ({m['median_return']:+.1%})",
            f"  Mean final value   : ${m['mean_final']:,.0f}",
            f"  5th pct (bad)      : ${m['pct_05_final']:,.0f}  ({m['pct_05_return']:+.1%})",
            f"  95th pct (good)    : ${m['pct_95_final']:,.0f}  ({m['pct_95_return']:+.1%})",
            f"  VaR 95% (horizon)  : {m['var_95']:+.2%}",
            f"  CVaR 95% (horizon) : {m['cvar_95']:+.2%}",
            f"  Prob of gain       : {m['prob_gain']:.1%}",
            f"  Prob of loss > 10% : {m['prob_loss_10pct']:.1%}",
            f"  Prob of loss > 20% : {m['prob_loss_20pct']:.1%}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _calc_metrics(
    final_values: np.ndarray,
    initial_value: float,
) -> dict:
    final_returns = final_values / initial_value - 1
    var_95  = float(np.percentile(final_returns, 5))
    cvar_95 = float(final_returns[final_returns <= var_95].mean())

    return {
        "mean_final":       float(final_values.mean()),
        "median_final":     float(np.median(final_values)),
        "std_final":        float(final_values.std()),
        "pct_05_final":     float(np.percentile(final_values, 5)),
        "pct_25_final":     float(np.percentile(final_values, 25)),
        "pct_75_final":     float(np.percentile(final_values, 75)),
        "pct_95_final":     float(np.percentile(final_values, 95)),
        "mean_return":      float(final_returns.mean()),
        "median_return":    float(np.median(final_returns)),
        "pct_05_return":    float(np.percentile(final_returns, 5)),
        "pct_95_return":    float(np.percentile(final_returns, 95)),
        "var_95":           var_95,
        "cvar_95":          cvar_95,
        "prob_gain":        float((final_returns > 0).mean()),
        "prob_loss_10pct":  float((final_returns < -0.10).mean()),
        "prob_loss_20pct":  float((final_returns < -0.20).mean()),
    }


def _build_paths(
    daily_port_returns: np.ndarray,   # shape (horizon, n_sims)
    initial_value: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Build paths DataFrame, percentile bands, and final values array."""
    # cumulative product: value at each day
    cum = np.cumprod(1 + daily_port_returns, axis=0)
    paths_arr = initial_value * np.vstack([np.ones((1, cum.shape[1])), cum])

    day_index = range(paths_arr.shape[0])
    paths_df  = pd.DataFrame(
        paths_arr,
        index=day_index,
        columns=[f"sim_{i}" for i in range(paths_arr.shape[1])],
    )

    pct_df = pd.DataFrame(
        {p: np.percentile(paths_arr, p, axis=1) for p in [5, 25, 50, 75, 95]},
        index=day_index,
    )

    final_values = paths_arr[-1]
    return paths_df, pct_df, final_values


# ---------------------------------------------------------------------------
# Simulation methods
# ---------------------------------------------------------------------------

def _simulate_historical(
    port_returns: pd.Series,
    horizon: int,
    n_sims: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Block bootstrap: resample contiguous blocks of historical returns.
    Preserves autocorrelation and volatility clustering better than IID sampling.
    Returns shape (horizon, n_sims).
    """
    returns_arr = port_returns.values
    n = len(returns_arr)
    n_blocks = int(np.ceil(horizon / block_size))

    all_sims = np.empty((horizon, n_sims))
    for s in range(n_sims):
        blocks = []
        for _ in range(n_blocks):
            start = rng.integers(0, n - block_size)
            blocks.append(returns_arr[start: start + block_size])
        sim = np.concatenate(blocks)[:horizon]
        all_sims[:, s] = sim
    return all_sims


def _simulate_normal(
    mean_returns: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    horizon: int,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Multivariate normal: draw correlated asset returns, apply weights.
    Returns shape (horizon, n_sims).
    """
    # draw (horizon, n_sims, n_assets) then apply weights
    draws = rng.multivariate_normal(mean_returns, cov, size=(horizon, n_sims))
    return (draws @ weights)   # (horizon, n_sims)


def _simulate_t(
    mean_returns: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    horizon: int,
    n_sims: int,
    df_t: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Multivariate Student-t via Cholesky decomposition.
    Heavier tails than normal — better for tail risk estimation.
    Returns shape (horizon, n_sims).
    """
    n_assets = len(weights)
    L        = np.linalg.cholesky(cov)
    chi2     = rng.chisquare(df_t, size=(horizon, n_sims)) / df_t  # (H, S)

    # standard normal shocks: (H, S, n_assets)
    Z = rng.standard_normal((horizon, n_sims, n_assets))
    # apply Cholesky: (H, S, n_assets)
    correlated = Z @ L.T
    # scale by chi2 for Student-t: broadcast chi2 (H, S, 1)
    t_draws = mean_returns + correlated / np.sqrt(chi2[:, :, np.newaxis])

    return (t_draws @ weights)  # (horizon, n_sims)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate(
    quant,
    symbols: list[str],
    weights: list[float] | np.ndarray | None = None,
    horizon: int = 252,
    n_sims: int = 1000,
    method: str = "historical",      # "historical" | "normal" | "t_dist"
    period: str = "3y",
    initial_value: float = 100_000.0,
    block_size: int = 10,            # days per block (historical method only)
    df_t: float = 5.0,               # degrees of freedom (t_dist method only)
    seed: int = 42,
) -> MonteCarloResult:
    """
    Run a Monte Carlo simulation for a portfolio.

    Parameters
    ----------
    quant         : QuantAnalytics instance
    symbols       : list of tickers
    weights       : portfolio weights — list/array summing to 1.
                    Defaults to equal weight.
    horizon       : simulation horizon in trading days (default 252 = 1 year)
    n_sims        : number of simulated paths (default 1000; use 5000+ for VaR)
    method        : simulation method
                    "historical" — block bootstrap of actual returns (recommended)
                    "normal"     — multivariate Gaussian (fastest, thin tails)
                    "t_dist"     — Student-t, heavier tails (best for tail risk)
    period        : historical window to fit the model on (default "3y")
    initial_value : starting portfolio value (default 100,000)
    block_size    : block length in days for historical bootstrap (default 10)
    df_t          : degrees of freedom for Student-t (default 5; lower = fatter tails)
    seed          : random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    # --- weights ---------------------------------------------------------
    n = len(symbols)
    if weights is None:
        weights = np.full(n, 1.0 / n)
    else:
        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum()

    # --- fetch historical returns ----------------------------------------
    ret_df = quant.returns_df(symbols, period=period, method="simple").dropna()

    # daily portfolio returns from history (for historical method + stats)
    port_hist = (ret_df * weights).sum(axis=1)

    # per-asset stats (for parametric methods)
    mean_daily = ret_df.mean().values
    cov_daily  = ret_df.cov().values

    # --- simulate --------------------------------------------------------
    if method == "historical":
        daily_sims = _simulate_historical(
            port_hist, horizon, n_sims, block_size, rng
        )
    elif method == "normal":
        daily_sims = _simulate_normal(
            mean_daily, cov_daily, weights, horizon, n_sims, rng
        )
    elif method == "t_dist":
        daily_sims = _simulate_t(
            mean_daily, cov_daily, weights, horizon, n_sims, df_t, rng
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Choose from: 'historical', 'normal', 't_dist'."
        )

    # --- build output ----------------------------------------------------
    paths, pct_df, final_values = _build_paths(daily_sims, initial_value)
    metrics = _calc_metrics(final_values, initial_value)

    return MonteCarloResult(
        paths=paths,
        percentiles=pct_df,
        final_values=final_values,
        metrics=metrics,
        symbols=symbols,
        weights=weights,
        initial_value=initial_value,
        horizon=horizon,
        n_sims=n_sims,
        method=method,
        period=period,
    )


def compare_methods(
    quant,
    symbols: list[str],
    weights: list[float] | np.ndarray | None = None,
    horizon: int = 252,
    n_sims: int = 1000,
    period: str = "3y",
    initial_value: float = 100_000.0,
) -> pd.DataFrame:
    """
    Run all three methods and return a comparison DataFrame.

    Useful for understanding how sensitive your risk estimates are
    to the distributional assumption.
    """
    methods  = ["historical", "normal", "t_dist"]
    rows = {}
    for m in methods:
        result = simulate(quant, symbols, weights,
                          horizon=horizon, n_sims=n_sims,
                          method=m, period=period,
                          initial_value=initial_value)
        rows[m] = result.metrics
    return pd.DataFrame(rows).T
