"""
portfolio.py — portfolio optimisation built on QuantAnalytics.

All optimisers return a PortfolioResult dataclass so results are
consistent and easy to pass to plots.efficient_frontier().

Available optimisers
--------------------
max_sharpe(quant, symbols, ...)      — maximise Sharpe ratio
min_variance(quant, symbols, ...)    — minimise portfolio volatility
risk_parity(quant, symbols, ...)     — equal risk contribution
equal_weight(quant, symbols, ...)    — 1/N baseline
efficient_frontier(quant, symbols, ...)— full curve of optimal portfolios

All accept:
  period          : historical window for covariance / returns (default "3y")
  risk_free_rate  : annual risk-free rate (default 0.05)
  allow_short     : allow negative weights (default False)
  weight_bounds   : (min, max) per asset weight (default (0, 1))
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from yfinance_api3.classes.quant_analytics import QuantAnalytics


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PortfolioResult:
    """
    Output of any optimiser.

    Attributes
    ----------
    symbols         : list of tickers
    weights         : optimised weights (sum to 1)
    expected_return : annualised expected return
    volatility      : annualised portfolio volatility
    sharpe_ratio    : (expected_return - rf) / volatility
    risk_free_rate  : risk-free rate used
    strategy        : name of the optimisation strategy
    period          : historical window used
    weights_df      : weights as a labelled Series (handy for display)
    """
    symbols:         list[str]
    weights:         np.ndarray
    expected_return: float
    volatility:      float
    sharpe_ratio:    float
    risk_free_rate:  float
    strategy:        str
    period:          str
    risk_contributions: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def weights_df(self) -> pd.Series:
        return pd.Series(self.weights, index=self.symbols, name="weight").sort_values(ascending=False)

    def summary(self) -> dict:
        return {
            "strategy":         self.strategy,
            "period":           self.period,
            "expected_return":  f"{self.expected_return:.2%}",
            "volatility":       f"{self.volatility:.2%}",
            "sharpe_ratio":     f"{self.sharpe_ratio:.3f}",
            "weights":          {s: f"{w:.2%}" for s, w in zip(self.symbols, self.weights)},
        }

    def __repr__(self) -> str:
        lines = [
            f"PortfolioResult — {self.strategy}",
            f"  Return     : {self.expected_return:.2%}",
            f"  Volatility : {self.volatility:.2%}",
            f"  Sharpe     : {self.sharpe_ratio:.3f}",
            "  Weights:",
        ] + [f"    {s}: {w:.2%}" for s, w in self.weights_df.items()]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_inputs(
    quant: QuantAnalytics,
    symbols: list[str],
    period: str,
    risk_free_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean_returns, cov_matrix) as numpy arrays."""
    cov = quant.covariance_matrix(symbols, period=period, annualise=True)
    rets_df = quant.returns_df(symbols, period=period).dropna()
    mean_returns = rets_df.mean().values * quant.trading_days
    return mean_returns, cov.values


def _portfolio_stats(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float,
) -> tuple[float, float, float]:
    """Return (expected_return, volatility, sharpe)."""
    ret = float(weights @ mean_returns)
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def _risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Marginal risk contribution of each asset (sums to portfolio vol)."""
    port_vol = np.sqrt(weights @ cov @ weights)
    marginal  = cov @ weights
    return weights * marginal / port_vol


def _build_result(
    weights: np.ndarray,
    symbols: list[str],
    mean_returns: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float,
    strategy: str,
    period: str,
) -> PortfolioResult:
    ret, vol, sharpe = _portfolio_stats(weights, mean_returns, cov, risk_free_rate)
    rc = _risk_contributions(weights, cov)
    return PortfolioResult(
        symbols=symbols,
        weights=weights,
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        risk_free_rate=risk_free_rate,
        strategy=strategy,
        period=period,
        risk_contributions=rc,
    )


def _default_bounds(n: int, allow_short: bool, weight_bounds: tuple) -> list:
    lo = -1.0 if allow_short else weight_bounds[0]
    hi = weight_bounds[1]
    return [(lo, hi)] * n


def _sum_to_one() -> dict:
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


# ---------------------------------------------------------------------------
# 1. Equal weight (baseline)
# ---------------------------------------------------------------------------

def equal_weight(
    quant: QuantAnalytics,
    symbols: list[str],
    period: str = "3y",
    risk_free_rate: float = 0.05,
) -> PortfolioResult:
    """1/N portfolio — equal allocation to each asset."""
    n = len(symbols)
    weights = np.full(n, 1.0 / n)
    mean_returns, cov = _get_inputs(quant, symbols, period, risk_free_rate)
    return _build_result(weights, symbols, mean_returns, cov,
                         risk_free_rate, "Equal weight", period)


# ---------------------------------------------------------------------------
# 2. Minimum variance
# ---------------------------------------------------------------------------

def min_variance(
    quant: QuantAnalytics,
    symbols: list[str],
    period: str = "3y",
    risk_free_rate: float = 0.05,
    allow_short: bool = False,
    weight_bounds: tuple = (0.0, 1.0),
) -> PortfolioResult:
    """
    Minimum variance portfolio — minimise portfolio volatility
    subject to weights summing to 1.
    """
    mean_returns, cov = _get_inputs(quant, symbols, period, risk_free_rate)
    n = len(symbols)
    w0 = np.full(n, 1.0 / n)

    result = minimize(
        fun=lambda w: float(w @ cov @ w),
        x0=w0,
        method="SLSQP",
        bounds=_default_bounds(n, allow_short, weight_bounds),
        constraints=[_sum_to_one()],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Min variance optimisation failed: {result.message}")

    weights = result.x
    return _build_result(weights, symbols, mean_returns, cov,
                         risk_free_rate, "Min variance", period)


# ---------------------------------------------------------------------------
# 3. Maximum Sharpe
# ---------------------------------------------------------------------------

def max_sharpe(
    quant: QuantAnalytics,
    symbols: list[str],
    period: str = "3y",
    risk_free_rate: float = 0.05,
    allow_short: bool = False,
    weight_bounds: tuple = (0.0, 1.0),
) -> PortfolioResult:
    """
    Maximum Sharpe ratio portfolio — tangency portfolio on the efficient frontier.
    Uses negative Sharpe as the objective so scipy can minimise it.
    """
    mean_returns, cov = _get_inputs(quant, symbols, period, risk_free_rate)
    n = len(symbols)
    w0 = np.full(n, 1.0 / n)

    def neg_sharpe(w):
        ret, vol, _ = _portfolio_stats(w, mean_returns, cov, risk_free_rate)
        return -ret / vol if vol > 1e-10 else 1e10

    result = minimize(
        fun=neg_sharpe,
        x0=w0,
        method="SLSQP",
        bounds=_default_bounds(n, allow_short, weight_bounds),
        constraints=[_sum_to_one()],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Max Sharpe optimisation failed: {result.message}")

    weights = result.x
    return _build_result(weights, symbols, mean_returns, cov,
                         risk_free_rate, "Max Sharpe", period)


# ---------------------------------------------------------------------------
# 4. Risk parity
# ---------------------------------------------------------------------------

def risk_parity(
    quant: QuantAnalytics,
    symbols: list[str],
    period: str = "3y",
    risk_free_rate: float = 0.05,
    weight_bounds: tuple = (0.001, 1.0),
) -> PortfolioResult:
    """
    Risk parity portfolio — each asset contributes equally to total portfolio risk.

    Uses a change-of-variables trick: w_i = y_i² / Σy_j²
    This automatically satisfies w_i > 0 and Σw_i = 1, turning the
    constrained problem into an unconstrained one that SLSQP handles cleanly.
    """
    mean_returns, cov = _get_inputs(quant, symbols, period, risk_free_rate)
    n = len(symbols)

    def _weights_from_y(y: np.ndarray) -> np.ndarray:
        """Map unconstrained y -> valid portfolio weights."""
        w = y ** 2
        return w / w.sum()

    def objective(y: np.ndarray) -> float:
        w  = _weights_from_y(y)
        rc = _risk_contributions(w, cov)
        target = rc.sum() / n          # equal share of total risk
        return float(np.sum((rc - target) ** 2))

    # multiple random restarts to avoid local minima
    best_result = None
    rng = np.random.default_rng(seed=42)

    for _ in range(10):
        y0 = rng.uniform(0.1, 1.0, n)
        res = minimize(
            fun=objective,
            x0=y0,
            method="SLSQP",
            options={"ftol": 1e-14, "maxiter": 5000},
        )
        if best_result is None or res.fun < best_result.fun:
            best_result = res

    weights = _weights_from_y(best_result.x)
    weights = np.clip(weights, 0, None)
    weights /= weights.sum()

    return _build_result(weights, symbols, mean_returns, cov,
                         risk_free_rate, "Risk parity", period)


# ---------------------------------------------------------------------------
# 5. Efficient frontier
# ---------------------------------------------------------------------------

@dataclass
class EfficientFrontier:
    """
    Full efficient frontier — collection of optimal portfolios
    spanning the return/volatility space.

    Attributes
    ----------
    points      : DataFrame with columns [volatility, return, sharpe, weights...]
    max_sharpe  : PortfolioResult for the tangency portfolio
    min_variance: PortfolioResult for the minimum variance portfolio
    symbols     : list of tickers
    period      : historical window used
    """
    points:       pd.DataFrame
    max_sharpe:   PortfolioResult
    min_variance: PortfolioResult
    symbols:      list[str]
    period:       str


def efficient_frontier(
    quant: QuantAnalytics,
    symbols: list[str],
    period: str = "3y",
    risk_free_rate: float = 0.05,
    n_points: int = 60,
    allow_short: bool = False,
    weight_bounds: tuple = (0.0, 1.0),
) -> EfficientFrontier:
    """
    Compute the efficient frontier by solving for minimum variance at each
    target return level between the min-variance return and the max return.

    Returns an EfficientFrontier object containing:
      - .points       : DataFrame of all frontier portfolios
      - .max_sharpe   : tangency portfolio
      - .min_variance : minimum variance portfolio
    """
    mean_returns, cov = _get_inputs(quant, symbols, period, risk_free_rate)
    n = len(symbols)
    bounds = _default_bounds(n, allow_short, weight_bounds)
    w0 = np.full(n, 1.0 / n)

    # anchor portfolios
    mv = min_variance(quant, symbols, period, risk_free_rate, allow_short, weight_bounds)
    ms = max_sharpe(quant, symbols, period, risk_free_rate, allow_short, weight_bounds)

    # return range: min_variance return → max individual asset return
    ret_min = mv.expected_return
    ret_max = float(mean_returns.max())
    target_returns = np.linspace(ret_min, ret_max, n_points)

    rows = []
    prev_weights = w0.copy()

    for target_ret in target_returns:
        constraints = [
            _sum_to_one(),
            {"type": "eq", "fun": lambda w, r=target_ret: float(w @ mean_returns) - r},
        ]
        res = minimize(
            fun=lambda w: float(w @ cov @ w),
            x0=prev_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if not res.success:
            continue

        w = res.x
        prev_weights = w
        ret, vol, sharpe = _portfolio_stats(w, mean_returns, cov, risk_free_rate)
        row = {"volatility": vol, "return": ret, "sharpe": sharpe}
        row.update({s: float(wi) for s, wi in zip(symbols, w)})
        rows.append(row)

    points = pd.DataFrame(rows)

    return EfficientFrontier(
        points=points,
        max_sharpe=ms,
        min_variance=mv,
        symbols=symbols,
        period=period,
    )


# ---------------------------------------------------------------------------
# Convenience: run all strategies and compare
# ---------------------------------------------------------------------------

def compare_strategies(
    quant: QuantAnalytics,
    symbols: list[str],
    period: str = "3y",
    risk_free_rate: float = 0.05,
) -> pd.DataFrame:
    """
    Run all four strategies and return a comparison DataFrame.

    Rows    : strategy names
    Columns : expected_return, volatility, sharpe_ratio, + one column per symbol
    """
    strategies = {
        "Equal weight": equal_weight(quant, symbols, period, risk_free_rate),
        "Min variance":  min_variance(quant, symbols, period, risk_free_rate),
        "Max Sharpe":    max_sharpe(quant, symbols, period, risk_free_rate),
        "Risk parity":   risk_parity(quant, symbols, period, risk_free_rate),
    }
    rows = {}
    for name, result in strategies.items():
        row = {
            "expected_return": result.expected_return,
            "volatility":      result.volatility,
            "sharpe_ratio":    result.sharpe_ratio,
        }
        row.update({s: float(w) for s, w in zip(result.symbols, result.weights)})
        rows[name] = row

    return pd.DataFrame(rows).T

