"""
factors.py — Fama-French factor exposure analysis.

Downloads factor data from Kenneth French's data library (no API key needed),
runs OLS regressions, and returns factor loadings, alpha, R², t-stats.

Models available
----------------
"ff3"  : Fama-French 3-factor  (Mkt-RF, SMB, HML)
"ff5"  : Fama-French 5-factor  (Mkt-RF, SMB, HML, RMW, CMA)
"mom"  : FF3 + Momentum        (Mkt-RF, SMB, HML, MOM)
"ff6"  : FF5 + Momentum        (all six factors)

Usage
-----
    from modules.factors import run, compare_models, factor_returns
    import modules.plots as plots

    result = run(quant, "AAPL", model="ff5", period="5y")
    print(result)

    # compare all models for one stock
    df = compare_models(quant, "AAPL", period="5y")

    # run for multiple stocks
    df = run_bulk(quant, ["AAPL","MSFT","NVDA"], model="ff3", period="3y")

    plots.factor_exposure(result).show()
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import pandas as pd
import requests
from scipy import stats

from classes.quant_analytics import QuantAnalytics


# ---------------------------------------------------------------------------
# Factor data download
# ---------------------------------------------------------------------------

_FF_URLS = {
    "ff3": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
    "ff5": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
}

_FF3_COLS  = ["Mkt-RF", "SMB", "HML", "RF"]
_FF5_COLS  = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
_MOM_COLS  = ["MOM"]

_FACTOR_LABELS = {
    "Mkt-RF": "Market excess return",
    "SMB":    "Size (Small Minus Big)",
    "HML":    "Value (High Minus Low)",
    "RMW":    "Profitability (Robust Minus Weak)",
    "CMA":    "Investment (Conservative Minus Aggressive)",
    "MOM":    "Momentum",
}

_MODEL_FACTORS = {
    "ff3": ["Mkt-RF", "SMB", "HML"],
    "ff5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "mom": ["Mkt-RF", "SMB", "HML", "MOM"],
    "ff6": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"],
}


@lru_cache(maxsize=8)
def _fetch_ff(url: str) -> pd.DataFrame:
    """Download and parse a French data library CSV zip. Cached in memory."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    zf   = zipfile.ZipFile(io.BytesIO(resp.content))
    name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
    raw  = zf.read(name).decode("utf-8", errors="replace")

    # skip header lines until we find the date column
    lines = raw.splitlines()
    # find header row (immediately before first data line)
    data_start = next(i for i, l in enumerate(lines) if l.strip()[:6].isdigit())
    header_start = max(0, data_start - 1)
    # find end of data block
    end = next(
        (i for i in range(data_start + 1, len(lines))
         if lines[i].strip() == "" or
         (lines[i].strip() and not lines[i].strip()[0].isdigit())),
        len(lines),
    )
    block = "\n".join(lines[header_start:end])
    df = pd.read_csv(io.StringIO(block), index_col=0)
    df.index.name = "Date"
    df.columns    = [c.strip() for c in df.columns]
    df.index      = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m%d")
    df = df.apply(pd.to_numeric, errors="coerce").dropna() / 100
    return df


def get_factors(model: str = "ff3") -> pd.DataFrame:
    """
    Return daily factor returns as a DataFrame.

    model : "ff3" | "ff5" | "mom" | "ff6"
    """
    if model in ("ff3", "mom_only"):
        base = _fetch_ff(_FF_URLS["ff3"])
    elif model == "ff5":
        base = _fetch_ff(_FF_URLS["ff5"])
    elif model in ("mom", "ff6"):
        base = _fetch_ff(_FF_URLS["ff5"])
        mom  = _fetch_ff(_FF_URLS["mom"])
        mom.columns = ["MOM"]
        base = base.join(mom, how="inner")
    else:
        raise ValueError(f"Unknown model '{model}'. Choose: ff3, ff5, mom, ff6")

    return base


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FactorResult:
    """
    Output of factors.run().

    Attributes
    ----------
    symbol      : ticker
    model       : factor model used
    period      : historical window
    alpha       : annualised Jensen's alpha
    alpha_pval  : p-value for alpha
    betas       : dict of factor name → beta loading
    t_stats     : dict of factor name → t-statistic
    p_values    : dict of factor name → p-value
    r_squared   : R² of the regression
    adj_r2      : adjusted R²
    n_obs       : number of observations used
    residuals   : regression residuals (idiosyncratic returns)
    """
    symbol:     str
    model:      str
    period:     str
    alpha:      float           # annualised
    alpha_pval: float
    betas:      dict[str, float]
    t_stats:    dict[str, float]
    p_values:   dict[str, float]
    r_squared:  float
    adj_r2:     float
    n_obs:      int
    residuals:  pd.Series

    def summary_df(self) -> pd.DataFrame:
        """Return a tidy DataFrame of factor loadings with stats."""
        rows = {}
        for f in self.betas:
            rows[f] = {
                "label":   _FACTOR_LABELS.get(f, f),
                "beta":    self.betas[f],
                "t_stat":  self.t_stats[f],
                "p_value": self.p_values[f],
                "significant": self.p_values[f] < 0.05,
            }
        df = pd.DataFrame(rows).T
        df.index.name = "factor"
        return df

    def __repr__(self) -> str:
        sig = lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        lines = [
            f"FactorResult — {self.symbol}  [{self.model.upper()}]  {self.period}",
            f"  Alpha (ann.)  : {self.alpha:+.3%} {sig(self.alpha_pval)}"
            f"  (p={self.alpha_pval:.3f})",
            f"  R²            : {self.r_squared:.3f}   "
            f"Adj-R²: {self.adj_r2:.3f}   N: {self.n_obs}",
            "",
            f"  {'Factor':<8} {'Beta':>8} {'t-stat':>8} {'p-val':>8}",
            f"  {'-'*38}",
        ]
        for f, b in self.betas.items():
            lines.append(
                f"  {f:<8} {b:>8.3f} {self.t_stats[f]:>8.2f}"
                f" {self.p_values[f]:>8.3f} {sig(self.p_values[f])}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core regression
# ---------------------------------------------------------------------------

def run(
    quant: QuantAnalytics,
    symbol: str,
    model: Literal["ff3", "ff5", "mom", "ff6"] = "ff5",
    period: str = "5y",
) -> FactorResult:
    """
    Run a Fama-French factor regression for *symbol*.

    Parameters
    ----------
    quant  : QuantAnalytics instance
    symbol : ticker
    model  : "ff3" | "ff5" | "mom" | "ff6"
    period : historical window for stock returns

    Returns
    -------
    FactorResult with alpha, betas, t-stats, p-values, R²
    """
    factor_names = _MODEL_FACTORS[model]

    # --- stock excess returns -------------------------------------------
    stock_ret = quant.returns(symbol, period=period, method="simple")

    # yfinance returns tz-aware index; FF data is tz-naive — normalise to date
    stock_ret.index = stock_ret.index.normalize().tz_localize(None)

    factors = get_factors(model)

    # align on common dates
    merged = stock_ret.to_frame("ret").join(factors, how="inner").dropna()
    if len(merged) < 60:
        raise ValueError(
            f"Only {len(merged)} overlapping observations — "
            "try a longer period or check the ticker."
        )

    excess_ret = merged["ret"] - merged["RF"]   # stock excess return
    X_factors  = merged[factor_names]

    # --- OLS via scipy ---------------------------------------------------
    X = np.column_stack([np.ones(len(X_factors)), X_factors.values])
    y = excess_ret.values

    result   = np.linalg.lstsq(X, y, rcond=None)
    coeffs   = result[0]                  # [alpha_daily, beta1, beta2, ...]
    y_hat    = X @ coeffs
    residuals = y - y_hat
    n, k     = len(y), X.shape[1]

    # standard errors
    sse      = float(residuals @ residuals)
    mse      = sse / (n - k)
    XtX_inv  = np.linalg.inv(X.T @ X)
    se       = np.sqrt(np.diag(XtX_inv) * mse)
    t_stats_all = coeffs / se
    p_vals_all  = [2 * (1 - stats.t.cdf(abs(t), df=n - k)) for t in t_stats_all]

    # R²
    ss_tot   = float(((y - y.mean()) ** 2).sum())
    r2       = 1 - sse / ss_tot
    adj_r2   = 1 - (1 - r2) * (n - 1) / (n - k)

    # annualise alpha (daily → annual)
    trading_days = quant.trading_days
    alpha_daily  = float(coeffs[0])
    alpha_ann    = (1 + alpha_daily) ** trading_days - 1

    betas   = {f: float(coeffs[i + 1]) for i, f in enumerate(factor_names)}
    t_stats = {f: float(t_stats_all[i + 1]) for i, f in enumerate(factor_names)}
    p_vals  = {f: float(p_vals_all[i + 1]) for i, f in enumerate(factor_names)}

    return FactorResult(
        symbol=symbol,
        model=model,
        period=period,
        alpha=alpha_ann,
        alpha_pval=float(p_vals_all[0]),
        betas=betas,
        t_stats=t_stats,
        p_values=p_vals,
        r_squared=float(r2),
        adj_r2=float(adj_r2),
        n_obs=n,
        residuals=pd.Series(residuals, index=merged.index, name="residual"),
    )


# ---------------------------------------------------------------------------
# Bulk and comparison helpers
# ---------------------------------------------------------------------------

def run_bulk(
    quant: QuantAnalytics,
    symbols: list[str],
    model: str = "ff5",
    period: str = "5y",
) -> pd.DataFrame:
    """
    Run factor regression for multiple symbols.

    Returns a DataFrame:
      rows    = symbols
      columns = alpha + one column per factor beta + r_squared
    """
    rows = {}
    for sym in symbols:
        try:
            r = run(quant, sym, model=model, period=period)
            row = {"alpha": r.alpha, "r_squared": r.r_squared}
            row.update(r.betas)
            rows[sym] = row
        except Exception as e:
            print(f"Skipping {sym}: {e}")

    return pd.DataFrame(rows).T


def compare_models(
    quant: QuantAnalytics,
    symbol: str,
    period: str = "5y",
) -> pd.DataFrame:
    """
    Run all four models for one symbol and compare alpha and R².

    Returns a DataFrame:
      rows    = model names
      columns = alpha, alpha_pval, r_squared, adj_r2
    """
    rows = {}
    for model in ["ff3", "ff5", "mom", "ff6"]:
        try:
            r = run(quant, symbol, model=model, period=period)
            rows[model.upper()] = {
                "alpha":      r.alpha,
                "alpha_pval": r.alpha_pval,
                "r_squared":  r.r_squared,
                "adj_r2":     r.adj_r2,
                "n_obs":      r.n_obs,
            }
        except Exception as e:
            print(f"Skipping {model}: {e}")

    return pd.DataFrame(rows).T


def rolling_betas(
    quant: QuantAnalytics,
    symbol: str,
    model: str = "ff3",
    period: str = "5y",
    window: int = 126,           # ~6 months of trading days
) -> pd.DataFrame:
    """
    Rolling factor betas over a *window*-day rolling window.

    Returns a DataFrame (dates × factors) showing how exposures
    shift over time — useful for detecting regime changes.
    """
    factor_names = _MODEL_FACTORS[model]
    stock_ret    = quant.returns(symbol, period=period, method="simple")
    stock_ret.index = stock_ret.index.normalize().tz_localize(None)
    factors      = get_factors(model)
    merged       = stock_ret.to_frame("ret").join(factors, how="inner").dropna()
    excess_ret   = (merged["ret"] - merged["RF"]).values
    F            = merged[factor_names].values
    dates        = merged.index
    n            = len(merged)

    results = []
    for i in range(window, n + 1):
        y  = excess_ret[i - window: i]
        X  = np.column_stack([np.ones(window), F[i - window: i]])
        try:
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            coeffs = np.full(len(factor_names) + 1, np.nan)
        row = {"alpha_daily": coeffs[0]}
        row.update({f: coeffs[j + 1] for j, f in enumerate(factor_names)})
        results.append((dates[i - 1], row))

    return pd.DataFrame(
        [r for _, r in results],
        index=[d for d, _ in results],
    )
