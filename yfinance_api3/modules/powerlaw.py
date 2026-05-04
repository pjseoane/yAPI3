"""
powerlaw.py — Bitcoin Power Law model.

Models Bitcoin's long-term price as a power law function of time:

    log(price) = a + b × log(days_since_genesis)

Corridor defined by percentile bands of OLS residuals — statistically
clean and updates automatically as new data arrives.

References
----------
Burger, H.C. (2019). "Bitcoin's natural long-term power-law corridor of growth"
Santostasi, G. — "Bitcoin Power Law Theory"

Usage
-----
    from yfinance_api3.modules.powerlaw import PowerLaw
    import yfinance_api3.modules.plots as plots

    pl = PowerLaw()
    pl.fit()

    print(pl.summary())
    print(pl.current_position())
    print(pl.fair_value("2026-12-31"))

    plots.powerlaw_chart(pl).show()
    plots.powerlaw_residuals(pl).show()
    plots.powerlaw_forecast(pl, years=4).show()
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

# Genesis block — January 3, 2009
GENESIS = date(2009, 1, 3)

# yfinance ticker for Bitcoin
BTC_TICKER = "BTC-USD"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _days_since_genesis(d) -> float:
    """Days elapsed since genesis block for a date or datetime."""
    if hasattr(d, "date"):
        d = d.date()
    return max((d - GENESIS).days, 1)


def fetch_btc(period: str = "max") -> pd.DataFrame:
    """
    Fetch Bitcoin OHLCV from yfinance.

    Returns DataFrame with columns:
      date, open, high, low, close, volume, days (since genesis)
    """
    import yfinance as yf
    raw = yf.download(BTC_TICKER, period=period, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("Failed to fetch BTC-USD data from yfinance")

    df = raw[["Open","High","Low","Close","Volume"]].copy()
    df.columns = ["open","high","low","close","volume"]
    df.index   = pd.to_datetime(df.index).normalize().tz_localize(None)
    df.index.name = "date"
    df = df.dropna(subset=["close"])
    df["days"] = df.index.map(lambda d: _days_since_genesis(d))
    return df


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PowerLawResult:
    """
    Output of PowerLaw.fit().

    Attributes
    ----------
    a            : intercept in log-log space (log(price) = a + b×log(days))
    b            : slope / power law exponent
    r_squared    : R² of the regression
    residuals    : log-price residuals from the fitted line
    percentiles  : dict of percentile → residual value (e.g. {10: -0.3, 90: 0.4})
    n_obs        : number of observations used
    data         : full DataFrame with model columns added
    fit_date     : date the model was fitted
    """
    a:           float
    b:           float
    r_squared:   float
    residuals:   pd.Series
    percentiles: dict[int, float]
    n_obs:       int
    data:        pd.DataFrame
    fit_date:    str

    def model_price(self, days: float) -> float:
        """Model price at N days since genesis."""
        return np.exp(self.a + self.b * np.log(days))

    def __repr__(self) -> str:
        return (
            f"PowerLawResult\n"
            f"  log(P) = {self.a:.4f} + {self.b:.4f} × log(days)\n"
            f"  R²     = {self.r_squared:.4f}\n"
            f"  N      = {self.n_obs:,}\n"
            f"  Fit    : {self.fit_date}"
        )


# ---------------------------------------------------------------------------
# PowerLaw
# ---------------------------------------------------------------------------

class PowerLaw:
    """
    Bitcoin Power Law model with percentile corridor.

    Steps
    -----
    1. Fetch BTC-USD price history
    2. Fit OLS: log(close) ~ a + b × log(days_since_genesis)
    3. Compute residuals
    4. Define corridor from percentile bands of residuals
    5. Analyse current position, fair value, cycle phase

    Example
    -------
        pl = PowerLaw()
        pl.fit()
        print(pl.summary())
        print(pl.current_position())
    """

    def __init__(
        self,
        corridor_low:  int = 10,    # lower band percentile
        corridor_high: int = 90,    # upper band percentile
        start_date:    str = None,  # trim data before this date (YYYY-MM-DD)
    ) -> None:
        self.corridor_low  = corridor_low
        self.corridor_high = corridor_high
        self.start_date    = start_date
        self.result:       PowerLawResult | None = None
        self._data:        pd.DataFrame | None   = None

    def fit(self, period: str = "max") -> "PowerLaw":
        """
        Fetch data and fit the power law model.

        Returns self for chaining.
        """
        df = fetch_btc(period=period)

        if self.start_date:
            df = df[df.index >= self.start_date]

        # log-log regression
        log_days  = np.log(df["days"].values.astype(float))
        log_price = np.log(df["close"].values.astype(float))

        # OLS: log_price = a + b × log_days
        X = np.column_stack([np.ones(len(log_days)), log_days])
        coeffs, *_ = np.linalg.lstsq(X, log_price, rcond=None)
        a, b = float(coeffs[0]), float(coeffs[1])

        # fitted values and residuals
        fitted    = a + b * log_days
        residuals = log_price - fitted

        # R²
        ss_res = float((residuals ** 2).sum())
        ss_tot = float(((log_price - log_price.mean()) ** 2).sum())
        r2     = 1 - ss_res / ss_tot

        # percentile bands
        pct_levels = [5, 10, 20, 25, 50, 75, 80, 90, 95]
        percentiles = {p: float(np.percentile(residuals, p)) for p in pct_levels}

        # enrich DataFrame
        df = df.copy()
        df["log_days"]      = log_days
        df["log_close"]     = log_price
        df["log_fitted"]    = fitted
        df["residual"]      = residuals
        df["model_price"]   = np.exp(fitted)
        df["floor_price"]   = np.exp(fitted + percentiles[corridor_low := self.corridor_low])
        df["ceiling_price"] = np.exp(fitted + percentiles[self.corridor_high])
        df["median_price"]  = np.exp(fitted + percentiles[50])
        # position within corridor (0 = floor, 1 = ceiling)
        corridor_width = percentiles[self.corridor_high] - percentiles[corridor_low]
        df["corridor_pct"]  = (residuals - percentiles[corridor_low]) / corridor_width

        self.result = PowerLawResult(
            a           = a,
            b           = b,
            r_squared   = r2,
            residuals   = pd.Series(residuals, index=df.index, name="residual"),
            percentiles = percentiles,
            n_obs       = len(df),
            data        = df,
            fit_date    = date.today().isoformat(),
        )
        self._data = df
        return self

    def _check_fitted(self) -> None:
        if self.result is None:
            raise RuntimeError("Call .fit() first")

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def fair_value(self, target_date: str | date) -> dict:
        """
        Model fair value at a future (or past) date.

        Returns
        -------
        dict:
          date         : target date
          days         : days since genesis
          model_price  : central model price
          floor_price  : lower corridor band
          median_price : 50th percentile
          ceiling_price: upper corridor band
        """
        self._check_fitted()
        r = self.result

        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date).date()

        days       = _days_since_genesis(target_date)
        log_fitted = r.a + r.b * np.log(days)

        return {
            "date":          target_date.isoformat(),
            "days":          days,
            "model_price":   round(np.exp(log_fitted), 2),
            "floor_price":   round(np.exp(log_fitted + r.percentiles[self.corridor_low]),  2),
            "median_price":  round(np.exp(log_fitted + r.percentiles[50]),                 2),
            "ceiling_price": round(np.exp(log_fitted + r.percentiles[self.corridor_high]), 2),
        }

    def current_position(self) -> dict:
        """
        Where is Bitcoin NOW relative to the power law corridor?

        Returns
        -------
        dict:
          price          : current price
          model_price    : central model price today
          floor_price    : lower corridor band
          ceiling_price  : upper corridor band
          corridor_pct   : 0=floor, 1=ceiling, >1=overbought, <0=oversold
          residual       : log-price deviation from model
          phase          : "oversold" | "floor" | "fair" | "ceiling" | "overbought"
          days_to_halving: approximate days to next halving
        """
        self._check_fitted()
        r      = self.result
        latest = self._data.iloc[-1]
        today  = fair_value = self.fair_value(date.today())

        corridor_pct = float(latest["corridor_pct"])

        if corridor_pct < 0:
            phase = "oversold"
        elif corridor_pct < 0.25:
            phase = "floor zone"
        elif corridor_pct < 0.75:
            phase = "fair value"
        elif corridor_pct <= 1.0:
            phase = "ceiling zone"
        else:
            phase = "overbought"

        # next halving (approximate — every 210,000 blocks ≈ 4 years)
        # last halving: April 20, 2024
        last_halving  = date(2024, 4, 20)
        next_halving  = last_halving + timedelta(days=4 * 365)
        days_to_halving = (next_halving - date.today()).days

        return {
            "date":            latest.name.date().isoformat(),
            "price":           round(float(latest["close"]), 2),
            "model_price":     today["model_price"],
            "floor_price":     today["floor_price"],
            "median_price":    today["median_price"],
            "ceiling_price":   today["ceiling_price"],
            "residual":        round(float(latest["residual"]), 4),
            "corridor_pct":    round(corridor_pct, 4),
            "phase":           phase,
            "days_to_halving": max(days_to_halving, 0),
        }

    def cycle_analysis(self) -> pd.DataFrame:
        """
        Identify bull/bear cycle peaks and troughs from residuals.

        Returns DataFrame with columns:
          date, price, residual, type (peak|trough), corridor_pct
        """
        self._check_fitted()
        from scipy.signal import find_peaks

        res    = self._data["residual"].values
        prices = self._data["close"].values
        dates  = self._data.index

        # find peaks (overbought) and troughs (oversold)
        peaks,  _ = find_peaks( res, distance=180, prominence=0.3)
        troughs,_ = find_peaks(-res, distance=180, prominence=0.3)

        rows = []
        for i in peaks:
            rows.append({"date": dates[i], "price": prices[i],
                         "residual": res[i],
                         "corridor_pct": self._data["corridor_pct"].iloc[i],
                         "type": "peak"})
        for i in troughs:
            rows.append({"date": dates[i], "price": prices[i],
                         "residual": res[i],
                         "corridor_pct": self._data["corridor_pct"].iloc[i],
                         "type": "trough"})

        return (pd.DataFrame(rows)
                .sort_values("date")
                .reset_index(drop=True))

    def forecast(
        self,
        years: int = 4,
        freq:  str = "ME",    # month-end
    ) -> pd.DataFrame:
        """
        Forward price projection at corridor bands.

        Parameters
        ----------
        years : number of years to project forward
        freq  : pandas date frequency for projection points

        Returns
        -------
        DataFrame: date, days, model_price, floor_price,
                   median_price, ceiling_price
        """
        self._check_fitted()
        end   = date.today() + timedelta(days=years * 365)
        dates = pd.date_range(date.today(), end, freq=freq)
        rows  = [self.fair_value(d.date()) for d in dates]
        return pd.DataFrame(rows)

    def summary(self) -> str:
        self._check_fitted()
        r   = self.result
        pos = self.current_position()
        fv1 = self.fair_value(date.today().replace(year=date.today().year + 1))
        fv4 = self.fair_value(date.today() + timedelta(days=4*365))

        lines = [
            "Bitcoin Power Law Model",
            "=" * 50,
            f"  Formula : log(P) = {r.a:.4f} + {r.b:.4f} × log(days)",
            f"  R²      : {r.r_squared:.4f}  (N={r.n_obs:,})",
            f"  Fit date: {r.fit_date}",
            "",
            "Current Position",
            "-" * 50,
            f"  Price       : ${pos['price']:>12,.0f}",
            f"  Floor       : ${pos['floor_price']:>12,.0f}",
            f"  Model       : ${pos['model_price']:>12,.0f}",
            f"  Median      : ${pos['median_price']:>12,.0f}",
            f"  Ceiling     : ${pos['ceiling_price']:>12,.0f}",
            f"  Position    : {pos['corridor_pct']:.1%} of corridor",
            f"  Phase       : {pos['phase']}",
            f"  Next halving: {pos['days_to_halving']} days",
            "",
            "Fair Value Projections",
            "-" * 50,
            f"  +1 year  floor/model/ceiling: "
            f"${fv1['floor_price']:>10,.0f} / "
            f"${fv1['model_price']:>10,.0f} / "
            f"${fv1['ceiling_price']:>10,.0f}",
            f"  +4 years floor/model/ceiling: "
            f"${fv4['floor_price']:>10,.0f} / "
            f"${fv4['model_price']:>10,.0f} / "
            f"${fv4['ceiling_price']:>10,.0f}",
        ]
        return "\n".join(lines)
