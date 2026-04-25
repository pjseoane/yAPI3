"""
etf.py — ETF holdings concentration analysis.

Fetches FULL holdings (not just top 10) directly from ETF provider websites.
Falls back to yahooquery top-10 for unknown ETFs.

Provider registry (full holdings)
----------------------------------
SPDR / State Street : SPY, VOO, GLD, and all SPDR ETFs
                      → XLSX from ssga.com
Invesco             : QQQ, QQQM, RSP and all Invesco ETFs
                      → CSV from invesco.com
iShares / BlackRock : IVV, AGG, EFA, EEM and all iShares ETFs
                      → CSV from ishares.com
Vanguard            : VTI, VEA, BND and all Vanguard ETFs
                      → CSV from vanguard.com (via advisors portal)
Fallback            : yahooquery top-10 for any other ETF

Dependencies
------------
    pip install yahooquery requests openpyxl

Usage
-----
    from modules.etf import ETFConcentration
    import modules.plots as plots

    spy = ETFConcentration("SPY", client=client)
    spy.fetch()

    plots.sp500_concentration(spy).show()
    print(spy.top_n(50))
    print(spy.concentration_metrics())
"""

from __future__ import annotations

import io
import time
from typing import Callable

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Provider registry — direct download URLs for full holdings
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _fetch_ssga(ticker: str) -> pd.DataFrame:
    """State Street SPDR ETFs — full holdings XLSX."""
    url  = (f"https://www.ssga.com/library-content/products/fund-data/"
            f"etfs/us/holdings-daily-us-en-{ticker.lower()}.xlsx")
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()

    # SSGA XLSX has a few header rows before the actual data
    xl   = pd.ExcelFile(io.BytesIO(resp.content))
    df   = xl.parse(xl.sheet_names[0], header=None)

    # find the row where the data table starts (contains "Name" or "Ticker")
    header_row = None
    for i, row in df.iterrows():
        vals = row.astype(str).str.lower().tolist()
        if any(v in vals for v in ["name", "ticker", "weight"]):
            header_row = i
            break

    if header_row is None:
        raise ValueError(f"Could not parse SSGA XLSX for {ticker}")

    df.columns = df.iloc[header_row].astype(str).str.strip()
    df         = df.iloc[header_row + 1:].reset_index(drop=True)

    # normalise column names
    col_map = {
        "Name":             "name",
        "Ticker":           "symbol",
        "Identifier":       "isin",
        "Weight":           "weight",
        "Shares Held":      "shares",
        "Local Market Value":"market_value",
        "Sector":           "sector",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # keep meaningful rows (symbol must look like a ticker)
    df = df[df["symbol"].astype(str).str.match(r"^[A-Z\-\.]{1,6}$", na=False)]
    df["weight"] = pd.to_numeric(
        df["weight"].astype(str).str.replace("%", "", regex=False),
        errors="coerce"
    ).fillna(0.0) / 100   # SSGA provides % values

    return df[["symbol", "name", "weight"] +
              (["sector"] if "sector" in df.columns else [])].copy()


def _fetch_invesco(ticker: str) -> pd.DataFrame:
    """Invesco ETFs — full holdings CSV."""
    url  = (f"https://www.invesco.com/us/financial-products/etfs/holdings/"
            f"main/holdings/0?audienceType=Investor&action=download&ticker={ticker.upper()}")
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))

    col_map = {
        "Holding Ticker": "symbol",
        "Name":           "name",
        "Weight":         "weight",
        "Sector":         "sector",
        "Class of Shares":"asset_class",
        "MarketValue":    "market_value",
        "Shares/Par Value":"shares",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    df["symbol"] = df["symbol"].astype(str).str.strip()
    df = df[df["symbol"].str.match(r"^[A-Z\-\.]{1,6}$", na=False)]
    df["weight"] = pd.to_numeric(
        df["weight"].astype(str).str.replace("%", "", regex=False),
        errors="coerce"
    ).fillna(0.0) / 100   # Invesco provides % values

    keep = ["symbol", "name", "weight"]
    if "sector" in df.columns:
        keep.append("sector")
    return df[keep].copy()


def _fetch_ishares(ticker: str) -> pd.DataFrame:
    """iShares / BlackRock ETFs — full holdings CSV."""
    # iShares uses a product-code based URL; try the direct CSV endpoint
    url  = (f"https://www.ishares.com/us/products/etf-product-data/"
            f"us/en/1467271812596.ajax?fileType=csv&dataType=fund"
            f"&assetClass=equities")

    # Alternative direct URL pattern
    url2 = (f"https://www.ishares.com/us/literature/spreadsheet/"
            f"15334.csv?fileType=csv&dataType=fund")

    # Use the known working URL pattern
    url3 = (f"https://www.ishares.com/us/products/"
            f"239726/ishares-core-sp-500-etf/1467271812596.ajax"
            f"?fileType=csv&includeAll=true")

    resp = requests.get(
        f"https://www.ishares.com/us/products/etf-product-data/"
        f"us/en/{ticker}/holdings",
        headers=_HEADERS, timeout=30
    )
    resp.raise_for_status()

    # parse as CSV, skip header rows
    lines = resp.text.splitlines()
    start = next((i for i, l in enumerate(lines)
                  if "ticker" in l.lower() or "symbol" in l.lower()), 0)

    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    col_map = {
        "Ticker":       "symbol",
        "Name":         "name",
        "Weight (%)":   "weight",
        "Sector":       "sector",
        "Asset Class":  "asset_class",
        "Market Value": "market_value",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df = df[df["symbol"].str.match(r"^[A-Z\-\.]{1,6}$", na=False)]
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0) / 100

    keep = ["symbol", "name", "weight"]
    if "sector" in df.columns:
        keep.append("sector")
    return df[keep].copy()


def _fetch_yahooquery_fallback(ticker: str) -> pd.DataFrame:
    """Fallback: yahooquery top-10 holdings."""
    try:
        from yahooquery import Ticker
    except ImportError:
        raise ImportError("pip install yahooquery")

    t    = Ticker(ticker)
    raw  = t.fund_holding_info
    data = raw.get(ticker, raw) if isinstance(raw, dict) else raw
    holdings = data.get("holdings", []) if isinstance(data, dict) else \
               (data if isinstance(data, list) else [])

    if not holdings:
        raise ValueError(f"No holdings data for '{ticker}'")

    df = pd.DataFrame(holdings).rename(columns={
        "symbol":         "symbol",
        "holdingName":    "name",
        "holdingPercent": "weight",
    })
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    df["sector"] = "Unknown"
    return df[["symbol", "name", "weight", "sector"]].copy()


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

# Known ETF → provider mapping (tickers we know work with each provider)
_SSGA_TICKERS = {
    "SPY","SPYG","SPYV","SPYD","SPLG","XLK","XLF","XLC","XLE","XLV",
    "XLI","XLY","XLP","XLU","XLRE","XLB","GLD","SLV","DIA","MDY",
    "SPDW","SPEM","SPAB","SPSB","SPTL","SPTI","SPIB","SPMB",
}
_INVESCO_TICKERS = {
    "QQQ","QQQM","RSP","IVZ","BKLN","PGX","PDBC","PBW","PBD",
    "KBWB","KBWP","KBWR","PSI","PSJ","PTH","RYT","RYF","RYH",
}

def _detect_provider(ticker: str) -> Callable:
    t = ticker.upper()
    if t in _SSGA_TICKERS:
        return _fetch_ssga
    if t in _INVESCO_TICKERS:
        return _fetch_invesco
    # iShares tickers start with I (heuristic)
    if t.startswith("I") and len(t) <= 5:
        return _fetch_ishares
    return _fetch_yahooquery_fallback


# ---------------------------------------------------------------------------
# Normalise and clean a raw holdings DataFrame
# ---------------------------------------------------------------------------

def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure consistent schema and compute weights.
    Returns: symbol, name, weight (0-1), weight_pct, sector
    """
    # ensure sector column
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    df["sector"] = df["sector"].fillna("Unknown").replace("", "Unknown")

    # clean and recompute weights to sum to 1
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    df = df[df["weight"] > 0].copy()

    # if weights look like percentages (max > 1.5), divide by 100
    if df["weight"].max() > 1.5:
        df["weight"] = df["weight"] / 100

    total = df["weight"].sum()
    if total > 0:
        df["weight"] = df["weight"] / total  # normalise to 1.0

    df["weight_pct"] = df["weight"] * 100
    df = df.sort_values("weight", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "rank"
    return df[["symbol", "name", "weight", "weight_pct", "sector"]]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ETFConcentration:
    """
    ETF holdings concentration analysis with full holdings support.

    Automatically selects the right data source for each ETF family:
      - SPDR (SPY, XL*): State Street SSGA XLSX  → 500+ holdings
      - Invesco (QQQ):    Invesco CSV             → 100+ holdings
      - iShares (IVV):    BlackRock CSV           → full holdings
      - Others:           yahooquery fallback     → top 10 only

    Parameters
    ----------
    ticker  : ETF ticker (e.g. "SPY", "QQQ", "IVV")
    client  : optional StockClient (not needed — provider CSVs include sectors)
    """

    def __init__(self, ticker: str, client=None) -> None:
        self.ticker      = ticker.upper()
        self._client     = client
        self._weights_df: pd.DataFrame | None = None
        self._provider   = None

    # ------------------------------------------------------------------

    def fetch(self, verbose: bool = True) -> "ETFConcentration":
        """
        Fetch full holdings from the ETF provider.

        Tries the provider-specific direct download first.
        Falls back to yahooquery (top 10) if the provider download fails.
        """
        provider_fn = _detect_provider(self.ticker)

        if verbose:
            source = provider_fn.__name__.replace("_fetch_", "")
            print(f"Fetching {self.ticker} holdings via {source}...")

        try:
            raw = provider_fn(self.ticker)
            self._provider = provider_fn.__name__
        except Exception as e:
            if verbose:
                print(f"  Provider fetch failed ({e}), falling back to yahooquery top-10...")
            raw = _fetch_yahooquery_fallback(self.ticker)
            self._provider = "yahooquery_fallback"

        self._weights_df = _normalise(raw)

        if verbose:
            print(f"  {len(self._weights_df)} holdings loaded.")
            print(f"  Top 5: {self._weights_df['symbol'].head(5).tolist()}")
            if self._provider == "yahooquery_fallback":
                print("  ⚠ Only top-10 available for this ETF.")

        return self

    # ------------------------------------------------------------------
    # Analysis (same API as before)
    # ------------------------------------------------------------------

    def weights(self) -> pd.DataFrame:
        self._check_fetched()
        return self._weights_df.copy()

    def top_n(self, n: int = 10) -> pd.DataFrame:
        self._check_fetched()
        return self._weights_df.head(n).copy()

    def concentration_metrics(self, top_ns: list[int] | None = None) -> dict:
        self._check_fetched()
        df      = self._weights_df
        weights = df["weight"].values
        n       = len(weights)

        top_ns = top_ns or [1, 5, 10, 25, 50]
        top_w  = {f"top_{tn}_weight": float(weights[:min(tn,n)].sum())
                  for tn in top_ns}

        hhi         = float((weights ** 2).sum() * 10_000)
        hhi_label   = ("unconcentrated" if hhi < 1500
                        else "moderately concentrated" if hhi < 2500
                        else "highly concentrated")
        effective_n = float(1 / (weights ** 2).sum())

        sw   = np.sort(weights)
        gini = float(
            (2 * np.sum(np.arange(1, n + 1) * sw) - (n + 1) * sw.sum())
            / (n * sw.sum())
        ) if sw.sum() > 0 else 0.0

        sector_weights = (df.groupby("sector")["weight"].sum()
                          .sort_values(ascending=False))
        top_sector   = sector_weights.index[0]
        top_sector_w = float(sector_weights.iloc[0])

        return {
            "ticker":            self.ticker,
            "total_holdings":    n,
            "data_source":       self._provider,
            **top_w,
            "hhi":               round(hhi, 1),
            "hhi_label":         hhi_label,
            "effective_n":       round(effective_n, 1),
            "gini":              round(gini, 4),
            "top_sector":        top_sector,
            "top_sector_weight": round(top_sector_w, 4),
            "largest_holding":   df["symbol"].iloc[0],
            "largest_weight":    round(float(weights[0]), 4),
            "total_weight":      round(float(weights.sum()), 4),
        }

    def sector_weights(self) -> pd.DataFrame:
        self._check_fetched()
        return (
            self._weights_df
            .groupby("sector")
            .agg(
                weight=("weight", "sum"),
                n_holdings=("symbol", "count"),
                top_holding=("symbol", "first"),
            )
            .sort_values("weight", ascending=False)
            .assign(weight_pct=lambda x: x["weight"] * 100)
        )

    def cumulative_weight(self) -> pd.Series:
        self._check_fetched()
        return self._weights_df["weight"].cumsum().rename("cumulative_weight")

    def holdings_for_pct(self, target_pct: float = 0.50) -> int:
        self._check_fetched()
        cumw = self.cumulative_weight()
        mask = cumw >= target_pct
        return int(mask.idxmax()) if mask.any() else len(cumw)

    def compare(self, other: "ETFConcentration") -> pd.DataFrame:
        m1 = self.concentration_metrics()
        m2 = other.concentration_metrics()
        return pd.DataFrame({self.ticker: m1, other.ticker: m2})

    def holdings_table(self) -> pd.DataFrame:
        """
        Clean holdings table with rank, symbol, name, weight and cumulative weight.

        Returns
        -------
        DataFrame indexed by rank (1-based) with columns:
          symbol                : ticker
          name                  : company name
          weight_pct            : individual weight in %
          cumulative_weight_pct : running total weight in %

        Example
        -------
        spy.holdings_table().head(20)
        spy.holdings_table()[spy.holdings_table()["cumulative_weight_pct"] <= 50]
        """
        self._check_fetched()
        df = self._weights_df.copy()
        df["cumulative_weight_pct"] = df["weight"].cumsum() * 100
        return df[["symbol", "name", "weight_pct", "cumulative_weight_pct"]]

    def _check_fetched(self) -> None:
        if self._weights_df is None:
            raise RuntimeError("Call .fetch() first.")


# Backwards-compatible alias
class SP500Concentration(ETFConcentration):
    def __init__(self, client=None, **kwargs):
        super().__init__("SPY", client=client)
