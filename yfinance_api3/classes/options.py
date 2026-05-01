"""
options.py — Options data analysis built on StockClient.

Fetches options chains via yfinance (no API key needed) and provides
analysis tools for expiration, strikes, volume, open interest, IV,
    put/call ratios, max pain, Greeks, and gamma exposure.

Usage
-----
    from yfinance_api3.classes.options import OptionsAnalyzer
    import yfinance_api3.modules.plots as plots

    opt = OptionsAnalyzer(client, "AAPL")

    print(opt.expiries())               # available expiration dates
    df  = opt.chain("2025-06-20")       # calls + puts for one expiry
    df  = opt.chain_all()               # all expiries combined
    pcr = opt.put_call_ratio()          # PCR per expiry
    mp  = opt.max_pain("2025-06-20")    # max pain strike
    gex = opt.gamma_exposure("2025-06-20")  # GEX by strike

    plots.options_chain(opt, "2025-06-20").show()
    plots.options_surface(opt).show()
    plots.options_oi_profile(opt, "2025-06-20").show()
    plots.options_put_call(opt).show()
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from yfinance_api3.classes.stock_client import StockClient


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OptionsAnalyzer:
    """
    Options data analysis for a single underlying symbol.

    All data is fetched via StockClient (yfinance) and cached
    automatically via the TTL cache.

    Parameters
    ----------
    client  : StockClient instance
    symbol  : underlying ticker (e.g. "AAPL", "SPY")

    Notes on Greeks
    ---------------
    Greeks are taken from yfinance when available, with a Black-Scholes
    fallback for missing values.
    """

    def __init__(self, client: StockClient, symbol: str) -> None:
        self.client = client
        self.symbol = symbol.upper()
        self._chain_cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Expiries
    # ------------------------------------------------------------------

    def expiries(self) -> list[str]:
        """
        Return all available option expiration dates as ISO strings.

        Example
        -------
        opt.expiries()
        → ['2025-05-02', '2025-05-09', '2025-05-16', ...]
        """
        return self.client.get_options_expiries(self.symbol)

    def nearest_expiry(self, n: int = 0) -> str:
        """
        Return the Nth nearest expiry (0 = front month, 1 = next, etc.)

        Example
        -------
        opt.nearest_expiry(0)   # front month
        opt.nearest_expiry(3)   # 4th expiry out
        """
        dates = self.expiries()
        if not dates:
            raise ValueError(f"No options data for {self.symbol}")
        return dates[min(n, len(dates) - 1)]

    # ------------------------------------------------------------------
    # Chain data
    # ------------------------------------------------------------------

    def chain(
        self,
        expiry: str | None = None,
        option_type: str = "both",     # "calls" | "puts" | "both"
        min_volume: int = 0,
        min_oi: int = 0,
    ) -> pd.DataFrame:
        """
        Return the options chain for a specific expiry as a clean DataFrame.

        Parameters
        ----------
        expiry      : expiration date string (default = front month)
                      use opt.expiries() to see available dates
        option_type : "calls", "puts", or "both" (default)
        min_volume  : filter rows with volume below this threshold
        min_oi      : filter rows with open interest below this threshold

        Returns
        -------
        DataFrame with columns:
          expiry, type, strike, bid, ask, last_price, volume,
          open_interest, implied_volatility, in_the_money,
          intrinsic_value (approx), time_value (approx)

        Example
        -------
        df = opt.chain("2025-06-20")
        calls = opt.chain("2025-06-20", option_type="calls")
        active = opt.chain("2025-06-20", min_volume=100, min_oi=500)
        """
        if expiry is None:
            expiry = self.nearest_expiry(0)

        if expiry in self._chain_cache:
            df = self._chain_cache[expiry]
        else:
            raw  = self.client.get_options_chain(self.symbol, expiry)
            calls = pd.DataFrame(raw.get("calls", []))
            puts  = pd.DataFrame(raw.get("puts",  []))

            if not calls.empty:
                calls["type"] = "call"
            if not puts.empty:
                puts["type"]  = "put"

            df = pd.concat([calls, puts], ignore_index=True)
            df["expiry"] = expiry
            df = self._normalise(df)
            self._chain_cache[expiry] = df

        # filter by type
        if option_type == "calls":
            df = df[df["type"] == "call"]
        elif option_type == "puts":
            df = df[df["type"] == "put"]

        # apply filters
        if min_volume > 0 and "volume" in df.columns:
            df = df[df["volume"] >= min_volume]
        if min_oi > 0 and "open_interest" in df.columns:
            df = df[df["open_interest"] >= min_oi]

        return df.reset_index(drop=True)

    def chain_all(
        self,
        max_expiries: int | None = None,
        option_type: str = "both",
        min_oi: int = 0,
    ) -> pd.DataFrame:
        """
        Return combined options chain across all (or first N) expiries.

        Parameters
        ----------
        max_expiries : limit to first N expiries (default all)
        option_type  : "calls", "puts", or "both"
        min_oi       : filter by minimum open interest

        Example
        -------
        df = opt.chain_all(max_expiries=6)   # next 6 expiries
        """
        dates = self.expiries()
        if max_expiries:
            dates = dates[:max_expiries]

        frames = []
        for exp in dates:
            try:
                frames.append(self.chain(exp, option_type=option_type,
                                         min_oi=min_oi))
            except Exception as e:
                print(f"Skipping {exp}: {e}")

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def summary(self, expiry: str | None = None) -> dict:
        """
        Key metrics for a single expiry.

        Returns
        -------
        dict with:
          expiry          : expiration date
          days_to_expiry  : calendar days until expiry
          n_strikes       : number of unique strikes
          total_call_oi   : total call open interest
          total_put_oi    : total put open interest
          total_call_vol  : total call volume today
          total_put_vol   : total put volume today
          put_call_ratio_oi  : put OI / call OI
          put_call_ratio_vol : put vol / call vol
          max_pain_strike : strike where option writers lose least
          max_oi_call_strike : strike with most call OI (resistance)
          max_oi_put_strike  : strike with most put OI (support)
          avg_call_iv     : average implied volatility for calls
          avg_put_iv      : average implied volatility for puts
          iv_skew         : avg_put_iv - avg_call_iv
        """
        if expiry is None:
            expiry = self.nearest_expiry(0)

        df    = self.chain(expiry)
        calls = df[df["type"] == "call"]
        puts  = df[df["type"] == "put"]

        # days to expiry
        from datetime import date
        dte = (pd.to_datetime(expiry).date() - date.today()).days

        # max OI strikes
        max_call_oi_strike = (calls.loc[calls["open_interest"].idxmax(), "strike"]
                              if not calls.empty else None)
        max_put_oi_strike  = (puts.loc[puts["open_interest"].idxmax(), "strike"]
                              if not puts.empty else None)

        # IV
        avg_call_iv = float(calls["implied_volatility"].mean()) if not calls.empty else 0.0
        avg_put_iv  = float(puts["implied_volatility"].mean())  if not puts.empty else 0.0

        total_call_oi  = int(calls["open_interest"].sum())
        total_put_oi   = int(puts["open_interest"].sum())
        total_call_vol = int(calls["volume"].sum())
        total_put_vol  = int(puts["volume"].sum())

        return {
            "symbol":              self.symbol,
            "expiry":              expiry,
            "days_to_expiry":      dte,
            "n_strikes":           int(df["strike"].nunique()),
            "total_call_oi":       total_call_oi,
            "total_put_oi":        total_put_oi,
            "total_call_vol":      total_call_vol,
            "total_put_vol":       total_put_vol,
            "put_call_ratio_oi":   round(total_put_oi  / total_call_oi,  3)
                                   if total_call_oi  > 0 else None,
            "put_call_ratio_vol":  round(total_put_vol / total_call_vol, 3)
                                   if total_call_vol > 0 else None,
            "max_pain_strike":     self.max_pain(expiry),
            "max_oi_call_strike":  max_call_oi_strike,
            "max_oi_put_strike":   max_put_oi_strike,
            "avg_call_iv":         round(avg_call_iv, 4),
            "avg_put_iv":          round(avg_put_iv,  4),
            "iv_skew":             round(avg_put_iv - avg_call_iv, 4),
        }

    def put_call_ratio(
        self,
        by: str = "oi",   # "oi" | "volume"
    ) -> pd.DataFrame:
        """
        Put/Call ratio across all expiries.

        Parameters
        ----------
        by : "oi" (open interest) or "volume"

        Returns
        -------
        DataFrame with columns: expiry, days_to_expiry, call_oi/vol,
        put_oi/vol, put_call_ratio — sorted by expiry ascending.

        A PCR > 1.0 signals more puts than calls (bearish sentiment).
        A PCR < 0.7 signals more calls (bullish / complacency).

        Example
        -------
        opt.put_call_ratio(by="oi")
        opt.put_call_ratio(by="volume")
        """
        from datetime import date
        today  = date.today()
        rows   = []

        for expiry in self.expiries():
            try:
                df    = self.chain(expiry)
                calls = df[df["type"] == "call"]
                puts  = df[df["type"] == "put"]

                col = "open_interest" if by == "oi" else "volume"
                call_val = int(calls[col].sum())
                put_val  = int(puts[col].sum())
                dte = (pd.to_datetime(expiry).date() - today).days

                rows.append({
                    "expiry":           expiry,
                    "days_to_expiry":   dte,
                    f"call_{by}":       call_val,
                    f"put_{by}":        put_val,
                    "put_call_ratio":   round(put_val / call_val, 3)
                                        if call_val > 0 else None,
                })
            except Exception:
                pass

        return pd.DataFrame(rows).sort_values("expiry").reset_index(drop=True)

    def max_pain(self, expiry: str | None = None) -> float:
        """
        Calculate the max pain strike for a given expiry.

        Max pain = the strike price at which the total dollar value of
        expiring options (calls + puts) is minimised — i.e. where option
        buyers lose the most and writers lose the least.

        Theory: underlying price tends to gravitate toward max pain near
        expiry as market makers hedge their books.

        Returns
        -------
        float : the max pain strike price

        Example
        -------
        opt.max_pain("2025-06-20")
        """
        if expiry is None:
            expiry = self.nearest_expiry(0)

        df     = self.chain(expiry)
        calls  = df[df["type"] == "call"][["strike", "open_interest"]]
        puts   = df[df["type"] == "put"][["strike",  "open_interest"]]
        strikes = sorted(df["strike"].unique())

        pain = {}
        for s in strikes:
            # call pain: sum of (strike - s) * OI for all calls with strike < s
            call_pain = float(
                calls[calls["strike"] < s]
                .apply(lambda r: (s - r["strike"]) * r["open_interest"], axis=1)
                .sum()
            )
            # put pain: sum of (s - strike) * OI for all puts with strike > s
            put_pain = float(
                puts[puts["strike"] > s]
                .apply(lambda r: (r["strike"] - s) * r["open_interest"], axis=1)
                .sum()
            )
            pain[s] = call_pain + put_pain

        return float(min(pain, key=lambda strike: pain[strike]))

    def gamma_exposure(
        self,
        expiry: str | None = None,
        spot: float | None = None,
    ) -> pd.DataFrame:
        """
        Backwards-compatible open-interest proxy for gamma exposure by strike.

        For model-based dollar GEX, use gex_by_strike().
        """
        if expiry is None:
            expiry = self.nearest_expiry(0)

        df    = self.chain(expiry)
        calls = df[df["type"] == "call"].groupby("strike")["open_interest"].sum()
        puts  = df[df["type"] == "put"].groupby("strike")["open_interest"].sum()

        strikes = sorted(set(calls.index) | set(puts.index))
        rows = []
        for strike in strikes:
            call_oi = float(calls.get(strike, 0))
            put_oi = float(puts.get(strike, 0))
            rows.append({
                "strike":   strike,
                "call_oi":  call_oi,
                "put_oi":   put_oi,
                "net_gex":  call_oi - put_oi,
            })

        gex = pd.DataFrame(rows)
        gex["cumulative_gex"] = gex["net_gex"].cumsum()
        if spot is not None:
            gex["distance_from_spot"] = gex["strike"] - float(spot)
        return gex


    def vol_surface(
        self,
        max_expiries: int = 8,
        moneyness_range: tuple[float, float] = (0.80, 1.20),
    ) -> pd.DataFrame:
        """
        Implied volatility surface: strike × expiry grid.

        Parameters
        ----------
        max_expiries    : number of expiries to include (default 8)
        moneyness_range : (min, max) moneyness filter around ATM
                          e.g. (0.80, 1.20) = strikes within ±20% of spot

        Returns
        -------
        DataFrame with strikes as index, expiries as columns,
        implied_volatility values as cells (NaN where no data).

        Example
        -------
        surface = opt.vol_surface(max_expiries=6)
        plots.options_surface(opt).show()
        """
        spot    = self._get_spot()
        expiries = self.expiries()[:max_expiries]

        frames = {}
        for exp in expiries:
            try:
                df = self.chain(exp)
                # filter to moneyness range
                lo = spot * moneyness_range[0]
                hi = spot * moneyness_range[1]
                df = df[(df["strike"] >= lo) & (df["strike"] <= hi)]
                # use mid of calls and puts IV per strike
                iv = df.groupby("strike")["implied_volatility"].mean()
                frames[exp] = iv
            except Exception:
                pass

        surface = pd.DataFrame(frames)
        surface.index.name = "strike"
        return surface.sort_index()

    def oi_by_strike(
        self,
        expiry: str | None = None,
    ) -> pd.DataFrame:
        """
        Open interest by strike for calls and puts side by side.

        Useful for identifying key support (high put OI) and
        resistance (high call OI) levels.

        Returns
        -------
        DataFrame: strike, call_oi, put_oi, total_oi, dominant
                   dominant = "call" | "put" (which side has more OI)

        Example
        -------
        oi = opt.oi_by_strike("2025-06-20")
        oi.sort_values("total_oi", ascending=False).head(10)
        """
        if expiry is None:
            expiry = self.nearest_expiry(0)

        # get OI directly from chain — no dependency on gamma_exposure
        df     = self.chain(expiry)
        calls  = df[df["type"]=="call"][["strike","open_interest"]].rename(
                     columns={"open_interest":"call_oi"})
        puts   = df[df["type"]=="put"][["strike","open_interest"]].rename(
                     columns={"open_interest":"put_oi"})
        oi     = calls.merge(puts, on="strike", how="outer").fillna(0)
        oi["total_oi"] = oi["call_oi"] + oi["put_oi"]
        oi["dominant"] = oi.apply(
            lambda r: "call" if r["call_oi"] >= r["put_oi"] else "put", axis=1
        )
        return oi[["strike","call_oi","put_oi","total_oi","dominant"]].sort_values("strike")


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Greeks (Black-Scholes with yfinance fallback)
    # ------------------------------------------------------------------
    #
    # ARCHITECTURE NOTE — Greeks computation
    # ----------------------------------------
    # Strategy: try yfinance first (faster, direct), fall back to
    # Black-Scholes when yfinance doesn't provide greeks.
    # B-S is embedded so the model is always available for strategy
    # analysis regardless of data provider.
    #
    # When the strategy analysis module is built, it will call:
    #   opt.greeks(expiry) → DataFrame with all greeks per contract
    # and use the embedded B-S model for P&L simulation, scenario
    # analysis, and position-level risk aggregation.
    # ------------------------------------------------------------------

    @staticmethod
    def _black_scholes_greeks(
        spot: float,
        strike: float,
        dte_years: float,
        iv: float,
        risk_free_rate: float = 0.05,
        option_type: str = "call",
    ) -> dict:
        """
        Compute option greeks via Black-Scholes-Merton model.

        Parameters
        ----------
        spot          : current underlying price
        strike        : option strike price
        dte_years     : time to expiry in years (days / 365)
        iv            : implied volatility (annualised, e.g. 0.25 = 25%)
        risk_free_rate: annualised risk-free rate (default 0.05)
        option_type   : "call" or "put"

        Returns
        -------
        dict: delta, gamma, theta, vega, rho, theoretical_price
        """
        from scipy.stats import norm
        import math

        # force all inputs to float — pandas/numpy types cause TypeError in math.*
        try:
            spot          = float(spot)
            strike        = float(strike)
            dte_years     = float(dte_years)
            iv            = float(iv)
            risk_free_rate = float(risk_free_rate)
        except (TypeError, ValueError):
            return {"delta": 0.0, "gamma": 0.0, "theta": 0.0,
                    "vega": 0.0, "rho": 0.0, "theoretical_price": 0.0}

        if dte_years <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
            return {"delta": 0.0, "gamma": 0.0, "theta": 0.0,
                    "vega": 0.0, "rho": 0.0, "theoretical_price": 0.0}

        S, K, T, r, σ = spot, strike, dte_years, risk_free_rate, iv

        d1 = (math.log(S / K) + (r + 0.5 * σ**2) * T) / (σ * math.sqrt(T))
        d2 = d1 - σ * math.sqrt(T)

        nd1  = norm.pdf(d1)
        Nd1  = norm.cdf(d1)
        Nd2  = norm.cdf(d2)
        Nd1_ = norm.cdf(-d1)
        Nd2_ = norm.cdf(-d2)

        if option_type == "call":
            price = S * Nd1 - K * math.exp(-r * T) * Nd2
            delta = Nd1
            rho   = K * T * math.exp(-r * T) * Nd2 / 100
        else:
            price = K * math.exp(-r * T) * Nd2_ - S * Nd1_
            delta = Nd1 - 1
            rho   = -K * T * math.exp(-r * T) * Nd2_ / 100

        gamma = nd1 / (S * σ * math.sqrt(T))
        vega  = S * nd1 * math.sqrt(T) / 100          # per 1% IV move
        theta = (-(S * nd1 * σ) / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) *
                 (Nd2 if option_type == "call" else Nd2_)) / 365

        return {
            "delta":             round(delta, 4),
            "gamma":             round(gamma, 6),
            "theta":             round(theta, 4),
            "vega":              round(vega,  4),
            "rho":               round(rho,   4),
            "theoretical_price": round(price, 4),
        }

    def greeks(
        self,
        expiry: str | None = None,
        risk_free_rate: float = 0.05,
    ) -> pd.DataFrame:
        """
        Return options chain enriched with greeks for all strikes.

        Strategy: try yfinance greeks first, compute Black-Scholes
        as fallback. Always adds theoretical_price from B-S for
        comparison even when yfinance greeks are available.

        Parameters
        ----------
        expiry         : expiration date (default front month)
        risk_free_rate : annualised risk-free rate (default 0.05)

        Returns
        -------
        DataFrame with all chain columns plus:
          delta, gamma, theta, vega, rho, theoretical_price, greek_source

        greek_source : "yfinance" | "black_scholes"

        Notes for strategy analysis
        ---------------------------
        delta  : price sensitivity — how much option moves per $1 in underlying
        gamma  : delta sensitivity — how fast delta changes (curvature)
        theta  : time decay per day (negative for long options)
        vega   : IV sensitivity — per 1% change in implied vol
        rho    : rate sensitivity — per 1% change in risk-free rate

        Example
        -------
        df = opt.greeks("2025-06-20")
        calls = df[df["type"] == "call"]
        print(calls[["strike","delta","gamma","theta","vega"]].head(10))
        """
        from datetime import date

        if expiry is None:
            expiry = self.nearest_expiry(0)

        df   = self.chain(expiry)
        spot = self._get_spot()
        dte_days  = max((pd.to_datetime(expiry).date() - date.today()).days, 1)
        dte_years = dte_days / 365.0

        # try yfinance greeks first
        raw = self.client.get_options_chain(self.symbol, expiry)
        yf_greeks = {}
        for side in ["calls", "puts"]:
            side_df = pd.DataFrame(raw.get(side, []))
            if not side_df.empty and "delta" in side_df.columns:
                for _, row in side_df.iterrows():
                    key = (round(float(row["strike"]), 2),
                           "call" if side == "calls" else "put")
                    yf_greeks[key] = {
                        "delta": row.get("delta", np.nan),
                        "gamma": row.get("gamma", np.nan),
                        "theta": row.get("theta", np.nan),
                        "vega":  row.get("vega",  np.nan),
                        "rho":   row.get("rho",   np.nan),
                    }

        greek_rows = []
        for _, row in df.iterrows():
            key    = (round(float(row["strike"]), 2), row["type"])
            iv     = float(row.get("implied_volatility", 0)) or 0.0
            # yfinance sometimes returns IV as raw % (4.18 = 418%) not decimal (0.0418)
            # normalise: anything > 5.0 is clearly a percentage, divide by 100
            if iv > 5.0:
                iv = iv / 100.0
            bs     = self._black_scholes_greeks(
                        float(spot), float(row["strike"]), float(dte_years),
                        float(iv), float(risk_free_rate), row["type"])

            yf = yf_greeks.get(key, {})
            has_yf = (yf and not np.isnan(yf.get("delta", np.nan)))

            greek_rows.append({
                **row.to_dict(),
                "delta":             yf.get("delta", bs["delta"]) if has_yf else bs["delta"],
                "gamma":             yf.get("gamma", bs["gamma"]) if has_yf else bs["gamma"],
                "theta":             yf.get("theta", bs["theta"]) if has_yf else bs["theta"],
                "vega":              yf.get("vega",  bs["vega"])  if has_yf else bs["vega"],
                "rho":               yf.get("rho",   bs["rho"])   if has_yf else bs["rho"],
                "theoretical_price": bs["theoretical_price"],
                "greek_source":      "yfinance" if has_yf else "black_scholes",
            })

        return pd.DataFrame(greek_rows).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Gamma Exposure (GEX)
    # ------------------------------------------------------------------
    #
    # GEX = Gamma × Open Interest × 100 (contract size) × Spot²  × 0.01
    #
    # Sign convention (from market maker perspective):
    #   Calls: market makers sold calls → short calls → long gamma → +GEX
    #   Puts:  market makers sold puts  → short puts  → short gamma → -GEX
    #
    # Total GEX > 0 → stabilising (MMs sell rallies, buy dips)
    # Total GEX < 0 → destabilising (MMs buy rallies, sell dips)
    # GEX flip level → strike where regime changes
    # ------------------------------------------------------------------

    def gex_by_strike(
        self,
        expiry: str | None = None,
        risk_free_rate: float = 0.05,
        contract_size: int = 100,
    ) -> pd.DataFrame:
        """
        Gamma Exposure (GEX) by strike for one expiry.

        GEX = Gamma × OI × contract_size × Spot² × 0.01

        Calls contribute positive GEX (market makers long gamma).
        Puts contribute negative GEX (market makers short gamma).

        Returns
        -------
        DataFrame: strike, call_gex, put_gex, net_gex, cumulative_gex
          net_gex > 0 at a strike → call gamma dominates → stabilising
          net_gex < 0 at a strike → put gamma dominates → destabilising

        Example
        -------
        gex = opt.gex_by_strike("2025-06-20")
        flip = gex[gex["cumulative_gex"] >= 0].iloc[0]["strike"]
        print(f"GEX flip level: ${flip:,.2f}")
        """
        if expiry is None:
            expiry = self.nearest_expiry(0)

        df   = self.greeks(expiry, risk_free_rate)
        spot = self._get_spot()
        scale = contract_size * spot**2 * 0.01

        calls = df[df["type"] == "call"].copy()
        puts  = df[df["type"] == "put"].copy()

        call_gex = calls.groupby("strike").apply(
            lambda x: float((x["gamma"] * x["open_interest"] * scale).sum())
        )
        put_gex = puts.groupby("strike").apply(
            lambda x: float(-(x["gamma"] * x["open_interest"] * scale).sum())
        )

        strikes = sorted(set(call_gex.index) | set(put_gex.index))
        rows = []
        for s in strikes:
            cg = call_gex.get(s, 0.0)
            pg = put_gex.get(s, 0.0)
            rows.append({
                "strike":   s,
                "call_gex": cg,
                "put_gex":  pg,
                "net_gex":  cg + pg,
            })

        result = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
        result["cumulative_gex"] = result["net_gex"].cumsum()
        return result

    def gex_by_expiry(
        self,
        max_expiries: int = 12,
        risk_free_rate: float = 0.05,
        contract_size: int = 100,
    ) -> pd.DataFrame:
        """
        Total Gamma Exposure aggregated per expiry.

        Shows which expiry carries the most gamma risk and how the
        gamma term structure is distributed.

        Returns
        -------
        DataFrame: expiry, days_to_expiry, call_gex, put_gex,
                   net_gex, abs_gex

        Example
        -------
        df = opt.gex_by_expiry(max_expiries=8)
        print(df.sort_values("abs_gex", ascending=False))
        """
        from datetime import date
        today    = date.today()
        expiries = self.expiries()[:max_expiries]
        rows     = []

        for exp in expiries:
            try:
                gex = self.gex_by_strike(exp, risk_free_rate, contract_size)
                dte = (pd.to_datetime(exp).date() - today).days
                rows.append({
                    "expiry":          exp,
                    "days_to_expiry":  dte,
                    "call_gex":        float(gex["call_gex"].sum()),
                    "put_gex":         float(gex["put_gex"].sum()),
                    "net_gex":         float(gex["net_gex"].sum()),
                    "abs_gex":         float(gex["net_gex"].abs().sum()),
                })
            except Exception as e:
                print(f"Skipping {exp}: {e}")

        return pd.DataFrame(rows).sort_values("expiry").reset_index(drop=True)

    def gex_total(
        self,
        max_expiries: int = 12,
        risk_free_rate: float = 0.05,
        contract_size: int = 100,
    ) -> dict:
        """
        Grand total Gamma Exposure across all expiries — the market regime indicator.

        Returns
        -------
        dict:
          total_call_gex  : sum of positive GEX (calls)
          total_put_gex   : sum of negative GEX (puts)
          total_net_gex   : grand total — positive = stabilising,
                            negative = destabilising
          regime          : "positive" | "negative"
          regime_label    : human-readable description
          flip_strike     : approximate strike where GEX crosses zero
                            (computed from front month only)
          dominant_expiry : expiry with largest absolute GEX

        Example
        -------
        gt = opt.gex_total()
        print(f"Market regime: {gt['regime_label']}")
        print(f"GEX flip: ${gt['flip_strike']:,.2f}")
        """
        by_exp = self.gex_by_expiry(max_expiries, risk_free_rate, contract_size)

        total_call = float(by_exp["call_gex"].sum())
        total_put  = float(by_exp["put_gex"].sum())
        total_net  = float(by_exp["net_gex"].sum())

        regime = "positive" if total_net >= 0 else "negative"
        label  = (
            "Positive GEX — market makers are net long gamma. "
            "They sell rallies and buy dips → price-stabilising, "
            "expect lower realised volatility."
            if regime == "positive" else
            "Negative GEX — market makers are net short gamma. "
            "They buy rallies and sell dips → price-amplifying, "
            "expect higher realised volatility and larger moves."
        )

        # flip strike from front month
        flip_strike = None
        try:
            front_gex = self.gex_by_strike(self.nearest_expiry(0),
                                            risk_free_rate, contract_size)
            above = front_gex[front_gex["cumulative_gex"] >= 0]
            if not above.empty:
                flip_strike = float(above.iloc[0]["strike"])
        except Exception:
            pass

        dominant = None
        if not by_exp.empty:
            dominant = by_exp.loc[by_exp["abs_gex"].idxmax(), "expiry"]

        return {
            "symbol":           self.symbol,
            "total_call_gex":   total_call,
            "total_put_gex":    total_put,
            "total_net_gex":    total_net,
            "regime":           regime,
            "regime_label":     label,
            "flip_strike":      flip_strike,
            "dominant_expiry":  dominant,
        }

    # ------------------------------------------------------------------
    # Unusual activity
    # ------------------------------------------------------------------

    def unusual_activity(
        self,
        expiry: str | None = None,
        vol_oi_threshold: float = 3.0,
        min_volume: int = 100,
        min_oi: int = 50,
    ) -> pd.DataFrame:
        """
        Detect unusual options activity — contracts where volume
        significantly exceeds open interest.

        A high volume/OI ratio suggests new positions are being opened
        (not just existing ones rolling), which can signal informed
        buying or hedging activity.

        Parameters
        ----------
        expiry           : expiration date (default front month)
        vol_oi_threshold : flag when volume/OI >= this ratio (default 3.0)
        min_volume       : minimum volume to consider (default 100)
        min_oi           : minimum OI to consider (default 50)

        Returns
        -------
        DataFrame sorted by vol_oi_ratio descending with columns:
          type, strike, volume, open_interest, vol_oi_ratio,
          implied_volatility, bid, ask, last_price, signal

        signal : "🚨 Unusual" | "⚠ Elevated" | "Normal"

        Example
        -------
        ua = opt.unusual_activity("2025-06-20", vol_oi_threshold=3.0)
        print(ua[ua["signal"] == "🚨 Unusual"][["type","strike","volume","vol_oi_ratio"]])
        """
        if expiry is None:
            expiry = self.nearest_expiry(0)

        df = self.chain(expiry, min_volume=min_volume)
        df = df[df["open_interest"] >= min_oi].copy()

        df["vol_oi_ratio"] = (
            df["volume"] / df["open_interest"].replace(0, np.nan)
        ).round(2)

        df["signal"] = df["vol_oi_ratio"].apply(
            lambda r: "🚨 Unusual" if r >= vol_oi_threshold
                      else "⚠ Elevated" if r >= vol_oi_threshold * 0.5
                      else "Normal"
        )

        return (df[df["vol_oi_ratio"].notna()]
                .sort_values("vol_oi_ratio", ascending=False)
                .reset_index(drop=True))

    def _get_spot(self) -> float:
        """Get current underlying price."""
        return float(self.client.get_price(self.symbol)["price"])

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardise column names and types from yfinance raw data."""
        col_map = {
            "strike":            "strike",
            "lastPrice":         "last_price",
            "bid":               "bid",
            "ask":               "ask",
            "volume":            "volume",
            "openInterest":      "open_interest",
            "impliedVolatility": "implied_volatility",
            "inTheMoney":        "in_the_money",
            "contractSymbol":    "contract",
            "lastTradeDate":     "last_trade_date",
            "percentChange":     "pct_change",
            "change":            "change",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # ensure numeric
        for col in ["strike","last_price","bid","ask","volume",
                    "open_interest","implied_volatility"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # keep useful columns only
        keep = ["expiry", "type", "strike", "bid", "ask", "last_price",
                "volume", "open_interest", "implied_volatility",
                "in_the_money", "contract"]
        df = df[[c for c in keep if c in df.columns]]

        return df.sort_values("strike").reset_index(drop=True)
