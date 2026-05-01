"""
options_strategy.py — Options strategy analysis and payoff modelling.

Builds multi-leg options strategies from real market data (via OptionsAnalyzer)
and computes P&L, Greeks, breakevens, and risk metrics.

Usage
-----
    from yfinance_api3.classes.options import OptionsAnalyzer
    from yfinance_api3.classes.options_strategy import OptionsStrategy
    import yfinance_api3.modules.plots as plots

    opt = OptionsAnalyzer(client, "AAPL")

    # build from factory
    strat = bull_call_spread(opt, low_strike=200, high_strike=210,
                             expiry=opt.nearest_expiry(1))

    print(strat.summary())
    plots.strategy_payoff(strat).show()
    plots.strategy_surface(strat).show()
    plots.strategy_greeks(strat).show()

    # or build manually
    strat = OptionsStrategy(opt, name="Custom spread")
    strat.add_leg("call", strike=200, expiry=opt.nearest_expiry(1),
                  quantity=1, direction="long")
    strat.add_leg("call", strike=210, expiry=opt.nearest_expiry(1),
                  quantity=1, direction="short")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes engine (standalone, no dependency on OptionsAnalyzer)
# ---------------------------------------------------------------------------

def _bs_price(S, K, T, sigma, r, is_call: bool) -> float:
    """Black-Scholes option price."""
    if T <= 0:
        return max(S - K, 0) if is_call else max(K - S, 0)
    if sigma <= 0:
        return max(S - K, 0) if is_call else max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_delta(S, K, T, sigma, r, is_call: bool) -> float:
    if T <= 0 or sigma <= 0:
        return 1.0 if is_call and S > K else (-1.0 if not is_call and S < K else 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1) if is_call else norm.cdf(d1) - 1


def _bs_gamma(S, K, T, sigma, r) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))


def _bs_theta(S, K, T, sigma, r, is_call: bool) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    if is_call:
        term2 = -r * K * math.exp(-r * T) * norm.cdf(d2)
    else:
        term2 = r * K * math.exp(-r * T) * norm.cdf(-d2)
    return (term1 + term2) / 365   # per day


def _bs_vega(S, K, T, sigma, r) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * norm.pdf(d1) * math.sqrt(T) * 0.01   # per 1% IV change


# ---------------------------------------------------------------------------
# OptionLeg — single option position
# ---------------------------------------------------------------------------

@dataclass
class OptionLeg:
    """
    One leg of an options strategy.

    Attributes
    ----------
    option_type : "call" or "put"
    strike      : strike price
    expiry      : expiration date string (YYYY-MM-DD)
    quantity    : number of contracts (1 contract = 100 shares)
    direction   : "long" (bought) or "short" (sold)
    premium     : price paid/received per share (mid-price from market)
    iv          : implied volatility used for modelling (from market)
    """
    option_type: Literal["call", "put"]
    strike:      float
    expiry:      str
    quantity:    int = 1
    direction:   Literal["long", "short"] = "long"
    premium:     float = 0.0
    iv:          float = 0.20

    @property
    def sign(self) -> int:
        """+1 for long, -1 for short."""
        return 1 if self.direction == "long" else -1

    @property
    def cost(self) -> float:
        """Net premium paid (negative = received). Per share."""
        return self.sign * self.premium * self.quantity * 100

    @property
    def dte(self) -> float:
        """Days to expiry from today."""
        return max((pd.to_datetime(self.expiry).date() - date.today()).days, 0)

    @property
    def T(self) -> float:
        """Time to expiry in years."""
        return self.dte / 365.0

    def intrinsic(self, spot: float) -> float:
        """Intrinsic value at expiry (ignores time value)."""
        if self.option_type == "call":
            return max(spot - self.strike, 0)
        else:
            return max(self.strike - spot, 0)

    def pnl_at_expiry(self, spot: float) -> float:
        """P&L per share at expiry for this leg."""
        return self.sign * (self.intrinsic(spot) - self.premium) * self.quantity * 100

    def theoretical_price(self, spot: float, days_ahead: int = 0,
                          risk_free_rate: float = 0.05) -> float:
        """B-S theoretical price at a given spot and days in the future."""
        T = max((self.dte - days_ahead) / 365.0, 0)
        return _bs_price(spot, self.strike, T, self.iv,
                         risk_free_rate, self.option_type == "call")

    def pnl_theoretical(self, spot: float, days_ahead: int = 0,
                        risk_free_rate: float = 0.05) -> float:
        """Theoretical P&L before expiry using B-S pricing."""
        price = self.theoretical_price(spot, days_ahead, risk_free_rate)
        return self.sign * (price - self.premium) * self.quantity * 100

    def delta(self, spot: float, risk_free_rate: float = 0.05) -> float:
        return self.sign * _bs_delta(spot, self.strike, self.T,
                                     self.iv, risk_free_rate,
                                     self.option_type == "call") * self.quantity

    def gamma(self, spot: float, risk_free_rate: float = 0.05) -> float:
        return self.sign * _bs_gamma(spot, self.strike, self.T,
                                     self.iv, risk_free_rate) * self.quantity

    def theta(self, spot: float, risk_free_rate: float = 0.05) -> float:
        return self.sign * _bs_theta(spot, self.strike, self.T,
                                     self.iv, risk_free_rate,
                                     self.option_type == "call") * self.quantity

    def vega(self, spot: float, risk_free_rate: float = 0.05) -> float:
        return self.sign * _bs_vega(spot, self.strike, self.T,
                                    self.iv, risk_free_rate) * self.quantity


# ---------------------------------------------------------------------------
# OptionsStrategy — collection of legs
# ---------------------------------------------------------------------------

class OptionsStrategy:
    """
    Multi-leg options strategy — P&L, Greeks, breakevens, risk metrics.

    Parameters
    ----------
    opt             : OptionsAnalyzer instance (for market data)
    name            : strategy label (e.g. "Bull Call Spread")
    risk_free_rate  : annual risk-free rate for B-S pricing (default 0.05)
    contract_size   : shares per contract (default 100)
    """

    def __init__(
        self,
        opt,
        name: str = "Custom Strategy",
        risk_free_rate: float = 0.05,
        contract_size: int = 100,
    ) -> None:
        self.opt           = opt
        self.name          = name
        self.risk_free_rate = risk_free_rate
        self.contract_size = contract_size
        self.legs: list[OptionLeg] = []
        self._spot         = opt._get_spot()

    # ------------------------------------------------------------------
    # Leg management
    # ------------------------------------------------------------------

    def add_leg(
        self,
        option_type: Literal["call", "put"],
        strike: float,
        expiry: str,
        quantity: int = 1,
        direction: Literal["long", "short"] = "long",
        risk_free_rate: float | None = None,
    ) -> "OptionsStrategy":
        """
        Add one leg to the strategy using real market prices.

        Fetches the mid-price and IV from the live options chain.
        Returns self for chaining.

        Example
        -------
        strat.add_leg("call", strike=200, expiry="2025-06-20",
                      quantity=1, direction="long")
        """
        # fetch real market data for this leg
        chain = self.opt.chain(expiry, option_type=option_type)
        row   = chain[chain["strike"] == strike]

        if row.empty:
            # find nearest available strike
            nearest = chain.iloc[(chain["strike"] - strike).abs().argsort()[:1]]
            row     = nearest
            strike  = float(row["strike"].iloc[0])

        bid  = float(row["bid"].iloc[0])
        ask  = float(row["ask"].iloc[0])
        mid  = (bid + ask) / 2 if ask > 0 else float(row["last_price"].iloc[0])
        iv   = float(row["implied_volatility"].iloc[0]) or 0.20

        leg = OptionLeg(
            option_type=option_type,
            strike=strike,
            expiry=expiry,
            quantity=quantity,
            direction=direction,
            premium=mid,
            iv=iv,
        )
        self.legs.append(leg)
        return self

    def add_leg_manual(
        self,
        option_type: Literal["call", "put"],
        strike: float,
        expiry: str,
        premium: float,
        iv: float,
        quantity: int = 1,
        direction: Literal["long", "short"] = "long",
    ) -> "OptionsStrategy":
        """Add a leg with manually specified premium and IV (no market lookup)."""
        self.legs.append(OptionLeg(
            option_type=option_type, strike=strike, expiry=expiry,
            quantity=quantity, direction=direction, premium=premium, iv=iv,
        ))
        return self

    # ------------------------------------------------------------------
    # P&L analysis
    # ------------------------------------------------------------------

    def net_premium(self) -> float:
        """Net premium paid (negative) or received (positive) for the strategy."""
        return sum(leg.cost for leg in self.legs)

    def payoff(
        self,
        spot_range: tuple[float, float] | None = None,
        n_points: int = 200,
    ) -> pd.DataFrame:
        """
        P&L at expiry across a range of underlying prices.

        Returns DataFrame with columns:
          spot      : underlying price
          pnl       : total P&L in dollars
          pnl_pct   : P&L as % of max risk (or net premium if undefined)
          per_leg   : individual leg P&L columns

        Example
        -------
        df = strat.payoff()
        df[df["pnl"] >= 0]   # profitable range
        """
        spot = self._spot
        if spot_range is None:
            lo = spot * 0.70
            hi = spot * 1.30
        else:
            lo, hi = spot_range

        spots = np.linspace(lo, hi, n_points)
        rows  = []
        for s in spots:
            row = {"spot": s}
            total = 0.0
            for i, leg in enumerate(self.legs):
                leg_pnl = leg.pnl_at_expiry(s)
                row[f"leg_{i+1}_{leg.direction}_{leg.option_type}_{leg.strike}"] = leg_pnl
                total += leg_pnl
            row["pnl"] = total
            rows.append(row)

        df = pd.DataFrame(rows)
        # compute max_risk directly from the pnl series — avoid calling
        # max_loss() here which would create payoff() → max_loss() → payoff()
        # circular recursion
        max_risk = abs(df["pnl"].min())
        df["pnl_pct"] = df["pnl"] / max_risk * 100 if max_risk > 0 else 0.0
        return df

    def pnl_surface(
        self,
        spot_range: tuple[float, float] | None = None,
        n_spots: int = 60,
        n_days: int = 20,
    ) -> pd.DataFrame:
        """
        P&L surface across underlying price AND time (days until expiry).

        Returns a pivot DataFrame:
          rows    : days_ahead (0 = today, max = days to front expiry)
          columns : spot prices
          values  : total theoretical P&L in dollars

        Example
        -------
        surface = strat.pnl_surface()
        plots.strategy_surface(strat).show()
        """
        spot = self._spot
        if spot_range is None:
            lo, hi = spot * 0.80, spot * 1.20
        else:
            lo, hi = spot_range

        spots     = np.linspace(lo, hi, n_spots)
        max_dte   = max(leg.dte for leg in self.legs) if self.legs else 30
        day_steps = np.linspace(0, max_dte, n_days, dtype=int)

        rows = []
        for days in day_steps:
            row: dict[object, float | int] = {"days_ahead": int(days)}
            for s in spots:
                total = sum(
                    leg.pnl_theoretical(s, int(days), self.risk_free_rate)
                    for leg in self.legs
                )
                row[round(s, 2)] = total
            rows.append(row)

        df = pd.DataFrame(rows).set_index("days_ahead")
        return df

    def greeks_profile(
        self,
        spot_range: tuple[float, float] | None = None,
        n_points: int = 100,
    ) -> pd.DataFrame:
        """
        Combined portfolio Greeks across a range of underlying prices.

        Returns DataFrame with columns:
          spot, delta, gamma, theta, vega

        Example
        -------
        df = strat.greeks_profile()
        df[df["delta"].abs() < 0.1]   # near-delta-neutral zone
        """
        spot = self._spot
        if spot_range is None:
            lo, hi = spot * 0.80, spot * 1.20
        else:
            lo, hi = spot_range

        spots = np.linspace(lo, hi, n_points)
        rows  = []
        for s in spots:
            rows.append({
                "spot":  s,
                "delta": sum(leg.delta(s, self.risk_free_rate) for leg in self.legs),
                "gamma": sum(leg.gamma(s, self.risk_free_rate) for leg in self.legs),
                "theta": sum(leg.theta(s, self.risk_free_rate) for leg in self.legs),
                "vega":  sum(leg.vega(s, self.risk_free_rate)  for leg in self.legs),
            })
        return pd.DataFrame(rows)

    def breakevens(self) -> list[float]:
        """
        Strike prices where P&L = 0 at expiry.

        Returns list of breakeven prices (usually 1-2 for spreads,
        2 for straddles/strangles, 4 for condors).
        """
        df = self.payoff(n_points=1000)
        be = []
        pnl = df["pnl"].values
        spots = df["spot"].values
        for i in range(len(pnl) - 1):
            if pnl[i] * pnl[i+1] <= 0:   # sign change = zero crossing
                # linear interpolation
                x = spots[i] - pnl[i] * (spots[i+1] - spots[i]) / (pnl[i+1] - pnl[i])
                be.append(round(x, 2))
        return be

    def max_profit(self) -> dict:
        """Maximum profit and at what spot price."""
        df  = self.payoff(n_points=1000)
        idx = df["pnl"].idxmax()
        return {
            "value": round(float(df.loc[idx, "pnl"]), 2),
            "at_spot": round(float(df.loc[idx, "spot"]), 2),
            "unlimited": df["pnl"].iloc[-1] > df["pnl"].iloc[-2],  # still rising at edge
        }

    def max_loss(self) -> dict:
        """Maximum loss and at what spot price."""
        df  = self.payoff(n_points=1000)
        idx = df["pnl"].idxmin()
        return {
            "value": round(float(df.loc[idx, "pnl"]), 2),
            "at_spot": round(float(df.loc[idx, "spot"]), 2),
            "unlimited": df["pnl"].iloc[0] < df["pnl"].iloc[1],  # still falling at edge
        }

    def summary(self) -> dict:
        """Full strategy summary — risk, reward, breakevens, Greeks at spot."""
        mp  = self.max_profit()
        ml  = self.max_loss()
        be  = self.breakevens()
        rr  = abs(mp["value"] / ml["value"]) if ml["value"] != 0 else float("inf")

        spot = self._spot
        return {
            "strategy":        self.name,
            "symbol":          self.opt.symbol,
            "spot":            round(spot, 2),
            "n_legs":          len(self.legs),
            "net_premium":     round(self.net_premium(), 2),
            "net_premium_dir": "paid" if self.net_premium() < 0 else "received",
            "max_profit":      mp["value"],
            "max_profit_at":   mp["at_spot"],
            "max_profit_unlimited": mp["unlimited"],
            "max_loss":        ml["value"],
            "max_loss_at":     ml["at_spot"],
            "max_loss_unlimited":   ml["unlimited"],
            "risk_reward":     round(rr, 2),
            "breakevens":      be,
            "delta":           round(sum(leg.delta(spot, self.risk_free_rate) for leg in self.legs), 4),
            "gamma":           round(sum(leg.gamma(spot, self.risk_free_rate) for leg in self.legs), 6),
            "theta":           round(sum(leg.theta(spot, self.risk_free_rate) for leg in self.legs), 4),
            "vega":            round(sum(leg.vega(spot,  self.risk_free_rate) for leg in self.legs), 4),
        }

    def __repr__(self) -> str:
        s = self.summary()
        lines = [
            f"OptionsStrategy — {s['strategy']}  [{s['symbol']}]",
            f"  Spot:         ${s['spot']:,.2f}",
            f"  Net premium:  ${abs(s['net_premium']):,.2f} {s['net_premium_dir']}",
            f"  Max profit:   ${s['max_profit']:,.2f}"
              + (" (unlimited)" if s["max_profit_unlimited"] else f" at ${s['max_profit_at']:,.2f}"),
            f"  Max loss:     ${s['max_loss']:,.2f}"
              + (" (unlimited)" if s["max_loss_unlimited"] else f" at ${s['max_loss_at']:,.2f}"),
            f"  Risk/reward:  {s['risk_reward']:.2f}x",
            f"  Breakevens:   {', '.join(f'${b:,.2f}' for b in s['breakevens'])}",
            f"  Greeks @spot: Δ={s['delta']:+.3f}  Γ={s['gamma']:+.5f}  "
              f"Θ={s['theta']:+.4f}/day  V={s['vega']:+.4f}",
            "",
            "  Legs:",
        ]
        for i, leg in enumerate(self.legs, 1):
            lines.append(
                f"    {i}. {leg.direction.upper()} {leg.quantity}x "
                f"{leg.option_type.upper()} ${leg.strike:,.0f}  "
                f"exp {leg.expiry}  @${leg.premium:.2f}  IV={leg.iv:.1%}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Strategy factories — common pre-built strategies
# ---------------------------------------------------------------------------

def long_call(opt, strike: float, expiry: str, quantity: int = 1,
              risk_free_rate: float = 0.05) -> OptionsStrategy:
    """
    Long Call — unlimited upside, limited downside (premium paid).
    Bullish directional bet.
    """
    s = OptionsStrategy(opt, "Long Call", risk_free_rate)
    s.add_leg("call", strike, expiry, quantity, "long")
    return s


def long_put(opt, strike: float, expiry: str, quantity: int = 1,
             risk_free_rate: float = 0.05) -> OptionsStrategy:
    """
    Long Put — profits as stock falls. Bearish or protective hedge.
    """
    s = OptionsStrategy(opt, "Long Put", risk_free_rate)
    s.add_leg("put", strike, expiry, quantity, "long")
    return s


def covered_call(opt, strike: float, expiry: str, quantity: int = 1,
                 risk_free_rate: float = 0.05) -> OptionsStrategy:
    """
    Covered Call — short call against existing stock position.
    Generates income, caps upside above the strike.
    Note: stock position P&L not included here (options legs only).
    """
    s = OptionsStrategy(opt, "Covered Call", risk_free_rate)
    s.add_leg("call", strike, expiry, quantity, "short")
    return s


def bull_call_spread(opt, low_strike: float, high_strike: float,
                     expiry: str, quantity: int = 1,
                     risk_free_rate: float = 0.05) -> OptionsStrategy:
    """
    Bull Call Spread — buy lower call, sell higher call.
    Capped upside, reduced cost vs naked long call.
    Best for moderately bullish view.
    """
    s = OptionsStrategy(opt, "Bull Call Spread", risk_free_rate)
    s.add_leg("call", low_strike,  expiry, quantity, "long")
    s.add_leg("call", high_strike, expiry, quantity, "short")
    return s


def bear_put_spread(opt, high_strike: float, low_strike: float,
                    expiry: str, quantity: int = 1,
                    risk_free_rate: float = 0.05) -> OptionsStrategy:
    """
    Bear Put Spread — buy higher put, sell lower put.
    Capped downside profit, reduced cost vs naked long put.
    Best for moderately bearish view.
    """
    s = OptionsStrategy(opt, "Bear Put Spread", risk_free_rate)
    s.add_leg("put", high_strike, expiry, quantity, "long")
    s.add_leg("put", low_strike,  expiry, quantity, "short")
    return s


def long_straddle(opt, strike: float, expiry: str, quantity: int = 1,
                  risk_free_rate: float = 0.05) -> OptionsStrategy:
    """
    Long Straddle — buy call + put at same strike.
    Profits from large moves in either direction.
    Best before high-uncertainty events (earnings, FOMC).
    """
    s = OptionsStrategy(opt, "Long Straddle", risk_free_rate)
    s.add_leg("call", strike, expiry, quantity, "long")
    s.add_leg("put",  strike, expiry, quantity, "long")
    return s


def short_straddle(opt, strike: float, expiry: str, quantity: int = 1,
                   risk_free_rate: float = 0.05) -> OptionsStrategy:
    """
    Short Straddle — sell call + put at same strike.
    Profits from low volatility / price pinning near strike.
    Maximum risk: unlimited on both sides.
    """
    s = OptionsStrategy(opt, "Short Straddle", risk_free_rate)
    s.add_leg("call", strike, expiry, quantity, "short")
    s.add_leg("put",  strike, expiry, quantity, "short")
    return s


def long_strangle(opt, put_strike: float, call_strike: float,
                  expiry: str, quantity: int = 1,
                  risk_free_rate: float = 0.05) -> OptionsStrategy:
    """
    Long Strangle — buy OTM put + OTM call.
    Cheaper than straddle, needs larger move to profit.
    """
    s = OptionsStrategy(opt, "Long Strangle", risk_free_rate)
    s.add_leg("put",  put_strike,  expiry, quantity, "long")
    s.add_leg("call", call_strike, expiry, quantity, "long")
    return s


def iron_condor(opt, put_low: float, put_high: float,
                call_low: float, call_high: float,
                expiry: str, quantity: int = 1,
                risk_free_rate: float = 0.05) -> OptionsStrategy:
    """
    Iron Condor — sell OTM put spread + sell OTM call spread.
    Profits when price stays in a range. Net credit strategy.
    Best in low-volatility environments with high IV.

    Put side  : buy put_low,  sell put_high   (below market)
    Call side : sell call_low, buy call_high  (above market)
    """
    s = OptionsStrategy(opt, "Iron Condor", risk_free_rate)
    s.add_leg("put",  put_low,   expiry, quantity, "long")
    s.add_leg("put",  put_high,  expiry, quantity, "short")
    s.add_leg("call", call_low,  expiry, quantity, "short")
    s.add_leg("call", call_high, expiry, quantity, "long")
    return s


def iron_butterfly(opt, put_strike: float, atm_strike: float,
                   call_strike: float, expiry: str, quantity: int = 1,
                   risk_free_rate: float = 0.05) -> OptionsStrategy:
    """
    Iron Butterfly — sell ATM straddle, buy OTM wings.
    Maximum profit at ATM strike. Higher credit than condor.

    Structure: buy put wing, sell ATM put, sell ATM call, buy call wing.
    """
    s = OptionsStrategy(opt, "Iron Butterfly", risk_free_rate)
    s.add_leg("put",  put_strike,  expiry, quantity, "long")
    s.add_leg("put",  atm_strike,  expiry, quantity, "short")
    s.add_leg("call", atm_strike,  expiry, quantity, "short")
    s.add_leg("call", call_strike, expiry, quantity, "long")
    return s


def calendar_spread(opt, strike: float, near_expiry: str, far_expiry: str,
                    option_type: Literal["call", "put"] = "call",
                    quantity: int = 1,
                    risk_free_rate: float = 0.05) -> OptionsStrategy:
    """
    Calendar Spread (Time Spread) — sell near-term, buy far-term at same strike.
    Profits from time decay of the near-term option and IV expansion.
    Best when expecting price to stay near strike short-term.
    """
    s = OptionsStrategy(opt, f"Calendar Spread ({option_type.capitalize()})",
                        risk_free_rate)
    s.add_leg(option_type, strike, near_expiry, quantity, "short")
    s.add_leg(option_type, strike, far_expiry,  quantity, "long")
    return s


def diagonal_spread(opt, near_strike: float, far_strike: float,
                    near_expiry: str, far_expiry: str,
                    option_type: Literal["call", "put"] = "call",
                    quantity: int = 1,
                    risk_free_rate: float = 0.05) -> OptionsStrategy:
    """
    Diagonal Spread — sell near-term option, buy far-term at different strike.
    Combines elements of calendar and vertical spreads.
    """
    s = OptionsStrategy(opt, f"Diagonal Spread ({option_type.capitalize()})",
                        risk_free_rate)
    s.add_leg(option_type, near_strike, near_expiry, quantity, "short")
    s.add_leg(option_type, far_strike,  far_expiry,  quantity, "long")
    return s
