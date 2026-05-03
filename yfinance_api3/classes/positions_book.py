"""
positions_book.py — Options position tracker.

Design
------
The BOOK records what you traded:
  - option legs  : type, expiry, strike, lots, price_paid
  - underlying   : lots, entry_price, direction

The MARKET provides current data via mark_to_market():
  - spot          : yesterday's close
  - leg.iv        : current IV from options chain (matched by strike + expiry)
  - risk_free_rate: 13-week T-bill (^IRX)
  - div_yield     : from yfinance info
  - book.vol      : historical vol — used ONLY for price range (±dStd)

The MODEL prices each leg using:
  - leg.iv        (current market IV per leg)
  - spot          (current underlying price)
  - days_to_expiry (computed daily from expiry date)
  - risk_free_rate (current)

Usage
-----
    book = PositionsBook("INTC")
    book.add_option("call", "2026-12-18", 85, -60, 18.11)
    book.add_option("call", "2026-12-18", 90, -40, 21.50)
    book.add_underlying(10000, 30.0, "long")

    book.mark_to_market(client, opt)   # fetch all market data
    print(book.summary())
    book.save("positions/INTC.json")
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Literal

import numpy as np
import pandas as pd

from yfinance_api3.classes.pricing import (
    BlackScholesModel, Binomial, BaroneAdesiWhaley,
    OptionData, OptionType, UnderlyingData,
    ContractType, ExerciseType, Space,
)


# ---------------------------------------------------------------------------
# Leg records
# ---------------------------------------------------------------------------

@dataclass
class OptionLegRecord:
    """
    One option leg — trade record only.

    Trade fields (set at entry, never change):
      leg_id, option_type, expiry, strike, lots, price_paid

    Market fields (updated by mark_to_market):
      iv          : current implied volatility from chain
      model_price : theoretical price at current market conditions

    Status:
      status      : "open" | "closed"
      close_price : price at which leg was closed
    """
    leg_id:      str
    option_type: str          # "call" | "put"
    expiry:      str          # YYYY-MM-DD
    strike:      float
    lots:        float        # negative = short
    price_paid:  float        # premium per share at entry
    iv:          float = 0.0  # current IV — set by mark_to_market
    model_price: float = 0.0  # theoretical price — set by mark_to_market
    instrument:  str   = "whaley"   # pricing model
    lot_size:    float = 100.0
    status:      str   = "open"
    close_price: float = None
    trade_date:  str   = None
    notes:       str   = ""

    @property
    def life_days(self) -> float:
        """Current days to expiry — always >= 1."""
        return max((pd.to_datetime(self.expiry).date() - date.today()).days, 1)

    @property
    def closed_pnl(self) -> float:
        if self.status != "closed" or self.close_price is None:
            return 0.0
        cp = 1 if self.option_type == "call" else -1
        return (self.close_price - self.price_paid) * self.lots * self.lot_size * cp

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OptionLegRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class UnderlyingRecord:
    """Stock or futures position."""
    lots:        float
    entry_price: float
    direction:   str   = "long"
    instrument:  str   = "equity"
    status:      str   = "open"
    close_price: float = None
    trade_date:  str   = None
    notes:       str   = ""

    @property
    def signed_lots(self) -> float:
        return self.lots if self.direction == "long" else -self.lots

    @property
    def closed_pnl(self) -> float:
        if self.status != "closed" or self.close_price is None:
            return 0.0
        return (self.close_price - self.entry_price) * self.signed_lots

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UnderlyingRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# PositionsBook
# ---------------------------------------------------------------------------

class PositionsBook:
    """
    Options position tracker for one underlying.

    The book stores what you traded.
    mark_to_market() fetches current market data and prices each leg.
    """

    def __init__(
        self,
        symbol:         str,
        name:           str          = None,
        contract_type:  ContractType = ContractType.STOCK,
        lot_size:       float        = 100.0,
        currency:       str          = "USD",
    ) -> None:
        self.symbol         = symbol.upper()
        self.name           = name or f"{symbol} positions"
        self.contract_type  = contract_type
        self.lot_size       = lot_size
        self.currency       = currency

        # market data — populated by mark_to_market()
        self.spot           = 0.0
        self.vol            = 0.30   # historical vol — for price range only
        self.risk_free_rate = 4.50   # % — updated from ^IRX
        self.div_yield      = 0.0    # % — updated from yfinance info

        self.option_legs:    list[OptionLegRecord]  = []
        self.underlying_pos: list[UnderlyingRecord] = []
        self.closed_pnl:     float                  = 0.0
        self.created_at:     str = datetime.now().isoformat()
        self.updated_at:     str = datetime.now().isoformat()

    # ------------------------------------------------------------------
    # Adding positions — trade record only
    # ------------------------------------------------------------------

    def add_option(
        self,
        option_type: Literal["call", "put"],
        expiry:      str,
        strike:      float,
        lots:        float,
        price_paid:  float,
        instrument:  Literal["whaley", "american", "equity", "futures"] = "whaley",
        lot_size:    float = None,
        notes:       str   = "",
    ) -> str:
        """
        Add an option leg — trade record only, no vol needed.

        Parameters
        ----------
        option_type : "call" or "put"
        expiry      : YYYY-MM-DD
        strike      : strike price
        lots        : contracts (negative = short, positive = long)
        price_paid  : premium paid (long) or received (short) per share
        instrument  : pricing model (default "whaley" = BAW American)
        lot_size    : units per contract (default = book lot_size)

        Returns
        -------
        leg_id : unique identifier for this leg
        """
        leg_id = str(uuid.uuid4())[:8]
        self.option_legs.append(OptionLegRecord(
            leg_id      = leg_id,
            option_type = option_type,
            expiry      = expiry,
            strike      = strike,
            lots        = lots,
            price_paid  = price_paid,
            instrument  = instrument,
            lot_size    = lot_size or self.lot_size,
            status      = "open",
            trade_date  = date.today().isoformat(),
            notes       = notes,
        ))
        self.updated_at = datetime.now().isoformat()
        return leg_id

    def add_underlying(
        self,
        lots:        float,
        entry_price: float,
        direction:   Literal["long", "short"] = "long",
        instrument:  Literal["equity", "futures"] = "equity",
        notes:       str = "",
    ) -> None:
        """Add underlying stock or futures position."""
        self.underlying_pos.append(UnderlyingRecord(
            lots        = lots,
            entry_price = entry_price,
            direction   = direction,
            instrument  = instrument,
            trade_date  = date.today().isoformat(),
            notes       = notes,
        ))
        self.updated_at = datetime.now().isoformat()

    def close_leg(self, leg_id: str, close_price: float) -> None:
        """Close an option leg and record realised P&L."""
        for leg in self.option_legs:
            if leg.leg_id == leg_id and leg.status == "open":
                leg.status      = "closed"
                leg.close_price = close_price
                self.closed_pnl += leg.closed_pnl
                self.updated_at  = datetime.now().isoformat()
                return
        raise ValueError(f"Leg {leg_id} not found or already closed")

    # ------------------------------------------------------------------
    # Mark to market — fetch all data from market
    # ------------------------------------------------------------------

    def mark_to_market(
        self,
        client,
        opt=None,
        quant=None,
        vol_period: str = "1y",
    ) -> dict:
        """
        Update all market data from yesterday's close.

        Fetches:
          spot          : current underlying price
          leg.iv        : current IV per leg from options chain (if opt provided)
          risk_free_rate: 13-week T-bill (^IRX)
          div_yield     : from yfinance info
          book.vol      : historical vol (if quant provided) — for price range only

        Parameters
        ----------
        client     : StockClient instance
        opt        : OptionsAnalyzer instance (for leg IV from chain)
        quant      : QuantAnalytics instance (for historical vol)
        vol_period : period for historical vol calculation

        Returns
        -------
        dict with all updated values
        """
        updated = {}

        # ── Spot ──────────────────────────────────────────────────────────
        price_data  = client.get_price(self.symbol)
        self.spot   = float(price_data["price"])
        updated["spot"] = self.spot

        # ── Risk-free rate from ^IRX (13-week T-bill) ─────────────────────
        try:
            tbill = client.get_price("^IRX")
            rf    = float(tbill.get("price", 0))
            if rf > 0:
                self.risk_free_rate = round(rf, 4)
                updated["risk_free_rate"] = self.risk_free_rate
        except Exception as e:
            print(f"  Risk-free rate update skipped: {e}")

        # ── Dividend yield from yfinance info ─────────────────────────────
        try:
            info = client.get_info(self.symbol)
            div  = (info.get("dividendYield")
                    or info.get("trailingAnnualDividendYield")
                    or 0.0)
            self.div_yield = round(float(div) * 100, 4)  # store as %
            updated["div_yield"] = self.div_yield
        except Exception as e:
            print(f"  Dividend yield update skipped: {e}")

        # ── Historical vol — for price range (±dStd) only ─────────────────
        if quant is not None:
            try:
                hv = quant.historical_volatility(self.symbol, period=vol_period)
                if hv and float(hv) > 0:
                    self.vol = round(float(hv), 4)
                    updated["vol"] = self.vol
            except Exception as e:
                print(f"  Historical vol update skipped: {e}")

        # ── IV per leg from options chain ─────────────────────────────────
        if opt is not None:
            updated["legs_iv"] = {}
            # group legs by expiry for efficient chain fetching
            expiries = set(l.expiry for l in self.option_legs if l.status == "open")
            for expiry in expiries:
                try:
                    chain = opt.chain(expiry)
                    for leg in self.option_legs:
                        if leg.expiry != expiry or leg.status != "open":
                            continue
                        # match by strike and type
                        match = chain[
                            (chain["strike"] == leg.strike) &
                            (chain["type"]   == leg.option_type)
                        ]
                        if not match.empty:
                            iv_raw = float(match["implied_volatility"].iloc[0])
                            # normalise: yfinance sometimes returns % not decimal
                            if iv_raw > 5.0:
                                iv_raw = iv_raw / 100.0
                            leg.iv = round(iv_raw, 4)
                            updated["legs_iv"][leg.leg_id] = leg.iv
                        else:
                            # fallback: nearest strike
                            sub = chain[chain["type"] == leg.option_type].copy()
                            if not sub.empty:
                                sub = sub.iloc[(sub["strike"] - leg.strike).abs().argsort()[:1]]
                                iv_raw = float(sub["implied_volatility"].iloc[0])
                                if iv_raw > 5.0:
                                    iv_raw = iv_raw / 100.0
                                leg.iv = round(iv_raw, 4)
                                updated["legs_iv"][leg.leg_id] = leg.iv
                except Exception as e:
                    print(f"  IV update for {expiry} skipped: {e}")

        self.updated_at = datetime.now().isoformat()
        return updated

    # ------------------------------------------------------------------
    # Pricing engine
    # ------------------------------------------------------------------

    def _make_engine(self, leg: OptionLegRecord, days_ahead: int = 0):
        """Build pricing engine for one leg using current market data."""
        from yfinance_api3.classes.pricing import Engine

        life = max(leg.life_days - days_ahead, 1)
        iv   = leg.iv if leg.iv > 0 else self.vol  # fallback to book vol

        opt_data = OptionData(
            strike      = leg.strike,
            option_type = OptionType.CALL if leg.option_type == "call" else OptionType.PUT,
            life_days   = life,
            lots        = leg.lots,
            price       = leg.price_paid,
            lot_size    = leg.lot_size,
        )
        und_data = UnderlyingData(
            underlyingValue = self.spot,
            underlyingVlt   = iv,
            dividendYield   = self.div_yield,
            riskFreeRate    = self.risk_free_rate,
            contractType    = self.contract_type,
            ticker          = self.symbol,
            currency        = self.currency,
        )

        if leg.instrument == "whaley":
            model = BaroneAdesiWhaley(opt_data, und_data)
        elif leg.instrument == "american":
            model = Binomial(opt_data, und_data, exercise=ExerciseType.AMERICAN)
        elif leg.instrument == "futures":
            und_data.contractType = ContractType.FUTURE
            model = BaroneAdesiWhaley(opt_data, und_data)
        else:
            model = BlackScholesModel(opt_data, und_data)

        return Engine(model=model)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def greeks_table(self, days_ahead: int = 0) -> pd.DataFrame:
        """Per-leg Greeks + totals row using current market data."""
        rows = []
        for leg in self.option_legs:
            if leg.status != "open":
                continue

            engine = self._make_engine(leg, days_ahead)
            out    = engine.get_model_outputs()
            sz     = leg.lot_size

            model_price = out["prima"]
            value       = model_price * leg.lots * sz
            cost        = leg.price_paid * leg.lots * sz
            pnl         = value - cost

            rows.append({
                "leg_id":      leg.leg_id,
                "type":        leg.option_type.capitalize(),
                "expiry":      leg.expiry,
                "strike":      leg.strike,
                "iv":          f"{leg.iv:.1%}" if leg.iv > 0 else "—",
                "model":       out.get("model", "—"),
                "lots":        leg.lots,
                "lot_size":    sz,
                "price_paid":  leg.price_paid,
                "model_price": round(model_price, 4),
                "value":       round(value, 2),
                "pnl":         round(pnl, 2),
                "delta":       round(out["delta"] * leg.lots * sz, 2),
                "gamma":       round(out["gamma"] * leg.lots * sz, 2),
                "vega":        round(out["vega"]  * leg.lots * sz, 2),
                "theta":       round(out["theta"] * leg.lots * sz, 2),
                "rho":         round(out["rho"]   * leg.lots * sz, 2),
            })

        # underlying
        for pos in self.underlying_pos:
            if pos.status != "open":
                continue
            sl      = pos.signed_lots
            pnl_und = (self.spot - pos.entry_price) * sl
            rows.append({
                "leg_id":      "UND",
                "type":        f"{'Long' if pos.direction=='long' else 'Short'} {pos.instrument}",
                "expiry":      "—",
                "strike":      pos.entry_price,
                "iv":          "—",
                "model":       "Linear",
                "lots":        sl,
                "lot_size":    1,
                "price_paid":  pos.entry_price,
                "model_price": self.spot,
                "value":       round(self.spot * sl, 2),
                "pnl":         round(pnl_und, 2),
                "delta":       round(sl, 2),
                "gamma":       0.0,
                "vega":        0.0,
                "theta":       0.0,
                "rho":         0.0,
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # totals
        num_cols = ["value","pnl","delta","gamma","vega","theta","rho"]
        totals   = {col: df[col].sum() for col in num_cols}
        totals.update({"leg_id":"TOTAL","type":"","expiry":"","strike":"",
                        "iv":"","model":"","lots":"","lot_size":"",
                        "price_paid":"","model_price":""})
        df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
        return df

    def summary(self, days_ahead: int = 0) -> dict:
        """Portfolio-level summary."""
        gt = self.greeks_table(days_ahead)
        if gt.empty:
            return {"symbol": self.symbol, "spot": self.spot}

        total      = gt[gt["leg_id"] == "TOTAL"].iloc[0]
        open_pnl   = float(total["pnl"])
        open_value = float(total["value"])
        exp_pnl    = open_pnl + self.closed_pnl
        delta      = float(total["delta"])
        model_date = (pd.Timestamp.today() + pd.DateOffset(days=days_ahead)).strftime("%Y-%m-%d")

        return {
            "symbol":         self.symbol,
            "spot":           round(self.spot, 4),
            "vol":            round(self.vol, 4),
            "risk_free_rate": self.risk_free_rate,
            "div_yield":      self.div_yield,
            "days_ahead":     days_ahead,
            "model_date":     model_date,
            "closed_pnl":     round(self.closed_pnl, 2),
            "open_value":     round(open_value, 2),
            "open_pnl":       round(open_pnl, 2),
            "expected_pnl":   round(exp_pnl, 2),
            "delta":          round(delta, 2),
            "delta_dollars":  round(delta * self.spot, 2),
            "gamma":          round(float(total["gamma"]), 4),
            "vega":           round(float(total["vega"]),  2),
            "theta":          round(float(total["theta"]), 2),
            "rho":            round(float(total["rho"]),   2),
        }

    def payoff_curves(
        self,
        days_ahead: int  = 0,
        d_std: float     = 3.0,
        days: int        = 30,
        steps: int       = 150,
    ) -> pd.DataFrame:
        """
        P&L curves — one row per price node, one column pair per leg.

        Price grid built ONCE from book.vol (historical vol) so all legs
        and the underlying share the exact same spot grid.

        Parameters
        ----------
        days_ahead : shift model N days forward (reduces leg life)
        d_std      : number of std devs for price range (default 3)
        days       : horizon in days for price range (default 30)
        steps      : number of price nodes (default 150)

        Columns
        -------
        underlyingValue              : spot price at each node
        leg_{id}_{type}_{K}_hoy     : P&L Today per option leg
        leg_{id}_{type}_{K}_vcto    : P&L at expiry per option leg
        und_hoy / und_vcto          : underlying contribution
        P&L Hoy                     : total P&L today
        P&L Vcto                    : total P&L at expiry

        Export: df.to_csv("payoff.csv")
        """
        # ── single price grid — ONE place, used everywhere ────────────────────
        coef   = np.sqrt(days / 365.0) * self.vol
        spot_lo = self.spot * np.exp(-coef * d_std)
        spot_hi = self.spot * np.exp( coef * d_std)
        spots   = np.linspace(spot_lo, spot_hi, steps)

        rows       = {"underlyingValue": spots}
        total_hoy  = np.zeros(steps)
        total_vcto = np.zeros(steps)

        # ── option legs ───────────────────────────────────────────────────────
        for leg in self.option_legs:
            if leg.status != "open":
                continue

            cp     = 1.0 if leg.option_type == "call" else -1.0
            col    = f"leg_{leg.leg_id}_{leg.option_type}_{int(leg.strike)}"
            engine = self._make_engine(leg, days_ahead)
            orig   = engine.model.underlyingObj.underlyingValue

            hoy_leg  = np.zeros(steps)
            vcto_leg = np.zeros(steps)

            for i, s in enumerate(spots):
                engine.model.underlyingObj.underlyingValue = float(s)
                out         = engine.model.get_model_outputs()
                intrinsic   = max(cp * (s - leg.strike), 0.0)
                hoy_leg[i]  = (out["prima"] - leg.price_paid) * leg.lots * leg.lot_size
                vcto_leg[i] = (intrinsic    - leg.price_paid) * leg.lots * leg.lot_size

            engine.model.underlyingObj.underlyingValue = orig

            rows[f"{col}_hoy"]  = hoy_leg
            rows[f"{col}_vcto"] = vcto_leg
            total_hoy  += hoy_leg
            total_vcto += vcto_leg

        # ── underlying ────────────────────────────────────────────────────────
        und_hoy  = np.zeros(steps)
        und_vcto = np.zeros(steps)
        for pos in self.underlying_pos:
            if pos.status != "open":
                continue
            sl        = pos.signed_lots
            und_hoy  += (spots - pos.entry_price) * sl
            und_vcto += (spots - pos.entry_price) * sl

        rows["und_hoy"]  = und_hoy
        rows["und_vcto"] = und_vcto
        total_hoy  += und_hoy
        total_vcto += und_vcto

        rows["P&L Hoy"]  = total_hoy
        rows["P&L Vcto"] = total_vcto

        return pd.DataFrame(rows).round(2)


    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "symbol":         self.symbol,
            "name":           self.name,
            "contract_type":  self.contract_type.value,
            "lot_size":       self.lot_size,
            "currency":       self.currency,
            "spot":           self.spot,
            "vol":            self.vol,
            "risk_free_rate": self.risk_free_rate,
            "div_yield":      self.div_yield,
            "closed_pnl":     self.closed_pnl,
            "created_at":     self.created_at,
            "updated_at":     datetime.now().isoformat(),
            "option_legs":    [l.to_dict() for l in self.option_legs],
            "underlying_pos": [p.to_dict() for p in self.underlying_pos],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "PositionsBook":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        book = cls(
            symbol        = data["symbol"],
            name          = data.get("name"),
            contract_type = ContractType(data.get("contract_type", "S")),
            lot_size      = data.get("lot_size", 100.0),
            currency      = data.get("currency", "USD"),
        )
        book.spot           = data.get("spot", 0.0)
        book.vol            = data.get("vol", 0.30)
        book.risk_free_rate = data.get("risk_free_rate", 4.50)
        book.div_yield      = data.get("div_yield", 0.0)
        book.closed_pnl     = data.get("closed_pnl", 0.0)
        book.created_at     = data.get("created_at", datetime.now().isoformat())
        book.updated_at     = data.get("updated_at", datetime.now().isoformat())
        book.option_legs    = [OptionLegRecord.from_dict(l)
                               for l in data.get("option_legs", [])]
        book.underlying_pos = [UnderlyingRecord.from_dict(p)
                               for p in data.get("underlying_pos", [])]
        return book

    def __repr__(self) -> str:
        open_legs = sum(1 for l in self.option_legs if l.status == "open")
        return (f"PositionsBook({self.symbol}  spot=${self.spot:,.2f}  "
                f"legs={open_legs}  closed_pnl=${self.closed_pnl:,.2f})")


# ---------------------------------------------------------------------------
# PortfolioBook
# ---------------------------------------------------------------------------

class PortfolioBook:
    """Aggregates multiple PositionsBooks into a portfolio view."""

    def __init__(self, name: str = "Portfolio") -> None:
        self.name        = name
        self.books:      dict[str, PositionsBook] = {}
        self.book_paths: dict[str, str]           = {}
        self.created_at  = datetime.now().isoformat()

    def add_book(self, book: PositionsBook, path: str = None) -> None:
        self.books[book.symbol] = book
        if path:
            self.book_paths[book.symbol] = path

    def remove_book(self, symbol: str) -> None:
        self.books.pop(symbol.upper(), None)
        self.book_paths.pop(symbol.upper(), None)

    def mark_to_market(self, client, opt_factory=None, quant=None) -> None:
        """Mark all books to market. opt_factory(symbol) returns OptionsAnalyzer."""
        for symbol, book in self.books.items():
            opt = opt_factory(symbol) if opt_factory else None
            book.mark_to_market(client, opt=opt, quant=quant)
            print(f"  ✓ {symbol}: spot=${book.spot:,.2f}  vol={book.vol:.1%}")

    def summary(self, days_ahead: int = 0) -> pd.DataFrame:
        rows = []
        for book in self.books.values():
            s = book.summary(days_ahead)
            rows.append(s)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        num_cols = ["open_pnl","closed_pnl","expected_pnl",
                    "delta","delta_dollars","gamma","vega","theta","rho"]
        totals = {col: df[col].sum() for col in num_cols if col in df.columns}
        totals["symbol"] = "TOTAL"
        totals["spot"]   = ""
        return pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "name":       self.name,
            "created_at": self.created_at,
            "updated_at": datetime.now().isoformat(),
            "book_paths": self.book_paths,
            "symbols":    list(self.books.keys()),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PortfolioBook":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        port            = cls(name=data.get("name", "Portfolio"))
        port.created_at = data.get("created_at", datetime.now().isoformat())
        port.book_paths = data.get("book_paths", {})
        for symbol, book_path in port.book_paths.items():
            if os.path.exists(book_path):
                port.books[symbol] = PositionsBook.load(book_path)
        return port

    def __repr__(self) -> str:
        return f"PortfolioBook({self.name}  books={list(self.books.keys())})"


# ---------------------------------------------------------------------------
# WatchList
# ---------------------------------------------------------------------------

class WatchList:
    """Saved tickers with notes."""

    def __init__(self, name: str = "Watchlist") -> None:
        self.name    = name
        self.tickers: dict[str, dict] = {}

    def add(self, symbol: str, notes: str = "", tags: list = None) -> None:
        self.tickers[symbol.upper()] = {
            "symbol":   symbol.upper(),
            "notes":    notes,
            "tags":     tags or [],
            "added_at": date.today().isoformat(),
        }

    def remove(self, symbol: str) -> None:
        self.tickers.pop(symbol.upper(), None)

    def scan(self, client) -> pd.DataFrame:
        rows = []
        for sym, meta in self.tickers.items():
            try:
                p  = client.get_price(sym)
                pc = client.get_price_change(sym)
                rows.append({"symbol": sym, "price": p["price"],
                             "change_pct": pc.get("change_pct"),
                             "notes": meta["notes"]})
            except Exception as e:
                rows.append({"symbol": sym, "price": None,
                             "change_pct": None, "notes": str(e)})
        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"name": self.name, "tickers": self.tickers}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "WatchList":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        wl         = cls(name=data.get("name", "Watchlist"))
        wl.tickers = data.get("tickers", {})
        return wl

    def __repr__(self) -> str:
        return f"WatchList({self.name}  tickers={list(self.tickers.keys())})"
