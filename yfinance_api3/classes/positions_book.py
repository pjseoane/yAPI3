"""
positions_book.py — Live options position tracker with persistence.

Replicates the Google Sheets workflow:
  - One PositionsBook per underlying / client / strategy
  - PortfolioBook aggregates multiple books
  - WatchList for saved tickers

All books persist to JSON and survive between sessions.

Usage
-----
    from yfinance_api3.classes.positions_book import (
        PositionsBook, PortfolioBook, WatchList
    )

    # create or load
    book = PositionsBook.load("positions/INTC.json")
    # or
    book = PositionsBook(symbol="INTC", name="INTC options book")

    # add legs
    book.add_option(
        option_type = "call",
        expiry      = "2026-12-18",
        strike      = 85,
        lots        = -60,          # negative = short
        price       = 18.11,        # premium received
        iv          = 0.61,
        instrument  = "equity",     # "equity" | "futures" | "american"
        lot_size    = 100,
    )
    book.add_underlying(lots=10000, entry_price=30.0, direction="long")

    # mark to market and analyse
    book.mark_to_market(client, days_ahead=0)
    print(book.greeks_table(days_ahead=0))
    print(book.summary(days_ahead=30))     # simulate 30 days ahead

    # save
    book.save("positions/INTC.json")

    # portfolio
    port = PortfolioBook()
    port.add_book(book)
    port.save("positions/portfolio.json")
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
    BlackScholesModel, Binomial, BaroneAdesiWhaley, Engine, PricingStrategy,
    OptionData, OptionType, UnderlyingData,
    ContractType, ExerciseType, Space,
)


# ---------------------------------------------------------------------------
# Leg record — stored in PositionsBook
# ---------------------------------------------------------------------------

@dataclass
class OptionLegRecord:
    """
    One option leg as stored in the book.

    leg_id      : unique identifier (auto-generated)
    option_type : "call" | "put"
    expiry      : YYYY-MM-DD
    strike      : strike price
    lots        : contracts (negative = short)
    price       : premium paid/received per unit
    iv          : implied volatility at time of trade (decimal)
    instrument  : "equity" | "futures" | "american"
    lot_size    : units per contract
    status      : "open" | "closed"
    close_price : price at which leg was closed (None if open)
    trade_date  : date leg was opened
    notes       : free text
    """
    leg_id:      str
    option_type: str
    expiry:      str
    strike:      float
    lots:        float
    price:       float
    iv:          float
    instrument:  str   = "equity"
    lot_size:    float = 100.0
    status:      str   = "open"
    close_price: float = None
    trade_date:  str   = None
    notes:       str   = ""

    @property
    def life_days(self) -> float:
        """Current days to expiry."""
        return max((pd.to_datetime(self.expiry).date() - date.today()).days, 0)

    @property
    def closed_pnl(self) -> float:
        """P&L for closed legs."""
        if self.status != "closed" or self.close_price is None:
            return 0.0
        cp = 1 if self.option_type == "call" else -1
        return (self.close_price - self.price) * self.lots * self.lot_size * cp

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OptionLegRecord":
        return cls(**d)


@dataclass
class UnderlyingRecord:
    """Stock or futures position in the book."""
    lots:        float
    entry_price: float
    direction:   str   = "long"     # "long" | "short"
    instrument:  str   = "equity"   # "equity" | "futures"
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
        return cls(**d)


# ---------------------------------------------------------------------------
# PositionsBook — one per underlying / client / strategy
# ---------------------------------------------------------------------------

class PositionsBook:
    """
    Options position tracker for one underlying.

    Tracks open and closed legs, computes live P&L and Greeks,
    and generates payoff curves using your pricing models.
    """

    def __init__(
        self,
        symbol:         str,
        name:           str          = None,
        spot:           float        = 0.0,
        vol:            float        = 0.30,
        risk_free_rate: float        = 3.5,    # % convention
        div_yield:      float        = 0.0,
        contract_type:  ContractType = ContractType.STOCK,
        lot_size:       float        = 100.0,
        currency:       str          = "USD",
    ) -> None:
        self.symbol         = symbol.upper()
        self.name           = name or f"{symbol} positions"
        self.spot           = spot
        self.vol            = vol
        self.risk_free_rate = risk_free_rate
        self.div_yield      = div_yield
        self.contract_type  = contract_type
        self.lot_size       = lot_size
        self.currency       = currency

        self.option_legs:    list[OptionLegRecord]   = []
        self.underlying_pos: list[UnderlyingRecord]  = []
        self.closed_pnl:     float                   = 0.0
        self.created_at:     str                     = datetime.now().isoformat()
        self.updated_at:     str                     = datetime.now().isoformat()

    # ------------------------------------------------------------------
    # Adding positions
    # ------------------------------------------------------------------

    def add_option(
        self,
        option_type: Literal["call", "put"],
        expiry:      str,
        strike:      float,
        lots:        float,
        price:       float,
        iv:          float,
        instrument:  Literal["equity", "futures", "american", "whaley"] = "whaley",
        lot_size:    float = None,
        notes:       str   = "",
    ) -> str:
        """
        Add an option leg to the book.

        Parameters
        ----------
        option_type : "call" or "put"
        expiry      : expiration date YYYY-MM-DD
        strike      : strike price
        lots        : contracts (negative = short, positive = long)
        price       : premium paid (long) or received (short) per unit
        iv          : implied volatility at time of trade (decimal, e.g. 0.25)
        instrument  : "whaley"   (BAW approximation — DEFAULT, fast American)
                      "american" (Binomial CRR — accurate, slow)
                      "equity"   (Black-Scholes European)
                      "futures"  (Black-76)
        lot_size    : units per contract (default = book lot_size)
        notes       : optional trade note

        Returns
        -------
        leg_id : unique identifier for this leg
        """
        leg_id = str(uuid.uuid4())[:8]
        leg = OptionLegRecord(
            leg_id      = leg_id,
            option_type = option_type,
            expiry      = expiry,
            strike      = strike,
            lots        = lots,
            price       = price,
            iv          = iv,
            instrument  = instrument,
            lot_size    = lot_size or self.lot_size,
            status      = "open",
            trade_date  = date.today().isoformat(),
            notes       = notes,
        )
        self.option_legs.append(leg)
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
        """
        Add underlying stock or futures position.

        lots        : number of shares / contracts (always positive)
        entry_price : price at which position was entered
        direction   : "long" or "short"
        """
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
        """
        Close an option leg and record the P&L.

        leg_id      : the ID returned by add_option()
        close_price : premium at which the leg was closed
        """
        for leg in self.option_legs:
            if leg.leg_id == leg_id and leg.status == "open":
                leg.status      = "closed"
                leg.close_price = close_price
                self.closed_pnl += leg.closed_pnl
                self.updated_at  = datetime.now().isoformat()
                return
        raise ValueError(f"Leg {leg_id} not found or already closed")

    # ------------------------------------------------------------------
    # Pricing engine helpers
    # ------------------------------------------------------------------

    def _underlying_data(self, days_ahead: int = 0) -> UnderlyingData:
        """Build UnderlyingData for the current book state."""
        return UnderlyingData(
            underlyingValue = self.spot,
            underlyingVlt   = self.vol,
            dividendYield   = self.div_yield,
            riskFreeRate    = self.risk_free_rate,
            contractType    = self.contract_type,
            ticker          = self.symbol,
            currency        = self.currency,
        )

    def _make_engine(
        self,
        leg: OptionLegRecord,
        days_ahead: int = 0,
    ) -> Engine:
        """Build an Engine for one leg, optionally shifted N days ahead."""
        life = max(leg.life_days - days_ahead, 0)

        opt_data = OptionData(
            strike      = leg.strike,
            option_type = OptionType.CALL if leg.option_type == "call" else OptionType.PUT,
            life_days   = life,
            lots        = leg.lots,
            price       = leg.price,
            lot_size    = leg.lot_size,
            expiry_date = leg.expiry,
        )
        und_data = self._underlying_data(days_ahead)
        und_data.underlyingVlt = leg.iv   # use trade IV for model

        if leg.instrument == "whaley":
            # BAW quadratic approximation — fast American, used as default
            model = BaroneAdesiWhaley(opt_data, und_data)
        elif leg.instrument == "american":
            # Binomial CRR — accurate but slow, use for edge cases
            model = Binomial(opt_data, und_data,
                             exercise=ExerciseType.AMERICAN)
        elif leg.instrument == "futures":
            und_data.contractType = ContractType.FUTURE
            model = BaroneAdesiWhaley(opt_data, und_data)  # BAW works for futures too
        else:
            # equity European (BS)
            model = BlackScholesModel(opt_data, und_data)

        return Engine(model=model)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def mark_to_market(self, client, days_ahead: int = 0) -> None:
        """
        Update spot price and optionally IV from live market data.

        Parameters
        ----------
        client     : StockClient instance
        days_ahead : shift model date N days forward (reduces life_days)
        """
        price_data  = client.get_price(self.symbol)
        self.spot   = float(price_data["price"])
        self.updated_at = datetime.now().isoformat()

    def greeks_table(self, days_ahead: int = 0) -> pd.DataFrame:
        """
        Per-leg Greeks table + totals row.

        Shows the model output for each open leg at current spot,
        optionally shifted N days ahead.

        Returns
        -------
        DataFrame matching your sheet layout:
          type, expiry, strike, iv, model, lots, price_paid,
          model_price, value, pnl, delta, gamma, vega, theta, rho
        """
        rows = []
        for leg in self.option_legs:
            if leg.status != "open":
                continue

            engine = self._make_engine(leg, days_ahead)
            out    = engine.get_model_outputs()
            lots   = leg.lots
            sz     = leg.lot_size
            sign   = 1 if leg.option_type == "call" else -1

            model_price = out["prima"]
            value       = model_price * lots * sz
            cost        = leg.price * lots * sz
            pnl         = value - cost

            rows.append({
                "leg_id":     leg.leg_id,
                "type":       leg.option_type.capitalize(),
                "expiry":     leg.expiry,
                "strike":     leg.strike,
                "iv":         f"{leg.iv:.1%}",
                "model":      out["model"],
                "exercise":   out["exercise"],
                "lots":       lots,
                "lot_size":   sz,
                "price_paid": leg.price,
                "model_price": round(model_price, 4),
                "value":      round(value, 2),
                "pnl":        round(pnl, 2),
                "delta":      round(out["delta"] * lots * sz, 2),
                "gamma":      round(out["gamma"] * lots * sz, 2),
                "vega":       round(out["vega"]  * lots * sz, 2),
                "theta":      round(out["theta"] * lots * sz, 2),
                "rho":        round(out["rho"]   * lots * sz, 2),
            })

        # underlying contribution
        for pos in self.underlying_pos:
            if pos.status != "open":
                continue
            sl = pos.signed_lots
            pnl_und = (self.spot - pos.entry_price) * sl
            rows.append({
                "leg_id":     "UND",
                "type":       f"{'Long' if pos.direction=='long' else 'Short'} {pos.instrument}",
                "expiry":     "—",
                "strike":     pos.entry_price,
                "iv":         "—",
                "model":      "Linear",
                "exercise":   "—",
                "lots":       sl,
                "lot_size":   1,
                "price_paid": pos.entry_price,
                "model_price": self.spot,
                "value":      round(self.spot * sl, 2),
                "pnl":        round(pnl_und, 2),
                "delta":      round(sl, 2),
                "gamma":      0.0,
                "vega":       0.0,
                "theta":      0.0,
                "rho":        0.0,
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # totals row
        num_cols = ["value", "pnl", "delta", "gamma",
                    "vega", "theta", "rho"]
        totals = {col: df[col].sum() for col in num_cols}
        totals.update({
            "leg_id": "TOTAL", "type": "", "expiry": "", "strike": "",
            "iv": "", "model": "", "exercise": "", "lots": "",
            "lot_size": "", "price_paid": "",
            "model_price": "",
        })
        df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
        return df

    def pnl_table(self, days_ahead: int = 0) -> pd.DataFrame:
        """
        P&L summary — open + closed positions.
        """
        rows = []
        for leg in self.option_legs:
            engine = self._make_engine(leg, days_ahead)
            out    = engine.get_model_outputs()
            sz     = leg.lot_size

            if leg.status == "open":
                model_price = out["prima"]
                open_pnl    = (model_price - leg.price) * leg.lots * sz
                closed_pnl  = 0.0
            else:
                open_pnl    = 0.0
                closed_pnl  = leg.closed_pnl

            rows.append({
                "leg_id":     leg.leg_id,
                "type":       leg.option_type.capitalize(),
                "expiry":     leg.expiry,
                "strike":     leg.strike,
                "lots":       leg.lots,
                "price_paid": leg.price,
                "status":     leg.status,
                "open_pnl":   round(open_pnl,   2),
                "closed_pnl": round(closed_pnl, 2),
                "total_pnl":  round(open_pnl + closed_pnl, 2),
            })

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def summary(self, days_ahead: int = 0) -> dict:
        """
        Portfolio-level summary — matches your sheet's Summary panel.

        Returns
        -------
        dict with:
          symbol, spot, days_ahead, model_date
          closed_pnl, open_value, open_pnl, expected_pnl (= open + closed)
          delta, delta_dollars, gamma, vega, theta, rho
        """
        gt     = self.greeks_table(days_ahead)
        if gt.empty:
            return {"symbol": self.symbol, "spot": self.spot}

        total  = gt[gt["leg_id"] == "TOTAL"].iloc[0]

        # open P&L from pnl column (excludes the TOTAL row text fields)
        open_pnl   = float(total["pnl"])
        open_value = float(total["value"])
        exp_pnl    = open_pnl + self.closed_pnl

        # delta in dollar terms
        delta       = float(total["delta"])
        delta_usd   = delta * self.spot

        model_date = (pd.to_datetime(date.today())
                      + pd.DateOffset(days=days_ahead)).strftime("%Y-%m-%d")

        return {
            "symbol":        self.symbol,
            "spot":          round(self.spot, 4),
            "days_ahead":    days_ahead,
            "model_date":    model_date,
            "risk_free_rate": self.risk_free_rate,
            "vol":           self.vol,
            "closed_pnl":   round(self.closed_pnl, 2),
            "open_value":   round(open_value, 2),
            "open_pnl":     round(open_pnl, 2),
            "expected_pnl": round(exp_pnl, 2),
            "delta":        round(delta, 2),
            "delta_dollars": round(delta_usd, 2),
            "gamma":        round(float(total["gamma"]), 4),
            "vega":         round(float(total["vega"]),  2),
            "theta":        round(float(total["theta"]), 2),
            "rho":          round(float(total["rho"]),   2),
        }

    def payoff_curves(
        self,
        days_ahead: int = 0,
        space: Space    = None,
    ) -> pd.DataFrame:
        """
        Combined strategy payoff curves across price range.

        Columns: underlyingValue, P&L Vcto, P&L Hoy, delta, gamma,
                 theta, vega, rho

        +days_ahead reduces life_days for all legs — simulates time passing.
        """
        if space is None:
            space = Space(dStd=3, days=days_ahead or 1, steps=100)

        engines = []
        for leg in self.option_legs:
            if leg.status == "open":
                engines.append(self._make_engine(leg, days_ahead))

        # add underlying P&L
        und_data = self._underlying_data()
        for pos in self.underlying_pos:
            if pos.status == "open":
                und_df = und_data.get_payoff(
                    space, lots=pos.signed_lots, price=pos.entry_price
                )
                engines.append(und_df)   # DataFrame, not Engine

        if not engines:
            return pd.DataFrame()

        # get option payoffs
        strategy_engines = [e for e in engines if isinstance(e, Engine)]
        und_frames       = [e for e in engines if isinstance(e, pd.DataFrame)]

        strat = PricingStrategy(engines=strategy_engines)
        totals = strat.get_strategy_totals(space)

        # add underlying contribution
        for und_df in und_frames:
            # align on underlyingValue
            und_aligned = np.interp(
                totals["underlyingValue"].values,
                und_df["underlyingValue"].values,
                und_df["P&L Hoy"].values,
            )
            totals["P&L Vcto"] += np.interp(
                totals["underlyingValue"].values,
                und_df["underlyingValue"].values,
                und_df["P&L Vcto"].values,
            )
            totals["P&L Hoy"]  += und_aligned
            totals["delta"]    += und_df["delta"].mean()

        return totals.round(2)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save book to JSON file. Creates directories if needed."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "symbol":         self.symbol,
            "name":           self.name,
            "spot":           self.spot,
            "vol":            self.vol,
            "risk_free_rate": self.risk_free_rate,
            "div_yield":      self.div_yield,
            "contract_type":  self.contract_type.value,
            "lot_size":       self.lot_size,
            "currency":       self.currency,
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
        """Load book from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        book = cls(
            symbol         = data["symbol"],
            name           = data.get("name"),
            spot           = data.get("spot", 0.0),
            vol            = data.get("vol", 0.30),
            risk_free_rate = data.get("risk_free_rate", 3.5),
            div_yield      = data.get("div_yield", 0.0),
            contract_type  = ContractType(data.get("contract_type", "S")),
            lot_size       = data.get("lot_size", 100.0),
            currency       = data.get("currency", "USD"),
        )
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
        return (f"PositionsBook({self.symbol}  "
                f"spot=${self.spot:,.2f}  "
                f"legs={open_legs}  "
                f"closed_pnl=${self.closed_pnl:,.2f})")


# ---------------------------------------------------------------------------
# PortfolioBook — aggregates multiple PositionsBooks
# ---------------------------------------------------------------------------

class PortfolioBook:
    """
    Aggregates multiple PositionsBooks into a portfolio view.

    Usage
    -----
        port = PortfolioBook(name="My portfolio")
        port.add_book(intc_book)
        port.add_book(spy_book)

        print(port.summary())
        port.save("positions/portfolio.json")
    """

    def __init__(self, name: str = "Portfolio") -> None:
        self.name       = name
        self.books:     dict[str, PositionsBook] = {}
        self.book_paths: dict[str, str]          = {}
        self.created_at = datetime.now().isoformat()

    def add_book(self, book: PositionsBook, path: str = None) -> None:
        """Add a PositionsBook. Optionally record its save path."""
        self.books[book.symbol] = book
        if path:
            self.book_paths[book.symbol] = path

    def remove_book(self, symbol: str) -> None:
        self.books.pop(symbol.upper(), None)
        self.book_paths.pop(symbol.upper(), None)

    def summary(self, days_ahead: int = 0) -> pd.DataFrame:
        """
        One row per ticker — matches your portfolio summary tab.

        Columns: symbol, spot, open_pnl, closed_pnl, expected_pnl,
                 delta, delta_dollars, gamma, vega, theta, rho
        """
        rows = []
        for sym, book in self.books.items():
            s = book.summary(days_ahead)
            rows.append(s)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # totals row
        num_cols = ["open_pnl", "closed_pnl", "expected_pnl",
                    "delta", "delta_dollars", "gamma", "vega", "theta", "rho"]
        totals   = {col: df[col].sum() for col in num_cols if col in df.columns}
        totals["symbol"] = "TOTAL"
        totals["spot"]   = ""
        df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
        return df

    def total_greeks(self, days_ahead: int = 0) -> dict:
        """Aggregate Greeks across all books."""
        totals = {g: 0.0 for g in ["delta", "delta_dollars",
                                    "gamma", "vega", "theta", "rho"]}
        for book in self.books.values():
            s = book.summary(days_ahead)
            for g in totals:
                totals[g] += s.get(g, 0.0)
        return {k: round(v, 4) for k, v in totals.items()}

    def save(self, path: str) -> None:
        """Save portfolio index to JSON (saves book paths, not book data)."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "name":        self.name,
            "created_at":  self.created_at,
            "updated_at":  datetime.now().isoformat(),
            "book_paths":  self.book_paths,
            "symbols":     list(self.books.keys()),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PortfolioBook":
        """Load portfolio and all referenced books from their JSON files."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        port            = cls(name=data.get("name", "Portfolio"))
        port.created_at = data.get("created_at", datetime.now().isoformat())
        port.book_paths = data.get("book_paths", {})

        for symbol, book_path in port.book_paths.items():
            if os.path.exists(book_path):
                book = PositionsBook.load(book_path)
                port.books[symbol] = book

        return port

    def __repr__(self) -> str:
        return (f"PortfolioBook({self.name}  "
                f"books={list(self.books.keys())})")


# ---------------------------------------------------------------------------
# WatchList — saved tickers + notes
# ---------------------------------------------------------------------------

class WatchList:
    """
    Saved ticker list with notes — persists between sessions.

    Usage
    -----
        wl = WatchList.load("positions/watchlist.json")
        wl.add("NVDA", notes="earnings play, IV crush expected")
        wl.scan(client)    # quick snapshot of all tickers
        wl.save("positions/watchlist.json")
    """

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
        """
        Fetch quick quote for all tickers.

        Returns DataFrame: symbol, price, change_pct, notes
        """
        rows = []
        for sym, meta in self.tickers.items():
            try:
                p  = client.get_price(sym)
                pc = client.get_price_change(sym)
                rows.append({
                    "symbol":     sym,
                    "price":      p["price"],
                    "change_pct": pc["change_pct"],
                    "notes":      meta["notes"],
                    "tags":       ", ".join(meta.get("tags", [])),
                })
            except Exception as e:
                rows.append({"symbol": sym, "price": None,
                             "change_pct": None, "notes": str(e)})
        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"name": self.name, "tickers": self.tickers},
                      f, indent=2)

    @classmethod
    def load(cls, path: str) -> "WatchList":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        wl         = cls(name=data.get("name", "Watchlist"))
        wl.tickers = data.get("tickers", {})
        return wl

    def __repr__(self) -> str:
        return f"WatchList({self.name}  tickers={list(self.tickers.keys())})"
