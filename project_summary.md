# yfinance_api3 — Project Summary
## Quantitative Finance Library

---

## Project Overview

A professional-grade quantitative finance Python library built around `yfinance`.
Covers: options analysis, position tracking, portfolio optimisation, seasonality,
factor models, Monte Carlo simulation, Bitcoin power law, and 40+ Plotly charts.

---

## Project Structure

```
yfinance_api3/
├── __init__.py
├── classes/
│   ├── stock_client.py        — yfinance wrapper + TTL cache
│   ├── quant_analytics.py     — metrics, returns, seasonality, Kelly...
│   ├── options.py             — OptionsAnalyzer: chain, Greeks, GEX, PCR, max pain
│   ├── options_strategy.py    — multi-leg strategy builder + B-S pricing
│   ├── pricing.py             — BlackScholesModel, Binomial, BAW, Engine
│   └── positions_book.py      — PositionsBook, PortfolioBook, WatchList
└── modules/
    ├── plots.py               — 40+ Plotly charts
    ├── portfolio.py           — efficient frontier, max Sharpe, risk parity
    ├── backtest.py            — strategy backtesting
    ├── montecarlo.py          — historical, normal, Student-t simulation
    ├── factors.py             — Fama-French FF3/FF5/MOM/FF6
    ├── etf.py                 — ETF concentration analysis
    ├── alerts.py              — AlertEngine + options alerts
    ├── report.py              — HTML report builder
    └── powerlaw.py            — Bitcoin Power Law model
```

---

## Key Architecture Decisions

### Pricing Models (pricing.py)
```
instrument="whaley"    → BaroneAdesiWhaley  ← DEFAULT (fast American)
instrument="american"  → Binomial CRR 200 steps
instrument="equity"    → BlackScholesModel (European)
instrument="futures"   → BaroneAdesiWhaley (Black-76)
```

**Conventions:**
- `riskFreeRate`, `dividendYield` stored as % (3.5 = 3.5%), divided by 100 internally via `.r` and `.q` properties
- `lots`: positive = long, negative = short
- Theta, Vega, Rho in Binomial use B-S approximation (standard industry practice)
- BAW uses inline `_call_price`/`_put_price` closures to avoid recursion

### PositionsBook Design
```
BOOK  = trade record (what you traded)
  - add_option(type, expiry, strike, lots, price_paid)  — NO vol
  - add_underlying(lots, entry_price, direction)

MARKET = current data from mark_to_market(client, opt, quant)
  - spot          ← client.get_price(symbol)
  - leg.iv        ← opt.chain() matched by strike+expiry
  - risk_free_rate← client.get_price("^IRX")
  - div_yield     ← client.get_info()["dividendYield"]
  - book.vol      ← quant.historical_volatility() — for price range ONLY

PRICING: each leg uses leg.iv (chain IV) for model, book.vol for range
```

### Payoff Curves Framework
```
1. Price grid: spot × exp(±dStd × book.vol × √days)
   - dStd=3, days=30 (30-day horizon, 3σ) — default
   - book.vol = historical vol (stable, not distorted by chain IV)

2. For each node:
   PL Vcto = intrinsic value at expiry (pure math, no model)
             = max(cp × (spot - strike), 0) - price_paid × lots × lot_size
   PL Hoy  = model_price(spot=node, T=actual_DTE, iv=leg.iv) - price_paid

3. Underlying: linear P&L added directly at each node
```

### Plotly Architecture Notes
1. Date vs categorical axis → use domain-based layout (no make_subplots)
2. `_apply_layout()` resets axis types → set types AFTER `_apply_layout`
3. Seasonality uses `_apply_layout` NOT `_apply_date_layout`
4. **go.Table + add_hline/add_vline**: NEVER use `row=` targeting a table subplot
   → use `add_shape(xref="xN", yref="yN")` with explicit axis references
5. `_apply_layout` auto-detects Table traces via `isinstance(t, go.Table)`
   and strips xaxis/yaxis keys to avoid PlotlyKeyError

---

## Options Module (options.py)

### OptionsAnalyzer Methods
```
expiries()              nearest_expiry(n)
chain(expiry)           chain_all()
summary(expiry)         greeks(expiry)        ← @staticmethod _black_scholes_greeks
put_call_ratio()        max_pain(expiry)       ← "knockout price"
gex_by_strike()         gex_by_expiry()
gex_total()             unusual_activity()
vol_surface()           oi_by_strike()
```

**Key fixes:**
- `_black_scholes_greeks` must be `@staticmethod` — without it, `self` is passed as `spot`
- IV normalisation: `if iv > 5.0: iv = iv / 100.0` — yfinance returns IV as % sometimes
- `oi_by_strike` uses `chain()` directly (not `gamma_exposure` which was removed)

### GEX Keys (gex_total returns)
```python
{
    "total_net_gex": float,
    "total_call_gex": float,
    "total_put_gex": float,
    "flip_strike": float | None,
    "regime": "long_gamma" | "short_gamma",
    "regime_label": str,
    "dominant_expiry": str,
}
```

### GEX by Expiry Columns
```
expiry, days_to_expiry, call_gex, put_gex, net_gex, abs_gex
```

---

## Seasonality Analysis (quant_analytics.py)

### Reliability Score (improved)
```python
size_score        = min(n / 10.0, 1.0)           # 40%
consistency_score = abs(win_rate - 0.5) * 2.0    # 40%
sharpe_score      = min(abs(sharpe) / 2.0, 1.0)  # 20%
skew_penalty      = max(min(-skew * 0.05, 0.15), 0.0)

score = size_score*0.4 + consistency_score*0.4 + sharpe_score*0.2 - skew_penalty
# >= 0.65 → "high", >= 0.35 → "medium", < 0.35 → "low"
```

### New Seasonality Methods
```python
seasonality_holding_sharpe(symbol, period, holding_years=[2,5])
  # Buy-and-hold Sharpe: enter month M, hold H years

seasonality_holding_drawdown(symbol, period, holding_years=[2,5])
  # Max drawdown during hold period

seasonality_decade_analysis(symbol, period, holding_years=1)
  # Hit rate by decade (2000s, 2010s, 2020s)

seasonality_cross_asset(symbols, period, holding_years=[2,5])
  # Compare seasonal edge across SPY, QQQ, GLD, BTC-USD

seasonality_combined_score(symbol, period, holding_years=[2,5])
  # score = (sharpe_2y×0.3 + sharpe_5y×0.3 + win_2y×0.2 + win_5y×0.2) × reliability
  # signals: "strong buy" | "buy" | "neutral" | "avoid"
```

---

## Bitcoin Power Law (modules/powerlaw.py)

```python
pl = PowerLaw(corridor_low=10, corridor_high=90)
pl.fit()

pl.summary()                    # full model summary
pl.current_position()           # phase, corridor %, price vs model
pl.fair_value("2028-01-01")    # floor/median/ceiling at future date
pl.cycle_analysis()             # historical peaks and troughs
pl.forecast(years=4)            # forward projection DataFrame

plots.powerlaw_chart(pl)        # log-log price + corridor
plots.powerlaw_residuals(pl)    # oscillator + corridor position
plots.powerlaw_forecast(pl, years=4)  # forward projection
```

---

## Plots Reference (plots.py — 46 functions)

### Equity
```
cumulative_returns, drawdown, rolling_volatility, rolling_sharpe,
returns_distribution, correlation_heatmap, metrics_bar, scatter
```

### Risk
```
monte_carlo, best_worst_days, extreme_days_concentration
```

### Seasonality
```
seasonality, seasonality_heatmap, seasonality_comparison_clean,
seasonality_box, seasonality_combined_score
```

### Portfolio
```
efficient_frontier, kelly, backtest, rolling_returns
```

### Factors
```
factor_exposure, rolling_factor_betas, factor_comparison
```

### Options
```
options_chain, options_surface, options_oi_profile, options_put_call,
options_max_pain, options_gex, options_unusual
```

### Strategy
```
strategy_payoff, strategy_surface, strategy_greeks
```

### Positions
```
positions_book(book, days_ahead, d_std)   — dark theme, PL curves + summary
positions_legs(book, days_ahead)          — legs table + parameters
portfolio_summary(portfolio)              — multi-ticker summary
```

### Bitcoin
```
powerlaw_chart, powerlaw_residuals, powerlaw_forecast
```

---

## Standard Test Cell

```python
import sys
for k in list(sys.modules):
    if 'yfinance_api3' in k:
        del sys.modules[k]

from yfinance_api3.classes.stock_client import StockClient
from yfinance_api3.classes.quant_analytics import QuantAnalytics
from yfinance_api3.classes.options import OptionsAnalyzer
from yfinance_api3.classes.positions_book import PositionsBook, PortfolioBook
import yfinance_api3.modules.plots as plots

client = StockClient()
quant  = QuantAnalytics(client)

# Options
opt    = OptionsAnalyzer(client, "SPY")
expiry = opt.nearest_expiry(0)

# PositionsBook — INTC covered call
book = PositionsBook("INTC")
book.add_option("call", "2026-12-18", 85, -60, 18.11)
book.add_option("call", "2026-12-18", 90, -40, 21.50)
book.add_underlying(10000, 30.0, "long")

opt_intc = OptionsAnalyzer(client, "INTC")
book.mark_to_market(client, opt=opt_intc, quant=quant)
```

---

## Known Issues / Pending

- `positions_legs()` plot — layout needs redesign (domain proportions)
- `portfolio_summary()` — not yet dark-themed
- Greeks profile (delta curve) removed from positions_book — can add back
- Dash dashboard callbacks need updating for new PositionsBook API

---

## File Locations

```
yfinance_api3/classes/    → stock_client, quant_analytics, options,
                             options_strategy, pricing, positions_book
yfinance_api3/modules/    → plots, portfolio, backtest, montecarlo,
                             factors, etf, alerts, report, powerlaw
positions/                → INTC.json, portfolio.json, watchlist.json
```

---

*Generated: 2026-05-28 | Session summary for account migration*
