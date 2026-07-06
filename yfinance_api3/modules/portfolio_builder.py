"""
portfolio_builder.py — end-to-end Q1 portfolio construction pipeline.

Pipeline
--------
1. scatter_quadrants()      — score all symbols vs benchmark
2. Filter to Q1             — stocks beating benchmark on BOTH periods
3. correlation_dedup()      — drop redundant stocks (keep higher score)
4. Top-N by score           — enforce max_stocks cap
5. All four optimisers      — equal_weight, min_variance, max_sharpe, risk_parity
6. Backtest each            — run over historical window vs benchmark
7. Return PortfolioBuilderResult — everything in one place

Usage
-----
    from yfinance_api3.modules.portfolio_builder import build_optimal_portfolio
    import yfinance_api3.modules.plots as plots

    result = build_optimal_portfolio(
        quant, client, symbols,
        max_stocks=8,
        metric="sharpe",
        period_x="2y",
        period_y="5y",
        period="3y",
        benchmark="SPY",
        risk_free_rate=0.05,
        corr_threshold=0.75,
    )

    print(result)                          # summary table
    result.plot_frontier().show()          # efficient frontier
    result.plot_backtest("max_sharpe").show()
    result.plot_heatmap().show()           # correlation dedup heatmap
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from yfinance_api3.classes.quant_analytics import QuantAnalytics
from yfinance_api3.classes.stock_client import StockClient
import yfinance_api3.modules.plots as plots
import yfinance_api3.modules.portfolio as portfolio
import yfinance_api3.modules.backtest as backtest


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PortfolioBuilderResult:
    """
    Full output of build_optimal_portfolio().

    Attributes
    ----------
    universe        : original input symbols
    q1_symbols      : symbols that passed the Q1 filter
    dedup_symbols   : after correlation deduplication
    final_symbols   : final list after top-N cap  ← use these
    quadrant_df     : full scatter_quadrants() DataFrame
    dedup_result    : dict from correlation_dedup()
    optimisers      : dict of strategy name → PortfolioResult
    backtests       : dict of strategy name → BacktestResult
    frontier        : EfficientFrontier object
    params          : dict of parameters used
    """
    universe:      list[str]
    q1_symbols:    list[str]
    dedup_symbols: list[str]
    final_symbols: list[str]
    quadrant_df:   pd.DataFrame
    dedup_result:  dict
    optimisers:    dict
    backtests:     dict
    frontier:      object   # EfficientFrontier
    params:        dict

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """
        Comparison table of all four optimised portfolios.

        Rows    : strategy name
        Columns : return, volatility, sharpe, max_drawdown, weights...
        """
        rows = {}
        for name, opt in self.optimisers.items():
            bt = self.backtests.get(name)
            row = {
                "exp_return":   f"{opt.expected_return:.2%}",
                "volatility":   f"{opt.volatility:.2%}",
                "sharpe":       f"{opt.sharpe_ratio:.3f}",
            }
            if bt:
                row["bt_cagr"]      = f"{bt.metrics.get('cagr', 0):.2%}"
                row["bt_sharpe"]    = f"{bt.metrics.get('sharpe_ratio', 0):.3f}"
                row["bt_max_dd"]    = f"{bt.metrics.get('max_drawdown', 0):.2%}"
                row["bt_win_rate"]  = f"{bt.metrics.get('win_rate', 0):.2%}"
            for sym, w in zip(opt.symbols, opt.weights):
                row[sym] = f"{w:.1%}"
            rows[name] = row
        return pd.DataFrame(rows).T

    def best_strategy(self, by: str = "sharpe") -> str:
        """Return the strategy name with the highest backtest Sharpe (or 'sharpe'/'cagr')."""
        metric_map = {"sharpe": "sharpe_ratio", "cagr": "cagr",
                      "sortino": "sortino_ratio"}
        key = metric_map.get(by, "sharpe_ratio")
        best, best_val = None, -np.inf
        for name, bt in self.backtests.items():
            v = bt.metrics.get(key, -np.inf)
            if v > best_val:
                best_val = v
                best = name
        return best

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_frontier(self):
        """Efficient frontier chart."""
        return plots.efficient_frontier(self.frontier)

    def plot_backtest(self, strategy: str | None = None):
        """Backtest equity curve for one strategy (default: best Sharpe)."""
        strategy = strategy or self.best_strategy()
        return plots.backtest(self.backtests[strategy])

    def plot_heatmap(self):
        """Correlation deduplication heatmap for the Q1 universe."""
        scores = (self.quadrant_df
                  .set_index("symbol")["score"]
                  .reindex(self.q1_symbols))
        return plots.correlation_dedup_heatmap(
            _quant_ref[0],
            self.q1_symbols,
            period=self.params["period"],
            threshold=self.params["corr_threshold"],
            method="score",
            scores=scores,
        )

    def plot_scatter(self):
        """Q1 zoom scatter for the final symbol list."""
        return plots.scatter_zoom(
            _quant_ref[0],
            self.universe,
            quadrant=1,
            metric=self.params["metric"],
            period_x=self.params["period_x"],
            period_y=self.params["period_y"],
            benchmark=self.params["benchmark"],
            risk_free_rate=self.params["risk_free_rate"],
        )

    def __repr__(self) -> str:
        p = self.params
        lines = [
            "PortfolioBuilderResult",
            "=" * 56,
            f"  Universe      : {len(self.universe)} symbols",
            f"  Q1 filter     : {len(self.q1_symbols)} symbols  "
            f"(metric={p['metric']}, {p['period_x']} vs {p['period_y']})",
            f"  After dedup   : {len(self.dedup_symbols)} symbols  "
            f"(corr threshold={p['corr_threshold']:.0%})",
            f"  Final (top-N) : {len(self.final_symbols)} symbols  "
            f"(max_stocks={p['max_stocks']})",
            f"  Symbols       : {', '.join(self.final_symbols)}",
            "",
            "  Strategy comparison (backtest)",
            "  " + "-" * 52,
        ]
        for name, bt in self.backtests.items():
            m = bt.metrics
            lines.append(
                f"  {name:<16}  "
                f"CAGR {m.get('cagr',0):+.1%}  "
                f"Sharpe {m.get('sharpe_ratio',0):.2f}  "
                f"MaxDD {m.get('max_drawdown',0):.1%}"
            )
        best = self.best_strategy()
        lines += ["", f"  Best strategy : {best}"]
        return "\n".join(lines)


# Module-level quant reference for plot methods (set by build_optimal_portfolio)
_quant_ref: list = [None]


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def build_optimal_portfolio(
    quant: QuantAnalytics,
    symbols: list[str],
    max_stocks: int = 8,
    metric: str = "sharpe",
    period_x: str = "2y",
    period_y: str = "5y",
    period: str = "3y",
    benchmark: str = "SPY",
    risk_free_rate: float = 0.05,
    corr_threshold: float = 0.75,
    allow_short: bool = False,
    weight_bounds: tuple = (0.0, 1.0),
    backtest_strategy: str = "buy_hold",
    backtest_rebalance: str = "monthly",
    transaction_cost_bps: float = 10.0,
    verbose: bool = True,
) -> PortfolioBuilderResult:
    """
    Build an optimised portfolio from a universe of symbols.

    Pipeline
    --------
    1. Score all symbols via scatter_quadrants() on the chosen metric
    2. Filter to Q1 (beats benchmark on BOTH period_x and period_y)
    3. Deduplicate by correlation — for each correlated pair (>= corr_threshold)
       keep the stock with the higher Q1 score
    4. Cap to top max_stocks by score
    5. Run all four optimisers: equal_weight, min_variance, max_sharpe, risk_parity
    6. Backtest each optimised portfolio
    7. Compute efficient frontier

    Parameters
    ----------
    quant               : QuantAnalytics instance
    symbols             : full universe to screen
    max_stocks          : maximum number of stocks in the final portfolio
    metric              : screening metric — "sharpe" | "sortino" | "calmar" | ...
    period_x            : shorter period for quadrant x-axis (default "2y")
    period_y            : longer  period for quadrant y-axis (default "5y")
    period              : historical window for optimisation + backtest (default "3y")
    benchmark           : benchmark ticker (default "SPY")
    risk_free_rate      : annual risk-free rate (default 0.05)
    corr_threshold      : correlation threshold for deduplication (default 0.75)
    allow_short         : allow short positions in optimiser (default False)
    weight_bounds       : (min, max) per-asset weight bounds (default (0, 1))
    backtest_strategy   : strategy for backtesting — "buy_hold" | "max_sharpe" |
                          "min_variance" | "risk_parity" (default "buy_hold")
    backtest_rebalance  : rebalance frequency (default "monthly")
    transaction_cost_bps: round-trip transaction cost in bps (default 10)
    verbose             : print progress (default True)

    Returns
    -------
    PortfolioBuilderResult
    """
    _quant_ref[0] = quant

    def _log(msg):
        if verbose:
            print(f"  [portfolio_builder] {msg}")

    params = dict(
        max_stocks=max_stocks, metric=metric,
        period_x=period_x, period_y=period_y, period=period,
        benchmark=benchmark, risk_free_rate=risk_free_rate,
        corr_threshold=corr_threshold,
    )

    # ── Step 1: Score all symbols ────────────────────────────────────────
    _log(f"Scoring {len(symbols)} symbols on {metric} ({period_x} vs {period_y})...")
    quadrant_df = plots.scatter_quadrants(
        quant, symbols,
        metric=metric,
        period_x=period_x,
        period_y=period_y,
        benchmark=benchmark,
        risk_free_rate=risk_free_rate,
    )

    # ── Step 2: Filter Q1 ───────────────────────────────────────────────
    q1_df      = quadrant_df[quadrant_df["quadrant"] == 1].copy()
    q1_symbols = q1_df["symbol"].tolist()
    _log(f"Q1 filter: {len(q1_symbols)}/{len(symbols)} symbols pass "
         f"(beat {benchmark} on both periods)")

    if len(q1_symbols) < 2:
        raise ValueError(
            f"Only {len(q1_symbols)} symbol(s) in Q1 — need at least 2 to build a portfolio. "
            f"Try a wider universe, lower threshold, or different periods."
        )

    # ── Step 3: Correlation deduplication ───────────────────────────────
    _log(f"Deduplicating by correlation (threshold={corr_threshold:.0%})...")
    scores     = q1_df.set_index("symbol")["score"]
    dedup      = plots.correlation_dedup(
        quant, q1_symbols,
        period=period,
        threshold=corr_threshold,
        method="score",
        scores=scores,
    )
    dedup_symbols = dedup["kept"]
    _log(f"After dedup: {len(dedup_symbols)} symbols kept, "
         f"{len(dedup['dropped'])} dropped "
         + (f"({', '.join(dedup['dropped'])})" if dedup["dropped"] else ""))

    # ── Step 4: Cap to top-N by score ───────────────────────────────────
    # re-rank dedup survivors by their original score
    dedup_scores = (quadrant_df
                    .set_index("symbol")["score"]
                    .reindex(dedup_symbols)
                    .sort_values(ascending=False))
    final_symbols = dedup_scores.head(max_stocks).index.tolist()
    _log(f"Top-{max_stocks} cap: final portfolio = {', '.join(final_symbols)}")

    if len(final_symbols) < 2:
        raise ValueError("Fewer than 2 stocks after filtering — cannot optimise.")

    # ── Step 5: Run all optimisers ───────────────────────────────────────
    _log("Running optimisers...")
    optimisers = {}
    kwargs = dict(period=period, risk_free_rate=risk_free_rate,
                  allow_short=allow_short, weight_bounds=weight_bounds)

    for name, fn in [
        ("equal_weight",  lambda: portfolio.equal_weight(quant, final_symbols,
                                                          period=period,
                                                          risk_free_rate=risk_free_rate)),
        ("min_variance",  lambda: portfolio.min_variance(quant, final_symbols, **kwargs)),
        ("max_sharpe",    lambda: portfolio.max_sharpe(quant, final_symbols, **kwargs)),
        ("risk_parity",   lambda: portfolio.risk_parity(quant, final_symbols,
                                                         period=period,
                                                         risk_free_rate=risk_free_rate)),
    ]:
        try:
            optimisers[name] = fn()
            _log(f"  {name:<16} Sharpe={optimisers[name].sharpe_ratio:.3f}  "
                 f"Vol={optimisers[name].volatility:.2%}  "
                 f"Ret={optimisers[name].expected_return:.2%}")
        except Exception as e:
            _log(f"  {name:<16} FAILED: {e}")

    # ── Step 6: Backtest each optimiser ──────────────────────────────────
    _log("Backtesting...")
    backtests = {}
    for name, opt in optimisers.items():
        try:
            weights_dict = dict(zip(opt.symbols, opt.weights))
            strat = backtest.target_weights(weights_dict)
            bt = backtest.run(
                quant, final_symbols,
                strategy=strat,
                period=period,
                rebalance=backtest_rebalance,
                transaction_cost_bps=transaction_cost_bps,
                risk_free_rate=risk_free_rate,
                benchmark=benchmark,
                strategy_name=name,
            )
            backtests[name] = bt
            _log(f"  {name:<16} CAGR={bt.metrics.get('cagr',0):+.1%}  "
                 f"Sharpe={bt.metrics.get('sharpe_ratio',0):.2f}  "
                 f"MaxDD={bt.metrics.get('max_drawdown',0):.1%}")
        except Exception as e:
            _log(f"  {name:<16} backtest FAILED: {e}")

    # ── Step 7: Efficient frontier ───────────────────────────────────────
    _log("Computing efficient frontier...")
    try:
        frontier = portfolio.efficient_frontier(
            quant, final_symbols,
            period=period,
            risk_free_rate=risk_free_rate,
            allow_short=allow_short,
            weight_bounds=weight_bounds,
        )
    except Exception as e:
        _log(f"Frontier FAILED: {e}")
        frontier = None

    _log("Done.")

    return PortfolioBuilderResult(
        universe=symbols,
        q1_symbols=q1_symbols,
        dedup_symbols=dedup_symbols,
        final_symbols=final_symbols,
        quadrant_df=quadrant_df,
        dedup_result=dedup,
        optimisers=optimisers,
        backtests=backtests,
        frontier=frontier,
        params=params,
    )
