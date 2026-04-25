"""
dashboard/callbacks.py — all Dash callback functions.

Each callback is registered against the app instance passed in.
Separated from app.py to keep file sizes manageable.
"""
from __future__ import annotations

from dash import Input, Output, State
import plotly.graph_objects as go

from yfinance_api3.classes.stock_client import StockClient
from yfinance_api3.classes.quant_analytics import QuantAnalytics
import yfinance_api3.modules.plots as plots
import yfinance_api3.modules.portfolio as portfolio
import yfinance_api3.modules.backtest as backtest
import yfinance_api3.modules.factors as factors


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_symbols(raw: str) -> list[str]:
    """Parse comma/space/newline separated symbol string."""
    import re
    return [s.strip().upper() for s in re.split(r"[,\s\n]+", raw) if s.strip()]


def _empty_fig(msg: str = "Run analysis to see results") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="#888780"),
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FAFAF9",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _fmt(v, pct=False, dollar=False, decimals=2):
    if v is None:
        return "—"
    if pct:
        return f"{v:.{decimals}%}"
    if dollar:
        return f"${v:,.{decimals}f}"
    return f"{v:.{decimals}f}"


# ---------------------------------------------------------------------------
# Register callbacks
# ---------------------------------------------------------------------------

def register(app, client: StockClient, quant: QuantAnalytics) -> None:
    """Register all callbacks against the Dash app."""

    # ── Overview tab ──────────────────────────────────────────────────────

    @app.callback(
        Output("graph-cumret",   "figure"),
        Output("graph-drawdown", "figure"),
        Output("graph-metrics-bar", "figure"),
        Output("metrics-row",    "children"),
        Output("status-msg",     "children"),
        Input("btn-run", "n_clicks"),
        State("input-symbols",   "value"),
        State("input-benchmark", "value"),
        State("input-period",    "value"),
        State("input-rfr",       "value"),
        prevent_initial_call=True,
    )
    def update_overview(n_clicks, symbols_raw, benchmark, period, rfr):
        symbols = _parse_symbols(symbols_raw)
        if not symbols:
            empty = _empty_fig("Enter symbols and click Run")
            return empty, empty, empty, [], "No symbols entered"

        try:
            import dash_bootstrap_components as _dbc
            from yfinance_api3.dashboard.components import metric_card, COLORS

            fig_cr  = plots.cumulative_returns(quant, symbols, period=period)
            fig_dd  = plots.drawdown(quant, symbols, period=period)
            fig_bar = plots.metrics_bar(quant, symbols, metric="sharpe",
                                        period=period, benchmark=benchmark,
                                        risk_free_rate=rfr)

            sym   = symbols[0]
            stats = quant.stock_report(sym, benchmark=benchmark,
                                       period=period, risk_free_rate=rfr)

            def _col(label, value, color=COLORS["text"]):
                return _dbc.Col(metric_card(label, value, color), md=2)

            cards = [
                _col("Volatility",   _fmt(stats["annualised_volatility"], pct=True)),
                _col("Sharpe",       _fmt(stats["sharpe_ratio"])),
                _col("Sortino",      _fmt(stats["sortino_ratio"])),
                _col("Max Drawdown", _fmt(stats["max_drawdown"], pct=True),
                     color=COLORS["red"]),
                _col("Beta",         _fmt(stats["beta"])),
                _col("VaR 95% 1d",   _fmt(stats["var_95_1d"], pct=True),
                     color=COLORS["red"]),
            ]
            return fig_cr, fig_dd, fig_bar, cards, f"✓ {len(symbols)} symbols loaded"

        except Exception as e:
            import traceback as _tb
            print(_tb.format_exc())
            empty = _empty_fig(str(e))
            return empty, empty, empty, [], f"Error: {e}"

    # ── Risk tab ──────────────────────────────────────────────────────────

    @app.callback(
        Output("graph-rolling-vol",    "figure"),
        Output("graph-rolling-sharpe", "figure"),
        Output("graph-corr",           "figure"),
        Output("graph-ret-dist",       "figure"),
        Input("btn-run", "n_clicks"),
        State("input-symbols",   "value"),
        State("input-benchmark", "value"),
        State("input-period",    "value"),
        State("input-rfr",       "value"),
        prevent_initial_call=True,
    )
    def update_risk(n_clicks, symbols_raw, benchmark, period, rfr):
        symbols = _parse_symbols(symbols_raw)
        if not symbols:
            e = _empty_fig()
            return e, e, e, e

        try:
            return (
                plots.rolling_volatility(quant, symbols, period=period),
                plots.rolling_sharpe(quant, symbols, period=period,
                                     risk_free_rate=rfr),
                plots.correlation_heatmap(quant, symbols, period=period),
                plots.returns_distribution(quant, symbols, period=period),
            )
        except Exception as ex:
            e = _empty_fig(str(ex))
            return e, e, e, e

    # ── Seasonality tab ───────────────────────────────────────────────────

    @app.callback(
        Output("graph-seasonality-bar",     "figure"),
        Output("graph-seasonality-heatmap", "figure"),
        Output("graph-seasonality-compare", "figure"),
        Output("graph-seasonality-box",     "figure"),
        Input("btn-run",           "n_clicks"),
        State("input-symbols",     "value"),
        State("input-period",      "value"),
        State("dd-season-symbol",  "value"),
        State("dd-season-gran",    "value"),
        State("dd-season-lt",      "value"),
        State("dd-season-st",      "value"),
        prevent_initial_call=True,
    )
    def update_seasonality(n_clicks, symbols_raw, period,
                           season_sym, gran, lt, st):
        symbols = _parse_symbols(symbols_raw)
        sym     = season_sym or (symbols[0] if symbols else "SPY")

        try:
            return (
                plots.seasonality(quant, sym, period=period,
                                  granularity=gran or "monthly"),
                plots.seasonality_heatmap(quant, sym, period=period),
                plots.seasonality_comparison_clean(quant, sym,
                                                   long_term=lt or "10y",
                                                   short_term=st or "5y"),
                plots.seasonality_box(quant, sym, period=period,
                                      granularity=gran or "monthly"),
            )
        except Exception as ex:
            e = _empty_fig(str(ex))
            return e, e, e, e

    # ── Portfolio tab ─────────────────────────────────────────────────────

    @app.callback(
        Output("graph-frontier",  "figure"),
        Output("graph-kelly",     "figure"),
        Output("graph-backtest",  "figure"),
        Input("btn-run",          "n_clicks"),
        State("input-symbols",    "value"),
        State("input-benchmark",  "value"),
        State("input-period",     "value"),
        State("input-rfr",        "value"),
        State("dd-bt-strategy",   "value"),
        prevent_initial_call=True,
    )
    def update_portfolio(n_clicks, symbols_raw, benchmark, period, rfr, strategy):
        symbols = _parse_symbols(symbols_raw)
        if not symbols:
            e = _empty_fig()
            return e, e, e

        try:
            frontier = portfolio.efficient_frontier(quant, symbols,
                                                    period=period,
                                                    risk_free_rate=rfr)
            fig_fr = plots.efficient_frontier(frontier)
        except Exception as ex:
            fig_fr = _empty_fig(str(ex))

        try:
            fig_kelly = plots.kelly(quant, symbols, period=period,
                                    risk_free_rate=rfr)
        except Exception as ex:
            fig_kelly = _empty_fig(str(ex))

        try:
            strategy_map = {
                "buy_hold":  backtest.buy_and_hold(),
                "ma_20_50":  backtest.ma_crossover(20, 50),
                "ma_50_200": backtest.ma_crossover(50, 200),
                "momentum":  backtest.momentum(lookback=63),
                "mean_rev":  backtest.mean_reversion(lookback=20),
            }
            strat = strategy_map.get(strategy or "buy_hold",
                                     backtest.buy_and_hold())
            bt_result = backtest.run(quant, symbols, strategy=strat,
                                     period=period, benchmark=benchmark,
                                     risk_free_rate=rfr)
            fig_bt = plots.backtest(bt_result)
        except Exception as ex:
            fig_bt = _empty_fig(str(ex))

        return fig_fr, fig_kelly, fig_bt

    # ── Factors tab ───────────────────────────────────────────────────────

    @app.callback(
        Output("graph-factor-exposure",  "figure"),
        Output("graph-factor-compare",   "figure"),
        Output("graph-factor-rolling",   "figure"),
        Input("btn-run",          "n_clicks"),
        State("input-symbols",    "value"),
        State("input-period",     "value"),
        State("dd-factor-model",  "value"),
        prevent_initial_call=True,
    )
    def update_factors(n_clicks, symbols_raw, period, model):
        symbols = _parse_symbols(symbols_raw)
        if not symbols:
            e = _empty_fig()
            return e, e, e

        model = model or "ff5"
        results = []
        for sym in symbols:
            try:
                results.append(factors.run(quant, sym, model=model,
                                           period=period))
            except Exception:
                pass

        if not results:
            e = _empty_fig("Factor data unavailable")
            return e, e, e

        try:
            fig_exp = plots.factor_exposure(results[0])
        except Exception as ex:
            fig_exp = _empty_fig(str(ex))

        try:
            fig_cmp = plots.factor_comparison(results) if len(results) > 1 \
                      else _empty_fig("Need 2+ symbols for comparison")
        except Exception as ex:
            fig_cmp = _empty_fig(str(ex))

        try:
            fig_roll = plots.rolling_factor_betas(
                quant, symbols[0], model=model, period=period
            )
        except Exception as ex:
            fig_roll = _empty_fig(str(ex))

        return fig_exp, fig_cmp, fig_roll

    # ── Sidebar extra controls (tab-specific) ─────────────────────────────

    @app.callback(
        Output("ctrl-seasonality", "style"),
        Output("ctrl-portfolio",   "style"),
        Output("ctrl-factors",     "style"),
        Input("main-tabs", "active_tab"),
    )
    def show_tab_controls(active_tab):
        """Show the relevant sidebar control panel for the active tab."""
        show = {"display": "block"}
        hide = {"display": "none"}
        return (
            show if active_tab == "tab-seasonality" else hide,
            show if active_tab == "tab-portfolio"   else hide,
            show if active_tab == "tab-factors"     else hide,
        )

    @app.callback(
        Output("dd-season-symbol", "options"),
        Output("dd-season-symbol", "value"),
        Input("btn-run", "n_clicks"),
        State("input-symbols", "value"),
        prevent_initial_call=True,
    )
    def update_season_symbols(n_clicks, symbols_raw):
        """Populate the seasonality symbol dropdown after Run is clicked."""
        symbols = _parse_symbols(symbols_raw or "AAPL")
        opts = [{"label": s, "value": s} for s in symbols]
        return opts, (symbols[0] if symbols else None)
