"""
dashboard/callbacks.py — all Dash callback functions.

Tabs covered
------------
  Overview    : cumulative returns, drawdown, metrics bar + KPI cards
  Risk        : rolling vol/Sharpe, correlation heatmap, return distribution
  Seasonality : bar, heatmap, comparison, box plots
  Portfolio   : efficient frontier, Kelly, backtest
  Factors     : factor exposure, comparison, rolling betas
  Options     : chain, PCR, max pain, OI profile, GEX, vol surface, unusual
  Positions   : positions book P&L curves + legs table (loaded from JSON)
  Bitcoin     : power law chart, residuals oscillator, forecast
  Reports     : generate HTML report via auto_report(), list + preview saved files

Sidebar visibility: show_tab_controls() hides/shows per-tab control panels.
"""
from __future__ import annotations

import glob
import json
import os
import traceback as _tb
from datetime import datetime

from dash import Input, Output, State, html, dcc
import dash_bootstrap_components as dbc
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
    import re
    return [s.strip().upper() for s in re.split(r"[,\s\n]+", raw or "") if s.strip()]


def _empty_fig(msg: str = "Run analysis to see results") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=13, color="#666666"),
    )
    fig.update_layout(
        paper_bgcolor="#1A1A1A",
        plot_bgcolor="#1A1A1A",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _fmt(v, pct=False, dollar=False, decimals=2):
    if v is None:
        return "—"
    try:
        if pct:
            return f"{v:.{decimals}%}"
        if dollar:
            return f"${v:,.{decimals}f}"
        return f"{v:.{decimals}f}"
    except Exception:
        return str(v)


def _err(e: Exception) -> go.Figure:
    return _empty_fig(f"⚠ {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Register all callbacks
# ---------------------------------------------------------------------------

def register(
    app,
    client: StockClient,
    quant: QuantAnalytics,
    reports_dir: str = "reports",
    positions_dir: str = "positions",
) -> None:
    """Register every callback against the Dash app instance."""

    # ══════════════════════════════════════════════════════════════════════
    # Sidebar: show/hide tab-specific control panels
    # ══════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("ctrl-seasonality", "style"),
        Output("ctrl-portfolio",   "style"),
        Output("ctrl-factors",     "style"),
        Output("ctrl-options",     "style"),
        Output("ctrl-positions",   "style"),
        Output("ctrl-bitcoin",     "style"),
        Output("ctrl-reports",     "style"),
        Input("main-tabs", "active_tab"),
    )
    def show_tab_controls(active_tab):
        show = {"display": "block"}
        hide = {"display": "none"}
        tab_map = {
            "ctrl-seasonality": "tab-seasonality",
            "ctrl-portfolio":   "tab-portfolio",
            "ctrl-factors":     "tab-factors",
            "ctrl-options":     "tab-options",
            "ctrl-positions":   "tab-positions",
            "ctrl-bitcoin":     "tab-bitcoin",
            "ctrl-reports":     "tab-reports",
        }
        return tuple(
            show if active_tab == tab else hide
            for tab in tab_map.values()
        )

    # ══════════════════════════════════════════════════════════════════════
    # OVERVIEW TAB
    # ══════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("graph-cumret",     "figure"),
        Output("graph-drawdown",   "figure"),
        Output("graph-metrics-bar","figure"),
        Output("metrics-row",      "children"),
        Output("status-msg",       "children"),
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
            e = _empty_fig("Enter symbols and click Run")
            return e, e, e, [], "No symbols entered"

        try:
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
                return dbc.Col(metric_card(label, value, color), md=2)

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
            return fig_cr, fig_dd, fig_bar, cards, f"✓ {len(symbols)} symbol(s) loaded"

        except Exception as e:
            print(_tb.format_exc())
            empty = _err(e)
            return empty, empty, empty, [], f"Error: {e}"

    # ══════════════════════════════════════════════════════════════════════
    # RISK TAB
    # ══════════════════════════════════════════════════════════════════════

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
            e = _err(ex)
            return e, e, e, e

    # ══════════════════════════════════════════════════════════════════════
    # SEASONALITY TAB
    # ══════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("graph-seasonality-bar",     "figure"),
        Output("graph-seasonality-heatmap", "figure"),
        Output("graph-seasonality-compare", "figure"),
        Output("graph-seasonality-box",     "figure"),
        Input("btn-run",          "n_clicks"),
        State("input-symbols",    "value"),
        State("input-period",     "value"),
        State("dd-season-symbol", "value"),
        State("dd-season-gran",   "value"),
        State("dd-season-lt",     "value"),
        State("dd-season-st",     "value"),
        prevent_initial_call=True,
    )
    def update_seasonality(n_clicks, symbols_raw, period,
                           season_sym, gran, lt, st):
        symbols = _parse_symbols(symbols_raw)
        sym     = season_sym or (symbols[0] if symbols else "SPY")
        season_period = lt or period or "10y"
        try:
            return (
                plots.seasonality(quant, sym, period=season_period,
                                  granularity=gran or "monthly"),
                plots.seasonality_heatmap(quant, sym, period=season_period),
                plots.seasonality_comparison_clean(quant, sym,
                                                   long_term=season_period,
                                                   short_term=st or "5y"),
                plots.seasonality_box(quant, sym, period=season_period,
                                      granularity=gran or "monthly"),
            )
        except Exception as ex:
            e = _err(ex)
            return e, e, e, e

    @app.callback(
        Output("dd-season-symbol", "options"),
        Output("dd-season-symbol", "value"),
        Input("btn-run", "n_clicks"),
        State("input-symbols", "value"),
        prevent_initial_call=True,
    )
    def update_season_symbols(n_clicks, symbols_raw):
        symbols = _parse_symbols(symbols_raw or "AAPL")
        opts = [{"label": s, "value": s} for s in symbols]
        return opts, (symbols[0] if symbols else None)

    # ══════════════════════════════════════════════════════════════════════
    # PORTFOLIO TAB
    # ══════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("graph-frontier", "figure"),
        Output("graph-kelly",    "figure"),
        Output("graph-backtest", "figure"),
        Input("btn-run",         "n_clicks"),
        State("input-symbols",   "value"),
        State("input-benchmark", "value"),
        State("input-period",    "value"),
        State("input-rfr",       "value"),
        State("dd-bt-strategy",  "value"),
        prevent_initial_call=True,
    )
    def update_portfolio(n_clicks, symbols_raw, benchmark, period, rfr, strategy):
        symbols = _parse_symbols(symbols_raw)
        if not symbols:
            e = _empty_fig()
            return e, e, e

        try:
            frontier = portfolio.efficient_frontier(quant, symbols,
                                                    period=period, risk_free_rate=rfr)
            fig_fr = plots.efficient_frontier(frontier)
        except Exception as ex:
            fig_fr = _err(ex)

        try:
            fig_kelly = plots.kelly(quant, symbols, period=period, risk_free_rate=rfr)
        except Exception as ex:
            fig_kelly = _err(ex)

        try:
            strategy_map = {
                "buy_hold":  backtest.buy_and_hold(),
                "ma_20_50":  backtest.ma_crossover(20, 50),
                "ma_50_200": backtest.ma_crossover(50, 200),
                "momentum":  backtest.momentum(lookback=63),
                "mean_rev":  backtest.mean_reversion(lookback=20),
            }
            strat = strategy_map.get(strategy or "buy_hold", backtest.buy_and_hold())
            bt_result = backtest.run(quant, symbols, strategy=strat,
                                     period=period, benchmark=benchmark,
                                     risk_free_rate=rfr)
            fig_bt = plots.backtest(bt_result)
        except Exception as ex:
            fig_bt = _err(ex)

        return fig_fr, fig_kelly, fig_bt

    # ══════════════════════════════════════════════════════════════════════
    # FACTORS TAB
    # ══════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("graph-factor-exposure", "figure"),
        Output("graph-factor-compare",  "figure"),
        Output("graph-factor-rolling",  "figure"),
        Input("btn-run",         "n_clicks"),
        State("input-symbols",   "value"),
        State("input-period",    "value"),
        State("dd-factor-model", "value"),
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
                results.append(factors.run(quant, sym, model=model, period=period))
            except Exception:
                pass

        if not results:
            e = _empty_fig("Factor data unavailable")
            return e, e, e

        try:
            fig_exp = plots.factor_exposure(results[0])
        except Exception as ex:
            fig_exp = _err(ex)

        try:
            fig_cmp = (plots.factor_comparison(results) if len(results) > 1
                       else _empty_fig("Need 2+ symbols for comparison"))
        except Exception as ex:
            fig_cmp = _err(ex)

        try:
            fig_roll = plots.rolling_factor_betas(
                quant, symbols[0], model=model, period=period
            )
        except Exception as ex:
            fig_roll = _err(ex)

        return fig_exp, fig_cmp, fig_roll

    # ══════════════════════════════════════════════════════════════════════
    # OPTIONS TAB
    # ══════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("dd-opt-expiry", "options"),
        Output("dd-opt-expiry", "value"),
        Input("btn-opt-load", "n_clicks"),
        State("input-opt-symbol", "value"),
        prevent_initial_call=True,
    )
    def load_option_expiries(n_clicks, symbol):
        """Populate expiry dropdown when user clicks Load Expiries."""
        if not symbol:
            return [], None
        try:
            from yfinance_api3.classes.options import OptionsAnalyzer
            opt = OptionsAnalyzer(client, symbol.strip().upper())
            expiries = opt.expiries()
            opts = [{"label": e, "value": e} for e in expiries]
            nearest = opt.nearest_expiry(0)
            return opts, nearest
        except Exception as ex:
            return [], None

    @app.callback(
        Output("graph-opt-chain",   "figure"),
        Output("graph-opt-pcr",     "figure"),
        Output("graph-opt-maxpain", "figure"),
        Output("graph-opt-oi",      "figure"),
        Output("graph-opt-gex",     "figure"),
        Output("graph-opt-surface", "figure"),
        Output("graph-opt-unusual", "figure"),
        Input("dd-opt-expiry", "value"),
        State("input-opt-symbol", "value"),
        prevent_initial_call=True,
    )
    def update_options(expiry, symbol):
        if not symbol or not expiry:
            e = _empty_fig("Select ticker and load expiries")
            return e, e, e, e, e, e, e

        try:
            from yfinance_api3.classes.options import OptionsAnalyzer
            opt = OptionsAnalyzer(client, symbol.strip().upper())
        except Exception as ex:
            e = _err(ex)
            return e, e, e, e, e, e, e

        def _safe(fn, *args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as ex:
                return _err(ex)

        fig_chain   = _safe(plots.options_chain,   opt, expiry=expiry)
        fig_pcr     = _safe(plots.options_put_call, opt)
        fig_maxpain = _safe(plots.options_max_pain, opt, expiry=expiry)
        fig_oi      = _safe(plots.options_oi_profile, opt, expiry=expiry)
        fig_gex     = _safe(plots.options_gex,     opt)
        fig_surface = _safe(plots.options_surface, opt)
        fig_unusual = _safe(plots.options_unusual, opt)

        return fig_chain, fig_pcr, fig_maxpain, fig_oi, fig_gex, fig_surface, fig_unusual

    # ══════════════════════════════════════════════════════════════════════
    # POSITIONS TAB
    # ══════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("dd-pos-book", "options"),
        Output("dd-pos-book", "value"),
        Input("main-tabs", "active_tab"),
    )
    def populate_positions_dropdown(active_tab):
        """Scan positions/ folder and populate dropdown when tab becomes active."""
        if active_tab != "tab-positions":
            return [], None
        try:
            pattern = os.path.join(positions_dir, "*.json")
            files = sorted(glob.glob(pattern))
            opts = []
            for f in files:
                name = os.path.splitext(os.path.basename(f))[0]
                opts.append({"label": name, "value": f})
            val = opts[0]["value"] if opts else None
            return opts, val
        except Exception:
            return [], None

    @app.callback(
        Output("graph-pos-book",  "figure"),
        Output("graph-pos-legs",  "figure"),
        Output("pos-mtm-status",  "children"),
        Input("btn-pos-mtm", "n_clicks"),
        State("dd-pos-book",  "value"),
        State("sl-pos-days",  "value"),
        State("sl-pos-dstd",  "value"),
        prevent_initial_call=True,
    )
    def update_positions(n_clicks, book_path, days_ahead, d_std):
        if not book_path or not os.path.exists(book_path):
            e = _empty_fig("Select a position file and click Mark to Market")
            return e, e, "No position file selected"

        try:
            from yfinance_api3.classes.positions_book import PositionsBook
            from yfinance_api3.classes.options import OptionsAnalyzer

            # Load book from JSON
            book = PositionsBook.from_json(book_path)
            symbol = book.symbol

            # Mark to market
            opt_inst = OptionsAnalyzer(client, symbol)
            book.mark_to_market(client, opt=opt_inst, quant=quant)

            fig_book = plots.positions_book(book,
                                            days_ahead=int(days_ahead or 0),
                                            d_std=float(d_std or 3.0))
            fig_legs = plots.positions_legs(book,
                                            days_ahead=int(days_ahead or 0))

            ts = datetime.now().strftime("%H:%M:%S")
            status = f"✓ Marked to market at {ts}"
            return fig_book, fig_legs, status

        except Exception as ex:
            print(_tb.format_exc())
            e = _err(ex)
            return e, e, f"⚠ Error: {ex}"

    # ══════════════════════════════════════════════════════════════════════
    # BITCOIN TAB
    # ══════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("graph-btc-chart",     "figure"),
        Output("graph-btc-residuals", "figure"),
        Output("graph-btc-forecast",  "figure"),
        Output("btc-metrics-row",     "children"),
        Input("btn-run", "n_clicks"),
        State("sl-btc-low",   "value"),
        State("sl-btc-high",  "value"),
        State("sl-btc-years", "value"),
        prevent_initial_call=True,
    )
    def update_bitcoin(n_clicks, low, high, years):
        try:
            from yfinance_api3.modules.powerlaw import PowerLaw
            from yfinance_api3.dashboard.components import metric_card, COLORS

            pl = PowerLaw(
                corridor_low=int(low or 10),
                corridor_high=int(high or 90),
            )
            pl.fit()

            pos = pl.current_position()

            fig_chart    = plots.powerlaw_chart(pl)
            fig_residuals= plots.powerlaw_residuals(pl)
            fig_forecast = plots.powerlaw_forecast(pl, years=int(years or 4))

            # KPI cards
            phase_color = {
                "oversold":    COLORS["green"],
                "floor zone":  COLORS["blue"],
                "fair value":  COLORS["text"],
                "ceiling zone":COLORS["orange"],
                "overbought":  COLORS["red"],
            }.get(pos["phase"], COLORS["text"])

            cards = [
                dbc.Col(metric_card("Price",
                    f"${pos['price']:,.0f}", COLORS["text"]), md=2),
                dbc.Col(metric_card("Model Price",
                    f"${pos['model_price']:,.0f}", COLORS["muted"]), md=2),
                dbc.Col(metric_card("Floor",
                    f"${pos['floor_price']:,.0f}", COLORS["green"]), md=2),
                dbc.Col(metric_card("Ceiling",
                    f"${pos['ceiling_price']:,.0f}", COLORS["red"]), md=2),
                dbc.Col(metric_card("Phase",
                    pos["phase"].title(), phase_color), md=2),
                dbc.Col(metric_card("Next Halving",
                    f"{pos['days_to_halving']}d", COLORS["yellow"]), md=2),
            ]

            return fig_chart, fig_residuals, fig_forecast, cards

        except Exception as ex:
            print(_tb.format_exc())
            e = _err(ex)
            return e, e, e, []

    # ══════════════════════════════════════════════════════════════════════
    # REPORTS TAB
    # ══════════════════════════════════════════════════════════════════════

    @app.callback(
        Output("rpt-status",    "children"),
        Output("rpt-file-list", "children"),
        Input("btn-rpt-gen", "n_clicks"),
        State("input-symbols",    "value"),
        State("input-benchmark",  "value"),
        State("input-period",     "value"),
        State("input-rfr",        "value"),
        State("input-rpt-title",  "value"),
        State("chk-rpt-sections", "value"),
        State("dd-rpt-mcsims",    "value"),
        prevent_initial_call=True,
    )
    def generate_report(n_clicks, symbols_raw, benchmark, period, rfr,
                        title, sections, mc_sims):
        symbols = _parse_symbols(symbols_raw)
        if not symbols:
            return "⚠ No symbols entered", _list_reports(reports_dir)

        try:
            from yfinance_api3.modules.report import auto_report
            os.makedirs(reports_dir, exist_ok=True)

            sections = sections or []
            r = auto_report(
                quant, client, symbols,
                title=title or "Equity Research Report",
                period=period or "3y",
                benchmark=benchmark or "SPY",
                risk_free_rate=rfr or 0.05,
                include_fundamentals= "fundamentals" in sections,
                include_frontier=     "frontier"     in sections,
                include_backtest=     "backtest"      in sections,
                include_montecarlo=   "montecarlo"    in sections,
                include_factors=      "factors"       in sections,
                mc_sims=int(mc_sims or 500),
            )

            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            syms  = "_".join(symbols[:3])
            fname = f"{syms}_{ts}.html"
            fpath = os.path.join(reports_dir, fname)
            r.save(fpath)

            status = f"✓ Report saved → {fpath}"
        except Exception as ex:
            print(_tb.format_exc())
            status = f"⚠ Error generating report: {ex}"

        return status, _list_reports(reports_dir)

    @app.callback(
        Output("rpt-file-list", "children", allow_duplicate=True),
        Input("main-tabs", "active_tab"),
        prevent_initial_call=True,
    )
    def refresh_report_list(active_tab):
        if active_tab != "tab-reports":
            return []
        return _list_reports(reports_dir)

    @app.callback(
        Output("rpt-preview", "src"),
        Input("store-rpt-path", "data"),
    )
    def preview_report(path):
        """Serve the selected report HTML inside the iframe."""
        if not path:
            return ""
        # Dash serves static assets from the assets/ folder.
        # Reports are served via a custom route registered below — see note.
        return f"/reports/{os.path.basename(path)}"

    # Register a simple static route so the iframe can load report HTML
    # (Dash Flask server exposes the underlying Flask app via app.server)
    from flask import send_from_directory

    @app.server.route("/reports/<path:filename>")
    def serve_report(filename):
        abs_dir = os.path.abspath(reports_dir)
        return send_from_directory(abs_dir, filename)

    # ── Store selected report path when user clicks a file link ──────────
    @app.callback(
        Output("store-rpt-path", "data"),
        Input({"type": "rpt-file-btn", "index": dash.ALL}, "n_clicks"),
        State({"type": "rpt-file-btn", "index": dash.ALL}, "id"),
        prevent_initial_call=True,
    )
    def select_report(n_clicks_list, ids):
        import dash
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        import json as _json
        id_dict = _json.loads(triggered_id)
        return id_dict["index"]


# ---------------------------------------------------------------------------
# Helper: build the saved-reports list UI
# ---------------------------------------------------------------------------

def _list_reports(reports_dir: str) -> list:
    """Scan reports/ folder and return a list of clickable file rows."""
    import dash
    pattern = os.path.join(reports_dir, "*.html")
    files   = sorted(glob.glob(pattern), reverse=True)   # newest first

    if not files:
        return [html.Div("No reports yet — generate one using the sidebar.",
                         style={"color": "#555555", "fontSize": "12px"})]

    rows = []
    for f in files:
        name  = os.path.basename(f)
        mtime = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M")
        size  = f"{os.path.getsize(f) / 1024:.0f} KB"
        rows.append(
            dbc.Row([
                dbc.Col(
                    html.Button(
                        name,
                        id={"type": "rpt-file-btn", "index": f},
                        style={
                            "background":  "none",
                            "border":      "none",
                            "color":       "#378ADD",
                            "fontSize":    "12px",
                            "cursor":      "pointer",
                            "padding":     "0",
                            "textAlign":   "left",
                        },
                    ), md=7,
                ),
                dbc.Col(html.Span(mtime, style={"color":"#555555","fontSize":"11px"}), md=3),
                dbc.Col(html.Span(size,  style={"color":"#555555","fontSize":"11px"}), md=2),
            ], style={
                "padding":      "8px 4px",
                "borderBottom": "1px solid #222222",
                "alignItems":   "center",
            })
        )
    return rows


# Import needed for pattern-matching callbacks
import dash
