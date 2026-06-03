"""
dashboard/app.py — Dash application factory.

Usage
-----
    from yfinance_api3.dashboard.app import create_app
    from yfinance_api3.classes.stock_client import StockClient
    from yfinance_api3.classes.quant_analytics import QuantAnalytics

    client = StockClient()
    quant  = QuantAnalytics(client)

    app = create_app(quant, client)
    app.run(debug=True, port=8050)

Or run directly:
    python -m yfinance_api3.dashboard.app

Dependencies
------------
    pip install dash dash-bootstrap-components
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from yfinance_api3.classes.stock_client import StockClient
from yfinance_api3.classes.quant_analytics import QuantAnalytics
from yfinance_api3.dashboard.components import (
    header, sidebar, plot_card, section_title, loading, COLORS
)
import logging

logging.basicConfig(level=logging.DEBUG)


# ---------------------------------------------------------------------------
# Tab content builders
# ---------------------------------------------------------------------------

def _tab_overview() -> html.Div:
    return html.Div([
        section_title("Performance Overview"),
        dbc.Row(id="metrics-row", style={"marginBottom": "16px"}),
        loading(plot_card("graph-cumret",      height=380)),
        loading(plot_card("graph-drawdown",    height=280)),
        loading(plot_card("graph-metrics-bar", height=380)),
    ], style={"padding": "24px"})


def _tab_risk() -> html.Div:
    return html.Div([
        section_title("Risk Analysis"),
        dbc.Row([
            dbc.Col(loading(plot_card("graph-rolling-vol",    height=350)), md=6),
            dbc.Col(loading(plot_card("graph-rolling-sharpe", height=350)), md=6),
        ]),
        dbc.Row([
            dbc.Col(loading(plot_card("graph-corr",     height=420)), md=5),
            dbc.Col(loading(plot_card("graph-ret-dist", height=420)), md=7),
        ]),
    ], style={"padding": "24px"})


def _tab_seasonality() -> html.Div:
    return html.Div([
        section_title("Seasonality Analysis"),
        dbc.Row([
            dbc.Col(loading(plot_card("graph-seasonality-bar", height=420)), md=6),
            dbc.Col(loading(plot_card("graph-seasonality-box", height=420)), md=6),
        ]),
        loading(plot_card("graph-seasonality-compare",  height=380)),
        loading(plot_card("graph-seasonality-heatmap",  height=400)),
    ], style={"padding": "24px"})


def _tab_portfolio() -> html.Div:
    return html.Div([
        section_title("Portfolio Optimisation & Backtesting"),
        loading(plot_card("graph-frontier", height=480)),
        loading(plot_card("graph-kelly",    height=500)),
        loading(plot_card("graph-backtest", height=550)),
    ], style={"padding": "24px"})


def _tab_factors() -> html.Div:
    return html.Div([
        section_title("Fama-French Factor Exposure"),
        loading(plot_card("graph-factor-exposure", height=420)),
        loading(plot_card("graph-factor-compare",  height=420)),
        loading(plot_card("graph-factor-rolling",  height=380)),
    ], style={"padding": "24px"})


def _tab_options() -> html.Div:
    return html.Div([
        section_title("Options Analysis"),
        dbc.Row([
            dbc.Col(loading(plot_card("graph-opt-chain",   height=480)), md=7),
            dbc.Col(loading(plot_card("graph-opt-pcr",     height=480)), md=5),
        ]),
        dbc.Row([
            dbc.Col(loading(plot_card("graph-opt-maxpain", height=420)), md=6),
            dbc.Col(loading(plot_card("graph-opt-oi",      height=420)), md=6),
        ]),
        loading(plot_card("graph-opt-gex",     height=450)),
        loading(plot_card("graph-opt-surface", height=500)),
        loading(plot_card("graph-opt-unusual", height=420)),
    ], style={"padding": "24px"})


def _tab_positions() -> html.Div:
    return html.Div([
        section_title("Positions Book"),
        # Mark-to-market trigger
        dbc.Row([
            dbc.Col(
                dbc.Button("Mark to Market", id="btn-pos-mtm",
                           color="warning", outline=True,
                           style={"fontSize":"12px","fontWeight":"600",
                                  "marginBottom":"16px"}),
                width="auto",
            ),
            dbc.Col(
                html.Div(id="pos-mtm-status", style={
                    "color": COLORS["muted"], "fontSize": "11px",
                    "paddingTop": "8px",
                }),
            ),
        ]),
        loading(plot_card("graph-pos-book", height=700)),
        loading(plot_card("graph-pos-legs", height=400)),
    ], style={"padding": "24px"})


def _tab_bitcoin() -> html.Div:
    return html.Div([
        section_title("Bitcoin Power Law"),
        # Current position summary cards
        dbc.Row(id="btc-metrics-row", style={"marginBottom": "16px"}),
        loading(plot_card("graph-btc-chart",    height=560)),
        loading(plot_card("graph-btc-residuals",height=400)),
        loading(plot_card("graph-btc-forecast", height=450)),
    ], style={"padding": "24px"})


def _tab_reports() -> html.Div:
    return html.Div([
        section_title("HTML Report Builder"),

        # Report generation status / progress
        html.Div(id="rpt-status", style={
            "color":      COLORS["muted"],
            "fontSize":   "12px",
            "marginBottom": "12px",
            "minHeight":  "20px",
        }),

        # Saved reports list
        html.Div([
            html.Div("Saved Reports", style={
                "color":         COLORS["muted"],
                "fontSize":      "11px",
                "fontWeight":    "600",
                "letterSpacing": "0.8px",
                "textTransform": "uppercase",
                "marginBottom":  "10px",
            }),
            html.Div(id="rpt-file-list"),
        ], style={
            "background":    COLORS["surface"],
            "border":        f"1px solid {COLORS['border']}",
            "borderRadius":  "8px",
            "padding":       "16px",
            "marginBottom":  "16px",
        }),

        # Report preview
        html.Div([
            html.Div("Preview", style={
                "color":         COLORS["muted"],
                "fontSize":      "11px",
                "fontWeight":    "600",
                "letterSpacing": "0.8px",
                "textTransform": "uppercase",
                "marginBottom":  "10px",
            }),
            html.Iframe(
                id="rpt-preview",
                style={
                    "width":       "100%",
                    "height":      "780px",
                    "border":      f"1px solid {COLORS['border']}",
                    "borderRadius":"6px",
                    "background":  "white",
                },
                src="",
            ),
        ], style={
            "background":   COLORS["surface"],
            "border":       f"1px solid {COLORS['border']}",
            "borderRadius": "8px",
            "padding":      "16px",
        }),

        # Hidden store for selected report path
        dcc.Store(id="store-rpt-path"),

    ], style={"padding": "24px"})


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    quant: QuantAnalytics,
    client: StockClient,
    title: str = "QuantDashboard",
    reports_dir: str = "reports",
    positions_dir: str = "positions",
) -> dash.Dash:
    """
    Create and configure the Dash application.

    Parameters
    ----------
    quant         : QuantAnalytics instance
    client        : StockClient instance
    title         : browser tab title
    reports_dir   : folder where HTML reports are saved
    positions_dir : folder where position JSON files live
    """
    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap",
        ],
        title=title,
        suppress_callback_exceptions=True,
    )

    app.layout = html.Div([

        # ── Top navbar ──────────────────────────────────────────────────
        header(),

        # ── Body: sidebar + main ─────────────────────────────────────────
        dbc.Row([

            # Sidebar
            dbc.Col(sidebar(), width=2, style={"padding": "0"}),

            # Main content
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(_tab_overview(),    label="Overview",    tab_id="tab-overview"),
                    dbc.Tab(_tab_risk(),        label="Risk",        tab_id="tab-risk"),
                    dbc.Tab(_tab_seasonality(), label="Seasonality", tab_id="tab-seasonality"),
                    dbc.Tab(_tab_portfolio(),   label="Portfolio",   tab_id="tab-portfolio"),
                    dbc.Tab(_tab_factors(),     label="Factors",     tab_id="tab-factors"),
                    dbc.Tab(_tab_options(),     label="Options",     tab_id="tab-options"),
                    dbc.Tab(_tab_positions(),   label="Positions",   tab_id="tab-positions"),
                    dbc.Tab(_tab_bitcoin(),     label="Bitcoin",     tab_id="tab-bitcoin"),
                    dbc.Tab(_tab_reports(),     label="Reports",     tab_id="tab-reports"),
                ],
                id="main-tabs",
                active_tab="tab-overview",
                style={"marginTop": "8px"},
                ),
            ], width=10, style={"padding": "0", "background": COLORS["bg"]}),

        ], style={"margin": "0"}),

    ], style={
        "fontFamily": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
        "background": COLORS["bg"],
        "minHeight":  "100vh",
        "color":      COLORS["text"],
    })

    # Register all callbacks
    from yfinance_api3.dashboard.callbacks import register
    register(app, client, quant,
             reports_dir=reports_dir,
             positions_dir=positions_dir)

    return app


# ---------------------------------------------------------------------------
# Direct run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client = StockClient()
    quant  = QuantAnalytics(client)
    app    = create_app(quant, client)
    app.run(debug=True, port=8050, host="0.0.0.0")
