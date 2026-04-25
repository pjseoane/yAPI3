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
from dash import html

from yfinance_api3.classes.stock_client import StockClient
from yfinance_api3.classes.quant_analytics import QuantAnalytics
from yfinance_api3.dashboard.components import (
    header, sidebar, plot_card, section_title, loading, COLORS
)
import logging

logging.basicConfig(level=logging.DEBUG)

# ---------------------------------------------------------------------------
# Tab layouts
# ---------------------------------------------------------------------------

def _tab_overview() -> html.Div:
    return html.Div([
        section_title("Performance Overview"),
        dbc.Row(id="metrics-row", style={"marginBottom": "16px"}),
        loading(plot_card("graph-cumret",    height=380)),
        loading(plot_card("graph-drawdown",  height=280)),
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
            dbc.Col(loading(plot_card("graph-corr",      height=420)), md=5),
            dbc.Col(loading(plot_card("graph-ret-dist",  height=420)), md=7),
        ]),
    ], style={"padding": "24px"})


def _tab_seasonality() -> html.Div:
    return html.Div([
        section_title("Seasonality Analysis"),
        dbc.Row([
            dbc.Col(loading(plot_card("graph-seasonality-bar",     height=420)), md=6),
            dbc.Col(loading(plot_card("graph-seasonality-box",     height=420)), md=6),
        ]),
        loading(plot_card("graph-seasonality-compare", height=380)),
        loading(plot_card("graph-seasonality-heatmap", height=400)),
    ], style={"padding": "24px"})


def _tab_portfolio() -> html.Div:
    return html.Div([
        section_title("Portfolio Optimisation & Backtesting"),
        loading(plot_card("graph-frontier", height=480)),
        dbc.Row([
            dbc.Col(loading(plot_card("graph-kelly",   height=500)), md=12),
        ]),
        loading(plot_card("graph-backtest", height=550)),
    ], style={"padding": "24px"})


def _tab_factors() -> html.Div:
    return html.Div([
        section_title("Fama-French Factor Exposure"),
        loading(plot_card("graph-factor-exposure", height=420)),
        loading(plot_card("graph-factor-compare",  height=420)),
        loading(plot_card("graph-factor-rolling",  height=380)),
    ], style={"padding": "24px"})


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    quant: QuantAnalytics,
    client: StockClient,
    title: str = "QuantDashboard",
) -> dash.Dash:
    """
    Create and configure the Dash application.

    Parameters
    ----------
    quant  : QuantAnalytics instance
    client : StockClient instance
    title  : browser tab title

    Returns
    -------
    dash.Dash app (call .run() to start)
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

        # ── Top navbar ───────────────────────────────────────────────────
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
        "minHeight": "100vh",
    })

    # Register all callbacks
    from yfinance_api3.dashboard.callbacks import register
    register(app, client, quant)

    return app


# ---------------------------------------------------------------------------
# Direct run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client = StockClient()
    quant  = QuantAnalytics(client)
    app    = create_app(quant, client)
    app.run(debug=True, port=8050, host="0.0.0.0")
