"""
dashboard/components.py — reusable UI components.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


# ---------------------------------------------------------------------------
# Colour tokens (match plots.py palette)
# ---------------------------------------------------------------------------
COLORS = {
    "bg":       "#F4F3EF",
    "sidebar":  "#1E1E1C",
    "card":     "#FFFFFF",
    "border":   "#D3D1C7",
    "text":     "#2C2C2A",
    "muted":    "#888780",
    "green":    "#1D9E75",
    "red":      "#E24B4A",
    "blue":     "#378ADD",
}

CARD_STYLE = {
    "background": COLORS["card"],
    "border":     f"0.5px solid {COLORS['border']}",
    "borderRadius": "10px",
    "padding":    "16px",
    "marginBottom": "16px",
}


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def header() -> dbc.Navbar:
    return dbc.Navbar(
        dbc.Container([
            html.Span("📈 QuantDashboard", style={
                "color": "white", "fontWeight": "600",
                "fontSize": "18px", "letterSpacing": "-0.3px",
            }),
            html.Span("powered by yfinance_api3", style={
                "color": COLORS["muted"], "fontSize": "11px",
                "marginLeft": "12px",
            }),
        ], fluid=True),
        color=COLORS["sidebar"],
        dark=True,
        style={"padding": "10px 24px"},
    )


# ---------------------------------------------------------------------------
# Sidebar — symbol input + controls
# ---------------------------------------------------------------------------

_lbl = {"color": "#D3D1C7", "fontSize": "11px", "marginTop": "10px", "display": "block"}
_dd  = {"fontSize": "12px"}


def sidebar() -> html.Div:
    return html.Div([
        html.Div("Configuration", style={
            "color": COLORS["muted"], "fontSize": "10px",
            "fontWeight": "600", "letterSpacing": "1.5px",
            "textTransform": "uppercase", "marginBottom": "16px",
        }),

        # Symbol input
        html.Label("Symbols", style={"color": "#D3D1C7", "fontSize": "12px"}),
        dcc.Textarea(
            id="input-symbols",
            value="AAPL, MSFT, NVDA, GOOGL, JPM",
            style={
                "width": "100%", "height": "80px",
                "background": "#2C2C2A", "color": "white",
                "border": f"1px solid #444441", "borderRadius": "6px",
                "padding": "8px", "fontSize": "12px",
                "resize": "none",
            },
        ),
        html.Div(style={"height": "12px"}),

        # Benchmark
        html.Label("Benchmark", style={"color": "#D3D1C7", "fontSize": "12px"}),
        dcc.Input(
            id="input-benchmark",
            value="SPY",
            style={
                "width": "100%", "background": "#2C2C2A",
                "color": "white", "border": "1px solid #444441",
                "borderRadius": "6px", "padding": "8px",
                "fontSize": "12px",
            },
        ),
        html.Div(style={"height": "12px"}),

        # Period
        html.Label("Period", style={"color": "#D3D1C7", "fontSize": "12px"}),
        dcc.Dropdown(
            id="input-period",
            options=[
                {"label": "1 Year",   "value": "1y"},
                {"label": "2 Years",  "value": "2y"},
                {"label": "3 Years",  "value": "3y"},
                {"label": "5 Years",  "value": "5y"},
                {"label": "10 Years", "value": "10y"},
            ],
            value="3y",
            clearable=False,
            style={"fontSize": "12px"},
        ),
        html.Div(style={"height": "12px"}),

        # Risk-free rate
        html.Label("Risk-free rate", style={"color": "#D3D1C7", "fontSize": "12px"}),
        dcc.Slider(
            id="input-rfr",
            min=0.0, max=0.10, step=0.005, value=0.05,
            marks={0: "0%", 0.05: "5%", 0.10: "10%"},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div(style={"height": "20px"}),

        # Run button
        dbc.Button(
            "Run Analysis",
            id="btn-run",
            color="success",
            style={"width": "100%", "fontWeight": "500"},
        ),

        html.Div(style={"height": "24px"}),
        html.Hr(style={"borderColor": "#444441", "marginTop": "20px"}),

        # ── Seasonality controls (always rendered, hidden when not on tab) ──
        html.Div("Tab options", style={
            "color": COLORS["muted"], "fontSize": "10px",
            "fontWeight": "600", "letterSpacing": "1.5px",
            "textTransform": "uppercase", "marginBottom": "12px",
        }),

        html.Div([
            html.Label("Season — Symbol", style=_lbl),
            dcc.Dropdown(id="dd-season-symbol",
                options=[], value=None, clearable=False, style=_dd),
            html.Label("Granularity", style=_lbl),
            dcc.Dropdown(id="dd-season-gran",
                options=[{"label":"Monthly","value":"monthly"},
                         {"label":"Weekly","value":"weekly"}],
                value="monthly", clearable=False, style=_dd),
            html.Label("Long-term", style=_lbl),
            dcc.Dropdown(id="dd-season-lt",
                options=[{"label":f"{n}y","value":f"{n}y"} for n in [5,10,15,20]],
                value="10y", clearable=False, style=_dd),
            html.Label("Short-term", style=_lbl),
            dcc.Dropdown(id="dd-season-st",
                options=[{"label":f"{n}y","value":f"{n}y"} for n in [1,2,3,5]],
                value="5y", clearable=False, style=_dd),
        ], id="ctrl-seasonality", style={"display":"none"}),

        html.Div([
            html.Label("Backtest strategy", style=_lbl),
            dcc.Dropdown(id="dd-bt-strategy",
                options=[
                    {"label":"Buy & Hold",    "value":"buy_hold"},
                    {"label":"MA 20/50",       "value":"ma_20_50"},
                    {"label":"MA 50/200",      "value":"ma_50_200"},
                    {"label":"Momentum (63d)", "value":"momentum"},
                    {"label":"Mean Reversion", "value":"mean_rev"},
                ],
                value="buy_hold", clearable=False, style=_dd),
        ], id="ctrl-portfolio", style={"display":"none"}),

        html.Div([
            html.Label("Factor model", style=_lbl),
            dcc.Dropdown(id="dd-factor-model",
                options=[
                    {"label":"FF3 (3-factor)",   "value":"ff3"},
                    {"label":"FF5 (5-factor)",   "value":"ff5"},
                    {"label":"FF3 + Momentum",   "value":"mom"},
                    {"label":"FF6 (all factors)","value":"ff6"},
                ],
                value="ff5", clearable=False, style=_dd),
        ], id="ctrl-factors", style={"display":"none"}),

        # placeholder for sidebar-extra output
        html.Div(id="sidebar-extra", style={"display":"none"}),

        # Status
        html.Div(id="status-msg", style={
            "color": COLORS["muted"], "fontSize": "11px",
            "marginTop": "16px",
        }),

    ], style={
        "background": COLORS["sidebar"],
        "padding": "24px 16px",
        "height": "100vh",
        "overflowY": "auto",
        "position": "sticky",
        "top": "0",
    })


# ---------------------------------------------------------------------------
# Metric card
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, color: str = COLORS["text"]) -> html.Div:
    return html.Div([
        html.Div(label, style={
            "fontSize": "10px", "fontWeight": "600",
            "color": COLORS["muted"], "textTransform": "uppercase",
            "letterSpacing": "0.8px", "marginBottom": "4px",
        }),
        html.Div(value, style={
            "fontSize": "22px", "fontWeight": "500",
            "color": color, "fontVariantNumeric": "tabular-nums",
        }),
    ], style=CARD_STYLE)


# ---------------------------------------------------------------------------
# Tab layout helpers
# ---------------------------------------------------------------------------

def plot_card(graph_id: str, height: int = 450) -> html.Div:
    return html.Div(
        dcc.Graph(id=graph_id, style={"height": f"{height}px"},
                  config={"displayModeBar": True, "responsive": True}),
        style=CARD_STYLE,
    )


def section_title(text: str) -> html.Div:
    return html.Div(text, style={
        "fontSize": "13px", "fontWeight": "500",
        "color": COLORS["text"], "marginBottom": "12px",
        "paddingBottom": "8px",
        "borderBottom": f"1px solid {COLORS['border']}",
    })


def loading(children) -> dcc.Loading:
    return dcc.Loading(
        children,
        type="dot",
        color=COLORS["green"],
    )
