"""
dashboard/components.py — reusable UI components.

Dark theme throughout — matches plots.py dark background (#111111 / #1E1E1C).
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


# ---------------------------------------------------------------------------
# Colour tokens — aligned with plots.py _LAYOUT palette
# ---------------------------------------------------------------------------
COLORS = {
    "bg":       "#0D0D0D",       # page background (darker than plots)
    "surface":  "#1A1A1A",       # card / panel surface
    "sidebar":  "#111111",       # sidebar background = plots bg
    "navbar":   "#0A0A0A",       # top navbar
    "border":   "#2A2A2A",       # card borders
    "border_l": "#333333",       # lighter border (inputs, dividers)
    "card":     "#1A1A1A",       # plot card background
    "text":     "#EEEEEE",       # primary text
    "muted":    "#888888",       # secondary / label text
    "dim":      "#555555",       # disabled / placeholder
    "green":    "#1D9E75",       # accent green (matches _LAYOUT palette)
    "red":      "#E24B4A",
    "blue":     "#378ADD",
    "orange":   "#FF8C00",
    "yellow":   "#FFD700",
    "input_bg": "#222222",       # input / dropdown background
    "input_bd": "#383838",       # input border
}

CARD_STYLE = {
    "background":    COLORS["card"],
    "border":        f"1px solid {COLORS['border']}",
    "borderRadius":  "8px",
    "padding":       "16px",
    "marginBottom":  "16px",
}

# Shared label / dropdown styles for sidebar
_LBL = {
    "color":      COLORS["muted"],
    "fontSize":   "11px",
    "marginTop":  "10px",
    "display":    "block",
    "fontWeight": "500",
    "letterSpacing": "0.5px",
}
_DD = {
    "fontSize":        "12px",
    "backgroundColor": COLORS["input_bg"],
    "color":           COLORS["text"],
}
_INPUT = {
    "width":        "100%",
    "background":   COLORS["input_bg"],
    "color":        COLORS["text"],
    "border":       f"1px solid {COLORS['input_bd']}",
    "borderRadius": "6px",
    "padding":      "8px",
    "fontSize":     "12px",
}


# ---------------------------------------------------------------------------
# Header / Navbar
# ---------------------------------------------------------------------------

def header() -> dbc.Navbar:
    return dbc.Navbar(
        dbc.Container([
            html.Span("📈  QuantDashboard", style={
                "color":        COLORS["text"],
                "fontWeight":   "600",
                "fontSize":     "17px",
                "letterSpacing": "-0.3px",
            }),
            html.Span("powered by yfinance_api3", style={
                "color":      COLORS["muted"],
                "fontSize":   "11px",
                "marginLeft": "14px",
            }),
            # right-side status badge
            html.Div(id="nav-status", style={
                "marginLeft": "auto",
                "color":      COLORS["muted"],
                "fontSize":   "11px",
            }),
        ], fluid=True),
        color=COLORS["navbar"],
        dark=True,
        style={"padding": "10px 24px", "borderBottom": f"1px solid {COLORS['border']}"},
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def sidebar() -> html.Div:
    return html.Div([

        _sidebar_label("Configuration"),

        # ── Symbols ─────────────────────────────────────────────────────
        html.Label("Symbols", style={**_LBL, "marginTop": "0"}),
        dcc.Textarea(
            id="input-symbols",
            value="AAPL, MSFT, NVDA, GOOGL, JPM",
            style={**_INPUT, "height": "72px", "resize": "none"},
        ),

        # ── Benchmark ───────────────────────────────────────────────────
        html.Label("Benchmark", style=_LBL),
        dcc.Input(id="input-benchmark", value="SPY", style=_INPUT),

        # ── Period ──────────────────────────────────────────────────────
        html.Label("Period", style=_LBL),
        dcc.Dropdown(
            id="input-period",
            options=[
                {"label": "1 Year",   "value": "1y"},
                {"label": "2 Years",  "value": "2y"},
                {"label": "3 Years",  "value": "3y"},
                {"label": "5 Years",  "value": "5y"},
                {"label": "10 Years", "value": "10y"},
            ],
            value="3y", clearable=False, style=_DD,
        ),

        # ── Risk-free rate ───────────────────────────────────────────────
        html.Label("Risk-free rate (%)", style=_LBL),
        dcc.Slider(
            id="input-rfr",
            min=0.0, max=0.10, step=0.005, value=0.05,
            marks={0: {"label": "0%",  "style": {"color": COLORS["muted"]}},
                   0.05: {"label": "5%", "style": {"color": COLORS["muted"]}},
                   0.10: {"label": "10%","style": {"color": COLORS["muted"]}}},
            tooltip={"placement": "bottom", "always_visible": True},
        ),

        html.Div(style={"height": "20px"}),

        # ── Run button ───────────────────────────────────────────────────
        dbc.Button(
            "▶  Run Analysis",
            id="btn-run",
            color="success",
            style={"width": "100%", "fontWeight": "600", "letterSpacing": "0.3px"},
        ),

        html.Hr(style={"borderColor": COLORS["border_l"], "margin": "20px 0 14px"}),

        # ── Tab-specific controls label ──────────────────────────────────
        _sidebar_label("Tab options"),

        # ── Seasonality controls ─────────────────────────────────────────
        html.Div([
            html.Label("Symbol", style=_LBL),
            dcc.Dropdown(id="dd-season-symbol", options=[], value=None,
                         clearable=False, style=_DD),
            html.Label("Granularity", style=_LBL),
            dcc.Dropdown(id="dd-season-gran",
                options=[{"label":"Monthly","value":"monthly"},
                         {"label":"Weekly","value":"weekly"}],
                value="monthly", clearable=False, style=_DD),
            html.Label("Long-term window", style=_LBL),
            dcc.Dropdown(id="dd-season-lt",
                options=[{"label":f"{n}y","value":f"{n}y"} for n in [5,10,15,20]],
                value="10y", clearable=False, style=_DD),
            html.Label("Short-term window", style=_LBL),
            dcc.Dropdown(id="dd-season-st",
                options=[{"label":f"{n}y","value":f"{n}y"} for n in [1,2,3,5]],
                value="5y", clearable=False, style=_DD),
        ], id="ctrl-seasonality", style={"display": "none"}),

        # ── Portfolio controls ───────────────────────────────────────────
        html.Div([
            html.Label("Backtest strategy", style=_LBL),
            dcc.Dropdown(id="dd-bt-strategy",
                options=[
                    {"label":"Buy & Hold",    "value":"buy_hold"},
                    {"label":"MA 20/50",       "value":"ma_20_50"},
                    {"label":"MA 50/200",      "value":"ma_50_200"},
                    {"label":"Momentum (63d)", "value":"momentum"},
                    {"label":"Mean Reversion", "value":"mean_rev"},
                ],
                value="buy_hold", clearable=False, style=_DD),
        ], id="ctrl-portfolio", style={"display": "none"}),

        # ── Factors controls ─────────────────────────────────────────────
        html.Div([
            html.Label("Factor model", style=_LBL),
            dcc.Dropdown(id="dd-factor-model",
                options=[
                    {"label":"FF3 (3-factor)",   "value":"ff3"},
                    {"label":"FF5 (5-factor)",   "value":"ff5"},
                    {"label":"FF3 + Momentum",   "value":"mom"},
                    {"label":"FF6 (all factors)","value":"ff6"},
                ],
                value="ff5", clearable=False, style=_DD),
        ], id="ctrl-factors", style={"display": "none"}),

        # ── Options controls ─────────────────────────────────────────────
        html.Div([
            html.Label("Ticker", style=_LBL),
            dcc.Input(id="input-opt-symbol", value="SPY", style=_INPUT),
            html.Label("Expiry", style=_LBL),
            dcc.Dropdown(id="dd-opt-expiry", options=[], value=None,
                         clearable=False, style=_DD,
                         placeholder="Click Load Expiries first"),
            dbc.Button("Load Expiries", id="btn-opt-load", size="sm",
                       color="secondary", outline=True,
                       style={"width":"100%","marginTop":"8px","fontSize":"11px"}),
        ], id="ctrl-options", style={"display": "none"}),

        # ── Positions controls ───────────────────────────────────────────
        html.Div([
            html.Label("Position file", style=_LBL),
            dcc.Dropdown(id="dd-pos-book", options=[], value=None,
                         clearable=False, style=_DD,
                         placeholder="Select position..."),
            html.Label("Days ahead", style=_LBL),
            dcc.Slider(id="sl-pos-days", min=0, max=60, step=1, value=0,
                marks={0:{"label":"0d","style":{"color":COLORS["muted"]}},
                       30:{"label":"30d","style":{"color":COLORS["muted"]}},
                       60:{"label":"60d","style":{"color":COLORS["muted"]}}},
                tooltip={"placement":"bottom","always_visible":True}),
            html.Label("± σ range", style=_LBL),
            dcc.Slider(id="sl-pos-dstd", min=1.0, max=5.0, step=0.5, value=3.0,
                marks={1:{"label":"1σ","style":{"color":COLORS["muted"]}},
                       3:{"label":"3σ","style":{"color":COLORS["muted"]}},
                       5:{"label":"5σ","style":{"color":COLORS["muted"]}}},
                tooltip={"placement":"bottom","always_visible":True}),
        ], id="ctrl-positions", style={"display": "none"}),

        # ── Bitcoin controls ─────────────────────────────────────────────
        html.Div([
            html.Label("Corridor low %", style=_LBL),
            dcc.Slider(id="sl-btc-low", min=5, max=25, step=5, value=10,
                marks={5:{"label":"5","style":{"color":COLORS["muted"]}},
                       10:{"label":"10","style":{"color":COLORS["muted"]}},
                       25:{"label":"25","style":{"color":COLORS["muted"]}}},
                tooltip={"placement":"bottom","always_visible":True}),
            html.Label("Corridor high %", style=_LBL),
            dcc.Slider(id="sl-btc-high", min=75, max=95, step=5, value=90,
                marks={75:{"label":"75","style":{"color":COLORS["muted"]}},
                       90:{"label":"90","style":{"color":COLORS["muted"]}},
                       95:{"label":"95","style":{"color":COLORS["muted"]}}},
                tooltip={"placement":"bottom","always_visible":True}),
            html.Label("Forecast years", style=_LBL),
            dcc.Slider(id="sl-btc-years", min=1, max=8, step=1, value=4,
                marks={1:{"label":"1","style":{"color":COLORS["muted"]}},
                       4:{"label":"4","style":{"color":COLORS["muted"]}},
                       8:{"label":"8","style":{"color":COLORS["muted"]}}},
                tooltip={"placement":"bottom","always_visible":True}),
        ], id="ctrl-bitcoin", style={"display": "none"}),

        # ── Reports controls ─────────────────────────────────────────────
        html.Div([
            html.Label("Report title", style=_LBL),
            dcc.Input(id="input-rpt-title", value="Equity Research Report",
                      style=_INPUT),
            html.Label("Include sections", style=_LBL),
            dcc.Checklist(
                id="chk-rpt-sections",
                options=[
                    {"label": " Fundamentals",  "value": "fundamentals"},
                    {"label": " Frontier",       "value": "frontier"},
                    {"label": " Backtest",       "value": "backtest"},
                    {"label": " Monte Carlo",    "value": "montecarlo"},
                    {"label": " Factor model",   "value": "factors"},
                ],
                value=["frontier", "backtest"],
                labelStyle={"display": "block", "color": COLORS["muted"],
                            "fontSize": "12px", "marginTop": "4px"},
                inputStyle={"marginRight": "6px"},
            ),
            html.Label("MC simulations", style=_LBL),
            dcc.Dropdown(id="dd-rpt-mcsims",
                options=[{"label":"200 (fast)","value":200},
                         {"label":"500","value":500},
                         {"label":"1000","value":1000},
                         {"label":"5000 (accurate)","value":5000}],
                value=500, clearable=False, style=_DD),
            dbc.Button("Generate Report", id="btn-rpt-gen", color="primary",
                       style={"width":"100%","marginTop":"14px",
                              "fontWeight":"600","fontSize":"12px"}),
        ], id="ctrl-reports", style={"display": "none"}),

        # Status / feedback line
        html.Div(id="status-msg", style={
            "color":      COLORS["muted"],
            "fontSize":   "11px",
            "marginTop":  "16px",
            "lineHeight": "1.5",
        }),

    ], style={
        "background":  COLORS["sidebar"],
        "padding":     "20px 14px",
        "height":      "100vh",
        "overflowY":   "auto",
        "position":    "sticky",
        "top":         "0",
        "borderRight": f"1px solid {COLORS['border']}",
    })


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _sidebar_label(text: str) -> html.Div:
    return html.Div(text, style={
        "color":          COLORS["dim"],
        "fontSize":       "10px",
        "fontWeight":     "600",
        "letterSpacing":  "1.5px",
        "textTransform":  "uppercase",
        "marginBottom":   "12px",
    })


def plot_card(graph_id: str, height: int = 450) -> html.Div:
    return html.Div(
        dcc.Graph(
            id=graph_id,
            style={"height": f"{height}px"},
            config={"displayModeBar": True, "responsive": True},
        ),
        style=CARD_STYLE,
    )


def section_title(text: str) -> html.Div:
    return html.Div(text, style={
        "fontSize":     "12px",
        "fontWeight":   "600",
        "color":        COLORS["muted"],
        "marginBottom": "12px",
        "paddingBottom": "8px",
        "borderBottom": f"1px solid {COLORS['border']}",
        "textTransform": "uppercase",
        "letterSpacing": "0.8px",
    })


def loading(children) -> dcc.Loading:
    return dcc.Loading(
        children,
        type="dot",
        color=COLORS["green"],
    )


def metric_card(label: str, value: str, color: str = COLORS["text"]) -> html.Div:
    return html.Div([
        html.Div(label, style={
            "fontSize":      "10px",
            "fontWeight":    "600",
            "color":         COLORS["muted"],
            "textTransform": "uppercase",
            "letterSpacing": "0.8px",
            "marginBottom":  "4px",
        }),
        html.Div(value, style={
            "fontSize":           "22px",
            "fontWeight":         "500",
            "color":              color,
            "fontVariantNumeric": "tabular-nums",
        }),
    ], style=CARD_STYLE)
