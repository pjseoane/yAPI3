"""
plots.py — interactive plot functions built on Plotly.

All functions share the same conventions:
  - First argument is QuantAnalytics or StockClient depending on data source
  - Return a plotly Figure (caller decides whether to show/save/embed)
  - Consistent visual style via _LAYOUT defaults

Available plots
---------------
scatter(quant, symbols, ...)             — any quant metric, two periods, benchmark quadrants
cumulative_returns(quant, symbols, ...)  — growth of $100 rebased chart
drawdown(quant, symbols, ...)            — underwater equity curves
rolling_volatility(quant, symbols, ...)  — rolling annualised vol
rolling_sharpe(quant, symbols, ...)      — rolling Sharpe ratio
correlation_heatmap(quant, symbols, ...) — Pearson correlation matrix
returns_distribution(quant, symbols, ...) — return histogram + KDE per symbol
metrics_bar(quant, symbols, ...)         — cross-sectional bar chart for any metric
fundamentals_scatter(client, symbols, ...)— two fundamental/computed fields, bubble sizing
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from yfinance_api3.classes.quant_analytics import QuantAnalytics
from yfinance_api3.modules.factors import _FACTOR_LABELS


# ---------------------------------------------------------------------------
# Architecture notes — lessons learned
# ---------------------------------------------------------------------------
#
# 1. DATE vs CATEGORICAL AXIS CONFLICT IN MULTI-SUBPLOT FIGURES
#    Symptom : chart renders title + legend but panels are completely empty
#              (no bars, no lines, no dots, no axis labels)
#    Cause   : make_subplots() + barmode="group" forces Plotly to reconcile
#              ALL subplot x-axes together. When one panel is categorical
#              (string labels) and another is date (ISO strings), Plotly
#              fails silently and renders nothing.
#    WRONG   : make_subplots(rows=2) + fig.update_xaxes(type="category", row=1)
#              + fig.update_xaxes(type="date", row=2) + update_layout(barmode="group")
#    RIGHT   : Use domain-based manual layout — define xaxis/xaxis2 directly
#              in update_layout() with explicit domain, type, and anchor.
#              Each trace declares xaxis="x" or xaxis="x2" explicitly.
#              No make_subplots(), no barmode conflict.
#    Example : see best_worst_days() — Panel 1 categorical, Panel 2 date.
#
# 2. _apply_layout() RESETS AXIS TYPES IN SUBPLOTS
#    Symptom : axis type set before _apply_layout() is lost afterwards.
#    Cause   : _apply_layout spreads **_LAYOUT which contains xaxis=dict(...)
#              This overwrites the entire xaxis object, clearing type="date"
#              or type="category" that was previously set.
#    Fix     : re-apply axis types AFTER _apply_layout(), or use domain-based
#              layout (note 1) which sets axis types inside update_layout()
#              directly and avoids the issue entirely.
#
# 3. SEASONALITY PLOTS — USE _apply_layout NOT _apply_date_layout
#    Symptom : empty chart when using _apply_date_layout on seasonality plots.
#    Cause   : _apply_date_layout calls update_xaxes(type="date") globally.
#              Seasonality x-axes are categorical ("Jan","Feb" or "W01","W02"),
#              so forcing type="date" silently drops all traces.
#    Fix     : seasonality plots always use _apply_layout (no date forcing).
#
# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

_PALETTE = px.colors.qualitative.T10

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FAFAF9",
    font=dict(family="sans-serif", color="#5F5E5A", size=12),
    hoverlabel=dict(bgcolor="white", font_size=12),
    margin=dict(l=60, r=40, t=70, b=60),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        font=dict(size=11),
    ),
    xaxis=dict(
        gridcolor="#D3D1C7", gridwidth=0.5,
        linecolor="#D3D1C7", zerolinecolor="#B4B2A9",
        tickfont=dict(size=10, color="#888780"),
        title_font=dict(size=12, color="#5F5E5A"),
    ),
    yaxis=dict(
        gridcolor="#D3D1C7", gridwidth=0.5,
        linecolor="#D3D1C7", zerolinecolor="#B4B2A9",
        tickfont=dict(size=10, color="#888780"),
        title_font=dict(size=12, color="#5F5E5A"),
    ),
)


def _apply_date_layout(fig: go.Figure, title: str, subtitle: str = "") -> go.Figure:
    """Like _apply_layout but forces type='date' on all x-axes after spreading
    _LAYOUT — prevents the xaxis dict in _LAYOUT from resetting Plotly's
    auto-detected axis type back to linear (which renders as 1970 epoch)."""
    _apply_layout(fig, title, subtitle)
    fig.update_xaxes(type="date", tickformat="%b %Y")
    return fig


def _apply_layout(fig: go.Figure, title: str, subtitle: str = "") -> go.Figure:
    full_title = f"<b>{title}</b>"
    if subtitle:
        full_title += f"<br><sup style='color:#888780'>{subtitle}</sup>"
    fig.update_layout(
        **_LAYOUT,
        title=dict(text=full_title, font=dict(size=15, color="#2C2C2A"), x=0.02),
    )
    return fig


def _pct_fmt(val: float) -> str:
    return f"{val * 100:.1f}%"


def _period_label(period: str) -> str:
    return {
        "1y": "1-year", "2y": "2-year", "3y": "3-year",
        "5y": "5-year", "10y": "10-year", "ytd": "YTD", "max": "max",
    }.get(period, period)


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

_METRICS: dict[str, tuple[Callable, str]] = {
    "sharpe":      (lambda q, s, p, rf: q.sharpe_ratio(s, p, rf),      "Sharpe ratio"),
    "sortino":     (lambda q, s, p, rf: q.sortino_ratio(s, p, rf),     "Sortino ratio"),
    "calmar":      (lambda q, s, p, rf: q.calmar_ratio(s, p),          "Calmar ratio"),
    "volatility":  (lambda q, s, p, rf: q.historical_volatility(s, p), "Volatility (ann.)"),
    "max_drawdown":(lambda q, s, p, rf: q.max_drawdown(s, p),          "Max drawdown"),
    "var":         (lambda q, s, p, rf: q.var(s, p),                   "VaR 95% (1d)"),
    "cvar":        (lambda q, s, p, rf: q.cvar(s, p),                  "CVaR 95% (1d)"),
    "beta":        (lambda q, s, p, rf: q.beta(s, period=p),           "Beta (vs SPY)"),
}


def list_metrics() -> list[str]:
    return list(_METRICS.keys())


def _resolve_metric(metric, metric_label):
    if callable(metric):
        if metric_label is None:
            raise ValueError("Pass metric_label='...' when supplying a custom callable.")
        return metric, metric_label
    if metric not in _METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Available: {list(_METRICS.keys())}")
    fn, default_label = _METRICS[metric]
    return fn, metric_label or default_label


# ---------------------------------------------------------------------------
# 1. Scatter — quant metric × two periods with benchmark cross-hair
# ---------------------------------------------------------------------------

def scatter(
    quant: QuantAnalytics,
    symbols: list[str],
    metric: str | Callable = "sharpe",
    period_x: str = "2y",
    period_y: str = "5y",
    benchmark: str = "SPY",
    risk_free_rate: float = 0.05,
    metric_label: str | None = None,
) -> go.Figure:
    """
    Scatter any quant metric across two time periods with a benchmark cross-hair.

    Quadrants are defined by the benchmark's own metric values.
    Built-in metrics: "sharpe", "sortino", "calmar", "volatility",
                      "max_drawdown", "var", "cvar", "beta"
    Custom: callable(quant, symbol, period, rf) -> float
    """
    fn, label = _resolve_metric(metric, metric_label)
    label_x   = _period_label(period_x)
    label_y   = _period_label(period_y)

    try:
        bx = fn(quant, benchmark, period_x, risk_free_rate)
        by = fn(quant, benchmark, period_y, risk_free_rate)
    except Exception as e:
        raise ValueError(f"Could not fetch benchmark '{benchmark}': {e}")

    data = {}
    for sym in symbols:
        if sym.upper() == benchmark.upper():
            continue
        try:
            data[sym] = {
                "sx": fn(quant, sym, period_x, risk_free_rate),
                "sy": fn(quant, sym, period_y, risk_free_rate),
            }
        except Exception as e:
            print(f"Skipping {sym}: {e}")

    if not data:
        raise ValueError("No data could be fetched.")

    all_x = [v["sx"] for v in data.values()] + [bx]
    all_y = [v["sy"] for v in data.values()] + [by]
    pad   = 0.35
    lo    = min(min(all_x), min(all_y)) - pad
    hi    = max(max(all_x), max(all_y)) + pad

    fig = go.Figure()

    # quadrant shading
    quad_configs = [
        (bx, hi, by, hi,  "rgba(29,158,117,0.07)",  f"Beat {benchmark} — both periods"),
        (lo, bx, by, hi,  "rgba(55,138,221,0.05)",  f"Beat {benchmark} — {label_y} only"),
        (bx, hi, lo, by,  "rgba(186,117,23,0.05)",  f"Beat {benchmark} — {label_x} only"),
        (lo, bx, lo, by,  "rgba(226,75,74,0.07)",   f"Lag {benchmark} — both periods"),
    ]
    for x0, x1, y0, y1, color, name in quad_configs:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=color, line_width=0, layer="below")

    # diagonal reference
    diag = [lo, hi]
    fig.add_trace(go.Scatter(
        x=diag, y=diag, mode="lines",
        line=dict(color="#B4B2A9", width=1, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ))

    # benchmark cross-hair
    fig.add_hline(y=by, line=dict(color="#5F5E5A", width=1, dash="dash"))
    fig.add_vline(x=bx, line=dict(color="#5F5E5A", width=1, dash="dash"))

    # symbols
    for (sym, vals), color in zip(data.items(), _PALETTE):
        fig.add_trace(go.Scatter(
            x=[vals["sx"]], y=[vals["sy"]],
            mode="markers+text",
            marker=dict(size=12, color=color),
            text=[sym], textposition="top right",
            textfont=dict(size=10, color=color),
            name=sym,
            customdata=[[sym, vals["sx"], vals["sy"]]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"{label} ({label_x}): %{{customdata[1]:.3f}}<br>"
                f"{label} ({label_y}): %{{customdata[2]:.3f}}<extra></extra>"
            ),
        ))

    # benchmark
    fig.add_trace(go.Scatter(
        x=[bx], y=[by],
        mode="markers+text",
        marker=dict(size=18, symbol="diamond", color="#2C2C2A",
                    line=dict(color="white", width=1.5)),
        text=[benchmark], textposition="top right",
        textfont=dict(size=11, color="#2C2C2A", family="sans-serif"),
        name=benchmark,
        hovertemplate=(
            f"<b>{benchmark}</b><br>"
            f"{label} ({label_x}): {bx:.3f}<br>"
            f"{label} ({label_y}): {by:.3f}<extra></extra>"
        ),
    ))

    _apply_layout(fig,
                  title=f"{label}: {label_x} vs {label_y}",
                  subtitle=f"benchmark: {benchmark}  ·  risk-free rate {risk_free_rate:.0%}")
    fig.update_xaxes(range=[lo, hi], title_text=f"{label} — {label_x}")
    fig.update_yaxes(range=[lo, hi], title_text=f"{label} — {label_y}")
    return fig


# ---------------------------------------------------------------------------
# 2. Cumulative returns
# ---------------------------------------------------------------------------

def cumulative_returns(
    quant: QuantAnalytics,
    symbols: list[str],
    period: str = "1y",
    base: float = 100.0,
) -> go.Figure:
    """Growth of *base* dollars invested at the start of *period*."""
    df = quant.cumulative_returns_df(symbols, period=period, base=base)

    fig = go.Figure()
    for sym, color in zip(symbols, _PALETTE):
        if sym not in df.columns:
            continue
        s = df[sym]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            mode="lines", name=sym,
            line=dict(color=color, width=1.8),
            hovertemplate=f"<b>{sym}</b><br>%{{x|%Y-%m-%d}}<br>${{y:.2f}}<extra></extra>",
        ))

    fig.add_hline(y=base, line=dict(color="#B4B2A9", width=1, dash="dash"))
    fig.update_yaxes(title_text=f"Value (started at {base:.0f})")
    fig.update_xaxes(title_text="Date")
    _apply_date_layout(fig, title=f"Cumulative returns — {_period_label(period)}")
    return fig


# ---------------------------------------------------------------------------
# 3. Drawdown
# ---------------------------------------------------------------------------

def drawdown(
    quant: QuantAnalytics,
    symbols: list[str],
    period: str = "1y",
) -> go.Figure:
    """Drawdown time-series for each symbol."""
    df = quant.drawdown_df(symbols, period=period)

    fig = go.Figure()
    for sym, color in zip(symbols, _PALETTE):
        if sym not in df.columns:
            continue
        s = df[sym]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            mode="lines", name=sym,
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor=color.replace("rgb", "rgba").replace(")", ",0.10)") if color.startswith("rgb") else color,
            hovertemplate=(
                f"<b>{sym}</b><br>%{{x|%Y-%m-%d}}<br>"
                "Drawdown: %{y:.1%}<extra></extra>"
            ),
        ))
        # mark max drawdown
        min_idx = s.idxmin()
        fig.add_trace(go.Scatter(
            x=[min_idx], y=[s.min()],
            mode="markers",
            marker=dict(size=8, color=color, symbol="x"),
            showlegend=False,
            hovertemplate=f"<b>{sym}</b> max DD: {s.min():.1%}<extra></extra>",
        ))

    fig.add_hline(y=0, line=dict(color="#B4B2A9", width=0.8))
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%")
    fig.update_xaxes(title_text="Date")
    _apply_date_layout(fig, title=f"Drawdown — {_period_label(period)}")
    return fig


# ---------------------------------------------------------------------------
# 4. Rolling volatility
# ---------------------------------------------------------------------------

def rolling_volatility(
    quant: QuantAnalytics,
    symbols: list[str],
    period: str = "1y",
    window: int = 21,
) -> go.Figure:
    """Rolling annualised volatility."""
    df = quant.rolling_volatility_df(symbols, period=period, window=window)

    fig = go.Figure()
    for sym, color in zip(symbols, _PALETTE):
        if sym not in df.columns:
            continue
        s = df[sym]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            mode="lines", name=sym,
            line=dict(color=color, width=1.8),
            hovertemplate=(
                f"<b>{sym}</b><br>%{{x|%Y-%m-%d}}<br>"
                "Vol: %{y:.1%}<extra></extra>"
            ),
        ))

    fig.update_yaxes(title_text="Annualised volatility", tickformat=".0%")
    fig.update_xaxes(title_text="Date")
    _apply_date_layout(fig,
                  title=f"Rolling volatility — {_period_label(period)}",
                  subtitle=f"{window}-day window")
    return fig


# ---------------------------------------------------------------------------
# 5. Rolling Sharpe
# ---------------------------------------------------------------------------

def rolling_sharpe(
    quant: QuantAnalytics,
    symbols: list[str],
    period: str = "1y",
    window: int = 63,
    risk_free_rate: float = 0.05,
) -> go.Figure:
    """Rolling Sharpe ratio."""
    df = quant.rolling_sharpe_df(symbols, period=period, window=window,
                                  risk_free_rate=risk_free_rate)

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#B4B2A9", width=0.8, dash="dash"))
    fig.add_hline(y=1, line=dict(color="#D3D1C7", width=0.6, dash="dot"))

    for sym, color in zip(symbols, _PALETTE):
        if sym not in df.columns:
            continue
        s = df[sym]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            mode="lines", name=sym,
            line=dict(color=color, width=1.8),
            hovertemplate=(
                f"<b>{sym}</b><br>%{{x|%Y-%m-%d}}<br>"
                "Sharpe: %{y:.2f}<extra></extra>"
            ),
        ))

    fig.update_yaxes(title_text="Sharpe ratio")
    fig.update_xaxes(title_text="Date")
    _apply_date_layout(fig,
                  title=f"Rolling Sharpe — {_period_label(period)}",
                  subtitle=f"{window}-day window  ·  risk-free rate {risk_free_rate:.0%}")
    return fig


# ---------------------------------------------------------------------------
# 6. Correlation heatmap
# ---------------------------------------------------------------------------

def correlation_heatmap(
    quant: QuantAnalytics,
    symbols: list[str],
    period: str = "1y",
) -> go.Figure:
    """Pearson correlation matrix as an annotated heatmap."""
    corr = quant.correlation_matrix(symbols, period=period)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        zmin=-1, zmax=1,
        colorscale="RdYlGn",
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="<b>%{y} / %{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
        colorbar=dict(
            thickness=12, len=0.8,
            tickfont=dict(size=10, color="#888780"),
            outlinewidth=0,
        ),
    ))

    fig.update_xaxes(tickangle=-40, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    _apply_layout(fig, title=f"Return correlation — {_period_label(period)}")
    fig.update_layout(width=600, height=520)
    return fig


# ---------------------------------------------------------------------------
# 7. Returns distribution
# ---------------------------------------------------------------------------

def returns_distribution(
    quant: QuantAnalytics,
    symbols: list[str],
    period: str = "1y",
    bins: int = 50,
) -> go.Figure:
    """Daily return histogram + KDE overlay for each symbol."""
    from scipy.stats import gaussian_kde

    df = quant.returns_df(symbols, period=period, method="simple")

    fig = go.Figure()
    for sym, color in zip(symbols, _PALETTE):
        if sym not in df.columns:
            continue
        rets = df[sym].dropna().values

        fig.add_trace(go.Histogram(
            x=rets, name=sym,
            histnorm="probability density",
            nbinsx=bins,
            marker_color=color,
            opacity=0.25,
            showlegend=True,
            hovertemplate=f"<b>{sym}</b><br>Return: %{{x:.2%}}<br>Density: %{{y:.2f}}<extra></extra>",
        ))

        kde_x = np.linspace(rets.min(), rets.max(), 300)
        kde_y = gaussian_kde(rets)(kde_x)
        fig.add_trace(go.Scatter(
            x=kde_x, y=kde_y,
            mode="lines", name=f"{sym} KDE",
            line=dict(color=color, width=2),
            showlegend=False,
            hovertemplate=f"<b>{sym}</b><br>Return: %{{x:.2%}}<br>Density: %{{y:.3f}}<extra></extra>",
        ))

    fig.add_vline(x=0, line=dict(color="#B4B2A9", width=1, dash="dash"))
    fig.update_layout(barmode="overlay")
    _apply_layout(fig, title=f"Return distribution — {_period_label(period)}")
    fig.update_xaxes(title_text="Daily return", tickformat=".1%")
    fig.update_yaxes(title_text="Density")
    return fig


# ---------------------------------------------------------------------------
# 8. Metrics bar
# ---------------------------------------------------------------------------

def metrics_bar(
    quant: QuantAnalytics,
    symbols: list[str],
    metric: str | Callable = "sharpe",
    period: str = "1y",
    benchmark: str | None = "SPY",
    risk_free_rate: float = 0.05,
    metric_label: str | None = None,
) -> go.Figure:
    """Horizontal bar chart ranking symbols by any single metric."""
    fn, label = _resolve_metric(metric, metric_label)

    values = {}
    for sym in symbols:
        try:
            values[sym] = fn(quant, sym, period, risk_free_rate)
        except Exception as e:
            print(f"Skipping {sym}: {e}")

    if not values:
        raise ValueError("No data could be fetched.")

    values = dict(sorted(values.items(), key=lambda x: x[1]))
    syms = list(values.keys())
    vals = list(values.values())
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(syms))]

    fig = go.Figure(go.Bar(
        x=vals, y=syms,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}" for v in vals],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>" + label + ": %{x:.3f}<extra></extra>",
    ))

    if benchmark:
        try:
            bval = fn(quant, benchmark, period, risk_free_rate)
            fig.add_vline(
                x=bval,
                line=dict(color="#2C2C2A", width=1.5, dash="dash"),
                annotation=dict(
                    text=benchmark, font=dict(size=10, color="#2C2C2A"),
                    bgcolor="rgba(255,255,255,0.7)",
                ),
            )
        except Exception:
            pass

    fig.add_vline(x=0, line=dict(color="#B4B2A9", width=0.8))
    fig.update_layout(height=max(350, len(syms) * 45 + 120))
    _apply_layout(fig, title=f"{label} — {_period_label(period)}")
    fig.update_xaxes(title_text=label)
    fig.update_yaxes(title_text="")
    return fig


# ---------------------------------------------------------------------------
# Fundamentals / computed fields
# ---------------------------------------------------------------------------

_FUNDAMENTALS: dict[str, tuple[str, str]] = {
    "trailingPE":                       ("Trailing P/E",          "ratio"),
    "forwardPE":                        ("Forward P/E",           "ratio"),
    "priceToBook":                      ("Price / Book",          "ratio"),
    "priceToSalesTrailingTwelveMonths": ("Price / Sales",         "ratio"),
    "enterpriseToEbitda":               ("EV / EBITDA",           "ratio"),
    "enterpriseToRevenue":              ("EV / Revenue",          "ratio"),
    "earningsGrowth":                   ("Earnings growth (YoY)", "pct"),
    "revenueGrowth":                    ("Revenue growth (YoY)",  "pct"),
    "grossMargins":                     ("Gross margin",          "pct"),
    "operatingMargins":                 ("Operating margin",      "pct"),
    "profitMargins":                    ("Net margin",            "pct"),
    "returnOnEquity":                   ("Return on equity",      "pct"),
    "returnOnAssets":                   ("Return on assets",      "pct"),
    "debtToEquity":                     ("Debt / Equity",         "ratio"),
    "currentRatio":                     ("Current ratio",         "ratio"),
    "dividendYield":                    ("Dividend yield",        "pct"),
    "beta":                             ("Beta",                  "ratio"),
    "trailingEps":                      ("EPS (trailing)",        "currency"),
    "marketCap":                        ("Market cap",            "currency"),
    "sector":                           ("Sector",                "text"),
    "industry":                         ("Industry",              "text"),
    # Short interest
    "shortRatio":                       ("Short ratio",           "ratio"),
    # Valuation extras
    "pegRatio":                         ("PEG ratio",             "ratio"),
    "trailingPegRatio":                 ("Trailing PEG ratio",    "ratio"),
    # 52-week changes
    "52WeekChange":                     ("52-week change",        "pct"),
    "SandP52WeekChange":                ("S&P 52-week change",    "pct"),
    "fiftyTwoWeekLowChangePercent":     ("% above 52w low",       "pct"),
    "fiftyTwoWeekHighChangePercent":    ("% below 52w high",      "pct"),
    "fiftyTwoWeekChangePercent":        ("52w change %",          "pct"),
    # Analyst opinions
    "recommendationMean":               ("Recommendation mean",   "ratio"),
    "recommendationKey":                ("Recommendation",        "text"),
    "numberOfAnalystOpinions":          ("# analyst opinions",    "ratio"),
    "averageAnalystRating":             ("Avg analyst rating",    "text"),
}

_COMPUTED_FIELDS: dict[str, tuple] = {
    "pct_from_52w_high": (
        lambda info: (
            (info.get("currentPrice") or info.get("regularMarketPrice", 0))
            / info["fiftyTwoWeekHigh"] - 1
            if info.get("fiftyTwoWeekHigh") else None
        ),
        "Distance from 52-week high", "pct",
    ),
    "pct_from_52w_low": (
        lambda info: (
            (info.get("currentPrice") or info.get("regularMarketPrice", 0))
            / info["fiftyTwoWeekLow"] - 1
            if info.get("fiftyTwoWeekLow") else None
        ),
        "Distance from 52-week low", "pct",
    ),
    "52w_range_position": (
        lambda info: (
            ((info.get("currentPrice") or info.get("regularMarketPrice", 0))
             - info["fiftyTwoWeekLow"])
            / (info["fiftyTwoWeekHigh"] - info["fiftyTwoWeekLow"])
            if info.get("fiftyTwoWeekHigh") and info.get("fiftyTwoWeekLow")
               and info["fiftyTwoWeekHigh"] != info["fiftyTwoWeekLow"]
            else None
        ),
        "52-week range position", "pct",
    ),
}


def list_fundamentals() -> list[str]:
    return list(_FUNDAMENTALS.keys()) + list(_COMPUTED_FIELDS.keys())


def _resolve_field_value(field: str, info: dict) -> float | str | None:
    if field in _COMPUTED_FIELDS:
        fn, _, _ = _COMPUTED_FIELDS[field]
        try:
            return fn(info)
        except Exception:
            return None
    val = info.get(field)
    if val is None:
        return None
    # text fields are returned as-is
    if _field_meta(field)[1] == "text":
        return str(val)
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _field_meta(field: str) -> tuple[str, str]:
    if field in _COMPUTED_FIELDS:
        _, label, hint = _COMPUTED_FIELDS[field]
        return label, hint
    if field in _FUNDAMENTALS:
        return _FUNDAMENTALS[field]
    return field, "raw"


def _axis_fmt(hint: str) -> str:
    return ".1%" if hint == "pct" else ("$,.0f" if hint == "currency" else ".2f")


# ---------------------------------------------------------------------------
# 9. Fundamentals scatter
# ---------------------------------------------------------------------------

def fundamentals_scatter(
    client,
    symbols: list[str],
    field_x: str = "trailingPE",
    field_y: str = "revenueGrowth",
    size_by: str | None = "pct_from_52w_high",
    label_x: str | None = None,
    label_y: str | None = None,
) -> go.Figure:
    """
    Scatter plot of two fundamental/computed fields with optional bubble sizing.

    field_x / field_y : raw yfinance info key OR computed field key.
    size_by           : field used to scale bubble size.
                        Default: "pct_from_52w_high" — stocks closer to
                        their 52-week high appear as larger bubbles.
                        Pass None for uniform dots.

    Raw fields: trailingPE, forwardPE, priceToBook, revenueGrowth,
                grossMargins, operatingMargins, profitMargins,
                returnOnEquity, returnOnAssets, debtToEquity,
                currentRatio, dividendYield, beta, marketCap, ...

    Computed fields: pct_from_52w_high, pct_from_52w_low, 52w_range_position

    Call list_fundamentals() to see all keys.
    """
    lx = label_x or _field_meta(field_x)[0]
    ly = label_y or _field_meta(field_y)[0]
    hx = _field_meta(field_x)[1]
    hy = _field_meta(field_y)[1]

    if hx == "text" or hy == "text":
        raise ValueError(
            "Text fields ('sector', 'industry') cannot be used as axes. "
            "They appear automatically in hover tooltips."
        )

    raw_info: dict[str, dict] = {}
    for sym in symbols:
        try:
            raw_info[sym] = client.get_info(sym)
        except Exception as e:
            print(f"Skipping {sym}: {e}")

    data: dict[str, dict] = {}
    for sym, info in raw_info.items():
        vx = _resolve_field_value(field_x, info)
        vy = _resolve_field_value(field_y, info)
        if vx is None or vy is None:
            print(f"Skipping {sym}: missing '{field_x}' or '{field_y}'")
            continue
        entry: dict = {"x": vx, "y": vy, "name": info.get("shortName", sym),
                       "sector": info.get("sector", ""),
                       "industry": info.get("industry", "")}
        if size_by:
            entry["s"] = _resolve_field_value(size_by, info)
        data[sym] = entry

    if not data:
        raise ValueError("No usable data — check field names or symbols.")

    # bubble size scale [10, 50] in plotly marker size (area ∝ value)
    if size_by and any(d.get("s") is not None for d in data.values()):
        raw_s = np.array(
            [d["s"] if d.get("s") is not None else 0.0 for d in data.values()],
            dtype=float,
        )
        lo_s, hi_s = raw_s.min(), raw_s.max()
        sizes = (10 + (raw_s - lo_s) / (hi_s - lo_s) * 40
                 if hi_s > lo_s else np.full(len(raw_s), 20.0))
    else:
        sizes = np.full(len(data), 14.0)

    xs     = [d["x"] for d in data.values()]
    ys     = [d["y"] for d in data.values()]
    syms   = list(data.keys())
    names  = [d["name"] for d in data.values()]
    fmt_x  = _axis_fmt(hx)
    fmt_y  = _axis_fmt(hy)

    fig = go.Figure()

    # median cross-hair
    fig.add_hline(y=float(np.median(ys)),
                  line=dict(color="#B4B2A9", width=0.8, dash="dot"),
                  annotation=dict(text="median", font=dict(size=9, color="#888780")))
    fig.add_vline(x=float(np.median(xs)),
                  line=dict(color="#B4B2A9", width=0.8, dash="dot"))
    fig.add_hline(y=0, line=dict(color="#D3D1C7", width=0.6, dash="dash"))
    fig.add_vline(x=0, line=dict(color="#D3D1C7", width=0.6, dash="dash"))

    size_label = _field_meta(size_by)[0] if size_by else ""
    for sym, name, vx, vy, size, color in zip(
        syms, names, xs, ys, sizes, _PALETTE
    ):
        sv = data[sym].get("s")
        hover = (
            f"<b>{sym}</b> — {name}<br>"
            + (f"{data[sym]['sector']} · {data[sym]['industry']}<br>" if data[sym].get("sector") else "")
            + f"{lx}: {{x:{fmt_x}}}<br>"
            f"{ly}: {{y:{fmt_y}}}"
            + (f"<br>{size_label}: {sv:{fmt_x}}" if sv is not None and size_by else "")
            + "<extra></extra>"
        )
        fig.add_trace(go.Scatter(
            x=[vx], y=[vy],
            mode="markers+text",
            marker=dict(size=size, color=color, opacity=0.82,
                        line=dict(color="white", width=1)),
            text=[sym], textposition="top right",
            textfont=dict(size=10, color=color),
            name=sym,
            hovertemplate=hover,
        ))

    fig.update_xaxes(title_text=lx, tickformat=fmt_x if hx in ("pct", "currency") else "")
    fig.update_yaxes(title_text=ly, tickformat=fmt_y if hy in ("pct", "currency") else "")

    subtitle = f"bubble size: {size_label}" if size_by else ""
    _apply_layout(fig, title=f"{lx}  vs  {ly}", subtitle=subtitle)
    return fig


# ---------------------------------------------------------------------------
# 10. Strip plot — categorical x-axis (e.g. sector) vs numeric field
# ---------------------------------------------------------------------------

def strip_plot(
    client,
    symbols: list[str],
    field_y: str = "trailingPE",
    group_by: str = "sector",       # "sector" | "industry" | any text info key
    label_y: str | None = None,
    jitter: float = 0.25,           # horizontal spread so points don't overlap
    show_median: bool = True,       # draw a median line per group
) -> go.Figure:
    """
    Strip plot with a categorical x-axis and a numeric y-axis.

    group_by : text field used to define columns — typically "sector" or "industry"
    field_y  : numeric field for the y-axis (any _FUNDAMENTALS or _COMPUTED_FIELDS key)
    jitter   : horizontal spread within each column (0 = stacked, 0.4 = wide spread)

    Each symbol is plotted as a dot. Symbols in the same group share the same
    column but sit at their individual y-values. A median marker per group is
    drawn optionally.

    Hover shows: symbol, full company name, group, and y-value.
    """
    ly = label_y or _field_meta(field_y)[0]
    hy = _field_meta(field_y)[1]

    if _field_meta(group_by)[1] != "text":
        raise ValueError(
            f"'{group_by}' is not a text field. "
            f"group_by must be a categorical field like 'sector' or 'industry'."
        )
    if _field_meta(field_y)[1] == "text":
        raise ValueError(
            f"'{field_y}' is a text field and cannot be used as the y-axis."
        )

    # --- fetch -----------------------------------------------------------
    raw_info: dict[str, dict] = {}
    for sym in symbols:
        try:
            raw_info[sym] = client.get_info(sym)
        except Exception as e:
            print(f"Skipping {sym}: {e}")

    records = []
    for sym, info in raw_info.items():
        vy    = _resolve_field_value(field_y, info)
        group = info.get(group_by, "Unknown") or "Unknown"
        if vy is None:
            print(f"Skipping {sym}: missing '{field_y}'")
            continue
        records.append({
            "sym":     sym,
            "name":    info.get("shortName", sym),
            "group":   group,
            "y":       float(vy),
            "sector":  info.get("sector", ""),
            "industry":info.get("industry", ""),
        })

    if not records:
        raise ValueError("No usable data.")

    # --- build ordered category list (sorted by median y, descending) ----
    import pandas as pd
    df = pd.DataFrame(records)
    group_order = (
        df.groupby("group")["y"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    group_index = {g: i for i, g in enumerate(group_order)}

    # --- jitter: deterministic per symbol so repeated calls are stable ---
    rng = np.random.default_rng(seed=42)
    df["x_jitter"] = df.apply(
        lambda r: group_index[r["group"]] + rng.uniform(-jitter, jitter),
        axis=1,
    )

    fmt_y = _axis_fmt(hy)
    fig   = go.Figure()

    # --- one trace per group for clean legend + color --------------------
    for i, (group, gdf) in enumerate(df.groupby("group")):
        color = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(go.Scatter(
            x=gdf["x_jitter"],
            y=gdf["y"],
            mode="markers+text",
            name=group,
            marker=dict(size=11, color=color,
                        line=dict(color="white", width=1)),
            text=gdf["sym"],
            textposition="top center",
            textfont=dict(size=9, color=color),
            customdata=gdf[["sym", "name", "group", "y", "sector", "industry"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b> — %{customdata[1]}<br>"
                "%{customdata[4]} · %{customdata[5]}<br>"
                f"{ly}: %{{customdata[3]:{fmt_y}}}"
                "<extra></extra>"
            ),
        ))

        # median line per group
        if show_median:
            med = gdf["y"].median()
            gi  = group_index[group]
            fig.add_shape(
                type="line",
                x0=gi - jitter - 0.05, x1=gi + jitter + 0.05,
                y0=med, y1=med,
                line=dict(color=color, width=2, dash="dot"),
                layer="below",
            )
            fig.add_annotation(
                x=gi + jitter + 0.08, y=med,
                text=f"med {med:{fmt_y}}",
                showarrow=False,
                font=dict(size=8, color=color),
                xanchor="left",
            )

    # --- categorical x-axis ----------------------------------------------
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(group_index.values()),
        ticktext=list(group_index.keys()),
        tickangle=-30,
        title_text=group_by.capitalize(),
    )
    fig.update_yaxes(
        title_text=ly,
        tickformat=fmt_y if hy in ("pct", "currency") else "",
    )

    _apply_layout(
        fig,
        title=f"{ly} by {group_by}",
        subtitle=f"median line per group  ·  {len(df)} stocks",
    )
    fig.update_layout(showlegend=True)
    return fig


# ---------------------------------------------------------------------------
# 11. Efficient frontier
# ---------------------------------------------------------------------------

def efficient_frontier(frontier, show_assets: bool = True) -> go.Figure:
    """
    Plot the efficient frontier from a portfolio.EfficientFrontier object.

    frontier    : EfficientFrontier returned by portfolio.efficient_frontier()
    show_assets : overlay individual asset risk/return as dots

    The plot shows:
      - The full frontier curve coloured by Sharpe ratio
      - Max Sharpe portfolio (star marker)
      - Min Variance portfolio (diamond marker)
      - Capital Market Line from risk-free rate through max Sharpe
      - Individual assets (optional)
    """

    pts = frontier.points
    ms  = frontier.max_sharpe
    mv  = frontier.min_variance
    rf  = ms.risk_free_rate

    fig = go.Figure()

    # --- frontier curve coloured by Sharpe --------------------------------
    fig.add_trace(go.Scatter(
        x=pts["volatility"],
        y=pts["return"],
        mode="lines+markers",
        marker=dict(
            size=6,
            color=pts["sharpe"],
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(
                title="Sharpe",
                thickness=12,
                len=0.6,
                tickfont=dict(size=9, color="#888780"),
                outlinewidth=0,
            ),
        ),
        line=dict(color="rgba(0,0,0,0.15)", width=1),
        customdata=np.stack([pts["return"], pts["sharpe"]], axis=1),
        hovertemplate=(
            "Vol: %{x:.2%}<br>"
            "Return: %{customdata[0]:.2%}<br>"
            "Sharpe: %{customdata[1]:.3f}"
            "<extra>Frontier</extra>"
        ),
        showlegend=False,
    ))

    # --- Capital Market Line (CML) ----------------------------------------
    cml_x = np.linspace(0, pts["volatility"].max() * 1.1, 100)
    slope  = (ms.expected_return - rf) / ms.volatility
    cml_y  = rf + slope * cml_x
    fig.add_trace(go.Scatter(
        x=cml_x, y=cml_y,
        mode="lines",
        line=dict(color="#888780", width=1.2, dash="dash"),
        name="Capital Market Line",
        hoverinfo="skip",
    ))

    # --- risk-free rate dot -----------------------------------------------
    fig.add_trace(go.Scatter(
        x=[0], y=[rf],
        mode="markers+text",
        marker=dict(size=8, color="#888780", symbol="circle"),
        text=["Rf"], textposition="top right",
        textfont=dict(size=9, color="#888780"),
        name=f"Risk-free ({rf:.1%})",
        hovertemplate=f"Risk-free rate: {rf:.2%}<extra></extra>",
    ))

    # --- individual assets ------------------------------------------------
    if show_assets:
        q_syms  = frontier.symbols
        # re-use already-fetched data from frontier points
        # compute each asset's own vol and return from the frontier's first/last weights
        for sym, color in zip(q_syms, _PALETTE):
            if sym not in pts.columns:
                continue
            # find a frontier point where this asset has ~100% weight (edge of frontier)
            # instead use max-weight point as proxy for individual asset risk/return
            asset_col = pts[sym]
            idx = asset_col.idxmax()
            if idx is not None and idx in pts.index:
                a_ret = pts.loc[idx, "return"]
                a_vol = pts.loc[idx, "volatility"]
                fig.add_trace(go.Scatter(
                    x=[a_vol], y=[a_ret],
                    mode="markers+text",
                    marker=dict(size=10, color=color, symbol="circle",
                                line=dict(color="white", width=1)),
                    text=[sym], textposition="top right",
                    textfont=dict(size=9, color=color),
                    name=sym,
                    hovertemplate=(
                        f"<b>{sym}</b><br>"
                        f"Vol: %{{x:.2%}}<br>"
                        f"Return: %{{y:.2%}}<extra></extra>"
                    ),
                ))

    # --- Max Sharpe -------------------------------------------------------
    fig.add_trace(go.Scatter(
        x=[ms.volatility], y=[ms.expected_return],
        mode="markers+text",
        marker=dict(size=18, symbol="star", color="#1D9E75",
                    line=dict(color="white", width=1.5)),
        text=["Max Sharpe"], textposition="top right",
        textfont=dict(size=10, color="#1D9E75", family="sans-serif"),
        name=f"Max Sharpe ({ms.sharpe_ratio:.2f})",
        hovertemplate=(
            "<b>Max Sharpe</b><br>"
            f"Sharpe: {ms.sharpe_ratio:.3f}<br>"
            f"Return: {ms.expected_return:.2%}<br>"
            f"Vol: {ms.volatility:.2%}<extra></extra>"
        ),
    ))

    # --- Min Variance -----------------------------------------------------
    fig.add_trace(go.Scatter(
        x=[mv.volatility], y=[mv.expected_return],
        mode="markers+text",
        marker=dict(size=14, symbol="diamond", color="#378ADD",
                    line=dict(color="white", width=1.5)),
        text=["Min Var"], textposition="top right",
        textfont=dict(size=10, color="#378ADD"),
        name="Min Variance",
        hovertemplate=(
            "<b>Min Variance</b><br>"
            f"Sharpe: {mv.sharpe_ratio:.3f}<br>"
            f"Return: {mv.expected_return:.2%}<br>"
            f"Vol: {mv.volatility:.2%}<extra></extra>"
        ),
    ))

    _apply_layout(
        fig,
        title="Efficient Frontier",
        subtitle=f"period: {_period_label(frontier.period)}  ·  "
                 f"rf: {rf:.1%}  ·  {len(frontier.symbols)} assets",
    )
    fig.update_xaxes(title_text="Annualised volatility", tickformat=".1%")
    fig.update_yaxes(title_text="Annualised return",     tickformat=".1%")
    return fig


# ---------------------------------------------------------------------------
# 12. Backtest — equity curve + drawdown + metrics
# ---------------------------------------------------------------------------

def backtest(result, show_drawdown: bool = True) -> go.Figure:
    """
    Plot a BacktestResult: equity curve(s) + optional drawdown panel.

    result       : BacktestResult from backtest.run()
    show_drawdown: add a drawdown subplot below the equity curve
    """
    rows   = 2 if show_drawdown else 1
    heights= [0.68, 0.32] if show_drawdown else [1.0]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=heights,
        vertical_spacing=0.04,
    )

    def _safe_index(s):
        """ISO date strings — unambiguous for Plotly."""
        idx = s.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_convert(None)
        return pd.DatetimeIndex(idx).normalize().strftime("%Y-%m-%d").tolist()

    # --- equity curve ----------------------------------------------------
    fig.add_trace(go.Scatter(
        x=_safe_index(result.equity_curve),
        y=result.equity_curve.values,
        mode="lines",
        name=result.strategy_name,
        line=dict(color=_PALETTE[0], width=2),
        hovertemplate=(
            "%{x|%Y-%m-%d}<br>"
            f"{result.strategy_name}: %{{y:$,.0f}}<extra></extra>"
        ),
    ), row=1, col=1)

    # --- benchmark equity curve ------------------------------------------
    if result.benchmark_result:
        bm = result.benchmark_result
        bm_eq = bm.equity_curve * (result.initial_capital / bm.equity_curve.iloc[0])
        fig.add_trace(go.Scatter(
            x=_safe_index(bm_eq),
            y=bm_eq.values,
            mode="lines",
            name=bm.strategy_name,
            line=dict(color=_PALETTE[2], width=1.5, dash="dot"),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                f"{bm.strategy_name}: %{{y:$,.0f}}<extra></extra>"
            ),
        ), row=1, col=1)

    # --- drawdown panel --------------------------------------------------
    if show_drawdown:
        roll_max = result.equity_curve.cummax()
        dd       = (result.equity_curve - roll_max) / roll_max

        fig.add_trace(go.Scatter(
            x=_safe_index(dd), y=dd.values,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(226,75,74,0.15)",
            line=dict(color="#E24B4A", width=1),
            name="Drawdown",
            hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.1%}<extra></extra>",
        ), row=2, col=1)

        if result.benchmark_result:
            bm = result.benchmark_result
            bm_eq  = bm.equity_curve * (result.initial_capital / bm.equity_curve.iloc[0])
            bm_max = bm_eq.cummax()
            bm_dd  = (bm_eq - bm_max) / bm_max
            fig.add_trace(go.Scatter(
                x=_safe_index(bm_dd), y=bm_dd.values,
                mode="lines",
                line=dict(color=_PALETTE[2], width=1, dash="dot"),
                name=f"{bm.strategy_name} DD",
                hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.1%}<extra></extra>",
            ), row=2, col=1)

        fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1,
                         tickfont=dict(size=9, color="#888780"),
                         gridcolor="#D3D1C7", zerolinecolor="#B4B2A9")

    # --- metrics annotation box ------------------------------------------
    m = result.metrics
    ann_lines = [
        f"CAGR: {m['cagr']:.1%}",
        f"Vol: {m['volatility']:.1%}",
        f"Sharpe: {m['sharpe_ratio']:.2f}",
        f"Max DD: {m['max_drawdown']:.1%}",
        f"Win rate: {m['win_rate']:.0%}",
    ]
    if result.benchmark_result:
        bm = result.benchmark_result.metrics
        ann_lines += [
            f"<br><b>vs {result.benchmark_result.strategy_name}</b>",
            f"CAGR: {bm['cagr']:.1%}",
            f"Sharpe: {bm['sharpe_ratio']:.2f}",
        ]
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.97,
        text="<br>".join(ann_lines),
        showarrow=False, align="left",
        font=dict(size=10, color="#5F5E5A"),
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="#D3D1C7", borderwidth=0.5,
        borderpad=8,
    )

    # --- styling ---------------------------------------------------------
    fig.update_yaxes(title_text="Portfolio value ($)", tickformat="$,.0f",
                     row=1, col=1,
                     tickfont=dict(size=9, color="#888780"),
                     gridcolor="#D3D1C7", zerolinecolor="#B4B2A9")

    _apply_date_layout(
        fig,
        title=f"Backtest — {result.strategy_name}",
        subtitle=f"period: {_period_label(result.period)}",
    )
    fig.update_layout(hovermode="x unified")
    return fig


def monte_carlo(result, show_paths: int = 50) -> go.Figure:
    """
    Two-panel Monte Carlo visualisation.

    Top panel  : fan chart — percentile bands + sample paths
    Bottom panel: terminal value distribution — histogram + VaR/CVaR markers

    result      : MonteCarloResult from montecarlo.simulate()
    show_paths  : number of individual sample paths to draw (default 50)
                  Set to 0 to hide individual paths.
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.08,
        subplot_titles=["Simulated portfolio paths", "Terminal value distribution"],
    )

    pct   = result.percentiles
    days  = list(pct.index)
    init  = result.initial_value
    m     = result.metrics

    # --- fan chart -------------------------------------------------------
    # 5–95 band
    fig.add_trace(go.Scatter(
        x=days + days[::-1],
        y=list(pct[95]) + list(pct[5])[::-1],
        fill="toself",
        fillcolor="rgba(55,138,221,0.10)",
        line=dict(width=0),
        name="5–95 percentile",
        hoverinfo="skip",
    ), row=1, col=1)

    # 25–75 band
    fig.add_trace(go.Scatter(
        x=days + days[::-1],
        y=list(pct[75]) + list(pct[25])[::-1],
        fill="toself",
        fillcolor="rgba(55,138,221,0.20)",
        line=dict(width=0),
        name="25–75 percentile",
        hoverinfo="skip",
    ), row=1, col=1)

    # median
    fig.add_trace(go.Scatter(
        x=days, y=pct[50],
        mode="lines",
        line=dict(color="#378ADD", width=2.5),
        name="Median",
        hovertemplate="Day %{x}<br>Median: $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    # 5th and 95th boundary lines
    fig.add_trace(go.Scatter(
        x=days, y=pct[5],
        mode="lines",
        line=dict(color="#E24B4A", width=1.2, dash="dot"),
        name="5th percentile",
        hovertemplate="Day %{x}<br>5th pct: $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=days, y=pct[95],
        mode="lines",
        line=dict(color="#1D9E75", width=1.2, dash="dot"),
        name="95th percentile",
        hovertemplate="Day %{x}<br>95th pct: $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    # sample paths
    if show_paths > 0:
        cols = result.paths.columns[:show_paths]
        for col in cols:
            fig.add_trace(go.Scatter(
                x=days, y=result.paths[col],
                mode="lines",
                line=dict(color="rgba(55,138,221,0.08)", width=0.6),
                showlegend=False,
                hoverinfo="skip",
            ), row=1, col=1)

    # initial value reference
    fig.add_hline(
        y=init, row=1, col=1,
        line=dict(color="#888780", width=1, dash="dash"),
        annotation=dict(
            text=f"Start ${init:,.0f}",
            font=dict(size=9, color="#888780"),
        ),
    )

    # --- terminal distribution histogram ---------------------------------
    final = result.final_values
    var_val  = init * (1 + m["var_95"])
    cvar_val = init * (1 + m["cvar_95"])

    fig.add_trace(go.Histogram(
        x=final,
        nbinsx=60,
        marker_color="#378ADD",
        opacity=0.65,
        name="Terminal value",
        hovertemplate="$%{x:,.0f}<br>Count: %{y}<extra></extra>",
    ), row=2, col=1)

    # initial value line
    fig.add_vline(
        x=init, row=2, col=1,
        line=dict(color="#888780", width=1.5, dash="dash"),
        annotation=dict(text="Start", font=dict(size=9, color="#888780")),
    )

    # VaR line
    fig.add_vline(
        x=var_val, row=2, col=1,
        line=dict(color="#E24B4A", width=1.5, dash="dot"),
        annotation=dict(
            text=f"VaR 95%<br>${var_val:,.0f}",
            font=dict(size=9, color="#E24B4A"),
        ),
    )

    # CVaR line
    fig.add_vline(
        x=cvar_val, row=2, col=1,
        line=dict(color="#993556", width=1.5, dash="dot"),
        annotation=dict(
            text=f"CVaR 95%<br>${cvar_val:,.0f}",
            font=dict(size=9, color="#993556"),
            xanchor="right",
        ),
    )

    # median line
    fig.add_vline(
        x=m["median_final"], row=2, col=1,
        line=dict(color="#1D9E75", width=1.5),
        annotation=dict(
            text=f"Median<br>${m['median_final']:,.0f}",
            font=dict(size=9, color="#1D9E75"),
        ),
    )

    # --- metrics annotation ----------------------------------------------
    ann = (
        f"Median return: {m['median_return']:+.1%}<br>"
        f"VaR 95%: {m['var_95']:+.1%}<br>"
        f"CVaR 95%: {m['cvar_95']:+.1%}<br>"
        f"Prob of gain: {m['prob_gain']:.0%}<br>"
        f"Prob loss >10%: {m['prob_loss_10pct']:.0%}<br>"
        f"Prob loss >20%: {m['prob_loss_20pct']:.0%}"
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.97,
        text=ann, showarrow=False, align="left",
        font=dict(size=10, color="#5F5E5A"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#D3D1C7", borderwidth=0.5, borderpad=8,
    )

    # --- axes & layout ---------------------------------------------------
    fig.update_yaxes(title_text="Portfolio value ($)", tickformat="$,.0f",
                     row=1, col=1, gridcolor="#D3D1C7",
                     tickfont=dict(size=9, color="#888780"))
    fig.update_yaxes(title_text="Count", row=2, col=1,
                     gridcolor="#D3D1C7",
                     tickfont=dict(size=9, color="#888780"))
    fig.update_xaxes(title_text="Trading days", row=1, col=1,
                     gridcolor="#D3D1C7", tickfont=dict(size=9, color="#888780"))
    fig.update_xaxes(title_text="Terminal value ($)", tickformat="$,.0f",
                     row=2, col=1, gridcolor="#D3D1C7",
                     tickfont=dict(size=9, color="#888780"))

    _apply_layout(
        fig,
        title=f"Monte Carlo Simulation — {result.method}",
        subtitle=(
            f"{result.n_sims:,} paths  ·  {result.horizon}-day horizon  ·  "
            f"fitted on {_period_label(result.period)}  ·  "
            f"{'equal weight' if len(set(result.weights)) == 1 else 'custom weights'}"
        ),
    )
    fig.update_layout(height=750, hovermode="x")
    # x-axis Panel 1 is trading day integers — must be linear not date
    fig.update_xaxes(type="linear", row=1, col=1)
    # x-axis Panel 2 is dollar values — also linear
    fig.update_xaxes(type="linear", row=2, col=1)
    return fig


# ---------------------------------------------------------------------------
# 14. Weekly seasonality — bar chart + cumulative drift
# ---------------------------------------------------------------------------

def seasonality(
    quant: QuantAnalytics,
    symbol: str,
    period: str = "10y",
    granularity: str = "weekly",    # "weekly" | "monthly"
    show_cumulative: bool = True,
) -> go.Figure:
    """
    Seasonality study — average return per week (or month) across all years.

    Bars are coloured green/red by sign. A cumulative drift line shows the
    seasonal bias building across the year. Unreliable weeks (< 3 observations)
    are shown with reduced opacity.

    granularity : "weekly"  → 52 bars, one per ISO week
                  "monthly" → 12 bars, one per calendar month
    """
    # both weekly and monthly now return a pivot (period × year)
    if granularity == "monthly":
        pivot   = quant.monthly_seasonality(symbol, period=period)
        x_vals  = list(pivot.index)   # ["Jan", "Feb", ...]
        x_title = "Month"
    else:
        pivot   = quant.weekly_seasonality(symbol, period=period)
        x_vals  = [f"W{int(w):02d}" for w in pivot.index]
        x_title = "ISO week"

    mean_ret   = pivot.mean(axis=1).values
    median_ret = pivot.median(axis=1).values
    win_rate   = (pivot > 0).mean(axis=1).values
    n_obs      = pivot.count(axis=1).values
    cumul      = pivot.mean(axis=1).cumsum().values
    reliable   = pivot.count(axis=1).values >= 3

    colors  = ["#1D9E75" if v >= 0 else "#E24B4A" for v in mean_ret]
    opacity = [0.85 if r else 0.35 for r in reliable]

    rows = 2 if show_cumulative else 1
    fig  = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38] if show_cumulative else [1.0],
        vertical_spacing=0.06,
        subplot_titles=[
            f"Mean {'weekly' if granularity=='weekly' else 'monthly'} return",
            "Cumulative seasonal drift",
        ] if show_cumulative else [
            f"Mean {'weekly' if granularity=='weekly' else 'monthly'} return"
        ],
    )

    # --- bar chart -------------------------------------------------------
    for i, (x, y, c, op, wr, n) in enumerate(
        zip(x_vals, mean_ret, colors, opacity, win_rate, n_obs)
    ):
        fig.add_trace(go.Bar(
            x=[x], y=[y],
            marker_color=c,
            marker_opacity=op,
            showlegend=False,
            customdata=[[wr, n, median_ret[i]]],
            hovertemplate=(
                f"<b>{x}</b><br>"
                "Mean return: %{y:.2%}<br>"
                "Median: %{customdata[0][2]:.2%}<br>"
                "Win rate: %{customdata[0][0]:.0%}<br>"
                "Observations: %{customdata[0][1]:.0f}"
                + ("" if reliable[i] else "<br><i>⚠ few observations</i>")
                + "<extra></extra>"
            ),
        ), row=1, col=1)

    # zero line
    fig.add_hline(y=0, row=1, col=1,
                  line=dict(color="#B4B2A9", width=0.8))

    # win-rate dots overlay
    fig.add_trace(go.Scatter(
        x=x_vals, y=mean_ret,
        mode="markers",
        marker=dict(
            size=6,
            color=win_rate,
            colorscale=[[0, "#E24B4A"], [0.5, "#F4F3EF"], [1, "#1D9E75"]],
            cmin=0, cmax=1,
            showscale=True,
            colorbar=dict(
                title="Win rate",
                thickness=10, len=0.45, y=0.78,
                tickformat=".0%",
                tickfont=dict(size=9, color="#888780"),
                outlinewidth=0,
            ),
            line=dict(color="white", width=1),
        ),
        showlegend=False,
        hoverinfo="skip",
    ), row=1, col=1)

    # --- cumulative drift ------------------------------------------------
    if show_cumulative:
        fig.add_trace(go.Scatter(
            x=x_vals, y=cumul,
            mode="lines+markers",
            line=dict(color="#378ADD", width=2),
            marker=dict(size=4, color="#378ADD"),
            fill="tozeroy",
            fillcolor="rgba(55,138,221,0.08)",
            name="Cumulative drift",
            hovertemplate="%{x}<br>Cumulative: %{y:.2%}<extra></extra>",
        ), row=2, col=1)

        fig.add_hline(y=0, row=2, col=1,
                      line=dict(color="#B4B2A9", width=0.8))

        fig.update_yaxes(
            tickformat=".1%", row=2, col=1,
            gridcolor="#D3D1C7",
            tickfont=dict(size=9, color="#888780"),
        )
        fig.update_xaxes(
            title_text=x_title, row=2, col=1,
            tickfont=dict(size=9, color="#888780"),
            gridcolor="#D3D1C7",
        )

    fig.update_yaxes(
        tickformat=".1%", row=1, col=1,
        gridcolor="#D3D1C7",
        tickfont=dict(size=9, color="#888780"),
        title_text="Mean return",
    )
    fig.update_xaxes(
        tickfont=dict(size=8 if granularity == "weekly" else 10, color="#888780"),
        gridcolor="#D3D1C7", row=1, col=1,
        tickangle=-45 if granularity == "weekly" else 0,
    )

    n_years = int(np.nanmedian(n_obs))
    _apply_layout(                           # categorical x-axis — NOT date type
        fig,
        title=f"{symbol} — {granularity.capitalize()} seasonality",
        subtitle=f"~{n_years} years of data  ·  period: {_period_label(period)}  ·  "
                 f"green = positive avg  ·  dot colour = win rate",
    )
    fig.update_layout(height=550 if show_cumulative else 380, bargap=0.15)
    return fig


# ---------------------------------------------------------------------------
# 15. Seasonality heatmap — year × week
# ---------------------------------------------------------------------------

def seasonality_heatmap(
    quant: QuantAnalytics,
    symbol: str,
    period: str = "10y",
) -> go.Figure:
    """
    Year × week heatmap of weekly returns.

    Rows   : calendar year
    Columns: ISO week (1–52)
    Colour : weekly return (red = negative, green = positive)

    Supports any "Ny" period string (e.g. "20y") — internally fetches
    "max" and filters to the last N years, same as seasonality_comparison.

    Lets you spot persistent seasonal patterns across individual years
    and identify outlier years that broke the seasonal norm.
    """
    # weekly_seasonality returns (week × year) — transpose to (year × week)
    # handles arbitrary "Ny" periods via "max" fallback in weekly_seasonality
    if period.endswith("y") and period[:-1].isdigit():
        n_years      = int(period[:-1])
        fetch_period = "max" if n_years > 10 else period
    else:
        n_years      = None
        fetch_period = period

    pivot_wk_yr = quant.weekly_seasonality(symbol, period=fetch_period)

    # filter to requested year range
    all_years = sorted(pivot_wk_yr.columns.tolist())
    if n_years is not None:
        all_years = all_years[-n_years:]

    pivot_wk_yr = pivot_wk_yr[all_years]

    # transpose → rows=year, cols=week
    pivot = pivot_wk_yr.T

    # symmetric colour scale around 0
    abs_max = float(np.nanpercentile(np.abs(pivot.values), 95))

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"W{int(c):02d}" for c in pivot.columns],
        y=[str(y) for y in pivot.index],
        zmin=-abs_max,
        zmax=abs_max,
        colorscale=[
            [0.0,  "#A32D2D"],
            [0.35, "#F5C4B3"],
            [0.5,  "#FAFAF9"],
            [0.65, "#9FE1CB"],
            [1.0,  "#0F6E56"],
        ],
        hovertemplate="<b>%{y}  %{x}</b><br>Return: %{z:.2%}<extra></extra>",
        colorbar=dict(
            title="Weekly return",
            thickness=12,
            tickformat=".1%",
            tickfont=dict(size=9, color="#888780"),
            outlinewidth=0,
        ),
    ))

    n_years = len(pivot)
    fig.update_xaxes(
        tickfont=dict(size=8, color="#888780"),
        tickangle=-45,
        title_text="ISO week",
    )
    fig.update_yaxes(
        tickfont=dict(size=10, color="#888780"),
        title_text="Year",
    )
    _apply_layout(
        fig,
        title=f"{symbol} — Weekly return heatmap",
        subtitle=f"{n_years} years  ·  period: {_period_label(period)}  ·  "
                 f"green = positive  ·  red = negative",
    )
    fig.update_layout(height=max(350, n_years * 28 + 120))
    return fig


# ---------------------------------------------------------------------------
# 16. Seasonality comparison — 10y / 5y average vs current year
# ---------------------------------------------------------------------------

def seasonality_comparison(
    quant: QuantAnalytics,
    symbol: str,
    long_term: str = "10y",
    short_term: str = "5y",
    extra_periods: list[str] | None = None,
) -> go.Figure:
    """
    Seasonality comparison with ±1σ band on the long-term average.

    Same logic as seasonality_comparison_clean() but adds a shaded band
    showing the standard deviation of historical cumulative paths around
    the long-term average.

    Parameters
    ----------
    long_term     : longer reference window — gray dotted  (default "10y")
    short_term    : shorter reference window — blue dashed (default "5y")
    extra_periods : additional windows, e.g. ["3y", "2y"]

    Supports any "Ny" period (e.g. "20y", "15y") — fetches "max" data
    and filters to the last N complete years.
    """
    import datetime

    current_year = datetime.date.today().year

    periods_to_plot = [long_term, short_term] + (extra_periods or [])
    seen = set()
    periods_to_plot = [p for p in periods_to_plot
                       if not (p in seen or seen.add(p))]

    # ── helper: per-year cumret pivot ──────────────────────────────────────
    def _get_hist(period: str):
        """Return hist DataFrame (week × year) and n_years used."""
        if period.endswith("y") and period[:-1].isdigit():
            n_years      = int(period[:-1])
            fetch_period = "max" if n_years > 10 else period
        else:
            n_years      = None
            fetch_period = period

        pivot     = quant.weekly_seasonality(symbol, period=fetch_period)
        hist_cols = sorted([c for c in pivot.columns if c != current_year])
        if n_years is not None:
            hist_cols = hist_cols[-n_years:]
        return pivot[hist_cols]

    def _col_cumret(col):
        valid = col.dropna()
        if valid.empty:
            return pd.Series(dtype=float)
        return ((1 + valid).cumprod() - 1).reindex(range(1, 53))

    def _avg_cumret(period: str) -> tuple[pd.Series, pd.Series | None]:
        """Return (mean_cumret, std_cumret) across all historical years."""
        hist   = _get_hist(period)
        if hist.empty:
            return pd.Series(dtype=float), None
        cum_df = hist.apply(_col_cumret)
        avg    = cum_df.mean(axis=1, skipna=True).reindex(range(1, 53))
        std    = cum_df.std(axis=1, skipna=True).reindex(range(1, 53))
        return avg, std

    # ── helper: current year ───────────────────────────────────────────────
    def _current_year_cumret() -> tuple[pd.Series, int]:
        pivot = quant.weekly_seasonality(symbol, period="1y")
        if current_year not in pivot.columns:
            return pd.Series(dtype=float), 0
        cy       = pivot[current_year].dropna()
        if cy.empty:
            return pd.Series(dtype=float), 0
        cumret   = (1 + cy).cumprod() - 1
        return cumret, int(cy.index[-1])

    def _to_xy(s: pd.Series) -> tuple[list, list]:
        s = s.dropna()
        return [0] + s.index.tolist(), [0.0] + s.tolist()

    # ── styles ─────────────────────────────────────────────────────────────
    _extra_colors = ["#BA7517", "#534AB7", "#D85A30", "#A32D2D"]
    _extra_dashes = ["dashdot", "longdash", "dot", "dash"]

    def _style(period, idx):
        if period == long_term:
            return "#B4B2A9", "dot"
        if period == short_term:
            return "#378ADD", "dash"
        ei = (idx - 2) % len(_extra_colors)
        return _extra_colors[ei], _extra_dashes[ei]

    # ── plot ───────────────────────────────────────────────────────────────
    fig      = go.Figure()
    all_y    = [0.0]
    hist_avgs = {}

    for i, period in enumerate(periods_to_plot):
        try:
            avg, std = _avg_cumret(period)
            color, dash = _style(period, i)
            x, y = _to_xy(avg)
            all_y.extend(y)

            # ±1σ band for the long-term period only
            if period == long_term and std is not None:
                upper = (avg + std).reindex(range(1, 53))
                lower = (avg - std).reindex(range(1, 53))
                xu, yu = _to_xy(upper)
                xl, yl = _to_xy(lower)
                fig.add_trace(go.Scatter(
                    x=xu + xl[::-1],
                    y=yu + yl[::-1],
                    fill="toself",
                    fillcolor="rgba(180,178,169,0.12)",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                all_y.extend(yu + yl)

            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode="lines",
                name=f"{period} avg",
                line=dict(color=color, width=2, dash=dash),
                hovertemplate=(
                    f"<b>{period} avg</b><br>"
                    "Week %{x}<br>Cumulative: %{y:.2%}<extra></extra>"
                ),
            ))
            hist_avgs[period] = dict(zip(x, y))
        except Exception as e:
            print(f"Skipping {period}: {e}")

    # current year
    cy_cumret, last_week = _current_year_cumret()
    if not cy_cumret.empty:
        x_cy = [0] + cy_cumret.index.tolist()
        y_cy = [0.0] + cy_cumret.tolist()
        all_y.extend(y_cy)

        fig.add_trace(go.Scatter(
            x=x_cy, y=y_cy,
            mode="lines+markers",
            name=str(current_year),
            line=dict(color="#1D9E75", width=2.5),
            marker=dict(size=5, color="#1D9E75",
                        line=dict(color="white", width=1)),
            hovertemplate=(
                f"<b>{current_year}</b><br>"
                "Week %{x}<br>Cumulative: %{y:.2%}<extra></extra>"
            ),
        ))

        fig.add_vline(
            x=last_week,
            line=dict(color="#D3D1C7", width=1, dash="dot"),
            annotation=dict(text=f"W{last_week:02d}",
                            font=dict(size=9, color="#888780")),
        )

        cy_now    = float(y_cy[-1])
        ann_lines = [f"<b>{current_year} at W{last_week:02d}: {cy_now:+.2%}</b>"]
        for period, avg_dict in hist_avgs.items():
            hist_now = avg_dict.get(last_week,
                       avg_dict.get(max(avg_dict.keys()), 0))
            diff     = cy_now - hist_now
            arrow    = "▲" if diff >= 0 else "▼"
            clr      = "#0F6E56" if diff >= 0 else "#A32D2D"
            ann_lines.append(
                f"vs {period}: <span style=\'color:{clr}\'>{arrow} {diff:+.2%}</span>"
            )
        fig.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.97,
            text="<br>".join(ann_lines),
            showarrow=False, align="left",
            font=dict(size=10, color="#5F5E5A"),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#D3D1C7", borderwidth=0.5, borderpad=8,
        )

    fig.add_hline(y=0, line=dict(color="#D3D1C7", width=0.8, dash="dash"))

    y_pad = 0.01
    y_min = min(all_y) - y_pad
    y_max = max(all_y) + y_pad

    fig.update_xaxes(
        title_text="ISO week",
        tickmode="array",
        tickvals=list(range(0, 53, 4)),
        ticktext=[f"W{w:02d}" if w > 0 else "Start" for w in range(0, 53, 4)],
        tickfont=dict(size=10, color="#888780"),
        gridcolor="#D3D1C7",
        range=[-0.5, 52.5],
    )
    fig.update_yaxes(
        title_text="Cumulative return",
        tickformat=".1%",
        tickfont=dict(size=10, color="#888780"),
        gridcolor="#D3D1C7",
        zerolinecolor="#B4B2A9",
        range=[y_min, y_max],
    )
    _apply_layout(
        fig,
        title=f"{symbol} — Seasonality",
        subtitle=(
            f"{current_year} vs {' / '.join(periods_to_plot)} historical averages  ·  "
            f"shaded band = {long_term} ±1σ"
        ),
    )
    fig.update_layout(
        height=460,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
    )
    return fig

def seasonality_comparison_clean(
    quant: QuantAnalytics,
    symbol: str,
    long_term: str = "10y",
    short_term: str = "5y",
    extra_periods: list[str] | None = None,
) -> go.Figure:
    """
    Seasonality comparison — current year vs historical averages.

    Logic (clean, from scratch)
    ---------------------------
    For each period (long_term, short_term, extras):
      1. Get the pivot table  (rows=week, cols=year)
      2. Drop the current year column (partial data would bias the average)
      3. For each remaining year-column compute cumulative return:
             (1+r_w1) * (1+r_w2) * ... - 1
      4. Average those per-year cumulative curves across all years
         → "average seasonal year" for that period
    Then plot the current year column as the live actual line.

    Parameters
    ----------
    long_term     : longer reference window — gray dotted  (default "10y")
    short_term    : shorter reference window — blue dashed (default "5y")
    extra_periods : additional windows, e.g. ["3y", "2y"]

    Examples
    --------
    plots.seasonality_comparison_clean(quant, "SPY").show()
    plots.seasonality_comparison_clean(quant, "SPY",
        long_term="20y", short_term="5y").show()
    """
    import datetime

    current_year = datetime.date.today().year

    # deduplicated list of periods to plot
    periods_to_plot = [long_term, short_term] + (extra_periods or [])
    seen = set()
    periods_to_plot = [p for p in periods_to_plot
                       if not (p in seen or seen.add(p))]

    # ── helper: average cumulative return curve for a period ─────────────
    def _avg_cumret(period: str) -> pd.Series:
        """
        Average cumulative return across all historical years in the period.

        Handles any "Ny" period string (e.g. "20y", "15y") by fetching
        "max" data and filtering to the last N complete years — since yfinance
        only supports up to "10y" as a valid period string.

        Steps:
          1. Parse period → number of years (N)
          2. Fetch "max" data if N > 10, else use period directly
          3. Filter pivot to last N years, exclude current year
          4. Drop any year with < 40 weeks of data (incomplete first year)
          5. Per year-column: cumulative return = (1+r1)(1+r2)... - 1
          6. Mean across year-columns → average seasonal curve
        """
        # parse period string → year count
        if period.endswith("y") and period[:-1].isdigit():
            n_years     = int(period[:-1])
            fetch_period = "max" if n_years > 10 else period
        else:
            n_years      = None
            fetch_period = period

        pivot     = quant.weekly_seasonality(symbol, period=fetch_period)

        # exclude current year, then take the last n_years complete years
        hist_cols = sorted([c for c in pivot.columns if c != current_year])
        if n_years is not None:
            hist_cols = hist_cols[-n_years:]

        if not hist_cols:
            return pd.Series(dtype=float)

        hist = pivot[hist_cols]
        # no week-count filter needed — mean(skipna=True) already handles
        # partial years gracefully (NaN weeks just don't contribute to the mean)

        # cumulative return per year-column, reindexed to full 52 weeks
        def _col_cumret(col):
            valid = col.dropna()
            if valid.empty:
                return pd.Series(dtype=float)
            return ((1 + valid).cumprod() - 1).reindex(range(1, 53))

        cum_df = hist.apply(_col_cumret)
        avg    = cum_df.mean(axis=1, skipna=True)
        return avg.reindex(range(1, 53))

    # ── helper: current year actual cumulative return ─────────────────────
    def _current_year_cumret() -> tuple[pd.Series, int]:
        pivot = quant.weekly_seasonality(symbol, period="1y")
        if current_year not in pivot.columns:
            return pd.Series(dtype=float), 0
        cy    = pivot[current_year].dropna()
        if cy.empty:
            return pd.Series(dtype=float), 0
        cumret   = (1 + cy).cumprod() - 1
        last_week = int(cy.index[-1])
        return cumret, last_week

    # ── build x-axis: prepend week 0 = 0% starting point ─────────────────
    def _to_xy(avg: pd.Series) -> tuple[list, list]:
        """Prepend (0, 0.0) so every line starts at the origin."""
        x = [0] + avg.dropna().index.tolist()
        y = [0.0] + avg.dropna().tolist()
        return x, y

    # ── styles ─────────────────────────────────────────────────────────────
    _extra_colors = ["#BA7517", "#534AB7", "#D85A30", "#A32D2D"]
    _extra_dashes = ["dashdot", "longdash", "dot", "dash"]

    def _style(period, idx):
        if period == long_term:
            return dict(color="#888780", width=1.8, dash="dot")
        if period == short_term:
            return dict(color="#378ADD", width=1.8, dash="dash")
        ei = (idx - 2) % len(_extra_colors)
        return dict(color=_extra_colors[ei], width=1.5, dash=_extra_dashes[ei])

    # ── plot ───────────────────────────────────────────────────────────────
    fig      = go.Figure()
    all_y    = [0.0]
    hist_avgs = {}

    for i, period in enumerate(periods_to_plot):
        try:
            avg = _avg_cumret(period)
            x, y = _to_xy(avg)
            all_y.extend(y)
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode="lines",
                name=f"{period} avg",
                line=_style(period, i),
                hovertemplate=(
                    f"<b>{period} avg</b><br>"
                    "Week %{x}<br>Cumulative: %{y:.2%}<extra></extra>"
                ),
            ))
            hist_avgs[period] = dict(zip(x, y))
        except Exception as e:
            print(f"Skipping {period}: {e}")

    # current year
    cy_cumret, last_week = _current_year_cumret()
    if not cy_cumret.empty:
        x_cy = [0] + cy_cumret.index.tolist()
        y_cy = [0.0] + cy_cumret.tolist()
        all_y.extend(y_cy)

        fig.add_trace(go.Scatter(
            x=x_cy, y=y_cy,
            mode="lines+markers",
            name=str(current_year),
            line=dict(color="#1D9E75", width=2.5),
            marker=dict(size=5, color="#1D9E75",
                        line=dict(color="white", width=1)),
            hovertemplate=(
                f"<b>{current_year}</b><br>"
                "Week %{x}<br>Cumulative: %{y:.2%}<extra></extra>"
            ),
        ))

        fig.add_vline(
            x=last_week,
            line=dict(color="#D3D1C7", width=1, dash="dot"),
            annotation=dict(text=f"W{last_week:02d}",
                            font=dict(size=9, color="#888780")),
        )

        # annotation box: how is current year doing vs each period?
        cy_now    = float(y_cy[-1])
        ann_lines = [f"<b>{current_year} at W{last_week:02d}: {cy_now:+.2%}</b>"]
        for period, avg_dict in hist_avgs.items():
            hist_now = avg_dict.get(last_week, avg_dict.get(max(avg_dict.keys())))
            diff     = cy_now - hist_now
            arrow    = "▲" if diff >= 0 else "▼"
            clr      = "#0F6E56" if diff >= 0 else "#A32D2D"
            ann_lines.append(
                f"vs {period}: <span style=\'color:{clr}\'>{arrow} {diff:+.2%}</span>"
            )
        fig.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.99,
            text="<br>".join(ann_lines),
            showarrow=False, align="left",
            font=dict(size=10, color="#5F5E5A"),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#D3D1C7", borderwidth=0.5, borderpad=8,
        )

    fig.add_hline(y=0, line=dict(color="#D3D1C7", width=0.8, dash="dash"))

    y_pad = 0.01
    y_min = min(all_y) - y_pad
    y_max = max(all_y) + y_pad

    fig.update_xaxes(
        title_text="ISO week",
        tickmode="array",
        tickvals=list(range(0, 53, 4)),
        ticktext=[f"W{w:02d}" if w > 0 else "Start" for w in range(0, 53, 4)],
        tickfont=dict(size=10, color="#888780"),
        gridcolor="#D3D1C7",
        range=[-0.5, 52.5],
    )
    fig.update_yaxes(
        title_text="Cumulative return",
        tickformat=".1%",
        tickfont=dict(size=10, color="#888780"),
        gridcolor="#D3D1C7",
        zerolinecolor="#B4B2A9",
        range=[y_min, y_max],
    )
    _apply_layout(
        fig,
        title=f"{symbol} — Seasonality",
        subtitle=f"{current_year} vs {' / '.join(periods_to_plot)} historical averages",
    )
    fig.update_layout(
        height=440,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
    )
    return fig


# ---------------------------------------------------------------------------
# 18. Factor exposure — loadings bar + rolling betas
# ---------------------------------------------------------------------------

def factor_exposure(result, show_significance: bool = True) -> go.Figure:
    """
    Two-panel factor exposure chart for a single FactorResult.

    Top panel   : factor beta loadings — bar per factor, coloured by sign,
                  error bars showing ±1 t-stat unit, significance stars
    Bottom panel: alpha and R² summary cards as annotations

    result             : FactorResult from factors.run()
    show_significance  : annotate bars with * / ** / *** significance stars
    """
    factors  = list(result.betas.keys())
    betas    = [result.betas[f] for f in factors]
    t_stats  = [result.t_stats[f] for f in factors]
    p_values = [result.p_values[f] for f in factors]
    labels   = [_FACTOR_LABELS.get(f, f) for f in factors]

    colors  = ["#1D9E75" if b >= 0 else "#E24B4A" for b in betas]
    opacity = [0.9 if p < 0.05 else 0.45 for p in p_values]

    def _sig(p):
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.72, 0.28],
        subplot_titles=["Factor loadings (beta)", "Model summary"],
        horizontal_spacing=0.08,
    )

    # --- beta bars -------------------------------------------------------
    for i, (f, b, t, p, lbl, clr, op) in enumerate(
        zip(factors, betas, t_stats, p_values, labels, colors, opacity)
    ):
        # error bar: ±se = ±|beta/t|
        se = abs(b / t) if t != 0 else 0
        sig_label = f"  {_sig(p)}" if show_significance else ""

        fig.add_trace(go.Bar(
            x=[lbl],
            y=[b],
            marker_color=clr,
            marker_opacity=op,
            error_y=dict(type="data", array=[se], visible=True,
                         color="#888780", thickness=1.5, width=6),
            name=f,
            showlegend=False,
            customdata=[[f, b, t, p, _sig(p)]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Beta: %{customdata[1]:.3f}<br>"
                "t-stat: %{customdata[2]:.2f}<br>"
                "p-value: %{customdata[3]:.3f}  %{customdata[4]}"
                "<extra></extra>"
            ),
            text=[f"{b:.2f}{sig_label}"],
            textposition="outside",
            textfont=dict(size=10, color=clr),
        ), row=1, col=1)

    fig.add_hline(y=0, row=1, col=1,
                  line=dict(color="#B4B2A9", width=0.8, dash="dash"))

    # --- summary panel (right col as annotations) ------------------------
    sig_alpha = _sig(result.alpha_pval)
    summary_items = [
        ("Alpha (ann.)",  f"{result.alpha:+.2%} {sig_alpha}"),
        ("Alpha p-value", f"{result.alpha_pval:.3f}"),
        ("R²",            f"{result.r_squared:.3f}"),
        ("Adj R²",        f"{result.adj_r2:.3f}"),
        ("Observations",  f"{result.n_obs:,}"),
        ("Model",         result.model.upper()),
        ("Period",        result.period),
    ]
    y_pos = 0.95
    for label, value in summary_items:
        color = ("#0F6E56" if "+" in str(value) and "Alpha" in label
                 else "#A32D2D" if "-" in str(value) and "Alpha" in label
                 else "#2C2C2A")
        fig.add_annotation(
            xref="x2 domain", yref="paper",
            x=0.05, y=y_pos,
            text=f"<b style='color:#888780;font-size:10px'>{label}</b><br>"
                 f"<span style='font-size:13px;color:{color}'>{value}</span>",
            showarrow=False, align="left",
            font=dict(size=11),
        )
        y_pos -= 0.13

    # legend for opacity
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=-0.12,
        text="<span style='color:#888780;font-size:9px'>"
             "Solid = significant (p<0.05)  ·  Faded = not significant  ·  "
             "Error bars = ±1 SE  ·  * p<0.1  ** p<0.05  *** p<0.01</span>",
        showarrow=False, align="left",
    )

    fig.update_yaxes(title_text="Beta", row=1, col=1,
                     gridcolor="#D3D1C7",
                     tickfont=dict(size=10, color="#888780"))
    fig.update_xaxes(tickfont=dict(size=10, color="#5F5E5A"),
                     row=1, col=1)

    _apply_layout(
        fig,
        title=f"{result.symbol} — Factor exposure [{result.model.upper()}]",
        subtitle=f"period: {_period_label(result.period)}",
    )
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)
    fig.update_layout(height=420, margin=dict(b=60))
    return fig


def rolling_factor_betas(
    quant,
    symbol: str,
    model: str = "ff3",
    period: str = "5y",
    window: int = 126,
) -> go.Figure:
    """
    Rolling factor betas over time — shows how exposures shift across regimes.

    One line per factor. A horizontal dashed line at 0 and at 1 (market beta).
    """
    from yfinance_api3.modules.factors import rolling_betas, _FACTOR_LABELS, _MODEL_FACTORS

    df      = rolling_betas(quant, symbol, model=model, period=period, window=window)
    factors = _MODEL_FACTORS[model]
    fig     = go.Figure()

    fig.add_hline(y=0, line=dict(color="#D3D1C7", width=0.8, dash="dash"))
    fig.add_hline(y=1, line=dict(color="#D3D1C7", width=0.6, dash="dot"),
                  annotation=dict(text="β=1", font=dict(size=9, color="#888780")))

    for factor, color in zip(factors, _PALETTE):
        if factor not in df.columns:
            continue
        label = _FACTOR_LABELS.get(factor, factor)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[factor],
            mode="lines",
            name=label,
            line=dict(color=color, width=1.8),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "%{x|%Y-%m-%d}<br>"
                "Beta: %{y:.3f}<extra></extra>"
            ),
        ))

    fig.update_xaxes(title_text="Date", gridcolor="#D3D1C7",
                     tickfont=dict(size=10, color="#888780"))
    fig.update_yaxes(title_text="Beta", gridcolor="#D3D1C7",
                     tickfont=dict(size=10, color="#888780"),
                     zerolinecolor="#B4B2A9")

    _apply_layout(
        fig,
        title=f"{symbol} — Rolling factor betas [{model.upper()}]",
        subtitle=f"{window}-day window  ·  period: {_period_label(period)}",
    )
    fig.update_layout(height=420, hovermode="x unified")
    return fig


def factor_comparison(
    results: list,
) -> go.Figure:
    """
    Side-by-side beta comparison across multiple symbols or models.

    results : list of FactorResult — all must use the same model.

    One grouped bar per factor, one bar per symbol within each group.
    Useful for spotting which stocks drive which factor exposures.
    """
    factors = list(results[0].betas.keys())
    try:
        from yfinance_api3.modules.factors import _FACTOR_LABELS
        factor_labels = [_FACTOR_LABELS.get(f, f) for f in factors]
    except ImportError:
        factor_labels = factors

    fig = go.Figure()

    for result, color in zip(results, _PALETTE):
        betas    = [result.betas[f] for f in factors]

        fig.add_trace(go.Bar(
            name=result.symbol,
            x=factor_labels,
            y=betas,
            marker_color=color,
            marker_opacity=0.85,
            hovertemplate=(
                f"<b>{result.symbol}</b><br>"
                "%{x}<br>Beta: %{y:.3f}<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line=dict(color="#B4B2A9", width=0.8, dash="dash"))

    fig.update_xaxes(tickfont=dict(size=10, color="#5F5E5A"),
                     gridcolor="#D3D1C7")
    fig.update_yaxes(title_text="Beta", gridcolor="#D3D1C7",
                     tickfont=dict(size=10, color="#888780"),
                     zerolinecolor="#B4B2A9")

    model = results[0].model.upper()
    _apply_layout(
        fig,
        title=f"Factor exposure comparison [{model}]",
        subtitle=f"{len(results)} symbols  ·  period: {_period_label(results[0].period)}",
    )
    fig.update_layout(barmode="group", height=420)
    return fig


# ---------------------------------------------------------------------------
# Kelly Criterion — position sizing chart
# ---------------------------------------------------------------------------

def kelly(
    quant: QuantAnalytics,
    symbols: list[str],
    period: str = "2y",
    fractional: float = 0.5,
    risk_free_rate: float = 0.05,
) -> go.Figure:
    """
    Kelly Criterion visualisation — three panels for a basket of stocks.

    Panel 1 (bar): Full Kelly vs Fractional Kelly per symbol.
                   Bars above 0 = positive edge. Red = no edge.
    Panel 2 (scatter): Kelly fraction vs Sharpe ratio — shows the
                       relationship between edge quality and sizing.
    Panel 3 (bar): Suggested max allocation (capped at 25%) as a
                   practical position size guide.

    Reading the chart
    -----------------
    • Tall green bar → strong historical edge, larger Kelly fraction
    • Red bar        → no edge over risk-free rate — Kelly says avoid
    • High Kelly + high Sharpe → genuinely good risk-adjusted opportunity
    • High Kelly + low Sharpe  → high return but also high vol — be cautious
    • Suggested max is min(half_kelly, 25%) — a conservative real-world cap
    """
    df = quant.kelly_bulk(symbols, period=period, fractional=fractional,
                          risk_free_rate=risk_free_rate)

    syms        = list(df.index)
    full_kelly  = df["full_kelly"].astype(float).tolist()
    frac_kelly  = df["fractional_kelly"].astype(float).tolist()
    sharpe      = df["sharpe_ratio"].astype(float).tolist()
    suggested   = df["suggested_max"].astype(float).tolist()
    has_edge    = df["has_edge"].tolist()
    mu          = df["mu_annual"].astype(float).tolist()
    sigma       = df["sigma_annual"].astype(float).tolist()

    colors = ["#1D9E75" if e else "#E24B4A" for e in has_edge]
    frac_label = {1.0: "Full", 0.5: "Half", 0.25: "Quarter"}.get(fractional,
                  f"{fractional:.0%}")

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.42, 0.30, 0.28],
        vertical_spacing=0.08,
        subplot_titles=[
            f"Full Kelly vs {frac_label} Kelly fraction",
            "Kelly fraction vs Sharpe ratio",
            "Suggested max allocation (capped at 25%)",
        ],
    )

    # ── Panel 1: Full Kelly bars + Fractional Kelly overlay ──────────────
    fig.add_trace(go.Bar(
        x=syms, y=full_kelly,
        name="Full Kelly",
        marker_color=[c.replace(")", ",0.35)").replace("rgb", "rgba")
                      if c.startswith("rgb") else c
                      for c in colors],
        marker_opacity=0.45,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Full Kelly: %{y:.1%}<extra>Full Kelly</extra>"
        ),
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=syms, y=frac_kelly,
        name=f"{frac_label} Kelly",
        marker_color=colors,
        customdata=list(zip(mu, sigma, sharpe)),
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"{frac_label} Kelly: %{{y:.1%}}<br>"
            "μ (ann.): %{customdata[0]:.1%}<br>"
            "σ (ann.): %{customdata[1]:.1%}<br>"
            "Sharpe: %{customdata[2]:.2f}"
            "<extra></extra>"
        ),
    ), row=1, col=1)

    fig.add_hline(y=0, row=1, col=1,
                  line=dict(color="#B4B2A9", width=0.8, dash="dash"))
    fig.add_hline(y=0.25, row=1, col=1,
                  line=dict(color="#BA7517", width=0.8, dash="dot"),
                  annotation=dict(text="25% cap", font=dict(size=9,
                  color="#BA7517")))

    fig.update_yaxes(tickformat=".0%", row=1, col=1,
                     gridcolor="#D3D1C7",
                     tickfont=dict(size=9, color="#888780"))

    # ── Panel 2: Kelly vs Sharpe scatter ─────────────────────────────────
    for sym, fk, sr, clr, has in zip(syms, frac_kelly, sharpe, colors, has_edge):
        fig.add_trace(go.Scatter(
            x=[sr], y=[fk],
            mode="markers+text",
            marker=dict(size=12, color=clr,
                        line=dict(color="white", width=1)),
            text=[sym], textposition="top right",
            textfont=dict(size=9, color=clr),
            showlegend=False,
            hovertemplate=(
                f"<b>{sym}</b><br>"
                f"Sharpe: {sr:.2f}<br>"
                f"{frac_label} Kelly: {fk:.1%}<extra></extra>"
            ),
        ), row=2, col=1)

    fig.add_hline(y=0, row=2, col=1,
                  line=dict(color="#B4B2A9", width=0.8, dash="dash"))
    fig.add_vline(x=0, row=2, col=1,
                  line=dict(color="#B4B2A9", width=0.8, dash="dash"))

    fig.update_xaxes(title_text="Sharpe ratio", row=2, col=1,
                     gridcolor="#D3D1C7",
                     tickfont=dict(size=9, color="#888780"))
    fig.update_yaxes(title_text=f"{frac_label} Kelly",
                     tickformat=".0%", row=2, col=1,
                     gridcolor="#D3D1C7",
                     tickfont=dict(size=9, color="#888780"))

    # ── Panel 3: Suggested max allocation ────────────────────────────────
    fig.add_trace(go.Bar(
        x=syms, y=suggested,
        name="Suggested max",
        marker_color=colors,
        text=[f"{v:.0%}" for v in suggested],
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Suggested max: %{y:.1%}"
            "<extra></extra>"
        ),
    ), row=3, col=1)

    fig.add_hline(y=0.25, row=3, col=1,
                  line=dict(color="#BA7517", width=0.8, dash="dot"))

    fig.update_yaxes(tickformat=".0%", row=3, col=1,
                     gridcolor="#D3D1C7",
                     tickfont=dict(size=9, color="#888780"),
                     range=[0, 0.30])

    # ── layout ───────────────────────────────────────────────────────────
    _apply_layout(
        fig,
        title="Kelly Criterion — position sizing analysis",
        subtitle=(
            f"period: {_period_label(period)}  ·  "
            f"{frac_label} Kelly shown  ·  "
            f"rf: {risk_free_rate:.0%}  ·  "
            "green = positive edge  ·  red = no edge"
        ),
    )
    fig.update_layout(
        height=700,
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=11)),
    )
    return fig


# ---------------------------------------------------------------------------
# Seasonality box plot — full return distribution per period
# ---------------------------------------------------------------------------

def seasonality_box(
    quant: QuantAnalytics,
    symbol: str,
    period: str = "10y",
    granularity: str = "monthly",    # "monthly" | "weekly"
    show_points: bool = True,        # overlay individual year dots
    show_mean: bool = True,          # mark the mean with a diamond
) -> go.Figure:
    """
    Box plot showing the full return distribution per month (or week).

    More informative than a bar chart — reveals:
      - Median vs mean (skew)
      - Interquartile range (typical outcome spread)
      - Outlier years (the dots beyond the whiskers)
      - Which months have consistent direction vs high variance

    Parameters
    ----------
    granularity  : "monthly" (12 boxes) or "weekly" (52 boxes)
    show_points  : overlay individual year returns as dots (default True)
    show_mean    : add a diamond marker at the mean (default True)

    Hover on any dot shows the exact year and return.
    """
    if granularity == "monthly":
        pivot   = quant.monthly_seasonality(symbol, period=period)
        x_title = "Month"
    else:
        pivot   = quant.weekly_seasonality(symbol, period=period)
        pivot.index = [f"W{int(w):02d}" for w in pivot.index]
        x_title = "ISO week"

    stats  = quant.seasonality_stats(pivot)
    labels = list(pivot.index)

    # colour each box by mean sign
    box_colors = [
        "#1D9E75" if stats.loc[lbl, "mean"] >= 0 else "#E24B4A"
        for lbl in labels
    ]

    fig = go.Figure()

    # --- one box per period -----------------------------------------------
    for lbl, color in zip(labels, box_colors):
        row_data = pivot.loc[lbl].dropna()
        years    = row_data.index.tolist()
        returns  = row_data.values.tolist()

        fig.add_trace(go.Box(
            y=returns,
            name=lbl,
            marker_color=color,
            line_color=color,
            fillcolor=color.replace("#", "rgba(").replace(
                "1D9E75", "29,158,117,0.25)"
            ).replace(
                "E24B4A", "226,75,74,0.25)"
            ) if color in ("#1D9E75", "#E24B4A") else "rgba(180,178,169,0.25)",
            boxmean=False,
            showlegend=False,
            hovertemplate=(
                f"<b>{lbl}</b><br>"
                "Return: %{y:.2%}<extra></extra>"
            ),
        ))

        # individual year dots
        if show_points:
            fig.add_trace(go.Scatter(
                x=[lbl] * len(years),
                y=returns,
                mode="markers",
                marker=dict(
                    size=7,
                    color=color,
                    opacity=0.75,
                    line=dict(color="white", width=0.8),
                ),
                showlegend=False,
                customdata=[[yr, ret] for yr, ret in zip(years, returns)],
                hovertemplate=(
                    f"<b>{lbl}</b><br>"
                    "Year: %{customdata[0]}<br>"
                    "Return: %{customdata[1]:.2%}<extra></extra>"
                ),
            ))

        # mean diamond
        if show_mean:
            mean_val = float(stats.loc[lbl, "mean"])
            fig.add_trace(go.Scatter(
                x=[lbl],
                y=[mean_val],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=9,
                    color="white",
                    line=dict(color=color, width=1.5),
                ),
                showlegend=False,
                hovertemplate=(
                    f"<b>{lbl} mean</b><br>"
                    f"Return: {mean_val:.2%}<extra></extra>"
                ),
            ))

    # zero line
    fig.add_hline(y=0, line=dict(color="#B4B2A9", width=0.8, dash="dash"))

    fig.update_yaxes(
        title_text="Return",
        tickformat=".1%",
        gridcolor="#D3D1C7",
        tickfont=dict(size=10, color="#888780"),
        zerolinecolor="#B4B2A9",
    )
    fig.update_xaxes(
        title_text=x_title,
        tickfont=dict(size=10 if granularity == "monthly" else 8,
                      color="#888780"),
        tickangle=-45 if granularity == "weekly" else 0,
        gridcolor="#D3D1C7",
    )

    n_years = int(stats["n_obs"].median())
    _apply_layout(
        fig,
        title=f"{symbol} — {granularity.capitalize()} return distribution",
        subtitle=(
            f"~{n_years} years  ·  period: {_period_label(period)}  ·  "
            "box = IQR  ·  whiskers = 1.5×IQR  ·  ◆ = mean  ·  "
            "green = positive avg  ·  red = negative avg"
        ),
    )
    fig.update_layout(
        height=520,
        boxmode="group",
        boxgap=0.3,
        boxgroupgap=0.1,
    )
    return fig


# ---------------------------------------------------------------------------
# Rolling holding period returns
# ---------------------------------------------------------------------------

def rolling_returns(
    quant: QuantAnalytics,
    symbol: str,
    hold_years: float = 5.0,
    period: str = "20y",
    compare_symbols: list[str] | None = None,
) -> go.Figure:
    """
    Distribution and timeline of all N-year holding period returns.

    Answers: "if I bought on any day in the last *period* and held for
    *hold_years* years, what return would I have gotten?"

    Three panels
    ------------
    Top    : timeline of returns by entry date — shows which eras were
             great/terrible entry points. Colour = return magnitude.
    Middle : histogram + KDE of all outcomes — full distribution shape.
    Bottom : cumulative distribution — probability of achieving at least X%.

    Parameters
    ----------
    hold_years       : holding period in years (default 5)
    period           : lookback window — must be > hold_years (default "20y")
    compare_symbols  : optional list to overlay on the histogram panel
    """
    from scipy.stats import gaussian_kde

    stats  = quant.rolling_returns_stats(symbol, hold_years, period)
    rets   = quant.rolling_returns(symbol, hold_years, period)

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.38, 0.37, 0.25],
        vertical_spacing=0.07,
        subplot_titles=[
            f"Return by entry date (hold {hold_years:.0f}y)",
            "Distribution of all outcomes",
            "Cumulative probability",
        ],
    )

    # ── Panel 1: timeline coloured by return ──────────────────────────────
    fig.add_trace(go.Scatter(
        x=rets.index,
        y=rets.values,
        mode="lines",
        line=dict(width=0.8, color="rgba(55,138,221,0.3)"),
        showlegend=False,
        hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:.1%}<extra></extra>",
    ), row=1, col=1)

    # colour-coded scatter overlay
    fig.add_trace(go.Scatter(
        x=rets.index,
        y=rets.values,
        mode="markers",
        marker=dict(
            size=2,
            color=rets.values,
            colorscale=[[0, "#A32D2D"], [0.4, "#F4F3EF"],
                        [0.6, "#F4F3EF"], [1, "#0F6E56"]],
            cmin=-0.5, cmax=1.5,
            showscale=True,
            colorbar=dict(
                title="Return",
                thickness=10, len=0.35, y=0.82,
                tickformat=".0%",
                tickfont=dict(size=8, color="#888780"),
                outlinewidth=0,
            ),
        ),
        showlegend=False,
        hoverinfo="skip",
    ), row=1, col=1)

    fig.add_hline(y=0, row=1, col=1,
                  line=dict(color="#B4B2A9", width=0.8, dash="dash"))
    fig.update_yaxes(tickformat=".0%", row=1, col=1,
                     gridcolor="#D3D1C7",
                     tickfont=dict(size=9, color="#888780"))
    fig.update_xaxes(gridcolor="#D3D1C7", row=1, col=1,
                     tickfont=dict(size=9, color="#888780"))

    # ── Panel 2: histogram + KDE ──────────────────────────────────────────
    all_series = {symbol: rets}
    if compare_symbols:
        for sym in compare_symbols:
            try:
                all_series[sym] = quant.rolling_returns(sym, hold_years, period)
            except Exception:
                pass

    colors_hist = [_PALETTE[i] for i in range(len(all_series))]

    for (sym, r), color in zip(all_series.items(), colors_hist):
        fig.add_trace(go.Histogram(
            x=r.values,
            histnorm="probability density",
            nbinsx=60,
            marker_color=color,
            opacity=0.3,
            name=sym,
            hovertemplate=f"<b>{sym}</b><br>Return: %{{x:.1%}}<br>Density: %{{y:.2f}}<extra></extra>",
        ), row=2, col=1)

        kde_x = np.linspace(r.min(), r.max(), 300)
        kde_y = gaussian_kde(r.values)(kde_x)
        fig.add_trace(go.Scatter(
            x=kde_x, y=kde_y,
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo="skip",
        ), row=2, col=1)

    fig.add_vline(x=0, row=2, col=1,
                  line=dict(color="#B4B2A9", width=0.8, dash="dash"))
    fig.add_vline(x=stats["mean_return"], row=2, col=1,
                  line=dict(color=_PALETTE[0], width=1.5),
                  annotation=dict(
                      text=f"mean {stats['mean_return']:+.0%}",
                      font=dict(size=9, color=_PALETTE[0])))
    fig.add_vline(x=stats["median_return"], row=2, col=1,
                  line=dict(color=_PALETTE[0], width=1.5, dash="dot"),
                  annotation=dict(
                      text=f"median {stats['median_return']:+.0%}",
                      font=dict(size=9, color=_PALETTE[0]),
                      yshift=-20))

    fig.update_xaxes(tickformat=".0%", row=2, col=1,
                     gridcolor="#D3D1C7",
                     tickfont=dict(size=9, color="#888780"))
    fig.update_yaxes(title_text="Density", row=2, col=1,
                     gridcolor="#D3D1C7",
                     tickfont=dict(size=9, color="#888780"))
    fig.update_layout(barmode="overlay")

    # ── Panel 3: cumulative distribution ──────────────────────────────────
    sorted_rets = np.sort(rets.values)
    cum_prob    = np.arange(1, len(sorted_rets) + 1) / len(sorted_rets)

    fig.add_trace(go.Scatter(
        x=sorted_rets, y=cum_prob,
        mode="lines",
        line=dict(color=_PALETTE[0], width=2),
        showlegend=False,
        hovertemplate="Return ≥ %{x:.1%}<br>Probability: %{y:.1%}<extra></extra>",
    ), row=3, col=1)

    # mark key thresholds
    for threshold, label in [(0, "break-even"), (0.5, "+50%"), (1.0, "+100% (2x)")]:
        p = float((rets >= threshold).mean())
        fig.add_vline(x=threshold, row=3, col=1,
                      line=dict(color="#D3D1C7", width=0.8, dash="dot"))
        fig.add_annotation(
            x=threshold, y=1.05, xref="x3", yref="y3",
            text=f"{p:.0%}", showarrow=False,
            font=dict(size=8, color="#888780"),
        )

    fig.update_xaxes(tickformat=".0%", row=3, col=1,
                     title_text=f"{hold_years:.0f}-year total return",
                     gridcolor="#D3D1C7",
                     tickfont=dict(size=9, color="#888780"))
    fig.update_yaxes(tickformat=".0%", row=3, col=1,
                     title_text="P(return ≤ x)",
                     gridcolor="#D3D1C7",
                     tickfont=dict(size=9, color="#888780"))

    # ── annotation box ────────────────────────────────────────────────────
    m = stats
    ann = (
        f"<b>{symbol} — {hold_years:.0f}y hold  ({m['n_entries']:,} entry dates)</b><br>"
        f"Mean: {m['mean_return']:+.1%}   CAGR: {m['cagr_mean']:+.1%}/yr<br>"
        f"Median: {m['median_return']:+.1%}   Std: {m['std_return']:.1%}<br>"
        f"Win rate: {m['win_rate']:.0%}   "
        f"Prob 2x: {m['prob_double']:.0%}   "
        f"Prob halve: {m['prob_halve']:.0%}<br>"
        f"Best entry: {m['best_entry']} ({m['max_return']:+.0%})<br>"
        f"Worst entry: {m['worst_entry']} ({m['min_return']:+.0%})"
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.995,
        text=ann, showarrow=False, align="left",
        font=dict(size=10, color="#5F5E5A"),
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="#D3D1C7", borderwidth=0.5, borderpad=8,
    )

    _apply_layout(
        fig,
        title=f"{symbol} — Rolling {hold_years:.0f}-year holding period returns",
        subtitle=f"every possible entry date in the last {_period_label(period)}  ·  "
                 f"{m['n_entries']:,} scenarios",
    )
    fig.update_layout(height=750, hovermode="x unified")
    return fig


# ---------------------------------------------------------------------------
# S&P 500 concentration
# ---------------------------------------------------------------------------

def sp500_concentration(sp, top_n: int = 50) -> go.Figure:
    """
    Four-panel ETF concentration analysis — one panel per row.

    Row 1 : Top N holdings bar chart  (colour-coded by sector)
    Row 2 : Cumulative weight curve   (how many stocks = X% of ETF)
    Row 3 : Sector weights bar chart
    Row 4 : Concentration metrics summary

    sp    : ETFConcentration instance (already fetched)
    top_n : number of holdings to show in the top bar (default 50)
    """
    from yfinance_api3.modules.etf import ETFConcentration

    df      = sp.weights()
    top_df  = sp.top_n(top_n)
    sectors = sp.sector_weights()
    cumw    = sp.cumulative_weight()
    metrics = sp.concentration_metrics()
    m       = metrics

    # sector colour palette
    unique_sectors = df["sector"].dropna().unique().tolist()
    sector_colors  = {s: _PALETTE[i % len(_PALETTE)]
                      for i, s in enumerate(sorted(unique_sectors))}

    fig = make_subplots(
        rows=4, cols=1,
        row_heights=[0.38, 0.20, 0.20, 0.22],
        vertical_spacing=0.08,
        specs=[
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "xy"}],
            [{"type": "table"}],
        ],
        subplot_titles=[
            f"Top {top_n} holdings by weight",
            "Cumulative weight — how concentrated is the index?",
            "Sector allocation",
            "Concentration metrics",
        ],
    )

    # ── Row 1: top-N bar chart ────────────────────────────────────────────
    bar_colors = [sector_colors.get(s, "#888780")
                  for s in top_df["sector"].fillna("Unknown")]

    # weight labels only on top 10 where bars are tall enough
    top10_mask = list(range(len(top_df)))
    bar_text   = [f"{v:.1f}%" if i < 10 else ""
                  for i, v in enumerate(top_df["weight_pct"])]

    fig.add_trace(go.Bar(
        x=top_df["symbol"],
        y=top_df["weight_pct"],
        marker_color=bar_colors,
        marker_line_width=0,
        showlegend=False,
        text=bar_text,
        textposition="outside",
        textfont=dict(size=10, color="#5F5E5A"),
        customdata=top_df[["name", "sector"]].values,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "%{customdata[0]}<br>"
            "Sector: %{customdata[1]}<br>"
            "Weight: %{y:.2f}%<extra></extra>"
        ),
    ), row=1, col=1)

    fig.update_xaxes(
        tickfont=dict(size=9, color="#2C2C2A", family="monospace"),
        tickangle=-90,
        row=1, col=1,
        gridcolor="#D3D1C7",
    )
    fig.update_yaxes(
        title_text="Weight (%)",
        title_font=dict(size=12),
        row=1, col=1,
        gridcolor="#D3D1C7",
        tickfont=dict(size=11, color="#888780"),
    )

    # sector legend — use Scatter with square marker instead of Bar
    # Bar with x=[None] pollutes the categorical x-axis in some Plotly versions
    for sector, color in sorted(sector_colors.items()):
        if sector in df["sector"].values:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=12, color=color, symbol="square"),
                name=sector,
                showlegend=True,
            ), row=1, col=1)

    # ── Row 2: cumulative weight curve ────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cumw) + 1)),
        y=cumw.values * 100,
        mode="lines",
        line=dict(color=_PALETTE[0], width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba(55,138,221,0.08)",
        showlegend=False,
        hovertemplate="Top %{x} holdings<br>Cumulative: %{y:.1f}%<extra></extra>",
    ), row=2, col=1)

    for pct in [25, 50, 75]:
        n_h = sp.holdings_for_pct(pct / 100)
        fig.add_hline(y=pct, row=2, col=1,
                      line=dict(color="#D3D1C7", width=1, dash="dot"))
        fig.add_vline(x=n_h, row=2, col=1,
                      line=dict(color="#D3D1C7", width=1, dash="dot"))
        fig.add_annotation(
            x=n_h + 2, y=pct + 3,
            xref="x2", yref="y2",
            text=f"<b>{n_h} holdings = {pct}%</b>",
            showarrow=False,
            font=dict(size=11, color="#378ADD"),
            xanchor="left",
        )

    fig.update_xaxes(
        title_text="Holdings ranked by weight",
        title_font=dict(size=12),
        row=2, col=1,
        gridcolor="#D3D1C7",
        tickfont=dict(size=11, color="#888780"),
    )
    fig.update_yaxes(
        title_text="Cumulative weight (%)",
        title_font=dict(size=12),
        row=2, col=1,
        gridcolor="#D3D1C7",
        tickfont=dict(size=11, color="#888780"),
        range=[0, 105],
    )

    # ── Row 3: sector bar chart ───────────────────────────────────────────
    sec_colors = [sector_colors.get(s, "#888780") for s in sectors.index]

    fig.add_trace(go.Bar(
        x=sectors.index,
        y=sectors["weight"] * 100,
        marker_color=sec_colors,
        marker_line_width=0,
        showlegend=False,
        customdata=sectors[["n_holdings", "top_holding", "weight_pct"]].values,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Weight: %{y:.1f}%<br>"
            "Holdings: %{customdata[0]}<br>"
            "Largest: %{customdata[1]}<extra></extra>"
        ),
        text=sectors["weight"].apply(lambda v: f"{v*100:.1f}%"),
        textposition="outside",
        textfont=dict(size=11, color="#5F5E5A"),
    ), row=3, col=1)

    fig.update_xaxes(
        tickangle=-30,
        tickfont=dict(size=11, color="#2C2C2A"),
        row=3, col=1,
        gridcolor="#D3D1C7",
    )
    fig.update_yaxes(
        title_text="Weight (%)",
        title_font=dict(size=12),
        row=3, col=1,
        gridcolor="#D3D1C7",
        tickfont=dict(size=11, color="#888780"),
    )

    # ── Row 4: metrics as a simple table trace ───────────────────────────
    kpi_labels = [
        "Total holdings", "Largest", "Top 5", "Top 10",
        "Top 25", "HHI", "Effective N", "Gini", "50% of ETF", "Top sector",
    ]
    kpi_values = [
        str(m["total_holdings"]),
        f"{m['largest_holding']} {m['largest_weight']:.1%}",
        f"{m.get('top_5_weight',0):.1%}",
        f"{m.get('top_10_weight',0):.1%}",
        f"{m.get('top_25_weight',0):.1%}",
        f"{m['hhi']:.0f}",
        f"{m['effective_n']:.0f}",
        f"{m['gini']:.3f}",
        f"{sp.holdings_for_pct(0.50)} stocks",
        f"{m['top_sector']} {m['top_sector_weight']:.1%}",
    ]

    fig.add_trace(go.Table(
        header=dict(
            values=[f"<b>{l}</b>" for l in kpi_labels],
            fill_color="#F4F3EF",
            font=dict(size=11, color="#888780"),
            align="center",
            height=28,
            line=dict(color="#D3D1C7", width=0.5),
        ),
        cells=dict(
            values=[[v] for v in kpi_values],
            fill_color="white",
            font=dict(size=13, color="#2C2C2A"),
            align="center",
            height=36,
            line=dict(color="#D3D1C7", width=0.5),
        ),
    ), row=4, col=1)

    # table subplot has no xy axes to hide

    # ── global layout ─────────────────────────────────────────────────────
    _apply_layout(
        fig,
        title=f"{sp.ticker} — ETF concentration analysis",
        subtitle=(
            f"{m['total_holdings']} holdings  ·  "
            f"top 10 = {m.get('top_10_weight', 0):.1%} of ETF  ·  "
            f"effective N = {m['effective_n']:.0f}  ·  "
            f"HHI = {m['hhi']:.0f}  ({m['hhi_label']})"
        ),
    )
    # only show legend if we have real sector data
    has_sectors = df["sector"].nunique() > 1 and "-" not in df["sector"].unique()

    fig.update_layout(
        height=1200,
        barmode="overlay",
        showlegend=has_sectors,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="right", x=1,
            font=dict(size=11),
            title=dict(text="Sector", font=dict(size=12)),
        ),
        margin=dict(l=60, r=40, t=100, b=120),
    )
    # MUST be after _apply_layout — _LAYOUT spread resets xaxis dict
    fig.update_xaxes(tickangle=-90, tickfont=dict(size=9), row=1, col=1)
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=11), row=3, col=1)
    return fig


def best_worst_days(
    quant: QuantAnalytics,
    symbol: str,
    period: str = "20y",
    miss_scenarios: list[int] | None = None,
    initial_value: float = 10_000.0,
) -> go.Figure:
    """
    "Cost of missing the best days" — two panels.

    Panel 1 (bar)    : final portfolio value for each scenario
    Panel 2 (scatter): timeline showing best/worst days cluster together
    """
    df     = quant.best_worst_days_impact(symbol, period, miss_scenarios, initial_value)
    detail = quant.best_worst_days_detail(symbol, period, n=20)

    bh_row    = df[df["type"] == "buy_and_hold"].iloc[0]
    bh_final  = bh_row["final_value"]
    miss_best = df[df["type"] == "miss_best"].sort_values("days_missed")
    miss_wrst = df[df["type"] == "miss_worst"].sort_values("days_missed")

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.58, 0.42],
        vertical_spacing=0.12,
        subplot_titles=[
            f"Final value of ${initial_value:,.0f} invested — {_period_label(period)}",
            "When did the best and worst days occur?",
        ],
    )

    # ── Panel 1: single bar trace with all scenarios ──────────────────────
    all_x = (
        ["Buy & Hold"] +
        [f"Miss best {n}d"  for n in miss_best["days_missed"]] +
        [f"Miss worst {n}d" for n in miss_wrst["days_missed"]]
    )
    all_y = (
        [bh_final] +
        list(miss_best["final_value"]) +
        list(miss_wrst["final_value"])
    )
    all_colors = (
        ["#2C2C2A"] +
        ["#E24B4A"] * len(miss_best) +
        ["#1D9E75"] * len(miss_wrst)
    )

    fig.add_trace(go.Bar(
        x=all_x, y=all_y,
        marker_color=all_colors,
        text=[f"${v:,.0f}" for v in all_y],
        textposition="outside",
        textfont=dict(size=8),
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>Final: $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    fig.add_hline(
        y=bh_final, row=1, col=1,
        line=dict(color="#2C2C2A", width=1.2, dash="dash"),
        annotation=dict(text=f"Buy & Hold ${bh_final:,.0f}",
                        font=dict(size=9, color="#2C2C2A"), xanchor="left"),
    )

    fig.update_yaxes(
        title_text="Final value ($)", tickformat="$,.0f",
        gridcolor="#D3D1C7", tickfont=dict(size=9, color="#888780"),
        row=1, col=1,
    )

    # ── Panel 2: scatter timeline with date strings ───────────────────────
    best_days  = detail[detail["type"] == "best"].copy()
    worst_days = detail[detail["type"] == "worst"].copy()
    best_days["ds"]  = pd.to_datetime(best_days["date"]).dt.strftime("%Y-%m-%d")
    worst_days["ds"] = pd.to_datetime(worst_days["date"]).dt.strftime("%Y-%m-%d")

    fig.add_trace(go.Scatter(
        x=best_days["ds"], y=best_days["return"],
        mode="markers", name="Best days",
        marker=dict(size=10, color="#1D9E75", line=dict(color="white", width=1)),
        hovertemplate="<b>Best day</b><br>%{x}<br>Return: %{y:.2%}<extra></extra>",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=worst_days["ds"], y=worst_days["return"],
        mode="markers", name="Worst days",
        marker=dict(size=10, color="#E24B4A", line=dict(color="white", width=1)),
        hovertemplate="<b>Worst day</b><br>%{x}<br>Return: %{y:.2%}<extra></extra>",
    ), row=2, col=1)

    fig.add_hline(y=0, row=2, col=1,
                  line=dict(color="#B4B2A9", width=0.8, dash="dash"))

    fig.update_yaxes(
        title_text="Daily return", tickformat=".0%",
        gridcolor="#D3D1C7", tickfont=dict(size=9, color="#888780"),
        row=2, col=1,
    )

    # annotation
    n_days = int(df["n_trading_days"].iloc[0])
    ann_text = (
        f"<b>{symbol}  ·  {_period_label(period)}</b><br>"
        f"{n_days:,} trading days  ·  B&H: ${bh_final:,.0f} ({bh_row['cagr']:.1%}/yr)"
    )
    if 10 in miss_best["days_missed"].values:
        v10 = miss_best[miss_best["days_missed"]==10]["final_value"].values[0]
        c10 = miss_best[miss_best["days_missed"]==10]["cagr"].values[0]
        ann_text += f"<br>Miss best 10d: ${v10:,.0f} ({c10:.1%}/yr)"

    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.99,
        text=ann_text, showarrow=False, align="left",
        font=dict(size=10, color="#5F5E5A"),
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="#D3D1C7", borderwidth=0.5, borderpad=8,
    )

    _apply_layout(
        fig,
        title=f"{symbol} — Cost of missing the best days",
        subtitle=(
            f"${initial_value:,.0f} invested  ·  {_period_label(period)}  ·  "
            "red = miss best (costly)  ·  green = miss worst (lucky)"
        ),
    )
    fig.update_layout(
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=11)),
        # explicit width helps notebook renderers size the figure correctly
        autosize=True,
    )
    # set axis types after _apply_layout to prevent reset
    fig.update_xaxes(type="category", row=1, col=1)
    fig.update_xaxes(type="date", tickformat="%Y", row=2, col=1)
    return fig



# ---------------------------------------------------------------------------
# Temporal concentration of best and worst days
# ---------------------------------------------------------------------------

def extreme_days_concentration(
    quant: QuantAnalytics,
    symbol: str,
    period: str = "20y",
    n: int = 50,
    window: int = 63,           # rolling window in trading days (~1 quarter)
) -> go.Figure:
    """
    Shows HOW CLOSE IN TIME the best and worst trading days are to each other.

    Three panels
    ------------
    Top    : price chart with best (green) and worst (red) days marked
    Middle : rolling density — how many extreme days fall in each window
             Shows that crises produce clusters of both best AND worst days
    Bottom : gap in days between each extreme day and the nearest opposite
             extreme day (best near a worst, worst near a best)
             A small gap = they happened close together

    The key insight: the best days are almost always within days or weeks
    of the worst days. You cannot avoid the bad without missing the good.

    Parameters
    ----------
    n      : number of best/worst days to include (default 50)
    window : rolling window in trading days for density count (default 63 = 1 quarter)
    """
    prices = quant._prices(symbol, period=period)
    rets   = prices.pct_change().dropna()

    best_days  = rets.nlargest(n)
    worst_days = rets.nsmallest(n)

    all_dates  = rets.index

    # ── rolling density ────────────────────────────────────────────────────
    # for each date, count how many best/worst days fall in the surrounding window
    is_best  = all_dates.isin(best_days.index).astype(float)
    is_worst = all_dates.isin(worst_days.index).astype(float)

    best_density  = pd.Series(is_best,  index=all_dates).rolling(window).sum()
    worst_density = pd.Series(is_worst, index=all_dates).rolling(window).sum()

    # ── gap analysis ───────────────────────────────────────────────────────
    # for each best day, find the nearest worst day (in trading days)
    best_idx  = [all_dates.get_loc(d) for d in best_days.index]
    worst_idx = [all_dates.get_loc(d) for d in worst_days.index]

    def _nearest_gap(src_indices, tgt_indices):
        gaps = []
        for i in src_indices:
            diffs = [abs(i - j) for j in tgt_indices]
            gaps.append(min(diffs))
        return gaps

    best_gaps  = _nearest_gap(best_idx,  worst_idx)   # best day → nearest worst day
    worst_gaps = _nearest_gap(worst_idx, best_idx)    # worst day → nearest best day

    # ── plot ───────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.35, 0.35, 0.30],
        vertical_spacing=0.07,
        subplot_titles=[
            f"Price with top-{n} best (▲) and worst (▼) days",
            f"Rolling {window}-day density of extreme days",
            "Gap (trading days) between each extreme day and nearest opposite",
        ],
        shared_xaxes=False,
    )

    # ── Panel 1: price + marked days ──────────────────────────────────────
    px_dates = prices.index.strftime("%Y-%m-%d").tolist()

    fig.add_trace(go.Scatter(
        x=px_dates, y=prices.values,
        mode="lines",
        line=dict(color="#B4B2A9", width=1.2),
        showlegend=False,
        hovertemplate="%{x}<br>Price: $%{y:,.2f}<extra></extra>",
    ), row=1, col=1)

    # best days
    fig.add_trace(go.Scatter(
        x=best_days.index.strftime("%Y-%m-%d").tolist(),
        y=prices.reindex(best_days.index).values,
        mode="markers",
        marker=dict(size=7, color="#1D9E75", symbol="triangle-up",
                    line=dict(color="white", width=0.5)),
        name=f"Top {n} best days",
        hovertemplate="%{x}<br>Return: %{customdata:.2%}<extra>Best day</extra>",
        customdata=best_days.values,
    ), row=1, col=1)

    # worst days
    fig.add_trace(go.Scatter(
        x=worst_days.index.strftime("%Y-%m-%d").tolist(),
        y=prices.reindex(worst_days.index).values,
        mode="markers",
        marker=dict(size=7, color="#E24B4A", symbol="triangle-down",
                    line=dict(color="white", width=0.5)),
        name=f"Top {n} worst days",
        hovertemplate="%{x}<br>Return: %{customdata:.2%}<extra>Worst day</extra>",
        customdata=worst_days.values,
    ), row=1, col=1)

    fig.update_yaxes(title_text="Price ($)", tickformat="$,.0f",
                     gridcolor="#D3D1C7", tickfont=dict(size=9, color="#888780"),
                     row=1, col=1)
    fig.update_xaxes(gridcolor="#D3D1C7", tickfont=dict(size=9, color="#888780"),
                     row=1, col=1)

    # ── Panel 2: rolling density ──────────────────────────────────────────
    density_dates = best_density.index.strftime("%Y-%m-%d").tolist()

    fig.add_trace(go.Scatter(
        x=density_dates, y=best_density.values,
        mode="lines", fill="tozeroy",
        fillcolor="rgba(29,158,117,0.20)",
        line=dict(color="#1D9E75", width=1.5),
        name=f"Best days per {window}d",
        hovertemplate="%{x}<br>Best days in window: %{y:.0f}<extra></extra>",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=density_dates, y=(-worst_density).values,
        mode="lines", fill="tozeroy",
        fillcolor="rgba(226,75,74,0.20)",
        line=dict(color="#E24B4A", width=1.5),
        name=f"Worst days per {window}d",
        customdata=worst_density.values,
        hovertemplate="%{x}<br>Worst days in window: %{customdata[0]:.0f}<extra></extra>",
    ), row=2, col=1)

    fig.add_hline(y=0, row=2, col=1,
                  line=dict(color="#B4B2A9", width=0.8))
    fig.update_yaxes(title_text="Days in window",
                     gridcolor="#D3D1C7", tickfont=dict(size=9, color="#888780"),
                     row=2, col=1)
    fig.update_xaxes(gridcolor="#D3D1C7", tickfont=dict(size=9, color="#888780"),
                     row=2, col=1)

    # ── Panel 3: gap histogram ────────────────────────────────────────────
    fig.add_trace(go.Histogram(
        x=best_gaps,
        nbinsx=30,
        marker_color="#1D9E75",
        opacity=0.7,
        name="Best day → nearest worst day",
        hovertemplate="Gap: %{x} days<br>Count: %{y}<extra>Best days</extra>",
    ), row=3, col=1)

    fig.add_trace(go.Histogram(
        x=worst_gaps,
        nbinsx=30,
        marker_color="#E24B4A",
        opacity=0.7,
        name="Worst day → nearest best day",
        hovertemplate="Gap: %{x} days<br>Count: %{y}<extra>Worst days</extra>",
    ), row=3, col=1)

    # median gap lines
    med_best  = float(np.median(best_gaps))
    med_worst = float(np.median(worst_gaps))
    fig.add_vline(x=med_best, row=3, col=1,
                  line=dict(color="#1D9E75", width=1.5, dash="dot"),
                  annotation=dict(text=f"median {med_best:.0f}d",
                                  font=dict(size=9, color="#1D9E75")))
    fig.add_vline(x=med_worst, row=3, col=1,
                  line=dict(color="#E24B4A", width=1.5, dash="dot"),
                  annotation=dict(text=f"median {med_worst:.0f}d",
                                  font=dict(size=9, color="#E24B4A"),
                                  yshift=-20))

    fig.update_xaxes(title_text="Gap (trading days)",
                     gridcolor="#D3D1C7", tickfont=dict(size=9, color="#888780"),
                     row=3, col=1)
    fig.update_yaxes(title_text="Count",
                     gridcolor="#D3D1C7", tickfont=dict(size=9, color="#888780"),
                     row=3, col=1)

    _apply_layout(
        fig,
        title=f"{symbol} — Temporal concentration of extreme days",
        subtitle=(
            f"Top {n} best and worst days  ·  {_period_label(period)}  ·  "
            f"median gap best↔worst: {med_best:.0f} trading days"
        ),
    )
    fig.update_layout(height=800, barmode="overlay",
                      hovermode="x unified")
    return fig

