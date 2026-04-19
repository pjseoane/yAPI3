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

from classes.quant_analytics import QuantAnalytics


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

    fig.update_xaxes(range=[lo, hi], title_text=f"{label} — {label_x}")
    fig.update_yaxes(range=[lo, hi], title_text=f"{label} — {label_y}")
    _apply_layout(fig,
                  title=f"{label}: {label_x} vs {label_y}",
                  subtitle=f"benchmark: {benchmark}  ·  risk-free rate {risk_free_rate:.0%}")
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
    _apply_layout(fig, title=f"Cumulative returns — {_period_label(period)}")
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
    _apply_layout(fig, title=f"Drawdown — {_period_label(period)}")
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
    _apply_layout(fig,
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
    _apply_layout(fig,
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
    fig.update_xaxes(title_text="Daily return", tickformat=".1%")
    fig.update_yaxes(title_text="Density")
    fig.update_layout(barmode="overlay")
    _apply_layout(fig, title=f"Return distribution — {_period_label(period)}")
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
    fig.update_xaxes(title_text=label)
    fig.update_yaxes(title_text="")
    fig.update_layout(height=max(350, len(syms) * 45 + 120))
    _apply_layout(fig, title=f"{label} — {_period_label(period)}")
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
            f"Text fields ('sector', 'industry') cannot be used as axes. "
            f"They appear automatically in hover tooltips."
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
