"""
report.py — professional HTML report builder for QuantAnalytics.

Usage — auto mode (one call):
------------------------------
    from modules.report import auto_report

    auto_report(
        quant, client, symbols,
        title="Equity Research Report",
        period="1y",
        benchmark="SPY",
    ).save("report.html")

Usage — manual mode (compose sections):
----------------------------------------
    from modules.report import QuantReport
    import modules.plots as plots

    r = QuantReport(title="My Portfolio Analysis", subtitle="Q4 2024")
    r.add_section("Market Overview")
    r.add_text("Performance summary for the selected basket.")
    r.add_plot(plots.cumulative_returns(quant, symbols))
    r.add_metrics({"Sharpe": 1.42, "Max DD": -0.18, "CAGR": 0.23})
    r.add_section("Risk Analysis")
    r.add_plot(plots.correlation_heatmap(quant, symbols))
    r.add_table(quant.metrics_df(symbols))
    r.save("report.html")
    r.show()
"""

from __future__ import annotations

import os
import webbrowser
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


# ---------------------------------------------------------------------------
# Element types
# ---------------------------------------------------------------------------

@dataclass
class _PlotElement:
    fig: go.Figure

@dataclass
class _TableElement:
    df: pd.DataFrame
    caption: str = ""

@dataclass
class _TextElement:
    text: str

@dataclass
class _MetricsElement:
    metrics: dict[str, Any]
    ncols: int = 4

@dataclass
class _Section:
    title: str
    elements: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# HTML / CSS templates
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #F4F3EF;
  color: #2C2C2A;
  font-size: 14px;
  line-height: 1.6;
}

/* ── Header ── */
.report-header {
  background: #1E1E1C;
  color: white;
  padding: 48px 64px 40px;
  position: relative;
  overflow: hidden;
}
.report-header::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, #1D9E75, #378ADD, #534AB7);
}
.report-title {
  font-size: 32px;
  font-weight: 600;
  letter-spacing: -0.5px;
  margin-bottom: 8px;
}
.report-subtitle {
  font-size: 15px;
  color: #888780;
  margin-bottom: 24px;
}
.report-meta {
  display: flex;
  gap: 32px;
  font-size: 12px;
  color: #888780;
}
.report-meta span b {
  color: #D3D1C7;
  font-weight: 500;
}

/* ── Layout ── */
.report-body {
  display: flex;
  min-height: calc(100vh - 160px);
}

/* ── Sidebar / TOC ── */
.toc {
  width: 220px;
  min-width: 220px;
  background: #1E1E1C;
  padding: 32px 0;
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
}
.toc-title {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: #5F5E5A;
  padding: 0 24px 16px;
}
.toc a {
  display: block;
  padding: 8px 24px;
  color: #888780;
  text-decoration: none;
  font-size: 12.5px;
  border-left: 2px solid transparent;
  transition: color 0.15s, border-color 0.15s;
}
.toc a:hover { color: #D3D1C7; border-left-color: #5F5E5A; }
.toc a.active { color: white; border-left-color: #1D9E75; }

/* ── Main content ── */
.report-content {
  flex: 1;
  padding: 40px 56px;
  max-width: 1200px;
}

/* ── Sections ── */
.report-section {
  margin-bottom: 56px;
  scroll-margin-top: 24px;
}
.section-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 24px;
  padding-bottom: 12px;
  border-bottom: 1px solid #D3D1C7;
}
.section-number {
  font-size: 11px;
  font-weight: 600;
  color: #888780;
  letter-spacing: 1px;
  min-width: 28px;
}
.section-title {
  font-size: 20px;
  font-weight: 500;
  color: #2C2C2A;
}

/* ── Plot card ── */
.plot-card {
  background: white;
  border-radius: 10px;
  border: 0.5px solid #D3D1C7;
  overflow: hidden;
  margin-bottom: 20px;
}
.plot-card .plotly-graph-div { width: 100% !important; }

/* ── Metrics grid ── */
.metrics-grid {
  display: grid;
  gap: 12px;
  margin-bottom: 20px;
}
.metric-card {
  background: white;
  border-radius: 8px;
  border: 0.5px solid #D3D1C7;
  padding: 16px 20px;
}
.metric-label {
  font-size: 11px;
  font-weight: 500;
  color: #888780;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-bottom: 6px;
}
.metric-value {
  font-size: 22px;
  font-weight: 500;
  color: #2C2C2A;
  font-variant-numeric: tabular-nums;
}
.metric-value.positive { color: #0F6E56; }
.metric-value.negative { color: #A32D2D; }

/* ── Table ── */
.table-wrap {
  background: white;
  border-radius: 10px;
  border: 0.5px solid #D3D1C7;
  overflow: auto;
  margin-bottom: 20px;
}
.table-caption {
  padding: 14px 20px 0;
  font-size: 12px;
  font-weight: 500;
  color: #888780;
  text-transform: uppercase;
  letter-spacing: 0.8px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
th {
  background: #F4F3EF;
  padding: 10px 16px;
  text-align: left;
  font-size: 11px;
  font-weight: 600;
  color: #888780;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  border-bottom: 1px solid #D3D1C7;
}
td {
  padding: 10px 16px;
  border-bottom: 0.5px solid #F4F3EF;
  color: #2C2C2A;
  font-variant-numeric: tabular-nums;
}
tr:last-child td { border-bottom: none; }
tr:hover td { background: #FAFAF9; }

/* ── Text block ── */
.text-block {
  color: #5F5E5A;
  font-size: 14px;
  line-height: 1.8;
  margin-bottom: 20px;
  max-width: 720px;
}

/* ── Footer ── */
.report-footer {
  text-align: center;
  padding: 32px;
  color: #888780;
  font-size: 11px;
  border-top: 1px solid #D3D1C7;
  margin-top: 40px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #B4B2A9; border-radius: 8px; }
"""

_JS = """
// Highlight active TOC link on scroll
const sections = document.querySelectorAll('.report-section');
const links    = document.querySelectorAll('.toc a');
const observer = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      links.forEach(l => l.classList.remove('active'));
      const id  = e.target.id;
      const lnk = document.querySelector(`.toc a[href="#${id}"]`);
      if (lnk) lnk.classList.add('active');
    }
  });
}, { threshold: 0.3 });
sections.forEach(s => observer.observe(s));
"""


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_fig(fig: go.Figure) -> str:
    return pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        config={"displayModeBar": True, "responsive": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "toImageButtonOptions": {"format": "png", "width": 1200, "height": 600}},
    )


def _render_table(df: pd.DataFrame, caption: str = "") -> str:
    def _fmt(v):
        if isinstance(v, float):
            if abs(v) < 10:
                return f"{v:.3f}"
            return f"{v:,.2f}"
        return str(v)

    cap_html = f'<div class="table-caption">{caption}</div>' if caption else ""
    header   = "".join(f"<th>{c}</th>" for c in [""] + list(df.columns))
    rows_html = ""
    for idx, row in df.iterrows():
        cells = "".join(f"<td>{_fmt(v)}</td>" for v in row)
        rows_html += f"<tr><td><b>{idx}</b></td>{cells}</tr>"
    return (
        f'<div class="table-wrap">{cap_html}'
        f"<table><thead><tr>{header}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table></div>"
    )


def _render_metrics(metrics: dict, ncols: int = 4) -> str:
    cards = []
    for label, value in metrics.items():
        if isinstance(value, float):
            if abs(value) <= 1 and "ratio" not in label.lower():
                fmt = f"{value:.2%}"
            else:
                fmt = f"{value:.3f}"
        else:
            fmt = str(value)

        # colour positive / negative values
        css = ""
        if isinstance(value, float):
            css = "positive" if value > 0 else "negative" if value < 0 else ""

        cards.append(
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value {css}">{fmt}</div>'
            f'</div>'
        )

    return (
        f'<div class="metrics-grid" '
        f'style="grid-template-columns: repeat({ncols}, 1fr);">'
        + "".join(cards)
        + "</div>"
    )


def _render_text(text: str) -> str:
    return f'<div class="text-block">{text}</div>'


def _slugify(text: str) -> str:
    return text.lower().replace(" ", "-").replace("/", "-")


# ---------------------------------------------------------------------------
# QuantReport
# ---------------------------------------------------------------------------

class QuantReport:
    """
    Composable HTML report builder.

    Methods
    -------
    add_section(title)           start a new named section
    add_plot(fig)                embed a Plotly figure
    add_table(df, caption)       embed a styled DataFrame table
    add_metrics(dict, ncols)     embed a row of metric cards
    add_text(str)                add a prose paragraph
    save(path) -> str            render and write HTML file, return path
    show()                       save to a temp file and open in browser
    """

    def __init__(
        self,
        title: str = "Quantitative Analysis Report",
        subtitle: str = "",
        author: str = "",
        symbols: list[str] | None = None,
        period: str | None = None,
    ) -> None:
        self.title    = title
        self.subtitle = subtitle
        self.author   = author
        self.symbols  = symbols or []
        self.period   = period
        self._sections: list[_Section] = []
        self._current: _Section | None = None

    # ------------------------------------------------------------------
    # Builder API
    # ------------------------------------------------------------------

    def add_section(self, title: str) -> "QuantReport":
        section = _Section(title=title)
        self._sections.append(section)
        self._current = section
        return self

    def _ensure_section(self) -> _Section:
        if self._current is None:
            self.add_section("Overview")
        return self._current  # type: ignore

    def add_plot(self, fig: go.Figure) -> "QuantReport":
        self._ensure_section().elements.append(_PlotElement(fig))
        return self

    def add_table(self, df: pd.DataFrame, caption: str = "") -> "QuantReport":
        self._ensure_section().elements.append(_TableElement(df, caption))
        return self

    def add_metrics(self, metrics: dict, ncols: int = 4) -> "QuantReport":
        self._ensure_section().elements.append(_MetricsElement(metrics, ncols))
        return self

    def add_text(self, text: str) -> "QuantReport":
        self._ensure_section().elements.append(_TextElement(text))
        return self

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_html(self) -> str:
        # TOC
        toc_links = "".join(
            f'<a href="#{_slugify(s.title)}">{s.title}</a>'
            for s in self._sections
        )

        # Meta bar
        meta_items = []
        if self.symbols:
            meta_items.append(f"<span><b>Symbols</b> &nbsp;{', '.join(self.symbols)}</span>")
        if self.period:
            meta_items.append(f"<span><b>Period</b> &nbsp;{self.period}</span>")
        meta_items.append(
            f"<span><b>Generated</b> &nbsp;{datetime.now().strftime('%d %b %Y %H:%M')}</span>"
        )
        if self.author:
            meta_items.append(f"<span><b>Author</b> &nbsp;{self.author}</span>")
        meta_html = "".join(meta_items)

        # Sections
        sections_html = ""
        for i, section in enumerate(self._sections, start=1):
            slug = _slugify(section.title)
            body = ""
            for el in section.elements:
                if isinstance(el, _PlotElement):
                    body += f'<div class="plot-card">{_render_fig(el.fig)}</div>'
                elif isinstance(el, _TableElement):
                    body += _render_table(el.df, el.caption)
                elif isinstance(el, _MetricsElement):
                    body += _render_metrics(el.metrics, el.ncols)
                elif isinstance(el, _TextElement):
                    body += _render_text(el.text)

            sections_html += f"""
            <div class="report-section" id="{slug}">
              <div class="section-header">
                <span class="section-number">{i:02d}</span>
                <h2 class="section-title">{section.title}</h2>
              </div>
              {body}
            </div>
            """

        subtitle_html = (
            f'<div class="report-subtitle">{self.subtitle}</div>'
            if self.subtitle else ""
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{self.title}</title>
  <style>{_CSS}</style>
</head>
<body>

<header class="report-header">
  <div class="report-title">{self.title}</div>
  {subtitle_html}
  <div class="report-meta">{meta_html}</div>
</header>

<div class="report-body">
  <nav class="toc">
    <div class="toc-title">Contents</div>
    {toc_links}
  </nav>
  <main class="report-content">
    {sections_html}
    <footer class="report-footer">
      Generated {datetime.now().strftime('%d %B %Y at %H:%M')} &nbsp;·&nbsp;
      For informational purposes only. Not financial advice.
    </footer>
  </main>
</div>

<script>{_JS}</script>
</body>
</html>"""

    def save(self, path: str = "report.html") -> str:
        """Render to HTML and write to *path*. Returns the absolute path."""
        html = self._render_html()
        abs_path = os.path.abspath(path)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Report saved → {abs_path}")
        return abs_path

    def show(self) -> None:
        """Save to a temp file and open in the default browser."""
        tmp = tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8"
        )
        tmp.write(self._render_html())
        tmp.close()
        webbrowser.open(f"file://{os.path.abspath(tmp.name)}")


# ---------------------------------------------------------------------------
# auto_report — one-call full report
# ---------------------------------------------------------------------------

def auto_report(
    quant,
    client,
    symbols: list[str],
    title: str = "Quantitative Analysis Report",
    subtitle: str = "",
    author: str = "",
    period: str = "1y",
    benchmark: str = "SPY",
    risk_free_rate: float = 0.05,
    include_backtest: bool = True,
    include_frontier: bool = True,
    include_fundamentals: bool = True,
    include_montecarlo: bool = True,
    mc_horizon: int = 252,
    mc_sims: int = 1000,
    include_factors: bool = True,
    factor_model: str = "ff5",
) -> QuantReport:
    """
    Build a full report automatically from symbols.

    Sections generated
    ------------------
    1. Executive Summary      — metric cards for each symbol
    2. Price Performance      — cumulative returns + drawdown
    3. Risk Analysis          — rolling vol, rolling Sharpe, correlation heatmap
    4. Return Distribution    — histogram + KDE
    5. Cross-sectional        — metrics bar (Sharpe, Sortino, vol, max DD)
    6. Sharpe Quadrant Study  — scatter: short vs long period, benchmark quadrants
    7. Fundamentals           — P/E vs revenue growth scatter (if include_fundamentals)
    8. Portfolio Optimisation — efficient frontier (if include_frontier)
    9. Backtest               — MA crossover vs benchmark (if include_backtest)
    10. Monte Carlo           — fan chart + terminal distribution (if include_montecarlo)
    11. Factor Exposure       — FF5 loadings + rolling betas (if include_factors)
    """
    import modules.plots as plots
    import modules.portfolio as portfolio
    import modules.backtest as backtest
    import modules.montecarlo as montecarlo

    r = QuantReport(
        title=title,
        subtitle=subtitle or f"Coverage: {', '.join(symbols)}",
        author=author,
        symbols=symbols,
        period=period,
    )

    # ------------------------------------------------------------------
    # 1. Executive Summary
    # ------------------------------------------------------------------
    r.add_section("Executive Summary")
    r.add_text(
        f"This report covers <b>{len(symbols)}</b> equities over a "
        f"<b>{period}</b> window. All return metrics use adjusted closing "
        f"prices with a risk-free rate of <b>{risk_free_rate:.1%}</b>."
    )
    try:
        mdf = quant.metrics_df(symbols, benchmark=benchmark,
                               period=period, risk_free_rate=risk_free_rate)
        r.add_table(mdf, caption="Key metrics")
    except Exception as e:
        r.add_text(f"<i>Metrics table unavailable: {e}</i>")

    # ------------------------------------------------------------------
    # 2. Price Performance
    # ------------------------------------------------------------------
    r.add_section("Price Performance")
    try:
        r.add_plot(plots.cumulative_returns(quant, symbols, period=period))
    except Exception as e:
        r.add_text(f"<i>Cumulative returns unavailable: {e}</i>")
    try:
        r.add_plot(plots.drawdown(quant, symbols, period=period))
    except Exception as e:
        r.add_text(f"<i>Drawdown unavailable: {e}</i>")

    # ------------------------------------------------------------------
    # 3. Risk Analysis
    # ------------------------------------------------------------------
    r.add_section("Risk Analysis")
    try:
        r.add_plot(plots.rolling_volatility(quant, symbols, period=period))
    except Exception as e:
        r.add_text(f"<i>Rolling volatility unavailable: {e}</i>")
    try:
        r.add_plot(plots.rolling_sharpe(quant, symbols, period=period,
                                         risk_free_rate=risk_free_rate))
    except Exception as e:
        r.add_text(f"<i>Rolling Sharpe unavailable: {e}</i>")
    try:
        r.add_plot(plots.correlation_heatmap(quant, symbols, period=period))
    except Exception as e:
        r.add_text(f"<i>Correlation heatmap unavailable: {e}</i>")

    # ------------------------------------------------------------------
    # 4. Return Distribution
    # ------------------------------------------------------------------
    r.add_section("Return Distribution")
    try:
        r.add_plot(plots.returns_distribution(quant, symbols, period=period))
    except Exception as e:
        r.add_text(f"<i>Return distribution unavailable: {e}</i>")

    # ------------------------------------------------------------------
    # 5. Cross-sectional Rankings
    # ------------------------------------------------------------------
    r.add_section("Cross-sectional Rankings")
    for metric in ["sharpe", "sortino", "volatility", "max_drawdown"]:
        try:
            r.add_plot(plots.metrics_bar(quant, symbols, metric=metric,
                                          period=period, benchmark=benchmark,
                                          risk_free_rate=risk_free_rate))
        except Exception as e:
            r.add_text(f"<i>{metric} bar unavailable: {e}</i>")

    # ------------------------------------------------------------------
    # 6. Sharpe Quadrant Study
    # ------------------------------------------------------------------
    r.add_section("Sharpe Quadrant Study")
    r.add_text(
        f"Each scatter compares the Sharpe ratio across two time windows. "
        f"Points above the diagonal improved over the longer horizon; "
        f"points below were stronger recently. "
        f"The <b>{benchmark}</b> diamond defines the four quadrants — "
        f"stocks in the top-right beat the benchmark on both periods."
    )
    # derive shorter and longer periods from the report period
    _period_pairs = {
        "1y": ("6mo", "1y"),   # not ideal but workable
        "2y": ("1y",  "2y"),
        "3y": ("1y",  "3y"),
        "5y": ("2y",  "5y"),
    }
    px_auto, py_auto = _period_pairs.get(period, ("1y", "2y"))
    for metric in ["sharpe", "sortino", "calmar"]:
        try:
            r.add_plot(plots.scatter(
                quant, symbols,
                metric=metric,
                period_x=px_auto,
                period_y=py_auto,
                benchmark=benchmark,
                risk_free_rate=risk_free_rate,
            ))
        except Exception as e:
            r.add_text(f"<i>{metric} scatter unavailable: {e}</i>")

    # ------------------------------------------------------------------
    # 6. Fundamentals
    # ------------------------------------------------------------------
    if include_fundamentals:
        r.add_section("Fundamentals")
        try:
            r.add_plot(plots.fundamentals_scatter(
                client, symbols,
                field_x="trailingPE",
                field_y="revenueGrowth",
                size_by="pct_from_52w_high",
            ))
        except Exception as e:
            r.add_text(f"<i>Fundamentals scatter unavailable: {e}</i>")
        try:
            r.add_plot(plots.fundamentals_scatter(
                client, symbols,
                field_x="operatingMargins",
                field_y="returnOnEquity",
                size_by="marketCap",
            ))
        except Exception as e:
            r.add_text(f"<i>Margins/ROE scatter unavailable: {e}</i>")

    # ------------------------------------------------------------------
    # 7. Portfolio Optimisation
    # ------------------------------------------------------------------
    if include_frontier:
        r.add_section("Portfolio Optimisation")
        try:
            frontier = portfolio.efficient_frontier(
                quant, symbols, period=period,
                risk_free_rate=risk_free_rate,
            )
            r.add_plot(plots.efficient_frontier(frontier))

            # comparison table
            cdf = portfolio.compare_strategies(
                quant, symbols, period=period,
                risk_free_rate=risk_free_rate,
            )
            r.add_table(cdf, caption="Strategy comparison")
        except Exception as e:
            r.add_text(f"<i>Portfolio optimisation unavailable: {e}</i>")

    # ------------------------------------------------------------------
    # 8. Backtest
    # ------------------------------------------------------------------
    if include_backtest:
        r.add_section("Backtest")
        r.add_text(
            "MA crossover strategy (20/50-day) versus buy-and-hold benchmark. "
            "Transaction cost: 10 bps per trade. Monthly rebalance."
        )
        try:
            bt = backtest.run(
                quant, symbols,
                strategy=backtest.ma_crossover(fast=20, slow=50),
                period=period,
                rebalance="monthly",
                transaction_cost_bps=10,
                benchmark=benchmark,
                risk_free_rate=risk_free_rate,
            )
            r.add_metrics({
                "CAGR":         bt.metrics["cagr"],
                "Volatility":   bt.metrics["volatility"],
                "Sharpe":       bt.metrics["sharpe_ratio"],
                "Max Drawdown": bt.metrics["max_drawdown"],
                "Win Rate":     bt.metrics["win_rate"],
                "# Trades":     bt.metrics["n_trades"],
            }, ncols=6)
            r.add_plot(plots.backtest(bt))
        except Exception as e:
            r.add_text(f"<i>Backtest unavailable: {e}</i>")


    if include_montecarlo:
        r.add_section("Monte Carlo Simulation")
        r.add_text(
            f"Forward simulation of {mc_sims:,} portfolio paths over a "
            f"{mc_horizon}-day horizon using three distributional assumptions. "
            "Block bootstrap preserves fat tails; Student-t adds heavier tails "
            "than the normal assumption."
        )
        for mc_method in ["historical", "normal", "t_dist"]:
            try:
                mc = montecarlo.simulate(
                    quant, symbols,
                    horizon=mc_horizon,
                    n_sims=mc_sims,
                    method=mc_method,
                    period=period,
                )
                r.add_metrics({
                    "Method":        mc_method,
                    "Median return": mc.metrics["median_return"],
                    "VaR 95%":       mc.metrics["var_95"],
                    "CVaR 95%":      mc.metrics["cvar_95"],
                    "Prob of gain":  mc.metrics["prob_gain"],
                    "Loss >10% prob":mc.metrics["prob_loss_10pct"],
                }, ncols=6)
                r.add_plot(plots.monte_carlo(mc, show_paths=30))
            except Exception as e:
                r.add_text(f"<i>Monte Carlo ({mc_method}) unavailable: {e}</i>")


    if include_factors:
        import modules.factors as factors
        r.add_section("Factor Exposure")
        r.add_text(
            f"Fama-French <b>{factor_model.upper()}</b> factor regression. "
            "Solid bars = significant (p<0.05). Error bars = ±1 SE. "
            "Rolling betas show how factor exposures shift over time."
        )
        for sym in symbols:
            try:
                fr = factors.run(quant, sym, model=factor_model, period=period)
                r.add_plot(plots.factor_exposure(fr))
            except Exception as e:
                r.add_text(f"<i>Factor exposure for {sym} unavailable: {e}</i>")
        try:
            results = []
            for sym in symbols:
                try:
                    results.append(factors.run(quant, sym, model=factor_model, period=period))
                except Exception:
                    pass
            if len(results) > 1:
                r.add_plot(plots.factor_comparison(results))
        except Exception as e:
            r.add_text(f"<i>Factor comparison unavailable: {e}</i>")

    return r
