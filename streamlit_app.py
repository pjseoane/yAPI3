"""
streamlit_app.py — Streamlit dashboard for yfinance_api3.

Run with:
    streamlit run yfinance_api3/dashboard/streamlit_app.py

Dependencies:
    pip install streamlit
"""

import streamlit as st
import pandas as pd

from yfinance_api3.classes.stock_client import StockClient
from yfinance_api3.classes.quant_analytics import QuantAnalytics
import yfinance_api3.modules.plots as plots
import yfinance_api3.modules.portfolio as portfolio
import yfinance_api3.modules.backtest as backtest
import yfinance_api3.modules.factors as factors


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="QuantDashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #F4F3EF; }
  [data-testid="stSidebar"]          { background: #1E1E1C; }
  [data-testid="stSidebar"] *        { color: #D3D1C7 !important; }
  .block-container                   { padding-top: 1.5rem; }
  .stTabs [data-baseweb="tab"]       { font-size: 13px; font-weight: 500; }
  div[data-testid="metric-container"] > div { background: white;
    border-radius: 8px; border: 0.5px solid #D3D1C7; padding: 12px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Shared client — cached so they survive reruns
# ---------------------------------------------------------------------------

@st.cache_resource
def get_client():
    return StockClient()

@st.cache_resource
def get_quant(_client):
    return QuantAnalytics(_client)

client = get_client()
quant  = get_quant(client)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📈 QuantDashboard")
    st.markdown("---")

    symbols_raw = st.text_area(
        "Symbols (comma separated)",
        value="AAPL, MSFT, NVDA, GOOGL, JPM",
        height=80,
    )
    symbols = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]

    benchmark = st.text_input("Benchmark", value="SPY")

    period = st.selectbox(
        "Period",
        ["1y", "2y", "3y", "5y", "10y"],
        index=2,
    )

    rfr = st.slider(
        "Risk-free rate",
        min_value=0.0, max_value=0.10,
        value=0.05, step=0.005,
        format="%.1f%%",
    )

    st.markdown("---")
    run = st.button("▶  Run Analysis", use_container_width=True, type="primary")

    st.markdown("---")
    st.caption(f"Loaded: {', '.join(symbols)}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _show(fig, use_container_width=True):
    st.plotly_chart(fig, use_container_width=use_container_width,
                    config={"displayModeBar": True})

def _safe(fn, *args, label="", **kwargs):
    """Run fn and show result, or show an error message."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        st.warning(f"{label or fn.__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_risk, tab_seasonality, tab_portfolio, tab_factors = st.tabs([
    "📊 Overview",
    "⚠️ Risk",
    "📅 Seasonality",
    "🎯 Portfolio",
    "🔬 Factors",
])


# ── Overview ────────────────────────────────────────────────────────────────
with tab_overview:
    if not run:
        st.info("Configure your symbols in the sidebar and click **Run Analysis**.")
    else:
        st.subheader(f"Performance Overview — {period}")

        # KPI metrics for the first symbol
        with st.spinner("Computing metrics..."):
            try:
                sym   = symbols[0]
                stats = quant.stock_report(sym, benchmark=benchmark,
                                           period=period, risk_free_rate=rfr)
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("Volatility",   f"{stats['annualised_volatility']:.1%}")
                c2.metric("Sharpe",       f"{stats['sharpe_ratio']:.2f}")
                c3.metric("Sortino",      f"{stats['sortino_ratio']:.2f}")
                c4.metric("Max Drawdown", f"{stats['max_drawdown']:.1%}")
                c5.metric("Beta",         f"{stats['beta']:.2f}")
                c6.metric("VaR 95% 1d",  f"{stats['var_95_1d']:.2%}")
            except Exception as e:
                st.warning(f"Metrics: {e}")

        with st.spinner("Loading charts..."):
            fig = _safe(plots.cumulative_returns, quant, symbols,
                        period=period, label="Cumulative returns")
            if fig: _show(fig)

            col1, col2 = st.columns(2)
            with col1:
                fig = _safe(plots.drawdown, quant, symbols,
                            period=period, label="Drawdown")
                if fig: _show(fig)
            with col2:
                fig = _safe(plots.metrics_bar, quant, symbols,
                            metric="sharpe", period=period,
                            benchmark=benchmark, risk_free_rate=rfr,
                            label="Sharpe bar")
                if fig: _show(fig)


# ── Risk ────────────────────────────────────────────────────────────────────
with tab_risk:
    if not run:
        st.info("Click **Run Analysis** to load charts.")
    else:
        st.subheader(f"Risk Analysis — {period}")

        with st.spinner("Loading risk charts..."):
            col1, col2 = st.columns(2)
            with col1:
                fig = _safe(plots.rolling_volatility, quant, symbols,
                            period=period, label="Rolling vol")
                if fig: _show(fig)
            with col2:
                fig = _safe(plots.rolling_sharpe, quant, symbols,
                            period=period, risk_free_rate=rfr,
                            label="Rolling Sharpe")
                if fig: _show(fig)

            col3, col4 = st.columns(2)
            with col3:
                fig = _safe(plots.correlation_heatmap, quant, symbols,
                            period=period, label="Correlation")
                if fig: _show(fig)
            with col4:
                fig = _safe(plots.returns_distribution, quant, symbols,
                            period=period, label="Return distribution")
                if fig: _show(fig)

            fig = _safe(plots.scatter, quant, symbols,
                        metric="sharpe", period_x="1y", period_y=period,
                        benchmark=benchmark, risk_free_rate=rfr,
                        label="Sharpe quadrant")
            if fig: _show(fig)


# ── Seasonality ─────────────────────────────────────────────────────────────
with tab_seasonality:
    if not run:
        st.info("Click **Run Analysis** to load charts.")
    else:
        st.subheader("Seasonality Analysis")

        s_col1, s_col2, s_col3 = st.columns(3)
        season_sym = s_col1.selectbox("Symbol", symbols, key="s_sym")
        season_gran = s_col2.selectbox("Granularity",
                                       ["monthly", "weekly"], key="s_gran")
        season_lt = s_col3.selectbox("Long-term",
                                     ["5y","10y","15y","20y"],
                                     index=1, key="s_lt")
        season_st = st.selectbox("Short-term",
                                 ["1y","2y","3y","5y"],
                                 index=2, key="s_st")

        with st.spinner("Loading seasonality..."):
            col1, col2 = st.columns(2)
            with col1:
                fig = _safe(plots.seasonality, quant, season_sym,
                            period=season_lt, granularity=season_gran,
                            label="Seasonality bar")
                if fig: _show(fig)
            with col2:
                fig = _safe(plots.seasonality_box, quant, season_sym,
                            period=season_lt, granularity=season_gran,
                            label="Seasonality box")
                if fig: _show(fig)

            fig = _safe(plots.seasonality_comparison_clean, quant, season_sym,
                        long_term=season_lt, short_term=season_st,
                        label="Seasonality comparison")
            if fig: _show(fig)

            fig = _safe(plots.seasonality_heatmap, quant, season_sym,
                        period=season_lt, label="Seasonality heatmap")
            if fig: _show(fig)


# ── Portfolio ────────────────────────────────────────────────────────────────
with tab_portfolio:
    if not run:
        st.info("Click **Run Analysis** to load charts.")
    else:
        st.subheader(f"Portfolio Optimisation — {period}")

        with st.spinner("Computing efficient frontier..."):
            fig = _safe(lambda: plots.efficient_frontier(
                portfolio.efficient_frontier(quant, symbols,
                                             period=period,
                                             risk_free_rate=rfr)
            ), label="Efficient frontier")
            if fig: _show(fig)

        with st.spinner("Kelly analysis..."):
            fig = _safe(plots.kelly, quant, symbols,
                        period=period, risk_free_rate=rfr,
                        label="Kelly")
            if fig: _show(fig)

        st.markdown("#### Backtest")
        bt_strategy = st.selectbox("Strategy", [
            "Buy & Hold", "MA 20/50", "MA 50/200",
            "Momentum (63d)", "Mean Reversion",
        ], key="bt_strat")

        strategy_map = {
            "Buy & Hold":       backtest.buy_and_hold(),
            "MA 20/50":         backtest.ma_crossover(20, 50),
            "MA 50/200":        backtest.ma_crossover(50, 200),
            "Momentum (63d)":   backtest.momentum(lookback=63),
            "Mean Reversion":   backtest.mean_reversion(lookback=20),
        }

        with st.spinner("Running backtest..."):
            try:
                bt_result = backtest.run(
                    quant, symbols,
                    strategy=strategy_map[bt_strategy],
                    period=period,
                    benchmark=benchmark,
                    risk_free_rate=rfr,
                )
                m = bt_result.metrics
                bc1, bc2, bc3, bc4 = st.columns(4)
                bc1.metric("CAGR",        f"{m['cagr']:.1%}")
                bc2.metric("Sharpe",      f"{m['sharpe_ratio']:.2f}")
                bc3.metric("Max DD",      f"{m['max_drawdown']:.1%}")
                bc4.metric("Win rate",    f"{m['win_rate']:.0%}")
                fig = _safe(plots.backtest, bt_result, label="Backtest")
                if fig: _show(fig)
            except Exception as e:
                st.warning(f"Backtest: {e}")


# ── Factors ──────────────────────────────────────────────────────────────────
with tab_factors:
    if not run:
        st.info("Click **Run Analysis** to load charts.")
    else:
        st.subheader("Fama-French Factor Exposure")

        factor_model = st.selectbox(
            "Model",
            ["ff3", "ff5", "mom", "ff6"],
            index=1,
            format_func=lambda x: {
                "ff3": "FF3 — 3 factor",
                "ff5": "FF5 — 5 factor",
                "mom": "FF3 + Momentum",
                "ff6": "FF6 — all factors",
            }[x],
            key="f_model",
        )

        with st.spinner("Running factor regressions..."):
            factor_results = []
            for sym in symbols:
                try:
                    fr = factors.run(quant, sym, model=factor_model,
                                     period=period)
                    factor_results.append(fr)
                    _show(plots.factor_exposure(fr))
                except Exception as e:
                    st.warning(f"{sym}: {e}")

            if len(factor_results) > 1:
                fig = _safe(plots.factor_comparison, factor_results,
                            label="Factor comparison")
                if fig: _show(fig)

            for sym in symbols[:3]:   # limit rolling to first 3 to avoid timeout
                fig = _safe(plots.rolling_factor_betas, quant, sym,
                            model=factor_model, period=period,
                            label=f"Rolling betas {sym}")
                if fig: _show(fig)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("QuantDashboard · powered by yfinance_api3 · "
           "for informational purposes only, not financial advice")
