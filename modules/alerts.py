"""
alerts.py — metric monitoring and alert engine.

Define rules (what to watch + threshold), run them against live data,
get notified when conditions are met.

Notification channels
---------------------
  console : always printed (default)
  log     : appended to a JSON-lines file
  email   : sent via SMTP (requires config)
  webhook : POST to any URL (Slack, Teams, Discord, etc.)

Usage
-----
    from modules.alerts import AlertEngine, price_change, metric_threshold
    from modules.alerts import volatility_spike, new_52w_high_low, drawdown_alert

    engine = AlertEngine(quant, client)

    # register alerts
    engine.add(price_change("NVDA", pct=0.05, direction="up"))
    engine.add(price_change("AAPL", pct=-0.05, direction="down"))
    engine.add(metric_threshold("TSLA", metric="volatility",
                                value=0.80, direction="above", period="1y"))
    engine.add(volatility_spike("SPY", threshold=0.25))
    engine.add(new_52w_high_low("MSFT"))
    engine.add(drawdown_alert("QQQ", threshold=-0.15))

    # run once
    results = engine.run()
    for r in results:
        print(r)

    # run on a schedule (background thread)
    engine.schedule(interval_minutes=30)
    engine.stop()

    # configure notifications
    engine.set_log("alerts.jsonl")
    engine.set_email(smtp_host="smtp.gmail.com", port=587,
                     user="you@gmail.com", password="...",
                     recipients=["you@gmail.com"])
    engine.set_webhook("https://hooks.slack.com/services/...")
"""

from __future__ import annotations

import json
import logging
import smtplib
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.text import MIMEText
from typing import Callable, Literal

import requests

from classes.quant_analytics import QuantAnalytics
from classes.stock_client import StockClient


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """
    A single monitoring rule.

    Attributes
    ----------
    name        : human-readable label shown in notifications
    symbol      : ticker being watched (or comma-separated list)
    check_fn    : callable(quant, client) -> AlertResult | None
                  Returns an AlertResult if the condition is met, else None.
    cooldown_min: minimum minutes between repeated triggers (default 60)
                  Prevents flooding when a condition persists.
    tags        : optional list of tags for grouping/filtering
    """
    name:         str
    symbol:       str
    check_fn:     Callable
    cooldown_min: int = 60
    tags:         list[str] = field(default_factory=list)
    _last_trigger: datetime | None = field(default=None, repr=False)

    def is_on_cooldown(self) -> bool:
        if self._last_trigger is None:
            return False
        elapsed = (datetime.now() - self._last_trigger).total_seconds() / 60
        return elapsed < self.cooldown_min

    def mark_triggered(self) -> None:
        self._last_trigger = datetime.now()


@dataclass
class AlertResult:
    """
    A triggered alert with full context.

    Attributes
    ----------
    alert_name  : name of the Alert that fired
    symbol      : ticker
    message     : human-readable description of what happened
    value       : current metric value that triggered the alert
    threshold   : the threshold that was crossed
    direction   : "above" | "below" | "up" | "down" | "new_high" | "new_low"
    severity    : "info" | "warning" | "critical"
    timestamp   : when the alert fired
    extra       : dict of additional context (e.g. previous value, period used)
    """
    alert_name: str
    symbol:     str
    message:    str
    value:      float
    threshold:  float
    direction:  str
    severity:   str = "warning"
    timestamp:  datetime = field(default_factory=datetime.now)
    extra:      dict = field(default_factory=dict)

    def __str__(self) -> str:
        sev_emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(
            self.severity, "⚠️"
        )
        return (
            f"{sev_emoji} [{self.timestamp.strftime('%Y-%m-%d %H:%M')}] "
            f"{self.alert_name} | {self.symbol}\n"
            f"   {self.message}\n"
            f"   Value: {self.value:.4f}  |  Threshold: {self.threshold:.4f}"
        )

    def to_dict(self) -> dict:
        return {
            "alert_name": self.alert_name,
            "symbol":     self.symbol,
            "message":    self.message,
            "value":      self.value,
            "threshold":  self.threshold,
            "direction":  self.direction,
            "severity":   self.severity,
            "timestamp":  self.timestamp.isoformat(),
            "extra":      self.extra,
        }


# ---------------------------------------------------------------------------
# Alert engine
# ---------------------------------------------------------------------------

class AlertEngine:
    """
    Registry and runner for Alert rules.

    Parameters
    ----------
    quant  : QuantAnalytics instance
    client : StockClient instance

    Notification channels (configure before running)
    ------------------------------------------------
    set_log(path)       → append AlertResults to a JSON-lines file
    set_email(...)      → send email via SMTP on each trigger
    set_webhook(url)    → POST to a webhook URL (Slack/Teams/Discord)
    """

    def __init__(self, quant: QuantAnalytics, client: StockClient) -> None:
        self.quant   = quant
        self.client  = client
        self._alerts: list[Alert] = []
        self._log_path:     str | None = None
        self._email_cfg:    dict | None = None
        self._webhook_url:  str | None = None
        self._thread:       threading.Thread | None = None
        self._stop_event    = threading.Event()

    # ------------------------------------------------------------------
    # Alert registry
    # ------------------------------------------------------------------

    def add(self, alert: Alert) -> "AlertEngine":
        """Register an alert. Returns self for chaining."""
        self._alerts.append(alert)
        return self

    def remove(self, name: str) -> "AlertEngine":
        """Remove all alerts with the given name."""
        self._alerts = [a for a in self._alerts if a.name != name]
        return self

    def list_alerts(self) -> list[dict]:
        """Return a summary of registered alerts."""
        return [
            {
                "name":         a.name,
                "symbol":       a.symbol,
                "cooldown_min": a.cooldown_min,
                "tags":         a.tags,
                "last_trigger": a.last_trigger.isoformat()
                                if a._last_trigger else None,
            }
            for a in self._alerts
        ]

    # ------------------------------------------------------------------
    # Notification config
    # ------------------------------------------------------------------

    def set_log(self, path: str) -> "AlertEngine":
        """Log triggered alerts to a JSON-lines file."""
        self._log_path = path
        return self

    def set_email(
        self,
        smtp_host: str,
        port: int,
        user: str,
        password: str,
        recipients: list[str],
        use_tls: bool = True,
    ) -> "AlertEngine":
        """Send email notifications via SMTP."""
        self._email_cfg = dict(
            host=smtp_host, port=port, user=user,
            password=password, recipients=recipients, use_tls=use_tls,
        )
        return self

    def set_webhook(self, url: str) -> "AlertEngine":
        """POST alert payloads to a webhook URL (Slack, Teams, Discord, etc.)."""
        self._webhook_url = url
        return self

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> list[AlertResult]:
        """
        Check all registered alerts now and return triggered ones.

        Alerts on cooldown are skipped.
        Triggered alerts are dispatched to all configured channels.
        """
        triggered = []

        for alert in self._alerts:
            if alert.is_on_cooldown():
                continue
            try:
                result = alert.check_fn(self.quant, self.client)
                if result is not None:
                    alert.mark_triggered()
                    triggered.append(result)
                    self._dispatch(result, verbose=verbose)
            except Exception as e:
                logger.warning(f"Alert '{alert.name}' check failed: {e}")

        if verbose and not triggered:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"No alerts triggered ({len(self._alerts)} checked).")

        return triggered

    def schedule(
        self,
        interval_minutes: float = 30,
        verbose: bool = True,
    ) -> "AlertEngine":
        """
        Run all alerts every *interval_minutes* in a background thread.

        Call engine.stop() to halt the scheduler.
        """
        self._stop_event.clear()

        def _loop():
            while not self._stop_event.is_set():
                if verbose:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Running {len(self._alerts)} alerts...")
                self.run(verbose=verbose)
                self._stop_event.wait(timeout=interval_minutes * 60)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        print(f"Alert scheduler started — checking every {interval_minutes} min. "
              f"Call engine.stop() to halt.")
        return self

    def stop(self) -> None:
        """Stop the background scheduler."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        print("Alert scheduler stopped.")

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, result: AlertResult, verbose: bool = True) -> None:
        if verbose:
            print(result)

        if self._log_path:
            self._log_to_file(result)

        if self._email_cfg:
            self._send_email(result)

        if self._webhook_url:
            self._send_webhook(result)

    def _log_to_file(self, result: AlertResult) -> None:
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_dict()) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log alert: {e}")

    def _send_email(self, result: AlertResult) -> None:
        cfg  = self._email_cfg
        body = str(result)
        msg  = MIMEText(body)
        msg["Subject"] = f"[Alert] {result.alert_name} — {result.symbol}"
        msg["From"]    = cfg["user"]
        msg["To"]      = ", ".join(cfg["recipients"])
        try:
            with smtplib.SMTP(cfg["host"], cfg["port"]) as smtp:
                if cfg["use_tls"]:
                    smtp.starttls()
                smtp.login(cfg["user"], cfg["password"])
                smtp.sendmail(cfg["user"], cfg["recipients"], msg.as_string())
        except Exception as e:
            logger.warning(f"Failed to send email alert: {e}")

    def _send_webhook(self, result: AlertResult) -> None:
        try:
            payload = {
                "text": str(result),
                "alert": result.to_dict(),
            }
            requests.post(self._webhook_url, json=payload, timeout=10)
        except Exception as e:
            logger.warning(f"Failed to send webhook alert: {e}")


# ---------------------------------------------------------------------------
# Built-in alert factories
# ---------------------------------------------------------------------------

def price_change(
    symbol: str,
    pct: float,
    direction: Literal["up", "down"] = "down",
    cooldown_min: int = 60,
) -> Alert:
    """
    Alert when a stock moves more than *pct* intraday.

    direction : "up"   → alert when price rises > pct  (e.g. gap up)
                "down" → alert when price falls > pct  (e.g. crash)
    pct       : fraction (e.g. 0.05 = 5%)

    Example
    -------
    engine.add(price_change("NVDA", pct=0.05, direction="up"))
    engine.add(price_change("AAPL", pct=0.05, direction="down"))
    """
    name = f"Price {'▲' if direction == 'up' else '▼'} {pct:.0%} — {symbol}"

    def check(quant: QuantAnalytics, client: StockClient) -> AlertResult | None:
        data   = client.get_price_change(symbol)
        change = data["change_pct"] / 100

        triggered = (
            (direction == "up"   and change >= pct) or
            (direction == "down" and change <= -pct)
        )
        if not triggered:
            return None

        severity = "critical" if abs(change) >= pct * 2 else "warning"
        return AlertResult(
            alert_name=name,
            symbol=symbol,
            message=(f"{symbol} moved {change:+.2%} today "
                     f"({'gained' if change > 0 else 'lost'} "
                     f"${abs(data['change']):.2f})"),
            value=change,
            threshold=pct if direction == "up" else -pct,
            direction=direction,
            severity=severity,
            extra={"price": data["price"],
                   "previous_close": data.get("previous_close")},
        )

    return Alert(name=name, symbol=symbol, check_fn=check,
                 cooldown_min=cooldown_min, tags=["price"])


def metric_threshold(
    symbol: str,
    metric: str,
    value: float,
    direction: Literal["above", "below"] = "above",
    period: str = "1y",
    risk_free_rate: float = 0.05,
    cooldown_min: int = 240,
) -> Alert:
    """
    Alert when a quant metric crosses a threshold.

    metric    : "sharpe", "sortino", "volatility", "max_drawdown",
                "beta", "var", "cvar", "calmar"
    direction : "above" → alert when metric > value
                "below" → alert when metric < value

    Example
    -------
    engine.add(metric_threshold("TSLA", "volatility", 0.80, "above"))
    engine.add(metric_threshold("SPY",  "sharpe",     1.0,  "below"))
    """
    _METRIC_FNS = {
        "sharpe":      lambda q, s, p, rf: q.sharpe_ratio(s, p, rf),
        "sortino":     lambda q, s, p, rf: q.sortino_ratio(s, p, rf),
        "volatility":  lambda q, s, p, rf: q.historical_volatility(s, p),
        "max_drawdown":lambda q, s, p, rf: q.max_drawdown(s, p),
        "beta":        lambda q, s, p, rf: q.beta(s, period=p),
        "calmar":      lambda q, s, p, rf: q.calmar_ratio(s, p),
        "var":         lambda q, s, p, rf: q.var(s, p),
        "cvar":        lambda q, s, p, rf: q.cvar(s, p),
    }

    if metric not in _METRIC_FNS:
        raise ValueError(f"Unknown metric '{metric}'. "
                         f"Available: {list(_METRIC_FNS)}")

    name = f"{metric.capitalize()} {direction} {value:.2f} — {symbol}"

    def check(quant: QuantAnalytics, client: StockClient) -> AlertResult | None:
        current = _METRIC_FNS[metric](quant, symbol, period, risk_free_rate)
        triggered = (
            (direction == "above" and current >= value) or
            (direction == "below" and current <= value)
        )
        if not triggered:
            return None

        return AlertResult(
            alert_name=name,
            symbol=symbol,
            message=(f"{symbol} {metric} is {current:.3f}, "
                     f"which is {direction} threshold {value:.3f}"),
            value=current,
            threshold=value,
            direction=direction,
            severity="warning",
            extra={"metric": metric, "period": period},
        )

    return Alert(name=name, symbol=symbol, check_fn=check,
                 cooldown_min=cooldown_min, tags=["metric", metric])


def volatility_spike(
    symbol: str,
    threshold: float = 0.40,
    period: str = "1y",
    cooldown_min: int = 120,
) -> Alert:
    """
    Alert when annualised volatility spikes above *threshold*.

    Example
    -------
    engine.add(volatility_spike("SPY", threshold=0.25))
    engine.add(volatility_spike("TSLA", threshold=0.80))
    """
    name = f"Volatility spike >{threshold:.0%} — {symbol}"

    def check(quant: QuantAnalytics, client: StockClient) -> AlertResult | None:
        vol = quant.historical_volatility(symbol, period=period)
        if vol < threshold:
            return None

        severity = "critical" if vol >= threshold * 1.5 else "warning"
        return AlertResult(
            alert_name=name,
            symbol=symbol,
            message=(f"{symbol} annualised volatility is {vol:.1%}, "
                     f"above spike threshold {threshold:.1%}"),
            value=vol,
            threshold=threshold,
            direction="above",
            severity=severity,
            extra={"period": period},
        )

    return Alert(name=name, symbol=symbol, check_fn=check,
                 cooldown_min=cooldown_min, tags=["volatility", "risk"])


def new_52w_high_low(
    symbol: str,
    watch: Literal["both", "high", "low"] = "both",
    cooldown_min: int = 1440,   # 1 day default — don't spam on sustained trend
) -> Alert:
    """
    Alert when a stock hits a new 52-week high or low.

    watch : "high" → only alert on 52w high
            "low"  → only alert on 52w low
            "both" → alert on either
    cooldown_min : default 1440 (1 day) — one alert per day max

    Example
    -------
    engine.add(new_52w_high_low("AAPL"))
    engine.add(new_52w_high_low("MSFT", watch="high"))
    """
    name = f"52w {'high/low' if watch == 'both' else watch} — {symbol}"

    def check(quant: QuantAnalytics, client: StockClient) -> AlertResult | None:
        info    = client.get_info(symbol)
        price   = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        high_52 = info.get("fiftyTwoWeekHigh", float("inf"))
        low_52  = info.get("fiftyTwoWeekLow",  0)

        hit_high = watch in ("both", "high") and price >= high_52 * 0.999
        hit_low  = watch in ("both", "low")  and price <= low_52  * 1.001

        if not hit_high and not hit_low:
            return None

        direction = "new_high" if hit_high else "new_low"
        threshold = high_52 if hit_high else low_52
        severity  = "info" if hit_high else "warning"

        return AlertResult(
            alert_name=name,
            symbol=symbol,
            message=(f"{symbol} hit a new 52-week "
                     f"{'HIGH' if hit_high else 'LOW'} "
                     f"at ${price:.2f} "
                     f"(52w range: ${low_52:.2f} – ${high_52:.2f})"),
            value=price,
            threshold=threshold,
            direction=direction,
            severity=severity,
            extra={"52w_high": high_52, "52w_low": low_52},
        )

    return Alert(name=name, symbol=symbol, check_fn=check,
                 cooldown_min=cooldown_min, tags=["price", "52w"])


def drawdown_alert(
    symbol: str,
    threshold: float = -0.15,
    period: str = "1y",
    cooldown_min: int = 480,
) -> Alert:
    """
    Alert when max drawdown exceeds *threshold* (negative number).

    threshold : e.g. -0.15 = alert when drawdown exceeds -15%

    Example
    -------
    engine.add(drawdown_alert("QQQ", threshold=-0.15))
    engine.add(drawdown_alert("TSLA", threshold=-0.30))
    """
    name = f"Drawdown >{abs(threshold):.0%} — {symbol}"

    def check(quant: QuantAnalytics, client: StockClient) -> AlertResult | None:
        dd = quant.max_drawdown(symbol, period=period)
        if dd > threshold:
            return None

        severity = "critical" if dd <= threshold * 1.5 else "warning"
        return AlertResult(
            alert_name=name,
            symbol=symbol,
            message=(f"{symbol} max drawdown is {dd:.1%}, "
                     f"exceeding threshold {threshold:.1%}"),
            value=dd,
            threshold=threshold,
            direction="below",
            severity=severity,
            extra={"period": period},
        )

    return Alert(name=name, symbol=symbol, check_fn=check,
                 cooldown_min=cooldown_min, tags=["drawdown", "risk"])


def kelly_edge_lost(
    symbol: str,
    period: str = "2y",
    risk_free_rate: float = 0.05,
    cooldown_min: int = 1440,
) -> Alert:
    """
    Alert when Kelly fraction drops below zero — the stock has lost
    its statistical edge over the risk-free rate.

    Example
    -------
    engine.add(kelly_edge_lost("NVDA"))
    """
    name = f"Kelly edge lost — {symbol}"

    def check(quant: QuantAnalytics, client: StockClient) -> AlertResult | None:
        k = quant.kelly(symbol, period=period,
                        risk_free_rate=risk_free_rate)
        if k["full_kelly"] >= 0:
            return None

        return AlertResult(
            alert_name=name,
            symbol=symbol,
            message=(f"{symbol} Kelly fraction is {k['full_kelly']:.3f} "
                     f"(negative = no edge over risk-free rate). "
                     f"μ={k['mu_annual']:.2%}, σ={k['sigma_annual']:.2%}"),
            value=k["full_kelly"],
            threshold=0.0,
            direction="below",
            severity="warning",
            extra=k,
        )

    return Alert(name=name, symbol=symbol, check_fn=check,
                 cooldown_min=cooldown_min, tags=["kelly", "edge"])


def ma_crossover_alert(
    symbol: str,
    fast: int = 20,
    slow: int = 50,
    cooldown_min: int = 1440,
) -> Alert:
    """
    Alert when the fast MA crosses above or below the slow MA.

    Golden cross (fast > slow): bullish signal → severity "info"
    Death cross  (fast < slow): bearish signal → severity "warning"

    Example
    -------
    engine.add(ma_crossover_alert("SPY", fast=50, slow=200))
    """
    name = f"MA {fast}/{slow} crossover — {symbol}"

    def check(quant: QuantAnalytics, client: StockClient) -> AlertResult | None:
        prices   = quant._prices(symbol, period="1y")
        fast_ma  = prices.rolling(fast).mean()
        slow_ma  = prices.rolling(slow).mean()

        # check last two bars for a crossover
        if len(fast_ma) < 2 or len(slow_ma) < 2:
            return None

        prev_above = fast_ma.iloc[-2] > slow_ma.iloc[-2]
        curr_above = fast_ma.iloc[-1] > slow_ma.iloc[-1]

        if prev_above == curr_above:
            return None   # no crossover

        golden = curr_above   # fast crossed above slow
        return AlertResult(
            alert_name=name,
            symbol=symbol,
            message=(
                f"{symbol} {'Golden' if golden else 'Death'} Cross: "
                f"MA{fast} ({fast_ma.iloc[-1]:.2f}) "
                f"{'crossed above' if golden else 'crossed below'} "
                f"MA{slow} ({slow_ma.iloc[-1]:.2f})"
            ),
            value=float(fast_ma.iloc[-1]),
            threshold=float(slow_ma.iloc[-1]),
            direction="above" if golden else "below",
            severity="info" if golden else "warning",
            extra={"fast": fast, "slow": slow,
                   "fast_ma": float(fast_ma.iloc[-1]),
                   "slow_ma": float(slow_ma.iloc[-1])},
        )

    return Alert(name=name, symbol=symbol, check_fn=check,
                 cooldown_min=cooldown_min, tags=["ma", "crossover"])
