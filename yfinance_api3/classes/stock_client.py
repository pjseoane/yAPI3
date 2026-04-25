"""
StockClient — yfinance wrapper with TTL caching.
"""

import functools
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any

import yfinance as yf


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

class _CacheEntry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl: float) -> None:
        self.value = value
        self.expires_at = time.monotonic() + ttl


class TTLCache:
    """Simple in-memory TTL cache."""

    def __init__(self, default_ttl: float = 60.0) -> None:
        self._store: dict[str, _CacheEntry] = {}
        self.default_ttl = default_ttl

    # ------------------------------------------------------------------
    def _make_key(self, namespace: str, *args, **kwargs) -> str:
        payload = json.dumps({"a": args, "k": kwargs}, sort_keys=True, default=str)
        digest = hashlib.md5(payload.encode()).hexdigest()
        return f"{namespace}:{digest}"

    def get(self, key: str) -> tuple[bool, Any]:
        entry = self._store.get(key)
        if entry is None or time.monotonic() > entry.expires_at:
            self._store.pop(key, None)
            return False, None
        return True, entry.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        self._store[key] = _CacheEntry(value, ttl if ttl is not None else self.default_ttl)

    def invalidate(self, prefix: str | None = None) -> int:
        """Remove all entries (or those whose key starts with *prefix*)."""
        if prefix is None:
            count = len(self._store)
            self._store.clear()
            return count
        keys = [k for k in self._store if k.startswith(prefix)]
        for k in keys:
            del self._store[k]
        return len(keys)

    def stats(self) -> dict:
        now = time.monotonic()
        alive = sum(1 for e in self._store.values() if e.expires_at > now)
        return {"total_entries": len(self._store), "alive": alive, "expired": len(self._store) - alive}


# ---------------------------------------------------------------------------
# Decorator factory (used internally by StockClient)
# ---------------------------------------------------------------------------

def _cached(namespace: str, ttl_attr: str):
    """Decorator that caches the result of a StockClient method."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            key = self._cache._make_key(namespace, *args, **kwargs)
            hit, value = self._cache.get(key)
            if hit:
                return value
            result = fn(self, *args, **kwargs)
            self._cache.set(key, result, ttl=getattr(self, ttl_attr))
            return result
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class StockClient:
    """
    yfinance wrapper with per-method TTL caching.

    TTL settings (seconds)
    ----------------------
    ttl_price   : real-time / fast-changing data  (default  60 s)
    ttl_info    : company metadata                (default 600 s)
    ttl_history : historical OHLCV bars           (default 300 s)
    ttl_news    : news headlines                  (default 120 s)
    """

    def __init__(
        self,
        ttl_price: float = 60.0,
        ttl_info: float = 600.0,
        ttl_history: float = 300.0,
        ttl_news: float = 120.0,
    ) -> None:
        self.ttl_price = ttl_price
        self.ttl_info = ttl_info
        self.ttl_history = ttl_history
        self.ttl_news = ttl_news
        self._cache = TTLCache()
        self._tickers: dict[str, yf.Ticker] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ticker(self, symbol: str) -> yf.Ticker:
        symbol = symbol.upper()
        if symbol not in self._tickers:
            self._tickers[symbol] = yf.Ticker(symbol)
        return self._tickers[symbol]

    # ------------------------------------------------------------------
    # Price / quote
    # ------------------------------------------------------------------

    @_cached("price", "ttl_price")
    def get_price(self, symbol: str) -> dict:
        """
        Return the latest price data for *symbol*.

        Returns a dict with keys: price, previous_close, open, day_high,
        day_low, volume, market_cap, currency, timestamp.
        """
        t = self._ticker(symbol)
        fast = t.fast_info        # lightweight; no full info call

        return {
            "symbol": symbol.upper(),
            "price": fast.last_price,
            "previous_close": fast.previous_close,
            "open": fast.open,
            "day_high": fast.day_high,
            "day_low": fast.day_low,
            "volume": fast.three_month_average_volume,
            "market_cap": fast.market_cap,
            "currency": fast.currency,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @_cached("price_change", "ttl_price")
    def get_price_change(self, symbol: str) -> dict:
        """Return the day's absolute and percentage change."""
        data = self.get_price(symbol)
        prev = data["previous_close"] or 0
        price = data["price"] or 0
        change = price - prev
        pct = (change / prev * 100) if prev else 0.0
        return {
            "symbol": data["symbol"],
            "price": price,
            "change": round(change, 4),
            "change_pct": round(pct, 4),
        }

    # ------------------------------------------------------------------
    # Company info
    # ------------------------------------------------------------------

    @_cached("info", "ttl_info")
    def get_info(self, symbol: str) -> dict:
        """Return full company metadata from yfinance."""
        return self._ticker(symbol).info

    @_cached("summary", "ttl_info")
    def get_summary(self, symbol: str) -> dict:
        """Return a concise company profile."""
        info = self.get_info(symbol)
        keys = (
            "shortName", "longName", "sector", "industry",
            "country", "website", "fullTimeEmployees",
            "longBusinessSummary", "currency",
        )
        return {k: info.get(k) for k in keys}

    # ------------------------------------------------------------------
    # Financials
    # ------------------------------------------------------------------

    @_cached("financials", "ttl_info")
    def get_financials(self, symbol: str) -> dict:
        """Return income statement, balance sheet, and cash-flow data."""
        t = self._ticker(symbol)
        return {
            "income_statement": t.financials.to_dict() if t.financials is not None else {},
            "balance_sheet": t.balance_sheet.to_dict() if t.balance_sheet is not None else {},
            "cash_flow": t.cashflow.to_dict() if t.cashflow is not None else {},
        }

    @_cached("dividends", "ttl_info")
    def get_dividends(self, symbol: str) -> list[dict]:
        """Return dividend history as a list of {date, dividend} dicts."""
        divs = self._ticker(symbol).dividends
        if divs is None or divs.empty:
            return []
        return [{"date": str(d.date()), "dividend": v} for d, v in divs.items()]

    # ------------------------------------------------------------------
    # Historical price data
    # ------------------------------------------------------------------

    # Period ordering — larger rank = longer history
    # Note: yfinance only supports up to "10y"; anything beyond should be "max"
    _PERIOD_RANK: dict = {
        "1d": 0, "5d": 1, "1mo": 2, "3mo": 3, "6mo": 4,
        "1y": 5, "2y": 6, "3y": 7, "5y": 8, "10y": 9,
        "ytd": 4, "max": 99,
    }

    # Days per period for cache slicing — extended to handle common strings
    _PERIOD_DAYS: dict = {
        "1d": 1, "5d": 5, "1mo": 30, "3mo": 91, "6mo": 182,
        "ytd": 365, "1y": 365, "2y": 730, "3y": 1095,
        "5y": 1825, "10y": 3650, "max": 999_999,
    }

    def get_history(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d",
        adjusted: bool = True,
    ) -> list[dict]:
        """
        Return OHLCV bars.

        period   : 1d 5d 1mo 3mo 6mo 1y 2y 3y 5y 10y ytd max
        interval : 1m 2m 5m 15m 30m 60m 90m 1h 1d 5d 1wk 1mo 3mo
        adjusted : True  -> adj_close (split + dividend adjusted). Default.
                   False -> raw close (actual traded price).

        Cache behaviour
        ---------------
        If a longer period for the same symbol/interval/adjusted combination
        is already cached, the result is sliced from it — no extra fetch.
        Requesting "2y" after "5y" costs zero API calls.
        """
        requested_rank = self._PERIOD_RANK.get(period, -1)

        # check if a longer cached period already covers this request
        for cached_period, rank in sorted(
            self._PERIOD_RANK.items(), key=lambda x: x[1], reverse=True
        ):
            if rank <= requested_rank:
                break
            ck = self._cache._make_key("history", symbol, cached_period, interval, adjusted)
            hit, cached_data = self._cache.get(ck)
            if hit and cached_data:
                sliced = self._slice_to_period(cached_data, period)
                if sliced:
                    return sliced

        # normal cache check for exact period
        ck = self._cache._make_key("history", symbol, period, interval, adjusted)
        hit, cached_data = self._cache.get(ck)
        if hit:
            return cached_data

        # fetch from yfinance
        df = self._ticker(symbol).history(
            period=period, interval=interval, auto_adjust=adjusted
        )
        if df is None or df.empty:
            return []
        df = df.reset_index()
        date_col    = "Datetime" if "Datetime" in df.columns else "Date"
        close_label = "adj_close" if adjusted else "close"
        result = (
            df[[date_col, "Open", "High", "Low", "Close", "Volume"]]
            .rename(columns={
                date_col: "date",
                "Open":   "open",
                "High":   "high",
                "Low":    "low",
                "Close":  close_label,
                "Volume": "volume",
            })
            .assign(adjusted=adjusted)
            .to_dict(orient="records")
        )
        self._cache.set(ck, result, ttl=self.ttl_history)
        return result

    @staticmethod
    def _slice_to_period(data: list, period: str) -> list:
        """Slice cached data to approximately cover *period* from the most recent bar."""
        from datetime import datetime, timezone, timedelta

        _period_days = {
            "1d": 1, "5d": 5, "1mo": 30, "3mo": 91, "6mo": 182,
            "1y": 365, "2y": 730, "3y": 1095, "5y": 1825,
            "10y": 3650, "ytd": 365, "max": 999_999,
        }
        n_days = _period_days.get(period, 365)
        if not data:
            return data

        def _parse(d):
            if isinstance(d, datetime):
                return d.replace(tzinfo=None) if d.tzinfo else d
            return datetime.fromisoformat(str(d).split("+")[0].split("T")[0])

        last_date = _parse(data[-1]["date"])
        cutoff    = last_date - timedelta(days=n_days)
        return [row for row in data if _parse(row["date"]) >= cutoff]

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------

    @_cached("news", "ttl_news")
    def get_news(self, symbol: str, limit: int = 10) -> list[dict]:
        """Return recent news articles for *symbol*."""
        raw = self._ticker(symbol).news or []
        out = []
        for item in raw[:limit]:
            content = item.get("content", {})
            out.append({
                "title": content.get("title"),
                "summary": content.get("summary"),
                "url": content.get("canonicalUrl", {}).get("url"),
                "publisher": content.get("provider", {}).get("displayName"),
                "published_at": content.get("pubDate"),
            })
        return out

    # ------------------------------------------------------------------
    # Analyst recommendations
    # ------------------------------------------------------------------

    @_cached("recommendations", "ttl_info")
    def get_recommendations(self, symbol: str) -> list[dict]:
        """Return latest analyst recommendations."""
        df = self._ticker(symbol).recommendations
        if df is None or df.empty:
            return []
        df = df.reset_index()
        return df.head(20).to_dict(orient="records")

    # ------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------

    @_cached("options_expiries", "ttl_price")
    def get_options_expiries(self, symbol: str) -> list[str]:
        """Return available options expiration dates."""
        return list(self._ticker(symbol).options or [])

    @_cached("options_chain", "ttl_price")
    def get_options_chain(self, symbol: str, expiry: str) -> dict:
        """Return calls and puts for a specific expiry date."""
        chain = self._ticker(symbol).option_chain(expiry)
        return {
            "calls": chain.calls.to_dict(orient="records"),
            "puts": chain.puts.to_dict(orient="records"),
        }

    # ------------------------------------------------------------------
    # Multiple symbols at once
    # ------------------------------------------------------------------

    def get_prices_bulk(self, symbols: list[str]) -> dict[str, dict]:
        """Fetch latest price for multiple tickers (each result cached individually)."""
        return {sym: self.get_price(sym) for sym in symbols}

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def cache_stats(self) -> dict:
        """Return current cache statistics."""
        return self._cache.stats()

    def invalidate(self, symbol: str | None = None) -> int:
        """
        Clear cached data.
        Pass a symbol to clear only that ticker's entries,
        or nothing to wipe the entire cache.
        """
        prefix = symbol.upper() if symbol else None
        return self._cache.invalidate(prefix)
