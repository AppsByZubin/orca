"""UpstoxProvider against a fake Upstox (httpx.MockTransport), using the
response shapes from https://upstox.com/developer/api-documentation/."""

from __future__ import annotations

from datetime import date, timedelta

import httpx
import pytest

from orca.config import get_universe
from orca.feeds.base import FeedError
from orca.feeds.upstox import UpstoxProvider, _iv_decimal

EXPIRY = date.today() + timedelta(days=5)


def ok(data) -> httpx.Response:
    return httpx.Response(200, json={"status": "success", "data": data})


def leg(key: str, ltp: float, iv: float, delta: float) -> dict:
    return {
        "instrument_key": key,
        "market_data": {
            "ltp": ltp,
            "volume": 1200,
            "oi": 50000,
            "close_price": ltp,
            "bid_price": ltp - 0.5,
            "bid_qty": 75,
            "ask_price": ltp + 0.5,
            "ask_qty": 75,
            "prev_oi": 45000,
        },
        "option_greeks": {
            "vega": 12.1,
            "theta": -9.8,
            "gamma": 0.001,
            "delta": delta,
            "iv": iv,
            "pop": 50.0,
        },
    }


class FakeUpstox:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path, q = request.url.path, request.url.params
        if path == "/v3/market-quote/ltp":
            keys = q["instrument_key"].split(",")
            prices = {
                "NSE_INDEX|Nifty 50": 26123.45,
                "BSE_INDEX|SENSEX": 85321.1,
                "NSE_FO|51834": 101.5,
            }
            names = {
                "NSE_INDEX|Nifty 50": "NSE_INDEX:Nifty 50",
                "BSE_INDEX|SENSEX": "BSE_INDEX:SENSEX",
                "NSE_FO|51834": "NSE_FO:NIFTY25D0226000CE",
            }
            return ok(
                {
                    names[k]: {
                        "last_price": prices[k],
                        "instrument_token": k,
                        "ltq": 0,
                        "volume": 0,
                        "cp": prices[k] - 50,
                    }
                    for k in keys
                    if k in prices
                }
            )
        if path == "/v2/option/contract":
            return ok(
                [
                    {"expiry": EXPIRY.isoformat(), "instrument_type": "CE", "strike_price": 26100},
                    {"expiry": EXPIRY.isoformat(), "instrument_type": "PE", "strike_price": 26100},
                    {"expiry": (EXPIRY + timedelta(days=7)).isoformat(), "instrument_type": "CE"},
                    {"expiry": "2020-01-02", "instrument_type": "CE"},  # past: dropped
                ]
            )
        if path == "/v2/option/chain":
            assert q["expiry_date"] == EXPIRY.isoformat()
            return ok(
                [
                    {
                        "expiry": EXPIRY.isoformat(),
                        "pcr": 1.1,
                        "strike_price": k,
                        "underlying_key": "NSE_INDEX|Nifty 50",
                        "underlying_spot_price": 26123.45,
                        "call_options": leg(f"NSE_FO|{k}CE", 120.0, 14.2, 0.5),
                        "put_options": leg(f"NSE_FO|{k}PE", 95.0, 15.1, -0.45),
                    }
                    for k in (26050.0, 26100.0, 26150.0)
                ]
            )
        if path.startswith("/v3/historical-candle/"):
            return ok(
                {
                    "candles": [
                        ["2026-11-25T00:00:00+05:30", 1, 2, 0.5, 26000.0, 100, 0],
                        ["2026-11-24T00:00:00+05:30", 1, 2, 0.5, 25900.0, 100, 0],
                    ]
                }
            )
        if path == "/v2/user/profile":
            return ok({"user_id": "X"})
        return httpx.Response(404, json={"status": "error", "errors": [{"message": path}]})


@pytest.fixture
def feed():
    fake = FakeUpstox()
    provider = UpstoxProvider(access_token="tok", transport=httpx.MockTransport(fake))
    provider.fake = fake  # type: ignore[attr-defined]
    yield provider
    provider.close()


def u(symbol: str):
    return get_universe().get(symbol)


def test_spot_uses_v3_ltp(feed) -> None:
    assert feed.get_spot(u("NIFTY")) == 26123.45
    assert feed.get_spot(u("SENSEX")) == 85321.1
    req = feed.fake.requests[0]
    assert req.url.path == "/v3/market-quote/ltp"
    assert req.url.params["instrument_key"] == "NSE_INDEX|Nifty 50"
    assert req.headers["Authorization"] == "Bearer tok"


def test_quotes_match_by_instrument_token(feed) -> None:
    quotes = feed.get_quotes(["NSE_FO|51834", "NSE_FO|99999"])
    assert set(quotes) == {"NSE_FO|51834"} and quotes["NSE_FO|51834"].ltp == 101.5


def test_expiries_future_only_and_cached(feed) -> None:
    assert feed.get_expiries(u("NIFTY")) == [EXPIRY, EXPIRY + timedelta(days=7)]
    feed.get_expiries(u("NIFTY"))
    assert sum(r.url.path == "/v2/option/contract" for r in feed.fake.requests) == 1


def test_option_chain_parsing(feed) -> None:
    chain = feed.get_option_chain(u("NIFTY"), EXPIRY)
    assert chain.spot == 26123.45 and chain.source == "upstox"
    assert len(chain.options) == 6 and chain.atm_strike() == 26100.0
    ce = next(o for o in chain.options if o.strike == 26100.0 and o.option_type.value == "CE")
    assert ce.iv == pytest.approx(0.142)  # percent -> decimal
    assert ce.delta == 0.5 and ce.oi_change == 5000 and ce.bid == 119.5
    assert chain.atm_iv() == pytest.approx((0.142 + 0.151) / 2)


def test_historical_uses_v3_and_encodes_key(feed) -> None:
    closes = feed.get_daily_closes(u("NIFTY"), days=10)
    assert [c for _d, c in closes] == [25900.0, 26000.0]  # oldest first
    raw = feed.fake.requests[0].url.raw_path.decode()
    assert raw.startswith("/v3/historical-candle/NSE_INDEX%7CNifty%2050/days/1/")


def test_errors_surface_as_feed_errors(monkeypatch) -> None:
    def handler(request):
        return httpx.Response(429, json={"status": "error"})

    provider = UpstoxProvider(access_token="tok", transport=httpx.MockTransport(handler))
    with pytest.raises(FeedError, match="rate limit"):
        provider.get_spot(u("NIFTY"))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(14.2, 0.142), (262.31, 2.6231), (0.15, 0.15), (0, None), (None, None), ("x", None)],
)
def test_iv_units(raw, expected) -> None:
    assert _iv_decimal(raw) == (pytest.approx(expected) if expected else None)
