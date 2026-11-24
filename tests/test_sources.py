"""Parquet chain source: column inference, IV scaling, and the context fix."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from orca.analytics.greeks import black_scholes, greeks, year_fraction
from orca.domain import OptionType
from orca.ml.dataset import frame_from_snapshots
from orca.ml.features import FEATURE_NAMES
from orca.ml.sources import (
    daily_atm_iv_from_snapshots,
    daily_closes_from_snapshots,
    describe_parquet,
    infer_columns,
    load_parquet_snapshots,
)

EXPIRY = date(2026, 12, 31)


def _archive(tmp_path, sessions: int = 26, bars: int = 4, strikes: int = 6):
    """A small chain archive using non-canonical column names and percent IV."""
    rows = []
    spot = 81000.0
    d = date(2026, 11, 2)
    done = 0
    while done < sessions:
        if d.weekday() >= 5:
            d += timedelta(days=1)
            continue
        done += 1
        spot *= 1.002 if done % 2 else 0.9985
        for b in range(bars):
            ts = datetime(d.year, d.month, d.day, 10 + b, 0, tzinfo=UTC)
            t = max(year_fraction(EXPIRY, ts), 1e-6)
            atm = round(spot / 100.0) * 100.0
            for i in range(-strikes, strikes + 1):
                k = atm + i * 100.0
                iv = 0.14 + 0.5 * ((k - spot) / spot) ** 2
                for ot in ("CE", "PE"):
                    e = OptionType(ot)
                    g = greeks(spot, k, t, iv, e)
                    rows.append(
                        {
                            "timestamp": ts,
                            "symbol": "SENSEX",
                            "expiry_date": EXPIRY,
                            "strike_price": k,
                            "instrument_type": ot,
                            "underlying_value": round(spot, 2),
                            "last_price": round(black_scholes(spot, k, t, iv, e), 2),
                            "bid_price": round(black_scholes(spot, k, t, iv, e) * 0.99, 2),
                            "ask_price": round(black_scholes(spot, k, t, iv, e) * 1.01, 2),
                            "open_interest": 600_000,
                            "implied_volatility": round(iv * 100, 2),  # percent
                            "delta": g["delta"],
                            "gamma": g["gamma"],
                            "theta": g["theta"],
                            "vega": g["vega"],
                        }
                    )
        d += timedelta(days=1)
    out = tmp_path / "chains.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


def test_infer_columns_matches_aliases_case_and_separator_insensitively() -> None:
    cmap = infer_columns(
        ["TimeStamp", "Symbol", "Expiry_Date", "strikeprice", "instrument type", "last_price"]
    )
    assert cmap.get("ts") == "TimeStamp"
    assert cmap.get("underlying") == "Symbol"
    assert cmap.get("expiry") == "Expiry_Date"
    assert cmap.get("strike") == "strikeprice"
    assert cmap.get("option_type") == "instrument type"
    assert cmap.get("ltp") == "last_price"


def test_describe_parquet_reports_mapping(tmp_path) -> None:
    info = describe_parquet(str(_archive(tmp_path, sessions=2, bars=1, strikes=2)))
    assert info["missing_required"] == []
    assert info["missing_advised"] == []
    assert info["mapped"]["spot"] == "underlying_value"
    assert info["mapped"]["iv"] == "implied_volatility"


def test_percent_iv_is_rescaled_to_decimal(tmp_path) -> None:
    snaps = list(load_parquet_snapshots(str(_archive(tmp_path, sessions=2, bars=1, strikes=2))))
    assert snaps
    ivs = [o.iv for s in snaps for o in s.options if o.iv is not None]
    assert ivs, "no IV survived the load"
    # written as ~14.0 (percent); must come back as ~0.14
    assert all(0.01 < iv < 1.0 for iv in ivs), max(ivs)


def test_missing_required_column_raises(tmp_path) -> None:
    src = tmp_path / "bad.parquet"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_parquet(src, index=False)
    with pytest.raises(ValueError, match="missing required field"):
        list(load_parquet_snapshots(str(src)))


def test_snapshots_are_grouped_and_time_ordered(tmp_path) -> None:
    snaps = list(load_parquet_snapshots(str(_archive(tmp_path, sessions=3, bars=2, strikes=2))))
    assert len(snaps) == 6  # 3 sessions x 2 bars, one expiry
    assert [s.ts for s in snaps] == sorted(s.ts for s in snaps)
    assert all(s.underlying == "SENSEX" and s.expiry == EXPIRY for s in snaps)
    assert all(s.options and s.spot > 0 for s in snaps)


def test_since_and_until_filter(tmp_path) -> None:
    src = str(_archive(tmp_path, sessions=6, bars=1, strikes=2))
    everything = list(load_parquet_snapshots(src))
    window = list(load_parquet_snapshots(src, since=date(2026, 11, 4), until=date(2026, 11, 6)))
    assert 0 < len(window) < len(everything)
    assert all(date(2026, 11, 4) <= s.ts.date() <= date(2026, 11, 6) for s in window)


def test_derived_daily_series_have_one_entry_per_session(tmp_path) -> None:
    snaps = list(load_parquet_snapshots(str(_archive(tmp_path, sessions=5, bars=3, strikes=2))))
    closes = daily_closes_from_snapshots(snaps)
    ivs = daily_atm_iv_from_snapshots(snaps)
    assert len(closes["SENSEX"]) == 5, "one close per session, not per bar"
    assert len(ivs["SENSEX"]) == 5
    assert [d for d, _c in closes["SENSEX"]] == sorted(d for d, _c in closes["SENSEX"])


def test_regime_features_are_not_constant_when_history_is_derived(tmp_path) -> None:
    """Regression test for the train/serve skew.

    Without daily closes, six regime features collapse to 0.0 in every row
    while the live scanner computes them for real.
    """
    snaps = list(load_parquet_snapshots(str(_archive(tmp_path, sessions=26, bars=4))))
    derived = frame_from_snapshots(snaps, sample_every=1)
    if derived.empty:
        pytest.skip("no candidates built from the toy archive")

    starved = frame_from_snapshots(snaps, sample_every=1, daily_closes={}, iv_history={})
    skew_prone = ["realized_vol_20", "iv_rv_spread", "trend_5d", "trend_20d", "spot_vs_ema20"]
    for col in skew_prone:
        assert starved[col].nunique() == 1, f"{col} unexpectedly varied without history"
        assert derived[col].nunique() > 1, f"{col} still constant with derived history"
    assert set(FEATURE_NAMES).issubset(derived.columns)
