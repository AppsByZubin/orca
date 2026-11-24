"""Chain-snapshot sources other than the local database.

ORCA's own `snapshot_job` writes chains to `chain_snapshots`/`option_quotes`,
but that table is only a convenience: anything that can yield
`list[ChainSnapshot]` can feed the ML pipeline, because `build_training_frame`
works on domain objects and never touches a session itself.

This module reads chains from Parquet - local files, a directory tree, or an
S3 prefix (`s3://bucket/key`, needs the `parquet` extra):

    orca train --parquet s3://my-bucket/option-chains/ --since 2026-09-01

Column names vary by whoever wrote the files, so `infer_columns()` matches a
set of common aliases and `describe_parquet()` prints what it found. Pass an
explicit mapping to override.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd

from orca.domain import ChainSnapshot, OptionQuote, OptionType
from orca.logging_setup import get_logger

log = get_logger(__name__)

# Candidate column names, in priority order, for each field we need.
ALIASES: dict[str, tuple[str, ...]] = {
    "ts": ("ts", "timestamp", "time", "datetime", "captured_at", "snapshot_ts", "bar_time"),
    "underlying": ("underlying", "symbol", "name", "index", "underlying_symbol"),
    "expiry": ("expiry", "expiry_date", "exp", "expirydate"),
    "strike": ("strike", "strike_price", "strikeprice"),
    "option_type": ("option_type", "instrument_type", "type", "opt_type", "right", "ce_pe"),
    "spot": ("spot", "underlying_spot", "spot_price", "underlying_value", "index_price"),
    "ltp": ("ltp", "last_price", "close", "last_traded_price", "price"),
    "bid": ("bid", "bid_price", "best_bid", "buy_price"),
    "ask": ("ask", "ask_price", "best_ask", "sell_price"),
    "volume": ("volume", "vol", "traded_volume", "qty"),
    "oi": ("oi", "open_interest", "openinterest"),
    "oi_change": ("oi_change", "change_in_oi", "oi_chg", "changeinopeninterest"),
    "iv": ("iv", "implied_volatility", "impliedvolatility", "implied_vol"),
    "delta": ("delta",),
    "gamma": ("gamma",),
    "theta": ("theta",),
    "vega": ("vega",),
    "futures_price": ("futures_price", "fut_price", "future_price", "fut"),
    "instrument_key": ("instrument_key", "instrument_token", "token"),
    "tradingsymbol": ("tradingsymbol", "trading_symbol", "symbol_name", "contract"),
}

REQUIRED = ("ts", "underlying", "expiry", "strike", "option_type", "spot", "ltp")
# Without delta, strategy templates cannot pick strikes (nearest_by_delta).
STRONGLY_ADVISED = ("delta", "iv")


@dataclass
class ColumnMap:
    """Resolved mapping of our field name -> the column name in the file."""

    mapping: dict[str, str]

    def get(self, field: str) -> str | None:
        return self.mapping.get(field)

    def missing(self, fields: Sequence[str]) -> list[str]:
        return [f for f in fields if f not in self.mapping]


def infer_columns(columns: Sequence[str]) -> ColumnMap:
    """Match the file's columns against ALIASES, case- and separator-insensitive."""

    def norm(s: str) -> str:
        return s.lower().replace("_", "").replace(" ", "").replace("-", "")

    available = {norm(c): c for c in columns}
    resolved: dict[str, str] = {}
    for field, aliases in ALIASES.items():
        for alias in aliases:
            hit = available.get(norm(alias))
            if hit is not None:
                resolved[field] = hit
                break
    return ColumnMap(resolved)


def _read(
    source: str, columns: Sequence[str] | None = None, storage_options: dict | None = None
) -> pd.DataFrame:
    return pd.read_parquet(
        source,
        columns=list(columns) if columns else None,
        storage_options=storage_options or None,
    )


def describe_parquet(source: str, storage_options: dict | None = None) -> dict[str, Any]:
    """Inspect a Parquet source and report how its columns map onto ORCA's fields."""
    import pyarrow.dataset as ds

    dataset = ds.dataset(source, format="parquet", partitioning="hive")
    names = list(dataset.schema.names)
    cmap = infer_columns(names)
    return {
        "source": source,
        "columns": names,
        "dtypes": {f.name: str(f.type) for f in dataset.schema},
        "mapped": cmap.mapping,
        "unmapped_columns": sorted(set(names) - set(cmap.mapping.values())),
        "missing_required": cmap.missing(REQUIRED),
        "missing_advised": cmap.missing(STRONGLY_ADVISED),
    }


def _as_option_type(value: Any) -> OptionType | None:
    s = str(value).strip().upper()
    if s in ("CE", "C", "CALL"):
        return OptionType.CE
    if s in ("PE", "P", "PUT"):
        return OptionType.PE
    return None


def _as_date(value: Any) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    ts = pd.Timestamp(value)
    return None if pd.isna(ts) else ts.date()


def _f(row: Any, col: str | None) -> float | None:
    if col is None:
        return None
    v = row.get(col)
    if v is None or pd.isna(v):
        return None
    return float(v)


def _i(row: Any, col: str | None) -> int:
    v = _f(row, col)
    return 0 if v is None else int(v)


def load_parquet_snapshots(
    source: str,
    underlying: str | None = None,
    since: date | None = None,
    until: date | None = None,
    columns: dict[str, str] | None = None,
    storage_options: dict | None = None,
    iv_as_percent: bool | None = None,
) -> Iterator[ChainSnapshot]:
    """Yield one ChainSnapshot per (underlying, expiry, ts) group, in time order.

    A generator on purpose: a year of 5-minute chains does not fit in memory as
    ORM rows, and the labeller only needs one (symbol, expiry) series at a time.

    `iv_as_percent=None` auto-detects: a median IV above 1.5 is read as percent
    and divided by 100, since ORCA stores IV as a decimal (0.14 == 14%).
    """
    df = _read(source, storage_options=storage_options)
    if df.empty:
        log.warning("parquet source %s produced no rows", source)
        return

    cmap = ColumnMap(columns) if columns else infer_columns(df.columns)
    missing = cmap.missing(REQUIRED)
    if missing:
        raise ValueError(
            f"parquet source is missing required field(s) {missing}; "
            f"columns present: {sorted(df.columns)}. "
            f"Pass columns={{'<field>': '<column>'}} to map them explicitly."
        )
    for field in cmap.missing(STRONGLY_ADVISED):
        log.warning(
            "parquet source has no %r column - strategy templates need delta to "
            "pick strikes, and features need iv",
            field,
        )

    c_ts, c_u = cmap.get("ts"), cmap.get("underlying")
    c_exp, c_strike, c_type = cmap.get("expiry"), cmap.get("strike"), cmap.get("option_type")
    c_spot = cmap.get("spot")

    df = df.copy()
    df[c_ts] = pd.to_datetime(df[c_ts], errors="coerce", utc=True)
    df = df.dropna(subset=[c_ts])
    if underlying:
        df = df[df[c_u].astype(str).str.upper() == underlying.upper()]
    if since:
        df = df[df[c_ts] >= pd.Timestamp(since, tz="UTC")]
    if until:
        df = df[df[c_ts] <= pd.Timestamp(until, tz="UTC")]
    if df.empty:
        log.warning("parquet source %s: no rows after filtering", source)
        return

    c_iv = cmap.get("iv")
    scale_iv = 1.0
    if c_iv is not None:
        med = pd.to_numeric(df[c_iv], errors="coerce").dropna().median()
        as_pct = iv_as_percent if iv_as_percent is not None else bool(med and med > 1.5)
        if as_pct:
            scale_iv = 0.01
            log.info("parquet iv median %.2f - treating iv as percent, scaling by 1/100", med)

    n_snapshots = 0
    for (sym, expiry_raw, ts), group in df.groupby([c_u, c_exp, c_ts], sort=True):
        expiry = _as_date(expiry_raw)
        if expiry is None:
            continue
        options: list[OptionQuote] = []
        for _, row in group.iterrows():
            otype = _as_option_type(row.get(c_type))
            if otype is None:
                continue
            raw_iv = _f(row, c_iv)
            options.append(
                OptionQuote(
                    underlying=str(sym),
                    expiry=expiry,
                    strike=float(row[c_strike]),
                    option_type=otype,
                    instrument_key=str(row.get(cmap.get("instrument_key") or "", "") or ""),
                    tradingsymbol=str(row.get(cmap.get("tradingsymbol") or "", "") or ""),
                    ltp=_f(row, cmap.get("ltp")) or 0.0,
                    bid=_f(row, cmap.get("bid")),
                    ask=_f(row, cmap.get("ask")),
                    volume=_i(row, cmap.get("volume")),
                    oi=_i(row, cmap.get("oi")),
                    oi_change=_i(row, cmap.get("oi_change")),
                    iv=None if raw_iv is None else raw_iv * scale_iv,
                    delta=_f(row, cmap.get("delta")),
                    gamma=_f(row, cmap.get("gamma")),
                    theta=_f(row, cmap.get("theta")),
                    vega=_f(row, cmap.get("vega")),
                )
            )
        if not options:
            continue
        first = group.iloc[0]
        n_snapshots += 1
        yield ChainSnapshot(
            underlying=str(sym),
            expiry=expiry,
            spot=float(first[c_spot]),
            ts=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            options=options,
            futures_price=_f(first, cmap.get("futures_price")),
            source="parquet",
        )
    log.info("parquet source %s: yielded %d snapshots", source, n_snapshots)


def daily_closes_from_snapshots(
    snapshots: Sequence[ChainSnapshot],
) -> dict[str, list[tuple[date, float]]]:
    """Per-symbol daily closes derived from the snapshots themselves.

    The training path has no feed to ask for candles, and without these the
    regime features (realized_vol_20, trend_*, spot_vs_ema20, gap_pct) are all
    computed as zero while the live scanner computes them for real - a
    train/serve skew. Close = the last spot seen on that date.
    """
    latest: dict[tuple[str, date], tuple[datetime, float]] = {}
    for snap in snapshots:
        key = (snap.underlying, snap.ts.date())
        prev = latest.get(key)
        if prev is None or snap.ts > prev[0]:
            latest[key] = (snap.ts, snap.spot)

    out: dict[str, list[tuple[date, float]]] = {}
    for (sym, d), (_ts, spot) in latest.items():
        out.setdefault(sym, []).append((d, spot))
    for sym in out:
        out[sym].sort(key=lambda pair: pair[0])
    return out


def daily_atm_iv_from_snapshots(
    snapshots: Sequence[ChainSnapshot],
) -> dict[str, list[float]]:
    """Per-symbol daily ATM-IV closes, mirroring what `update_daily` stores.

    The live scanner derives iv_rank/iv_percentile from `underlying_daily`
    (one ATM IV per day). Ranking against within-session snapshots instead
    would give the model a different notion of "IV rank" at train time than at
    score time, so the training path derives the same daily series.
    """
    latest: dict[tuple[str, date], tuple[datetime, float]] = {}
    for snap in snapshots:
        iv = snap.atm_iv()
        if not iv:
            continue
        key = (snap.underlying, snap.ts.date())
        prev = latest.get(key)
        if prev is None or snap.ts > prev[0]:
            latest[key] = (snap.ts, iv)

    by_symbol: dict[str, list[tuple[date, float]]] = {}
    for (sym, d), (_ts, iv) in latest.items():
        by_symbol.setdefault(sym, []).append((d, iv))
    return {
        sym: [iv for _d, iv in sorted(pairs, key=lambda pair: pair[0])]
        for sym, pairs in by_symbol.items()
    }
