"""Market-data orchestration: fetch, persist, and summarise the regime.

Persisting every chain snapshot is what makes the ML layer possible later -
there is no public history of intraday option greeks, so you have to collect
your own from day one.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from orca.config import Underlying, get_universe
from orca.domain import ChainSnapshot
from orca.feeds import MarketDataProvider, get_feed
from orca.logging_setup import get_logger
from orca.ml.features import market_context
from orca.models import ChainSnapshot as ChainSnapshotRow
from orca.models import OptionQuoteRow, UnderlyingDaily

log = get_logger(__name__)


class MarketDataService:
    def __init__(self, feed: MarketDataProvider | None = None) -> None:
        self.feed = feed or get_feed()

    # -- expiries --------------------------------------------------------
    def expiries(self, underlying: Underlying) -> list[date]:
        try:
            expiries = self.feed.get_expiries(underlying)
        except Exception as exc:  # noqa: BLE001 - never let the UI die on a feed hiccup
            log.warning("expiry lookup failed for %s: %s", underlying.symbol, exc)
            expiries = []
        if expiries:
            return expiries
        # Fallback: the configured weekly weekday (see config/instruments.yaml).
        today = date.today()
        delta = (underlying.weekly_expiry_weekday - today.weekday()) % 7
        first = today + timedelta(days=delta)
        return [first + timedelta(days=7 * i) for i in range(4)]

    def nearest_expiry(
        self, underlying: Underlying, min_dte: int = 0, max_dte: int = 10
    ) -> date | None:
        today = date.today()
        for expiry in self.expiries(underlying):
            dte = (expiry - today).days
            if min_dte <= dte <= max_dte:
                return expiry
        return None

    # -- chains ----------------------------------------------------------
    def chain(self, underlying: Underlying, expiry: date) -> ChainSnapshot:
        return self.feed.get_option_chain(underlying, expiry)

    def persist_chain(self, session: Session, chain: ChainSnapshot) -> ChainSnapshotRow:
        row = ChainSnapshotRow(
            underlying=chain.underlying,
            expiry=chain.expiry,
            ts=chain.ts,
            spot=chain.spot,
            futures_price=chain.futures_price,
            atm_iv=chain.atm_iv(),
            dte=chain.dte(),
            source=chain.source,
            quotes=[
                OptionQuoteRow(
                    instrument_key=o.instrument_key,
                    tradingsymbol=o.tradingsymbol,
                    strike=o.strike,
                    option_type=o.option_type.value,
                    ltp=o.ltp,
                    bid=o.bid,
                    ask=o.ask,
                    volume=o.volume,
                    oi=o.oi,
                    oi_change=o.oi_change,
                    iv=o.iv,
                    delta=o.delta,
                    gamma=o.gamma,
                    theta=o.theta,
                    vega=o.vega,
                )
                for o in chain.options
            ],
        )
        session.add(row)
        session.flush()
        return row

    def capture_all(self, session: Session, max_dte: int = 10) -> list[ChainSnapshotRow]:
        """One snapshot per enabled underlying - call this on a timer."""
        saved: list[ChainSnapshotRow] = []
        for underlying in get_universe().enabled:
            expiry = self.nearest_expiry(underlying, max_dte=max_dte)
            if expiry is None:
                log.info("%s: no expiry within %dd", underlying.symbol, max_dte)
                continue
            try:
                chain = self.chain(underlying, expiry)
            except Exception as exc:  # noqa: BLE001
                log.warning("chain fetch failed for %s %s: %s", underlying.symbol, expiry, exc)
                continue
            saved.append(self.persist_chain(session, chain))
            self.update_daily(session, underlying, chain)
        return saved

    # -- history / context ----------------------------------------------
    def update_daily(self, session: Session, underlying: Underlying, chain: ChainSnapshot) -> None:
        """Keep one row per day per symbol with the latest close + ATM IV."""
        today = chain.ts.date()
        row = session.scalar(
            select(UnderlyingDaily).where(
                UnderlyingDaily.symbol == underlying.symbol, UnderlyingDaily.d == today
            )
        )
        if row is None:
            row = UnderlyingDaily(symbol=underlying.symbol, d=today, open=chain.spot)
            session.add(row)
        row.close = chain.spot
        row.high = max(row.high or chain.spot, chain.spot)
        row.low = min(row.low or chain.spot, chain.spot)
        row.atm_iv_close = chain.atm_iv()

    def iv_history(self, session: Session, symbol: str, lookback_days: int = 60) -> list[float]:
        since = date.today() - timedelta(days=lookback_days)
        rows = session.scalars(
            select(UnderlyingDaily)
            .where(UnderlyingDaily.symbol == symbol, UnderlyingDaily.d >= since)
            .order_by(UnderlyingDaily.d)
        ).all()
        return [r.atm_iv_close for r in rows if r.atm_iv_close]

    def daily_closes(
        self, session: Session, underlying: Underlying, days: int = 60
    ) -> list[tuple[date, float]]:
        """Prefer the feed's candles; fall back to whatever we have stored."""
        try:
            closes = self.feed.get_daily_closes(underlying, days)
        except Exception as exc:  # noqa: BLE001
            log.debug("daily candles unavailable: %s", exc)
            closes = []
        if closes:
            return closes
        since = date.today() - timedelta(days=days * 2)
        rows = session.scalars(
            select(UnderlyingDaily)
            .where(UnderlyingDaily.symbol == underlying.symbol, UnderlyingDaily.d >= since)
            .order_by(UnderlyingDaily.d)
        ).all()
        return [(r.d, r.close) for r in rows if r.close]

    def context(
        self, session: Session, underlying: Underlying, chain: ChainSnapshot
    ) -> dict[str, float]:
        """Regime feature block for this snapshot."""
        return market_context(
            chain,
            daily_closes=self.daily_closes(session, underlying),
            iv_history=self.iv_history(session, underlying.symbol),
            now=datetime.now(UTC),
        )
