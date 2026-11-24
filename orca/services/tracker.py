"""Multi-leg position tracking.

Every tick, for each open Trade:

  1. mark every open leg off the live chain (or a direct quote),
  2. aggregate P&L and greeks at the *structure* level,
  3. store a TradeSnapshot so the P&L curve and ML outcome labels exist,
  4. evaluate the exit rules and either alert or auto-exit.

Risk is deliberately read at trade level, never leg level: a strangle whose
call leg is losing 3x is fine if the put leg is paying for it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from orca.analytics.greeks import greeks as bs_greeks
from orca.analytics.greeks import implied_vol, year_fraction
from orca.config import get_universe
from orca.domain import ChainSnapshot, LegStatus, OptionType, TradeStatus
from orca.logging_setup import get_logger
from orca.models import Trade, TradeLeg, TradeSnapshot
from orca.services.market_data import MarketDataService
from orca.services.trade_service import TradeService

log = get_logger(__name__)

DELTA_BREACH_PER_LOT = 0.45  # |net delta| per lot that counts as directional
EXPIRY_CLOSE_WARN_MIN = 45


@dataclass
class TrackResult:
    trade_id: int
    status: str
    mtm: float = 0.0
    realized: float = 0.0
    unrealized: float = 0.0
    net_delta: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    spot: float | None = None
    action: str = "hold"  # hold | exit | adjust
    alerts: list[dict] = field(default_factory=list)
    leg_marks: dict[int, float] = field(default_factory=dict)


class Tracker:
    def __init__(
        self, market: MarketDataService | None = None, trades: TradeService | None = None
    ) -> None:
        self.market = market or MarketDataService()
        self.trades = trades or TradeService()

    # -- entrypoints -----------------------------------------------------
    def open_trades(self, session: Session) -> list[Trade]:
        return list(
            session.scalars(
                select(Trade).where(
                    Trade.status.in_(
                        [
                            TradeStatus.OPEN.value,
                            TradeStatus.PENDING.value,
                            TradeStatus.CLOSING.value,
                        ]
                    )
                )
            ).all()
        )

    def track_all(self, session: Session) -> list[TrackResult]:
        trades = self.open_trades(session)
        if not trades:
            return []

        # One chain fetch per (underlying, expiry) serves every trade on it.
        groups: dict[tuple[str, date], list[Trade]] = defaultdict(list)
        for trade in trades:
            groups[(trade.underlying, trade.expiry)].append(trade)

        results: list[TrackResult] = []
        for (symbol, expiry), group in groups.items():
            chain = self._load_chain(symbol, expiry)
            for trade in group:
                try:
                    results.append(self.track(session, trade, chain))
                except Exception as exc:  # noqa: BLE001 - one bad trade must not stop the loop
                    log.exception("tracking failed for trade %s: %s", trade.id, exc)
        return results

    def _load_chain(self, symbol: str, expiry: date) -> ChainSnapshot | None:
        try:
            return self.market.chain(get_universe().get(symbol), expiry)
        except Exception as exc:  # noqa: BLE001
            log.warning("chain unavailable for %s %s: %s", symbol, expiry, exc)
            return None

    # -- the per-trade tick ----------------------------------------------
    def track(
        self, session: Session, trade: Trade, chain: ChainSnapshot | None = None
    ) -> TrackResult:
        chain = chain or self._load_chain(trade.underlying, trade.expiry)
        result = TrackResult(trade_id=trade.id, status=trade.status)
        spot = chain.spot if chain else None
        result.spot = spot

        quotes: dict[str, float] = {}
        if chain is None:
            keys = [leg.instrument_key for leg in trade.open_legs if leg.instrument_key]
            if keys:
                quotes = {k: q.ltp for k, q in self.market.feed.get_quotes(keys).items() if q.ltp}

        t_years = year_fraction(trade.expiry, datetime.now(UTC))
        net_delta = net_gamma = net_theta = net_vega = 0.0
        unrealized = 0.0

        for leg in trade.legs:
            if leg.status != LegStatus.OPEN.value:
                continue
            price, leg_greeks = self._mark_leg(leg, chain, quotes, spot, t_years)
            if price is None:
                continue
            unrealized += leg.mark(price)
            result.leg_marks[leg.id] = price
            if leg_greeks:
                sign_qty = leg.sign * leg.quantity
                leg.delta, leg.gamma = leg_greeks.get("delta"), leg_greeks.get("gamma")
                leg.theta, leg.vega = leg_greeks.get("theta"), leg_greeks.get("vega")
                net_delta += (leg.delta or 0.0) * sign_qty
                net_gamma += (leg.gamma or 0.0) * sign_qty
                net_theta += (leg.theta or 0.0) * sign_qty
                net_vega += (leg.vega or 0.0) * sign_qty

        realized = sum(leg.realized_pnl for leg in trade.legs)
        trade.realized_pnl = round(realized, 2)
        trade.unrealized_pnl = round(unrealized, 2)
        trade.net_delta, trade.net_gamma = round(net_delta, 2), round(net_gamma, 4)
        trade.net_theta, trade.net_vega = round(net_theta, 2), round(net_vega, 2)
        trade.last_tracked_at = datetime.now(UTC)

        total = trade.total_pnl
        trade.max_profit_seen = round(max(trade.max_profit_seen, total), 2)
        trade.max_loss_seen = round(min(trade.max_loss_seen, total), 2)

        result.mtm = round(total, 2)
        result.realized, result.unrealized = trade.realized_pnl, trade.unrealized_pnl
        result.net_delta, result.net_theta, result.net_vega = (
            trade.net_delta,
            trade.net_theta,
            trade.net_vega,
        )

        session.add(
            TradeSnapshot(
                trade_id=trade.id,
                ts=datetime.now(UTC),
                spot=spot,
                mtm=result.mtm,
                net_delta=trade.net_delta,
                net_theta=trade.net_theta,
                net_vega=trade.net_vega,
                leg_marks={str(k): v for k, v in result.leg_marks.items()},
            )
        )

        self._evaluate_rules(session, trade, result, chain)
        session.flush()
        return result

    # -- marking ---------------------------------------------------------
    def _mark_leg(
        self,
        leg: TradeLeg,
        chain: ChainSnapshot | None,
        quotes: dict[str, float],
        spot: float | None,
        t_years: float,
    ) -> tuple[float | None, dict[str, float] | None]:
        option_type = OptionType(leg.option_type)
        if chain is not None:
            quote = chain.get(leg.strike, option_type)
            if quote is not None and quote.ltp > 0:
                if quote.delta is not None:
                    return quote.ltp, {
                        "delta": quote.delta,
                        "gamma": quote.gamma,
                        "theta": quote.theta,
                        "vega": quote.vega,
                    }
                return quote.ltp, self._compute_greeks(
                    quote.ltp, chain.spot, leg.strike, t_years, option_type
                )
        price = quotes.get(leg.instrument_key)
        if price is None:
            return None, None
        if spot:
            return price, self._compute_greeks(price, spot, leg.strike, t_years, option_type)
        return price, None

    @staticmethod
    def _compute_greeks(
        price: float, spot: float, strike: float, t_years: float, option_type: OptionType
    ) -> dict[str, float] | None:
        """Back out IV from the mark, then compute greeks off that."""
        iv = implied_vol(price, spot, strike, t_years, option_type)
        if iv is None:
            return None
        return bs_greeks(spot, strike, t_years, iv, option_type)

    # -- exit rules ------------------------------------------------------
    def _evaluate_rules(
        self,
        session: Session,
        trade: Trade,
        result: TrackResult,
        chain: ChainSnapshot | None,
    ) -> None:
        alerts: list[dict] = []
        total = result.mtm

        def add(level: str, code: str, message: str) -> None:
            alerts.append(
                {
                    "level": level,
                    "code": code,
                    "message": message,
                    "ts": datetime.now(UTC).isoformat(),
                }
            )

        hit_target = trade.target_amount is not None and total >= trade.target_amount
        hit_stop = trade.stop_loss_amount is not None and total <= -abs(trade.stop_loss_amount)

        if hit_target:
            add("good", "target", f"target hit: P&L {total:,.0f}")
            result.action = "exit"
        if hit_stop:
            add("bad", "stop", f"stop hit: P&L {total:,.0f}")
            result.action = "exit"

        if (
            trade.trail_to_breakeven
            and trade.target_amount
            and trade.max_profit_seen >= 0.6 * trade.target_amount
            and total <= 0
        ):
            add("warn", "trail_be", "gave back the move - breakeven trail triggered")
            result.action = "exit"

        delta_limit = DELTA_BREACH_PER_LOT * max(trade.lots, 1) * max(trade.lot_size, 1)
        if abs(result.net_delta) > delta_limit:
            add(
                "warn",
                "delta_breach",
                f"net delta {result.net_delta:,.0f} beyond "
                f"+/-{delta_limit:,.0f} - consider adjusting",
            )
            if result.action == "hold":
                result.action = "adjust"

        breakevens = (trade.meta or {}).get("breakevens") or []
        if chain and breakevens:
            lower, upper = min(breakevens), max(breakevens)
            if chain.spot < lower or chain.spot > upper:
                add("warn", "beyond_be", f"spot {chain.spot:,.0f} outside breakevens")

        if trade.expiry == date.today() and chain:
            minutes_left = (15 * 60 + 30) - (
                chain.ts.astimezone(get_universe().market.tz).hour * 60
                + chain.ts.astimezone(get_universe().market.tz).minute
            )
            if 0 < minutes_left <= EXPIRY_CLOSE_WARN_MIN:
                add("warn", "expiry_close", f"{minutes_left}m to close on expiry day")

        if alerts:
            existing = list(trade.alerts or [])
            trade.alerts = (existing + alerts)[-25:]
        result.alerts = alerts

        if result.action == "exit" and trade.auto_exit:
            reason = "target" if hit_target else "stop"
            log.info("auto-exiting trade %s (%s)", trade.id, reason)
            self.trades.exit_trade(session, trade, reason=reason)
            result.status = trade.status


# ---------------------------------------------------------------------------
# portfolio level
# ---------------------------------------------------------------------------


def portfolio_summary(session: Session) -> dict:
    """Aggregate open risk, served by GET /api/portfolio."""
    trades = list(
        session.scalars(
            select(Trade).where(
                Trade.status.in_([TradeStatus.OPEN.value, TradeStatus.PENDING.value])
            )
        ).all()
    )
    today = datetime.now(UTC).date()
    closed_today = [
        t
        for t in session.scalars(
            select(Trade).where(Trade.status == TradeStatus.CLOSED.value)
        ).all()
        if t.exit_ts and t.exit_ts.date() == today
    ]

    by_underlying: dict[str, dict] = {}
    for t in trades:
        node = by_underlying.setdefault(
            t.underlying, {"mtm": 0.0, "delta": 0.0, "theta": 0.0, "vega": 0.0, "trades": 0}
        )
        node["mtm"] += t.total_pnl
        node["delta"] += t.net_delta or 0.0
        node["theta"] += t.net_theta or 0.0
        node["vega"] += t.net_vega or 0.0
        node["trades"] += 1

    return {
        "open_trades": len(trades),
        "open_mtm": round(sum(t.total_pnl for t in trades), 2),
        "realized_today": round(sum(t.total_pnl for t in closed_today), 2),
        "margin_blocked": round(sum(t.margin_blocked or 0 for t in trades), 2),
        "net_delta": round(sum(t.net_delta or 0 for t in trades), 2),
        "net_theta": round(sum(t.net_theta or 0 for t in trades), 2),
        "net_vega": round(sum(t.net_vega or 0 for t in trades), 2),
        "by_underlying": {
            k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in by_underlying.items()
        },
        "alerts": [a for t in trades for a in (t.alerts or [])][-10:],
    }
