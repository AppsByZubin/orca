"""The multi-leg lifecycle: scan -> enter -> track -> partial exit -> close."""

from __future__ import annotations

from orca.domain import LegStatus, Side, TradeStatus
from orca.execution.paper import PaperGateway
from orca.services.scanner import Scanner
from orca.services.tracker import Tracker, portfolio_summary
from orca.services.trade_service import TradeService
from orca.strategies import get_template


def _candidate(chain, nifty):
    return get_template("iron_condor")(
        chain, nifty, {"short_delta": [0.15], "wing_width_strikes": [4], "_defaults": {}}, 1
    )[0]


def test_enter_records_credit_and_opens_legs(session, chain, nifty, feed) -> None:
    service = TradeService(gateway=PaperGateway(feed=feed))
    trade = service.create_from_candidate(session, _candidate(chain, nifty), lots=1)

    assert trade.status == TradeStatus.DRAFT.value
    assert len(trade.legs) == 4
    assert all(leg.status == LegStatus.PENDING.value for leg in trade.legs)

    service.enter(session, trade)
    assert trade.status == TradeStatus.OPEN.value
    assert all(leg.status == LegStatus.OPEN.value for leg in trade.legs)
    assert all(leg.entry_price for leg in trade.legs)
    assert trade.entry_credit > 0  # net credit structure
    assert trade.charges > 0
    assert trade.stop_loss_amount > trade.target_amount > 0
    assert len(trade.orders) == 4


def test_tracker_marks_legs_and_aggregates_greeks(session, chain, nifty, feed) -> None:
    service = TradeService(gateway=PaperGateway(feed=feed))
    trade = service.create_from_candidate(session, _candidate(chain, nifty), lots=1)
    service.enter(session, trade)

    result = Tracker(trades=service).track(session, trade, chain)

    assert result.leg_marks and len(result.leg_marks) == 4
    assert trade.last_tracked_at is not None
    assert trade.net_theta is not None
    # P&L identity: trade total == sum of leg P&Ls minus charges.
    legs_pnl = sum(leg.unrealized_pnl + leg.realized_pnl for leg in trade.legs)
    assert abs(trade.total_pnl - (legs_pnl - trade.charges)) < 0.01

    summary = portfolio_summary(session)
    assert summary["open_trades"] == 1


class ScaledFeed:
    """Wraps the mock feed so exit fills can be moved away from entry prices."""

    def __init__(self, inner, factor: float = 1.0) -> None:
        self.inner, self.factor = inner, factor

    def get_quotes(self, keys):
        quotes = self.inner.get_quotes(keys)
        for quote in quotes.values():
            quote.ltp = max(round(quote.ltp * self.factor, 2), 0.05)
        return quotes


def test_target_hit_triggers_auto_exit(session, chain, nifty, feed) -> None:
    scaled = ScaledFeed(feed)
    service = TradeService(gateway=PaperGateway(feed=scaled))
    trade = service.create_from_candidate(session, _candidate(chain, nifty), lots=1, auto_exit=True)
    service.enter(session, trade)

    # Collapse premium everywhere: the tracker marks off the chain, the paper
    # gateway fills off the feed, so both have to move together.
    scaled.factor = 0.05
    for option in chain.options:
        option.ltp = max(round(option.ltp * 0.05, 2), 0.05)
        option.bid, option.ask = max(option.ltp - 0.05, 0.05), option.ltp + 0.05

    result = Tracker(trades=service).track(session, trade, chain)

    assert any(a["code"] == "target" for a in result.alerts)
    assert trade.status == TradeStatus.CLOSED.value
    assert not trade.open_legs
    assert all(leg.exit_price is not None for leg in trade.legs)
    assert trade.realized_pnl > 0


def test_exit_single_leg_leaves_the_rest_open(session, chain, nifty, feed) -> None:
    service = TradeService(gateway=PaperGateway(feed=feed))
    trade = service.create_from_candidate(session, _candidate(chain, nifty), lots=1)
    service.enter(session, trade)

    short_call = next(
        leg for leg in trade.legs if leg.side == Side.SELL.value and leg.option_type == "CE"
    )
    service.exit_leg(session, short_call, reason="test")

    assert short_call.status == LegStatus.CLOSED.value
    assert short_call.exit_price is not None
    assert len(trade.open_legs) == 3
    assert trade.status == TradeStatus.OPEN.value


def test_scanner_ranks_and_persists_signals(session, nifty) -> None:
    candidates = Scanner().scan(session, symbols=["NIFTY"], lots=1, persist=True)
    assert candidates
    scores = [c.score for c in candidates if c.score is not None]
    assert scores == sorted(scores, reverse=True)

    from orca.models import Signal

    rows = session.query(Signal).all()
    assert len(rows) == len(candidates)
    assert rows[0].legs and rows[0].features
