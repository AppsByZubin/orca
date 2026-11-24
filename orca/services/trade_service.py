"""Trade lifecycle: build -> enter -> (track) -> exit.

A Trade owns its legs. Entry and exit always go through the configured
OrderGateway, so the paper and live paths are identical except for the gateway.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from orca.analytics.charges import leg_charges
from orca.config import get_settings, get_universe
from orca.domain import Candidate, LegSpec, LegStatus, OptionType, Side, TradeStatus
from orca.execution import OrderGateway, OrderRequest, get_gateway
from orca.logging_setup import get_logger
from orca.models import BrokerOrder, Trade, TradeLeg
from orca.symbols import kite_tradingsymbol

log = get_logger(__name__)


class TradeService:
    def __init__(self, gateway: OrderGateway | None = None) -> None:
        self.gateway = gateway or get_gateway()

    # -- construction ----------------------------------------------------
    def create_from_candidate(
        self,
        session: Session,
        candidate: Candidate,
        lots: int = 1,
        auto_exit: bool = False,
        signal_id: int | None = None,
        notes: str | None = None,
    ) -> Trade:
        """Materialise a DRAFT trade. Nothing is sent to the broker yet."""
        settings = get_settings()
        underlying = get_universe().get(candidate.underlying)
        trade = Trade(
            underlying=candidate.underlying,
            expiry=candidate.expiry,
            strategy_id=candidate.strategy_id,
            template=candidate.template,
            status=TradeStatus.DRAFT.value,
            signal_id=signal_id,
            lots=lots,
            lot_size=underlying.lot_size,
            entry_spot=candidate.spot,
            margin_blocked=candidate.margin_estimate,
            auto_exit=auto_exit,
            notes=notes,
            meta={
                "score": candidate.score,
                "score_source": candidate.score_source,
                "expected_return": candidate.expected_return,
                "breakevens": candidate.breakevens,
                "prob_of_profit": candidate.prob_of_profit,
                "reasons": candidate.reasons,
                "features": candidate.features,
            },
        )
        for leg in candidate.legs:
            scaled_lots = max(1, int(round(leg.lots * lots / max(leg.lots, 1))))
            trade.legs.append(
                TradeLeg(
                    instrument_key=leg.instrument_key,
                    tradingsymbol=leg.tradingsymbol
                    or kite_tradingsymbol(underlying, leg.expiry, leg.strike, leg.option_type),
                    exchange=underlying.option_exchange,
                    strike=leg.strike,
                    option_type=leg.option_type.value,
                    side=leg.side.value,
                    lots=scaled_lots,
                    lot_size=underlying.lot_size,
                    quantity=scaled_lots * underlying.lot_size,
                    status=LegStatus.PENDING.value,
                )
            )

        # Default risk envelope from the credit we expect to collect.
        expected_credit = abs(candidate.net_credit) * lots
        trade.stop_loss_amount = expected_credit * settings.default_stop_multiple
        trade.target_amount = expected_credit * settings.default_target_pct

        session.add(trade)
        session.flush()
        log.info("draft trade %s created: %s", trade.id, candidate.describe())
        return trade

    def create_manual(
        self,
        session: Session,
        underlying_symbol: str,
        expiry: date,
        legs: list[dict],
        lots: int = 1,
        strategy_id: str = "manual",
        auto_exit: bool = False,
    ) -> Trade:
        """Build a trade from explicit legs: [{strike, option_type, side, lots}]."""
        underlying = get_universe().get(underlying_symbol)
        trade = Trade(
            underlying=underlying.symbol,
            expiry=expiry,
            strategy_id=strategy_id,
            template="custom",
            status=TradeStatus.DRAFT.value,
            lots=lots,
            lot_size=underlying.lot_size,
            auto_exit=auto_exit,
        )
        for spec in legs:
            option_type = OptionType(str(spec["option_type"]).upper())
            side = Side(str(spec["side"]).upper())
            leg_lots = int(spec.get("lots", lots))
            trade.legs.append(
                TradeLeg(
                    instrument_key=spec.get("instrument_key", ""),
                    tradingsymbol=spec.get("tradingsymbol")
                    or kite_tradingsymbol(underlying, expiry, float(spec["strike"]), option_type),
                    exchange=underlying.option_exchange,
                    strike=float(spec["strike"]),
                    option_type=option_type.value,
                    side=side.value,
                    lots=leg_lots,
                    lot_size=underlying.lot_size,
                    quantity=leg_lots * underlying.lot_size,
                    status=LegStatus.PENDING.value,
                )
            )
        session.add(trade)
        session.flush()
        return trade

    # -- entry -----------------------------------------------------------
    def enter(self, session: Session, trade: Trade, order_type: str = "MARKET") -> Trade:
        """Send the entry basket and record the fills."""
        requests = [
            OrderRequest(
                tradingsymbol=leg.tradingsymbol,
                exchange=leg.exchange,
                transaction_type=leg.side,
                quantity=leg.quantity,
                order_type=order_type,
                product="NRML",
                instrument_key=leg.instrument_key,
                tag=f"orca-{trade.id}",
                purpose="ENTRY",
            )
            for leg in trade.legs
            if leg.status == LegStatus.PENDING.value
        ]
        if not requests:
            return trade

        by_symbol = {leg.tradingsymbol: leg for leg in trade.legs}
        results = self.gateway.place_basket(requests)

        credit = 0.0
        charges = 0.0
        for request, result in zip(
            sorted(requests, key=lambda r: 0 if r.transaction_type == "BUY" else 1),
            results,
            strict=False,
        ):
            leg = by_symbol.get(request.tradingsymbol)
            if leg is None:
                continue
            session.add(
                BrokerOrder(
                    trade_id=trade.id,
                    leg_id=leg.id,
                    gateway=self.gateway.name,
                    broker_order_id=result.order_id,
                    tradingsymbol=request.tradingsymbol,
                    exchange=request.exchange,
                    transaction_type=request.transaction_type,
                    order_type=request.order_type,
                    quantity=request.quantity,
                    price=result.average_price,
                    filled_quantity=result.filled_quantity,
                    average_price=result.average_price,
                    status=result.status.value,
                    message=result.message,
                    purpose="ENTRY",
                )
            )
            if result.ok and result.average_price:
                leg.entry_price = result.average_price
                leg.status = LegStatus.OPEN.value
                leg.entry_ts = datetime.now(UTC)
                credit += -leg.sign * result.average_price * leg.quantity
                charges += leg_charges(result.average_price, leg.quantity, Side(leg.side)).total
            else:
                leg.status = LegStatus.REJECTED.value

        open_legs = [leg for leg in trade.legs if leg.status == LegStatus.OPEN.value]
        rejected = [leg for leg in trade.legs if leg.status == LegStatus.REJECTED.value]

        trade.entry_credit = round(credit, 2)
        trade.charges = round(charges, 2)
        trade.entry_ts = datetime.now(UTC)
        if open_legs and not rejected:
            trade.status = TradeStatus.OPEN.value
        elif open_legs:
            trade.status = TradeStatus.OPEN.value
            self._alert(trade, "warn", "partial_fill", f"{len(rejected)} leg(s) rejected")
        else:
            trade.status = TradeStatus.REJECTED.value

        settings = get_settings()
        if trade.entry_credit:
            trade.stop_loss_amount = abs(trade.entry_credit) * settings.default_stop_multiple
            trade.target_amount = abs(trade.entry_credit) * settings.default_target_pct

        session.flush()
        log.info(
            "trade %s %s | credit %.2f | %d legs open",
            trade.id,
            trade.status,
            trade.entry_credit,
            len(open_legs),
        )
        return trade

    # -- exit ------------------------------------------------------------
    def exit_leg(self, session: Session, leg: TradeLeg, reason: str = "manual") -> TradeLeg:
        if leg.status != LegStatus.OPEN.value:
            return leg
        opposite = "BUY" if leg.side == Side.SELL.value else "SELL"
        result = self.gateway.place_order(
            OrderRequest(
                tradingsymbol=leg.tradingsymbol,
                exchange=leg.exchange,
                transaction_type=opposite,
                quantity=leg.quantity,
                order_type="MARKET",
                product="NRML",
                instrument_key=leg.instrument_key,
                tag=f"orca-{leg.trade_id}",
                purpose="EXIT",
            )
        )
        session.add(
            BrokerOrder(
                trade_id=leg.trade_id,
                leg_id=leg.id,
                gateway=self.gateway.name,
                broker_order_id=result.order_id,
                tradingsymbol=leg.tradingsymbol,
                exchange=leg.exchange,
                transaction_type=opposite,
                quantity=leg.quantity,
                average_price=result.average_price,
                filled_quantity=result.filled_quantity,
                status=result.status.value,
                message=f"{reason}: {result.message}",
                purpose="EXIT",
            )
        )
        if result.ok and result.average_price is not None:
            leg.exit_price = result.average_price
            leg.exit_ts = datetime.now(UTC)
            leg.status = LegStatus.CLOSED.value
            leg.realized_pnl = round(
                (leg.exit_price - (leg.entry_price or 0.0)) * leg.sign * leg.quantity, 2
            )
            leg.unrealized_pnl = 0.0
            if leg.trade is not None:
                leg.trade.charges = round(
                    leg.trade.charges
                    + leg_charges(
                        leg.exit_price,
                        leg.quantity,
                        Side.BUY if leg.side == Side.SELL.value else Side.SELL,
                    ).total,
                    2,
                )
        session.flush()
        return leg

    def exit_trade(self, session: Session, trade: Trade, reason: str = "manual") -> Trade:
        """Close every open leg. Longs are closed last so hedges stay on longest."""
        trade.status = TradeStatus.CLOSING.value
        session.flush()
        legs = sorted(trade.open_legs, key=lambda lg: 0 if lg.side == Side.SELL.value else 1)
        for leg in legs:
            self.exit_leg(session, leg, reason)

        trade.realized_pnl = round(sum(leg.realized_pnl for leg in trade.legs), 2)
        trade.unrealized_pnl = 0.0
        trade.exit_ts = datetime.now(UTC)
        trade.status = (
            TradeStatus.CLOSED.value
            if not trade.open_legs
            else TradeStatus.OPEN.value  # something failed to close - stays open
        )
        self._alert(trade, "info", "closed", f"closed ({reason}) P&L {trade.total_pnl:,.0f}")
        session.flush()
        log.info("trade %s closed (%s) P&L %.2f", trade.id, reason, trade.total_pnl)
        return trade

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _alert(trade: Trade, level: str, code: str, message: str) -> None:
        alerts = list(trade.alerts or [])
        alerts.append(
            {
                "level": level,
                "code": code,
                "message": message,
                "ts": datetime.now(UTC).isoformat(),
            }
        )
        trade.alerts = alerts[-25:]


def candidate_from_signal(signal) -> Candidate:  # noqa: ANN001 - models.Signal
    """Rebuild a Candidate from a persisted Signal row."""
    legs = [
        LegSpec(
            underlying=signal.underlying,
            expiry=date.fromisoformat(leg["expiry"]),
            strike=float(leg["strike"]),
            option_type=OptionType(leg["option_type"]),
            side=Side(leg["side"]),
            lots=int(leg.get("lots", 1)),
            lot_size=int(leg.get("lot_size", 1)),
            instrument_key=leg.get("instrument_key", ""),
            tradingsymbol=leg.get("tradingsymbol", ""),
            ref_price=float(leg.get("ref_price", 0.0)),
        )
        for leg in (signal.legs or [])
    ]
    return Candidate(
        strategy_id=signal.strategy_id,
        template=signal.template,
        underlying=signal.underlying,
        expiry=signal.expiry,
        legs=legs,
        spot=signal.spot,
        net_credit=signal.net_credit,
        max_loss=signal.max_loss,
        breakevens=list(signal.breakevens or []),
        prob_of_profit=signal.prob_of_profit,
        margin_estimate=signal.margin_estimate,
        features=dict(signal.features or {}),
        score=signal.score,
        expected_return=signal.expected_return,
        score_source=signal.score_source,
        reasons=list(signal.reasons or []),
    )


def get_trade(session: Session, trade_id: int) -> Trade | None:
    return session.scalar(select(Trade).where(Trade.id == trade_id))


def list_trades(session: Session, status: str | None = None, limit: int = 100) -> list[Trade]:
    stmt = select(Trade).order_by(Trade.created_at.desc()).limit(limit)
    if status == "open":
        stmt = stmt.where(
            Trade.status.in_(
                [TradeStatus.OPEN.value, TradeStatus.PENDING.value, TradeStatus.CLOSING.value]
            )
        )
    elif status:
        stmt = stmt.where(Trade.status == status.upper())
    return list(session.scalars(stmt).all())
