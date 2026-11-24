"""Pydantic DTOs for the web API, plus the domain -> DTO converters."""

from __future__ import annotations

from orca.domain import Candidate, ChainSnapshot
from orca.models import Signal, Trade
from orca.schemas.api import (
    CandidateOut,
    ChainOut,
    HealthOut,
    LegOut,
    ManualLegIn,
    ManualTradeRequest,
    ModelInfoOut,
    OptionQuoteOut,
    PayoffOut,
    ScanRequest,
    TradeFromSignalRequest,
    TradeLegOut,
    TradeOut,
    TradeUpdateRequest,
    TrainRequest,
    UnderlyingOut,
)

__all__ = [
    "CandidateOut",
    "ChainOut",
    "HealthOut",
    "LegOut",
    "ManualLegIn",
    "ManualTradeRequest",
    "ModelInfoOut",
    "OptionQuoteOut",
    "PayoffOut",
    "ScanRequest",
    "TradeFromSignalRequest",
    "TradeLegOut",
    "TradeOut",
    "TradeUpdateRequest",
    "TrainRequest",
    "UnderlyingOut",
    "candidate_to_out",
    "chain_to_out",
    "signal_to_out",
    "trade_to_out",
]


def chain_to_out(chain: ChainSnapshot) -> ChainOut:
    return ChainOut(
        underlying=chain.underlying,
        expiry=chain.expiry,
        spot=chain.spot,
        ts=chain.ts,
        atm_strike=chain.atm_strike(),
        atm_iv=chain.atm_iv(),
        dte=chain.dte(),
        source=chain.source,
        options=[
            OptionQuoteOut(
                strike=o.strike,
                option_type=o.option_type.value,
                ltp=o.ltp,
                bid=o.bid,
                ask=o.ask,
                oi=o.oi,
                oi_change=o.oi_change,
                volume=o.volume,
                iv=o.iv,
                delta=o.delta,
                theta=o.theta,
                vega=o.vega,
                instrument_key=o.instrument_key,
                tradingsymbol=o.tradingsymbol,
            )
            for o in sorted(chain.options, key=lambda q: (q.strike, q.option_type.value))
        ],
    )


def candidate_to_out(candidate: Candidate, signal_id: int | None = None) -> CandidateOut:
    return CandidateOut(
        signal_id=signal_id,
        strategy_id=candidate.strategy_id,
        template=candidate.template,
        underlying=candidate.underlying,
        expiry=candidate.expiry,
        spot=candidate.spot,
        score=candidate.score,
        expected_return=candidate.expected_return,
        score_source=candidate.score_source,
        net_credit=candidate.net_credit,
        max_loss=candidate.max_loss,
        margin_estimate=candidate.margin_estimate,
        prob_of_profit=candidate.prob_of_profit,
        breakevens=candidate.breakevens,
        reasons=candidate.reasons,
        features=candidate.features,
        ts=candidate.ts,
        legs=[
            LegOut(
                strike=leg.strike,
                option_type=leg.option_type.value,
                side=leg.side.value,
                lots=leg.lots,
                lot_size=leg.lot_size,
                quantity=leg.quantity,
                ref_price=leg.ref_price,
                instrument_key=leg.instrument_key,
                tradingsymbol=leg.tradingsymbol,
            )
            for leg in candidate.legs
        ],
    )


def signal_to_out(signal: Signal) -> CandidateOut:
    return CandidateOut(
        signal_id=signal.id,
        strategy_id=signal.strategy_id,
        template=signal.template,
        underlying=signal.underlying,
        expiry=signal.expiry,
        spot=signal.spot,
        score=signal.score,
        expected_return=signal.expected_return,
        score_source=signal.score_source,
        net_credit=signal.net_credit,
        max_loss=signal.max_loss,
        margin_estimate=signal.margin_estimate,
        prob_of_profit=signal.prob_of_profit,
        breakevens=list(signal.breakevens or []),
        reasons=list(signal.reasons or []),
        features=dict(signal.features or {}),
        ts=signal.ts,
        legs=[
            LegOut(
                strike=float(leg["strike"]),
                option_type=leg["option_type"],
                side=leg["side"],
                lots=int(leg.get("lots", 1)),
                lot_size=int(leg.get("lot_size", 1)),
                quantity=int(leg.get("quantity", 0)),
                ref_price=float(leg.get("ref_price", 0.0)),
                instrument_key=leg.get("instrument_key", ""),
                tradingsymbol=leg.get("tradingsymbol", ""),
            )
            for leg in (signal.legs or [])
        ],
    )


def trade_to_out(trade: Trade) -> TradeOut:
    return TradeOut(
        id=trade.id,
        underlying=trade.underlying,
        expiry=trade.expiry,
        strategy_id=trade.strategy_id,
        template=trade.template,
        status=trade.status,
        lots=trade.lots,
        lot_size=trade.lot_size,
        entry_credit=trade.entry_credit,
        realized_pnl=trade.realized_pnl,
        unrealized_pnl=trade.unrealized_pnl,
        charges=trade.charges,
        total_pnl=round(trade.total_pnl, 2),
        margin_blocked=trade.margin_blocked,
        stop_loss_amount=trade.stop_loss_amount,
        target_amount=trade.target_amount,
        auto_exit=trade.auto_exit,
        trail_to_breakeven=trade.trail_to_breakeven,
        net_delta=trade.net_delta,
        net_theta=trade.net_theta,
        net_vega=trade.net_vega,
        max_profit_seen=trade.max_profit_seen,
        max_loss_seen=trade.max_loss_seen,
        entry_ts=trade.entry_ts,
        exit_ts=trade.exit_ts,
        last_tracked_at=trade.last_tracked_at,
        alerts=list(trade.alerts or []),
        meta=dict(trade.meta or {}),
        legs=[
            TradeLegOut(
                id=leg.id,
                strike=leg.strike,
                option_type=leg.option_type,
                side=leg.side,
                lots=leg.lots,
                quantity=leg.quantity,
                status=leg.status,
                entry_price=leg.entry_price,
                exit_price=leg.exit_price,
                ltp=leg.ltp,
                realized_pnl=leg.realized_pnl,
                unrealized_pnl=leg.unrealized_pnl,
                delta=leg.delta,
                theta=leg.theta,
                vega=leg.vega,
                tradingsymbol=leg.tradingsymbol,
            )
            for leg in sorted(trade.legs, key=lambda lg: (lg.option_type, lg.strike))
        ],
    )
