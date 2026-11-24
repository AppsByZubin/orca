"""Transaction cost model for Indian index options.

RATES CHANGE. These are the F&O charges as configured below, not a promise -
check your broker's current charge sheet (and the exchange circulars) and
update `RATES` before relying on net numbers:

    https://zerodha.com/charges/

Charges are applied per *order leg*, since brokerage and STT are per order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from orca.domain import LegSpec, Side

RATES = {
    "brokerage_per_order": 20.0,  # flat, or 0.03% whichever is lower
    "brokerage_pct_cap": 0.0003,
    "stt_sell_pct": 0.001,  # on sell-side premium turnover
    "exchange_txn_pct": 0.0003503,  # NSE options; BSE differs slightly
    "sebi_pct": 0.000001,  # Rs 10 per crore
    "stamp_duty_buy_pct": 0.00003,  # buy side only
    "gst_pct": 0.18,  # on brokerage + exchange txn charges
}


@dataclass
class ChargeBreakdown:
    brokerage: float = 0.0
    stt: float = 0.0
    exchange: float = 0.0
    sebi: float = 0.0
    stamp: float = 0.0
    gst: float = 0.0

    @property
    def total(self) -> float:
        return round(
            self.brokerage + self.stt + self.exchange + self.sebi + self.stamp + self.gst, 2
        )

    def as_dict(self) -> dict:
        return {
            "brokerage": round(self.brokerage, 2),
            "stt": round(self.stt, 2),
            "exchange": round(self.exchange, 2),
            "sebi": round(self.sebi, 2),
            "stamp": round(self.stamp, 2),
            "gst": round(self.gst, 2),
            "total": self.total,
        }


def leg_charges(price: float, quantity: int, side: Side) -> ChargeBreakdown:
    turnover = price * quantity
    r = RATES
    brokerage = min(r["brokerage_per_order"], turnover * r["brokerage_pct_cap"])
    c = ChargeBreakdown(
        brokerage=brokerage,
        stt=turnover * r["stt_sell_pct"] if side is Side.SELL else 0.0,
        exchange=turnover * r["exchange_txn_pct"],
        sebi=turnover * r["sebi_pct"],
        stamp=turnover * r["stamp_duty_buy_pct"] if side is Side.BUY else 0.0,
    )
    c.gst = (c.brokerage + c.exchange) * r["gst_pct"]
    return c


def round_trip_charges(legs: Sequence[LegSpec]) -> float:
    """Entry + exit charges for a whole structure."""
    total = 0.0
    for leg in legs:
        entry = leg_charges(leg.ref_price, leg.quantity, leg.side)
        exit_side = Side.BUY if leg.side is Side.SELL else Side.SELL
        exit_ = leg_charges(leg.ref_price, leg.quantity, exit_side)
        total += entry.total + exit_.total
    return round(total, 2)
