"""Request/response models for the web API."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


# --- session ----------------------------------------------------------------
class LoginRequest(BaseModel):
    username: str
    password: str


class SessionOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    expires_at: int  # epoch seconds


# --- broker tokens ----------------------------------------------------------
class BrokerTokenIn(BaseModel):
    access_token: str = Field(min_length=1)


class BrokerTokenOut(BaseModel):
    broker: str
    label: str
    has_token: bool
    source: str | None = None  # "saved" (Tokens page) | "env" | None
    token_hint: str | None = None  # last 4 characters, never the token
    saved_at: datetime | None = None
    expires_at: datetime | None = None
    expired: bool
    rejected: bool
    in_use: bool  # this broker is the configured feed / gateway
    api_key_configured: bool
    needs_attention: bool  # missing, expired, or only an env token of unknown age
    verified: bool | None = None  # only on save/verify: did the broker accept it
    verify_detail: str | None = None


# --- market -----------------------------------------------------------------
class UnderlyingOut(BaseModel):
    symbol: str
    name: str
    lot_size: int
    strike_step: float
    option_exchange: str
    enabled: bool


class OptionQuoteOut(BaseModel):
    strike: float
    option_type: str
    ltp: float
    bid: float | None = None
    ask: float | None = None
    oi: int = 0
    oi_change: int = 0
    volume: int = 0
    iv: float | None = None
    delta: float | None = None
    theta: float | None = None
    vega: float | None = None
    instrument_key: str = ""
    tradingsymbol: str = ""


class ChainOut(BaseModel):
    underlying: str
    expiry: date
    spot: float
    ts: datetime
    atm_strike: float
    atm_iv: float | None = None
    dte: int
    source: str
    options: list[OptionQuoteOut]


# --- probability model --------------------------------------------------------
class SigmaRangeOut(BaseModel):
    sigmas: int
    lower: float
    upper: float
    prob: float


class StrikeProbabilityOut(BaseModel):
    strike: float
    iv: float  # the IV used for this strike
    iv_source: str  # "otm" (that strike's OTM option) | "itm" | "atm" (fallback)
    p_below: float
    p_above: float
    p_touch: float
    call_delta: float | None = None
    put_delta: float | None = None


class CurvePointOut(BaseModel):
    price: float
    density: float
    p_below: float


class ProbabilityOut(BaseModel):
    underlying: str
    expiry: date
    ts: datetime
    source: str
    spot: float
    atm_strike: float
    atm_iv: float
    dte: int
    t_years: float
    sigma: float  # atm_iv * sqrt(t_years): one standard deviation of log return
    ranges: list[SigmaRangeOut]
    strikes: list[StrikeProbabilityOut]
    curve: list[CurvePointOut]


# --- candidates / signals ---------------------------------------------------
class LegOut(BaseModel):
    strike: float
    option_type: str
    side: str
    lots: int
    lot_size: int
    quantity: int
    ref_price: float = 0.0
    instrument_key: str = ""
    tradingsymbol: str = ""


class CandidateOut(BaseModel):
    signal_id: int | None = None
    strategy_id: str
    template: str
    underlying: str
    expiry: date
    spot: float
    score: float | None = None
    expected_return: float | None = None
    score_source: str = "heuristic"
    net_credit: float
    max_loss: float | None = None
    margin_estimate: float | None = None
    prob_of_profit: float | None = None
    breakevens: list[float] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    legs: list[LegOut] = Field(default_factory=list)
    features: dict[str, float] = Field(default_factory=dict)
    ts: datetime | None = None


class ScanRequest(BaseModel):
    symbols: list[str] | None = None
    lots: int = 1
    max_dte: int | None = None
    persist: bool = True
    limit: int = 25


# --- trades -----------------------------------------------------------------
class TradeLegOut(BaseModel):
    id: int
    strike: float
    option_type: str
    side: str
    lots: int
    quantity: int
    status: str
    entry_price: float | None = None
    exit_price: float | None = None
    ltp: float | None = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    delta: float | None = None
    theta: float | None = None
    vega: float | None = None
    tradingsymbol: str = ""


class TradeOut(BaseModel):
    id: int
    underlying: str
    expiry: date
    strategy_id: str
    template: str
    status: str
    lots: int
    lot_size: int
    entry_credit: float
    realized_pnl: float
    unrealized_pnl: float
    charges: float
    total_pnl: float
    margin_blocked: float | None = None
    stop_loss_amount: float | None = None
    target_amount: float | None = None
    auto_exit: bool = False
    trail_to_breakeven: bool = False
    net_delta: float | None = None
    net_theta: float | None = None
    net_vega: float | None = None
    max_profit_seen: float = 0.0
    max_loss_seen: float = 0.0
    entry_ts: datetime | None = None
    exit_ts: datetime | None = None
    last_tracked_at: datetime | None = None
    alerts: list[dict] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)
    legs: list[TradeLegOut] = Field(default_factory=list)


class TradeFromSignalRequest(BaseModel):
    signal_id: int
    lots: int = 1
    auto_exit: bool = False
    place: bool = True  # send the entry basket immediately
    notes: str | None = None


class ManualLegIn(BaseModel):
    strike: float
    option_type: str
    side: str
    lots: int = 1
    instrument_key: str = ""
    tradingsymbol: str = ""


class ManualTradeRequest(BaseModel):
    underlying: str
    expiry: date
    legs: list[ManualLegIn]
    lots: int = 1
    strategy_id: str = "manual"
    auto_exit: bool = False
    place: bool = True


class TradeUpdateRequest(BaseModel):
    stop_loss_amount: float | None = None
    target_amount: float | None = None
    auto_exit: bool | None = None
    trail_to_breakeven: bool | None = None
    notes: str | None = None


class PayoffOut(BaseModel):
    """Expiry payoff. max_profit/max_loss are P&L values, so a loss is negative
    (unlike CandidateOut.max_loss, which is a positive magnitude). max_loss is
    null when a naked short leg makes it undefined."""

    spot: float
    net_credit: float
    max_profit: float | None = None
    max_loss: float | None = None
    breakevens: list[float] = Field(default_factory=list)
    prob_of_profit: float | None = None
    curve: list[list[float]] = Field(default_factory=list)
    current_mtm: float | None = None


# --- model / system ---------------------------------------------------------
class ModelInfoOut(BaseModel):
    name: str
    is_fitted: bool
    trained_at: str = ""
    n_samples: int = 0
    synthetic: bool = False
    metrics: dict = Field(default_factory=dict)
    label_definition: str = ""
    top_features: dict[str, float] = Field(default_factory=dict)


class TrainRequest(BaseModel):
    name: str = "default"
    synthetic: bool = False
    rows: int = 4000
    underlying: str | None = None
    target_pct: float = 0.5
    stop_multiple: float = 2.0


class HealthOut(BaseModel):
    ok: bool
    env: str
    version: str
    market_open: bool
    feed: dict
    gateway: dict
    trading_enabled: bool
    model: ModelInfoOut
    universe: list[str]
