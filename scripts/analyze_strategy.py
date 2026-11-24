#!/usr/bin/env python
"""Sweep one strategy template over a strike/width grid and explain the ranking.

Analysis only - builds and scores candidates, writes nothing, places nothing.
Independent of config/strategies.yaml so you can explore a wider grid than the
scanner is configured for.

    python scripts/analyze_strategy.py --symbol SENSEX
    python scripts/analyze_strategy.py --symbol SENSEX --deltas 0.08,0.12 --widths 2,4,8
    python scripts/analyze_strategy.py --template short_strangle --deltas 0.10,0.15,0.20
"""

from __future__ import annotations

import argparse
from datetime import date

from orca.config import get_settings, get_strategy_config, get_universe
from orca.db import session_scope
from orca.ml.features import candidate_features
from orca.ml.heuristic import score as heuristic_score
from orca.ml.model import get_model
from orca.services.market_data import MarketDataService
from orca.strategies import get_template
from orca.strategies.base import meets_credit_floor


def floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--symbol", default="SENSEX")
    p.add_argument("--template", default="iron_condor")
    p.add_argument("--deltas", default="0.08,0.10,0.12,0.15,0.18,0.22,0.25")
    p.add_argument(
        "--widths", default="2,4,6,8,10,12", help="wing width in strikes (strangle ignores)"
    )
    p.add_argument("--lots", type=int, default=1)
    p.add_argument("--min-dte", type=int, default=1, help="skip expiry day by default")
    p.add_argument("--max-dte", type=int, default=10)
    p.add_argument("--top", type=int, default=12)
    p.add_argument(
        "--no-filters", action="store_true", help="ignore the credit floor / liquidity gates"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    universe = get_universe()
    underlying = universe.get(args.symbol)
    market = MarketDataService()
    model = get_model()

    expiry = market.nearest_expiry(underlying, min_dte=args.min_dte, max_dte=args.max_dte)
    if expiry is None:
        print(f"no expiry for {args.symbol} within {args.min_dte}-{args.max_dte} dte")
        return 1
    chain = market.chain(underlying, expiry)
    dte = (expiry - date.today()).days

    print("=" * 108)
    print(f"  {args.template.upper()}  on  {underlying.symbol}   expiry {expiry}  (dte {dte})")
    print(
        f"  feed={settings.feed_provider}   spot={chain.spot:,.1f}"
        f"   atm_iv={chain.atm_iv() or 0:.1%}"
        f"   lot_size={underlying.lot_size}   lots={args.lots}"
    )
    if model.is_fitted:
        tag = " [SYNTHETIC - not tradable]" if model.metadata.synthetic else ""
        print(f"  model={model.metadata.name}{tag}")
    else:
        print("  model=<none>  (heuristic only)")
    print("=" * 108)

    with session_scope() as session:
        context = market.context(session, underlying, chain)

        params = {
            "short_delta": floats(args.deltas),
            "wing_width_strikes": ints(args.widths),
            "side": ["PE"],
            "_defaults": {} if args.no_filters else get_strategy_config().defaults,
        }
        cands = get_template(args.template)(chain, underlying, params, args.lots)
        if not cands:
            print("\n  no candidates built - loosen --deltas/--widths or pass --no-filters\n")
            return 1

        rows = []
        for c in cands:
            c.features = candidate_features(c, chain, context)
            h_score, h_reasons = heuristic_score(c.features)
            m_score = None
            if model.is_fitted:
                ((prob, _exp),) = model.score_many([c.features])
                m_score = prob
            kept = args.no_filters or meets_credit_floor(
                c, get_strategy_config().defaults, args.lots
            )
            rows.append((c, m_score, h_score, h_reasons, kept))

        rows.sort(key=lambda r: (r[1] if r[1] is not None else r[2]) or 0.0, reverse=True)

        hdr = (
            f"{'model':>6} {'heur':>6} {'credit':>8} {'maxloss':>9} {'margin':>9} "
            f"{'RoM%':>6} {'POP':>6} {'breakevens':>19} {'theta':>7} {'vega':>7}  contracts"
        )
        print("\n" + hdr)
        print("-" * 108)
        for c, m_score, h_score, _r, kept in rows[: args.top]:
            rom = (c.net_credit / c.margin_estimate * 100) if c.margin_estimate else 0.0
            be = "-".join(f"{b:,.0f}" for b in c.breakevens) if c.breakevens else "n/a"
            legs = " ".join(
                f"{'S' if leg.side.value == 'SELL' else 'B'}"
                f"{int(leg.strike)}{leg.option_type.value}"
                for leg in c.legs
            )
            flag = "" if kept else "  <-- filtered out"
            print(
                f"{(m_score or 0):>6.3f} {h_score:>6.3f} {c.net_credit:>8,.0f} "
                f"{(c.max_loss or 0):>9,.0f} {(c.margin_estimate or 0):>9,.0f} {rom:>6.1f} "
                f"{(c.prob_of_profit or 0):>6.1%} {be:>19} "
                f"{c.features.get('net_theta', 0):>7,.0f} "
                f"{c.features.get('net_vega', 0):>7,.0f}  {legs}{flag}"
            )

        best, m_score, h_score, h_reasons, _k = rows[0]
        print("\n" + "=" * 108)
        print(f"  TOP PICK   {best.strategy_id}")
        print("=" * 108)
        for leg in best.legs:
            print(
                f"    {leg.side.value:<4} {leg.lots} lot x {leg.lot_size} "
                f"{int(leg.strike):>6} {leg.option_type.value}  @ {leg.ref_price:>8,.2f}"
                f"   {leg.tradingsymbol or ''}"
            )
        print(f"\n    net credit     {best.net_credit:>12,.0f}")
        print(
            f"    max loss       {(best.max_loss or 0):>12,.0f}"
            f"   (credit/risk {(best.credit_to_risk or 0):.2f})"
        )
        rom_best = best.net_credit / best.margin_estimate * 100 if best.margin_estimate else 0.0
        print(
            f"    margin est.    {(best.margin_estimate or 0):>12,.0f}"
            f"   return on margin {rom_best:.1f}%"
        )
        print(
            f"    target +50%    {best.net_credit * 0.5:>12,.0f}"
            f"   stop -200% {-best.net_credit * 2:,.0f}   (.env defaults)"
        )
        print(f"    breakevens     {', '.join(f'{b:,.0f}' for b in best.breakevens) or 'n/a'}")
        print(f"    prob of profit {(best.prob_of_profit or 0):>12.1%}")

        print("\n    why it ranks here (heuristic view):")
        for r in h_reasons:
            print(f"      - {r}")

        print("\n    regime features driving the score:")
        for k in (
            "atm_iv",
            "iv_rank",
            "iv_rv_spread",
            "iv_skew",
            "pcr_oi",
            "trend_5d",
            "dte",
            "short_delta_avg",
            "net_delta",
            "theta_per_vega",
            "credit_to_max_loss",
            "credit_to_margin",
            "nearest_be_sigma",
            "prob_of_profit",
        ):
            v = best.features.get(k)
            if v is not None:
                print(f"      {k:<22} {v:>10.4f}")
        if model.is_fitted and model.metadata.synthetic:
            print("\n  NOTE: model column is from a SYNTHETIC-trained model: plumbing only.")
        if settings.feed_provider == "mock":
            print(
                "  NOTE: feed=mock - these strikes and prices are generated, not real market data."
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
