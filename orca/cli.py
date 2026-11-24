"""ORCA command line.

orca init-db                 create the schema
orca serve                   run the web app + scheduler
orca worker                  run only the background jobs
orca health                  check feed / gateway / model wiring
orca chain NIFTY             print the nearest-expiry chain around ATM
orca scan                    rank option-selling candidates now
orca snapshot                capture one chain snapshot per underlying
orca track                   mark every open trade once
orca train --synthetic       train the scoring model
"""

from __future__ import annotations

import time
from datetime import date

import typer

from orca import __version__
from orca.config import get_settings, get_universe
from orca.db import init_db as _init_db
from orca.db import session_scope
from orca.logging_setup import get_logger, setup_logging

app = typer.Typer(add_completion=False, help=f"ORCA {__version__}")
log = get_logger("orca.cli")


@app.callback()
def _main() -> None:
    setup_logging(get_settings().log_level)


@app.command("init-db")
def init_db_cmd() -> None:
    """Create all tables."""
    _init_db()
    typer.echo("schema created")


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = typer.Option(False, help="auto-reload on code changes"),
) -> None:
    """Run the JSON API (interactive docs at http://host:port/docs)."""
    import uvicorn

    uvicorn.run("orca.web.app:app", host=host, port=port, reload=reload)


@app.command()
def worker() -> None:
    """Run the scheduler without the web server."""
    from orca.jobs.scheduler import build_scheduler

    _init_db()
    scheduler = build_scheduler()
    scheduler.start()
    typer.echo(f"worker started: {[j.id for j in scheduler.get_jobs()]} - Ctrl-C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        scheduler.shutdown()


@app.command()
def health() -> None:
    """Print feed / gateway / model status."""
    from orca.execution import get_gateway
    from orca.feeds import get_feed
    from orca.jobs.scheduler import is_market_open
    from orca.ml.model import get_model

    settings = get_settings()
    model = get_model()
    typer.echo(f"env              {settings.env}")
    typer.echo(f"market open      {is_market_open()}")
    typer.echo(f"feed             {get_feed().health()}")
    typer.echo(f"gateway          {get_gateway().health()}")
    typer.echo(f"trading enabled  {settings.trading_enabled}")
    typer.echo(
        f"model            {model.metadata.name} fitted={model.is_fitted} "
        f"synthetic={model.metadata.synthetic} metrics={model.metadata.metrics or '{}'}"
    )
    typer.echo(f"universe         {[u.symbol for u in get_universe().enabled]}")


@app.command()
def chain(
    symbol: str, expiry: str = typer.Option(None, help="YYYY-MM-DD"), strikes: int = 8
) -> None:
    """Print the option chain around ATM."""
    from orca.domain import OptionType
    from orca.services.market_data import MarketDataService

    underlying = get_universe().get(symbol)
    service = MarketDataService()
    exp = date.fromisoformat(expiry) if expiry else service.nearest_expiry(underlying, max_dte=60)
    if exp is None:
        raise typer.BadParameter(f"no expiry found for {symbol}")
    snap = service.chain(underlying, exp)
    atm = snap.atm_strike()
    typer.echo(
        f"{symbol} {exp}  spot={snap.spot:,.2f}  "
        f"ATM IV={(snap.atm_iv() or 0):.2%}  src={snap.source}"
    )
    typer.echo(
        f"{'CE bid':>9} {'CE iv':>7} {'CE Δ':>7} | {'strike':^8} | "
        f"{'PE Δ':>7} {'PE iv':>7} {'PE bid':>9}"
    )
    for strike in snap.strikes():
        if abs(strike - atm) > strikes * underlying.strike_step:
            continue
        ce = snap.get(strike, OptionType.CE)
        pe = snap.get(strike, OptionType.PE)
        marker = "<" if abs(strike - atm) < 1e-6 else " "
        typer.echo(
            f"{(ce.bid or 0):>9.2f} {(ce.iv or 0):>7.2%} {(ce.delta or 0):>7.3f} |"
            f"{strike:^8.0f}{marker}| {(pe.delta or 0):>7.3f} "
            f"{(pe.iv or 0):>7.2%} {(pe.bid or 0):>9.2f}"
        )


@app.command()
def scan(
    symbol: str = typer.Option(None, help="restrict to one underlying"),
    lots: int = 1,
    top: int = 10,
    persist: bool = True,
) -> None:
    """Rank option-selling candidates."""
    from orca.services.scanner import Scanner

    with session_scope() as session:
        candidates = Scanner().scan(
            session, symbols=[symbol] if symbol else None, lots=lots, persist=persist
        )
        typer.echo(
            f"{'score':>6} {'underlying':<10} {'strategy':<26} "
            f"{'credit':>10} {'maxloss':>10} {'margin':>11}  legs"
        )
        for c in candidates[:top]:
            typer.echo(
                f"{(c.score or 0):>6.3f} {c.underlying:<10} {c.strategy_id:<26} "
                f"{c.net_credit:>10,.0f} "
                f"{('n/a' if c.max_loss is None else f'{c.max_loss:,.0f}'):>10} "
                f"{(c.margin_estimate or 0):>11,.0f}  " + " ".join(leg.label() for leg in c.legs)
            )
        if not candidates:
            typer.echo("(no candidates passed the filters)")


@app.command()
def snapshot() -> None:
    """Capture one chain snapshot per enabled underlying."""
    from orca.services.market_data import MarketDataService

    _init_db()
    with session_scope() as session:
        rows = MarketDataService().capture_all(session)
        for row in rows:
            typer.echo(
                f"saved {row.underlying} {row.expiry} spot={row.spot:,.2f} quotes={len(row.quotes)}"
            )


@app.command()
def track() -> None:
    """Mark every open trade once and apply the exit rules."""
    from orca.services.tracker import Tracker, portfolio_summary

    with session_scope() as session:
        results = Tracker().track_all(session)
        for r in results:
            typer.echo(
                f"trade {r.trade_id}: mtm={r.mtm:>10,.0f} Δ={r.net_delta:>8,.0f} "
                f"Θ={r.net_theta:>8,.0f} action={r.action}"
            )
            for alert in r.alerts:
                typer.echo(f"    [{alert['level']}] {alert['message']}")
        summary = portfolio_summary(session)
    typer.echo(f"open={summary['open_trades']} mtm={summary['open_mtm']:,.0f}")


@app.command("parquet-schema")
def parquet_schema(
    source: str = typer.Argument(..., help="local path, directory, or s3://bucket/prefix"),
) -> None:
    """Show how a Parquet chain archive's columns map onto ORCA's fields."""
    import json

    from orca.ml.sources import describe_parquet

    info = describe_parquet(source)
    typer.echo(f"source: {info['source']}")
    typer.echo(f"\ncolumns found ({len(info['columns'])}):")
    for c in info["columns"]:
        typer.echo(f"  {c:<28} {info['dtypes'].get(c, '')}")
    typer.echo("\nmapped onto ORCA fields:")
    for field, col in sorted(info["mapped"].items()):
        typer.echo(f"  {field:<16} <- {col}")
    if info["unmapped_columns"]:
        typer.echo(f"\nignored columns: {', '.join(info['unmapped_columns'])}")
    if info["missing_required"]:
        typer.echo(f"\nMISSING REQUIRED: {info['missing_required']}")
        typer.echo("  pass columns={'<field>': '<column>'} or rename in the archive")
        raise typer.Exit(1)
    if info["missing_advised"]:
        typer.echo(f"\nWARNING - missing {info['missing_advised']}:")
        typer.echo("  without `delta`, strategy templates cannot select strikes")
    typer.echo("\nschema OK" if not info["missing_required"] else "")
    typer.echo(json.dumps({"mapped": info["mapped"]}, indent=2)[:0])


@app.command()
def train(
    synthetic: bool = typer.Option(False, help="bootstrap on synthetic data"),
    rows: int = 4000,
    name: str = "default",
    symbol: str = typer.Option(None, help="train on one underlying only"),
    parquet: str = typer.Option(
        None, help="train from a Parquet archive (local path or s3://bucket/prefix)"
    ),
    since: str = typer.Option(None, help="ISO date; only snapshots on/after it"),
    until: str = typer.Option(None, help="ISO date; only snapshots on/before it"),
    sample_every: int = typer.Option(
        None, help="use every Nth snapshot as an entry point (default 1 for parquet, 5 for db)"
    ),
) -> None:
    """Train the candidate-scoring model."""
    from datetime import date as _date

    from orca.ml.dataset import (
        build_training_frame,
        build_training_frame_from_parquet,
        synthetic_frame,
    )
    from orca.ml.train import train_model

    d_since = _date.fromisoformat(since) if since else None
    d_until = _date.fromisoformat(until) if until else None

    if synthetic:
        frame = synthetic_frame(rows)
    elif parquet:
        frame = build_training_frame_from_parquet(
            parquet,
            underlying=symbol,
            since=d_since,
            until=d_until,
            sample_every=1 if sample_every is None else sample_every,
        )
    else:
        with session_scope() as session:
            frame = build_training_frame(
                session,
                underlying=symbol,
                since=d_since,
                sample_every=5 if sample_every is None else sample_every,
            )
    if frame.empty:
        typer.echo(
            "no training rows - pass --parquet <path|s3://...>, "
            "run `orca snapshot` on a few sessions, or pass --synthetic"
        )
        raise typer.Exit(1)

    _model, metrics = train_model(frame, name=name, synthetic=synthetic)
    for key, value in metrics.items():
        typer.echo(f"{key:>12}  {value}")


@app.command()
def demo(lots: int = 1) -> None:
    """End-to-end smoke test on the mock feed: scan -> enter -> track -> exit."""
    from orca.services.scanner import Scanner
    from orca.services.tracker import Tracker
    from orca.services.trade_service import TradeService

    _init_db()
    with session_scope() as session:
        candidates = Scanner().scan(session, lots=lots, persist=True)
        if not candidates:
            typer.echo("no candidates; nothing to demo")
            raise typer.Exit(1)
        best = candidates[0]
        typer.echo(f"best: {best.describe()} score={best.score} credit={best.net_credit:,.0f}")

        service = TradeService()
        trade = service.create_from_candidate(session, best, lots=lots)
        service.enter(session, trade)
        typer.echo(
            f"entered trade {trade.id} status={trade.status} credit={trade.entry_credit:,.2f}"
        )

        result = Tracker().track(session, trade)
        typer.echo(
            f"tracked: mtm={result.mtm:,.2f} delta={result.net_delta:,.0f} action={result.action}"
        )

        service.exit_trade(session, trade, reason="demo")
        typer.echo(
            f"exited: realized={trade.realized_pnl:,.2f} "
            f"charges={trade.charges:,.2f} total={trade.total_pnl:,.2f}"
        )


if __name__ == "__main__":
    app()
