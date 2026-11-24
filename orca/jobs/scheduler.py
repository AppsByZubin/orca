"""Background jobs: snapshot capture, position tracking, periodic scan.

Runs inside the web process via APScheduler (`ORCA_SCHEDULER_ENABLED=true`),
or standalone with `orca worker`. Every job is market-hours gated and wrapped
so one failure never kills the schedule.
"""

from __future__ import annotations

from datetime import date, datetime

from apscheduler.schedulers.background import BackgroundScheduler

from orca.config import get_settings, get_universe
from orca.db import session_scope
from orca.logging_setup import get_logger
from orca.services.market_data import MarketDataService
from orca.services.scanner import Scanner
from orca.services.tracker import Tracker

log = get_logger(__name__)


def is_market_open(now: datetime | None = None) -> bool:
    market = get_universe().market
    now = (now or datetime.now(market.tz)).astimezone(market.tz)
    if now.weekday() >= 5:
        return False
    if now.date().isoformat() in set(market.holidays):
        return False
    return market.open <= now.time() <= market.close


def snapshot_job() -> None:
    """Capture chains - this is how the ML training set gets built."""
    if not is_market_open():
        return
    with session_scope() as session:
        rows = MarketDataService().capture_all(session)
        log.debug("captured %d chain snapshot(s)", len(rows))


def tracker_job() -> None:
    """Mark open multi-leg positions and fire exit rules."""
    with session_scope() as session:
        results = Tracker().track_all(session)
    for r in results:
        if r.action != "hold":
            log.info("trade %s -> %s (P&L %.0f)", r.trade_id, r.action, r.mtm)


def scan_job() -> None:
    """Refresh the ranked candidate list."""
    if not is_market_open():
        return
    with session_scope() as session:
        Scanner().scan(session, persist=True)


def daily_rollup_job() -> None:
    """End-of-day housekeeping hook (retrain, export, report)."""
    log.info("daily rollup for %s - add retraining/reporting here", date.today())


def build_scheduler() -> BackgroundScheduler:
    settings = get_settings()
    tz = get_universe().market.timezone
    scheduler = BackgroundScheduler(
        timezone=tz, job_defaults={"coalesce": True, "max_instances": 1}
    )

    scheduler.add_job(
        snapshot_job,
        "interval",
        seconds=settings.snapshot_interval_sec,
        id="snapshot",
        name="capture option chains",
    )
    scheduler.add_job(
        tracker_job,
        "interval",
        seconds=settings.tracker_interval_sec,
        id="tracker",
        name="track open trades",
    )
    scheduler.add_job(
        scan_job,
        "interval",
        seconds=max(settings.snapshot_interval_sec * 5, 300),
        id="scan",
        name="rank candidates",
    )
    scheduler.add_job(daily_rollup_job, "cron", hour=16, minute=0, id="rollup", name="daily rollup")
    return scheduler


_scheduler: BackgroundScheduler | None = None


def start_scheduler() -> BackgroundScheduler | None:
    global _scheduler
    if not get_settings().scheduler_enabled:
        log.info("scheduler disabled (ORCA_SCHEDULER_ENABLED=false)")
        return None
    if _scheduler is None:
        _scheduler = build_scheduler()
        _scheduler.start()
        log.info("scheduler started: %s", [j.id for j in _scheduler.get_jobs()])
    return _scheduler


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
