"""Single place that configures logging.

Dev prints wall-clock time only; anything deployed needs the date too, so the
format is chosen from ORCA_LOG_FORMAT ("plain" | "full" | "json"). Reconstructing
a bad trading session from logs that say "14:32:07" with no date is miserable,
so "full" is the right choice for staging and prod.
"""

from __future__ import annotations

import json
import logging
import os
import sys

_CONFIGURED = False

PLAIN = ("%(asctime)s %(levelname)-7s %(name)-28s %(message)s", "%H:%M:%S")
FULL = ("%(asctime)s %(levelname)-7s %(name)-28s %(message)s", "%Y-%m-%dT%H:%M:%S%z")


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for log shippers."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _formatter(style: str) -> logging.Formatter:
    if style == "json":
        return JsonFormatter()
    fmt, datefmt = FULL if style == "full" else PLAIN
    return logging.Formatter(fmt, datefmt)


def setup_logging(level: str = "INFO", style: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    style = (style or os.environ.get("ORCA_LOG_FORMAT", "plain")).lower()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_formatter(style))
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers = [handler]
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
