"""Strategy templates. Importing this module registers every template."""

from orca.strategies import credit_spread, iron_condor, short_strangle  # noqa: F401
from orca.strategies.base import (
    available_templates,
    build_candidate,
    get_template,
    make_leg,
    template,
)

__all__ = [
    "available_templates",
    "build_candidate",
    "get_template",
    "make_leg",
    "template",
]
