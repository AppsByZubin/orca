"""The market-data contract. Anything that can answer these five questions can
feed ORCA - Upstox today, something else tomorrow, the mock in tests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from orca.config import Underlying
from orca.domain import ChainSnapshot, Quote


class MarketDataProvider(ABC):
    """Read-only market data. Implementations must not place orders."""

    name: str = "base"

    @abstractmethod
    def get_spot(self, underlying: Underlying) -> float:
        """Last traded price of the underlying index/stock."""

    @abstractmethod
    def get_expiries(self, underlying: Underlying) -> list[date]:
        """Available option expiries, nearest first."""

    @abstractmethod
    def get_option_chain(self, underlying: Underlying, expiry: date) -> ChainSnapshot:
        """Full chain with prices, OI, IV and greeks where the feed provides them."""

    @abstractmethod
    def get_quotes(self, instrument_keys: list[str]) -> dict[str, Quote]:
        """Live quotes for arbitrary instruments, keyed by instrument_key."""

    def get_daily_closes(self, underlying: Underlying, days: int = 60) -> list[tuple[date, float]]:
        """Daily closes, oldest first. Optional - used for realized-vol features."""
        return []

    def health(self) -> dict:
        return {"provider": self.name, "ok": True}

    def close(self) -> None:  # noqa: B027 - optional hook, not every feed holds resources
        """Release sockets/sessions."""


class FeedError(RuntimeError):
    """Raised when the upstream feed fails or returns something unusable."""
