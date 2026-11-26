"""Configuration: environment settings plus the YAML universe/strategy files."""

from __future__ import annotations

import functools
import os
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# PROJECT_ROOT is only a sensible default when ORCA is installed editable from a
# checkout. A plain `pip install` puts the package in site-packages, where
# ../config does not exist - so both directories are overridable. Containers and
# system installs set ORCA_CONFIG_DIR; everything else inherits the checkout.
CONFIG_DIR = Path(os.environ.get("ORCA_CONFIG_DIR") or PROJECT_ROOT / "config")
DATA_DIR = Path(os.environ.get("ORCA_DATA_DIR") or PROJECT_ROOT / "data")


class Settings(BaseSettings):
    """Everything that comes from the environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    env: str = Field(default="dev", alias="ORCA_ENV")
    log_level: str = Field(default="INFO", alias="ORCA_LOG_LEVEL")
    database_url: str = Field(default="sqlite:///./data/orca.db", alias="ORCA_DATABASE_URL")

    # --- data feed (Upstox) ---
    feed_provider: str = Field(default="mock", alias="ORCA_FEED_PROVIDER")  # mock | upstox
    upstox_api_key: str = Field(default="", alias="UPSTOX_API_KEY")
    upstox_api_secret: str = Field(default="", alias="UPSTOX_API_SECRET")
    upstox_redirect_uri: str = Field(default="", alias="UPSTOX_REDIRECT_URI")
    upstox_access_token: str = Field(default="", alias="UPSTOX_ACCESS_TOKEN")

    # --- execution (Zerodha Kite) ---
    execution_gateway: str = Field(default="paper", alias="ORCA_EXECUTION_GATEWAY")  # paper | kite
    kite_api_key: str = Field(default="", alias="KITE_API_KEY")
    kite_api_secret: str = Field(default="", alias="KITE_API_SECRET")
    kite_access_token: str = Field(default="", alias="KITE_ACCESS_TOKEN")

    # --- risk guardrails ---
    trading_enabled: bool = Field(default=False, alias="ORCA_TRADING_ENABLED")
    max_lots_per_trade: int = Field(default=2, alias="ORCA_MAX_LOTS_PER_TRADE")
    max_open_trades: int = Field(default=3, alias="ORCA_MAX_OPEN_TRADES")
    max_day_loss: float = Field(default=15000.0, alias="ORCA_MAX_DAY_LOSS")
    default_stop_multiple: float = Field(default=2.0, alias="ORCA_DEFAULT_STOP_MULTIPLE")
    default_target_pct: float = Field(default=0.5, alias="ORCA_DEFAULT_TARGET_PCT")

    # --- web ---
    # Origins allowed to call the API. The UI is a separate project, so this is
    # the only thing standing between it and a cross-origin block.
    # Comma-separated.
    cors_origins_raw: str = Field(default="*", alias="ORCA_CORS_ORIGINS")

    # Dashboard login. One operator account; every /api route except /health,
    # /session/login and the broker OAuth callbacks requires its bearer token.
    # The secret signs those tokens - leave it empty to derive one from the
    # credentials (so changing the password also logs everyone out).
    auth_username: str = Field(default="Admin", alias="ORCA_AUTH_USERNAME")
    auth_password: str = Field(default="passwordpassword", alias="ORCA_AUTH_PASSWORD")
    auth_secret: str = Field(default="", alias="ORCA_AUTH_SECRET")
    auth_token_ttl_hours: float = Field(default=12.0, alias="ORCA_AUTH_TOKEN_TTL_HOURS")

    # --- scheduler ---
    scheduler_enabled: bool = Field(default=True, alias="ORCA_SCHEDULER_ENABLED")
    snapshot_interval_sec: int = Field(default=60, alias="ORCA_SNAPSHOT_INTERVAL_SEC")
    tracker_interval_sec: int = Field(default=15, alias="ORCA_TRACKER_INTERVAL_SEC")
    # How often the worker re-verifies the saved Zerodha token with Kite (0 = off).
    token_check_interval_sec: int = Field(default=900, alias="ORCA_TOKEN_CHECK_INTERVAL_SEC")

    # --- ml ---
    model_dir: Path = Field(default=DATA_DIR / "models", alias="ORCA_MODEL_DIR")
    active_model: str = Field(default="default", alias="ORCA_ACTIVE_MODEL")

    # --- pricing assumptions ---
    risk_free_rate: float = 0.065  # used by the Black-Scholes helpers

    @property
    def cors_origins(self) -> list[str]:
        raw = (self.cors_origins_raw or "*").strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]


class Underlying(BaseModel):
    """One tradable index/stock in the universe (see config/instruments.yaml)."""

    symbol: str
    name: str = ""
    spot_exchange: str = "NSE_INDEX"
    option_exchange: str = "NFO"
    upstox_instrument_key: str = ""
    kite_name: str = ""
    lot_size: int = 1
    strike_step: float = 50.0
    weekly_expiry_weekday: int = 3
    has_weekly: bool = True
    enabled: bool = True

    @property
    def kite_symbol_prefix(self) -> str:
        return self.kite_name or self.symbol


class MarketHours(BaseModel):
    timezone: str = "Asia/Kolkata"
    open: time = time(9, 15)
    close: time = time(15, 30)
    holidays: list[str] = Field(default_factory=list)

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


class Universe(BaseModel):
    underlyings: list[Underlying] = Field(default_factory=list)
    market: MarketHours = Field(default_factory=MarketHours)

    def get(self, symbol: str) -> Underlying:
        for u in self.underlyings:
            if u.symbol.upper() == symbol.upper():
                return u
        raise KeyError(f"unknown underlying {symbol!r}; add it to config/instruments.yaml")

    @property
    def enabled(self) -> list[Underlying]:
        return [u for u in self.underlyings if u.enabled]


class StrategySpec(BaseModel):
    id: str
    template: str
    enabled: bool = True
    params: dict = Field(default_factory=dict)


class StrategyConfig(BaseModel):
    defaults: dict = Field(default_factory=dict)
    strategies: list[StrategySpec] = Field(default_factory=list)

    @property
    def enabled(self) -> list[StrategySpec]:
        return [s for s in self.strategies if s.enabled]


def _load_yaml(path: Path, required: bool = True) -> dict:
    """Load a YAML config file.

    Missing files used to return {} silently, which surfaced much later as an
    unrelated `unknown underlying 'SENSEX'` - the universe was simply empty.
    Fail loudly at the point of the actual problem instead.
    """
    if not path.exists():
        if not required:
            return {}
        raise FileNotFoundError(
            f"config file not found: {path}\n"
            f"  CONFIG_DIR resolves to {CONFIG_DIR}\n"
            f"  set ORCA_CONFIG_DIR to the directory holding instruments.yaml "
            f"and strategies.yaml"
        )
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@functools.lru_cache(maxsize=1)
def get_universe() -> Universe:
    return Universe.model_validate(_load_yaml(CONFIG_DIR / "instruments.yaml"))


@functools.lru_cache(maxsize=1)
def get_strategy_config() -> StrategyConfig:
    return StrategyConfig.model_validate(_load_yaml(CONFIG_DIR / "strategies.yaml"))


def reload_config() -> None:
    """Drop cached config so edited YAML/.env is picked up without a restart."""
    get_settings.cache_clear()
    get_universe.cache_clear()
    get_strategy_config.cache_clear()
