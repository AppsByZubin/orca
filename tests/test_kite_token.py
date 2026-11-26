"""Zerodha Kite access-token verification (no network: httpx.get is faked)."""

from __future__ import annotations

import httpx
import pytest

from orca.execution import kite
from orca.execution.kite import KiteAuth, normalize_kite_token


class FakeKite:
    """Stands in for httpx.get; records the Authorization header it was sent."""

    def __init__(self, status: int = 200, body: dict | None = None, exc: Exception | None = None):
        self.status, self.body, self.exc = status, body or {}, exc
        self.auth: str | None = None

    def __call__(self, url, headers=None, timeout=None):
        self.auth = (headers or {}).get("Authorization")
        if self.exc:
            raise self.exc
        return httpx.Response(self.status, json=self.body, request=httpx.Request("GET", url))


PROFILE = {
    "status": "success",
    "data": {"user_id": "AB1234", "user_name": "Test Trader", "exchanges": ["NSE", "NFO", "BSE"]},
}


@pytest.fixture
def fake(monkeypatch):
    def install(**kw) -> FakeKite:
        f = FakeKite(**kw)
        monkeypatch.setattr(kite.httpx, "get", f)
        return f

    return install


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  abc123  ", "abc123"),
        ("token key1:abc123", "abc123"),
        ("key1:abc123", "abc123"),
        ('"abc123"', "abc123"),
        ("other:abc123", "other:abc123"),  # not our api_key: leave it alone
    ],
)
def test_normalize(raw, expected) -> None:
    assert normalize_kite_token(raw, "key1") == expected


def test_accepted_reports_account(fake) -> None:
    f = fake(body=PROFILE)
    check = KiteAuth.verify_token("token key1:abc123", api_key="key1", required_exchanges=["NFO"])
    assert check.ok is True
    assert check.user_id == "AB1234" and "AB1234 (Test Trader)" in check.detail
    assert check.missing_exchanges == []
    assert f.auth == "token key1:abc123"  # normalized before use


def test_accepted_but_segment_missing(fake) -> None:
    fake(body=PROFILE)
    check = KiteAuth.verify_token("abc123", api_key="key1", required_exchanges=["BFO", "NFO"])
    assert check.ok is True
    assert check.missing_exchanges == ["BFO"] and "not enabled for BFO" in check.detail


def test_token_exception_is_rejection(fake) -> None:
    fake(
        status=403,
        body={"status": "error", "message": "Incorrect `api_key` or `access_token`.",
              "error_type": "TokenException"},
    )
    check = KiteAuth.verify_token("abc123", api_key="key1")
    assert check.ok is False and "Incorrect" in check.detail


@pytest.mark.parametrize(
    "kw",
    [
        {"status": 429, "body": {"status": "error", "message": "Too many requests"}},
        {"status": 503, "body": {"status": "error", "message": "Service unavailable"}},
        {"exc": httpx.ConnectError("no route")},
    ],
)
def test_inconclusive(fake, kw) -> None:
    fake(**kw)
    assert KiteAuth.verify_token("abc123", api_key="key1").ok is None


def test_no_api_key_or_bad_shape(fake) -> None:
    f = fake(body=PROFILE)
    assert KiteAuth.verify_token("abc123", api_key="").ok is None
    assert KiteAuth.verify_token("abc 123", api_key="key1").ok is False
    assert f.auth is None  # neither reached Kite


def test_recheck_marks_revoked_token(monkeypatch) -> None:
    from orca.config import get_settings
    from orca.db import init_db
    from orca.services import broker_tokens as svc

    init_db()
    monkeypatch.setattr(get_settings(), "kite_api_key", "key1")
    monkeypatch.setattr(get_settings(), "kite_access_token", "")
    svc.delete_token("kite")
    try:
        svc.save_token("kite", "token key1:abc123", verify=False)
        assert svc.current_token("kite") == "abc123"  # stored normalized

        monkeypatch.setattr(kite.httpx, "get", FakeKite(body=PROFILE))
        svc.recheck_saved()
        assert svc.current_token("kite") == "abc123"

        monkeypatch.setattr(
            kite.httpx, "get", FakeKite(status=403, body={"error_type": "TokenException"})
        )
        svc.recheck_saved()
        assert svc.current_token("kite") == ""
        assert svc.status("kite")["rejected"] is True
    finally:
        svc.delete_token("kite")
