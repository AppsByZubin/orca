"""Broker token storage, expiry, and the /api/tokens surface."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from orca.services import broker_tokens as svc
from orca.services.broker_tokens import IST, next_cutoff


@pytest.fixture
def clean(monkeypatch):
    """Empty token table, no env tokens, no broker network calls."""
    from orca.config import get_settings
    from orca.db import init_db

    init_db()
    settings = get_settings()
    monkeypatch.setattr(settings, "upstox_access_token", "")
    monkeypatch.setattr(settings, "kite_access_token", "")
    calls: list[tuple[str, str]] = []
    result: dict = {"value": (True, "ok")}

    def fake_probe(broker, token):
        calls.append((broker, token))
        return result["value"]

    monkeypatch.setattr(svc, "probe", fake_probe)
    for b in svc.BROKERS:
        svc.delete_token(b)
    yield {"calls": calls, "result": result, "settings": settings}
    for b in svc.BROKERS:
        svc.delete_token(b)


@pytest.fixture
def client(clean):
    from orca.web.app import create_app

    with TestClient(create_app()) as c:
        r = c.post("/api/session/login", json={"username": "Admin", "password": "passwordpassword"})
        c.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
        yield c


def ist(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=IST)


@pytest.mark.parametrize(
    ("broker", "saved", "expected"),
    [
        ("upstox", ist(2026, 11, 25, 10, 0), ist(2026, 11, 26, 3, 30)),
        ("upstox", ist(2026, 11, 25, 2, 0), ist(2026, 11, 25, 3, 30)),
        ("upstox", ist(2026, 11, 25, 3, 30), ist(2026, 11, 26, 3, 30)),
        ("kite", ist(2026, 11, 25, 5, 59), ist(2026, 11, 25, 6, 0)),
        ("kite", ist(2026, 11, 25, 9, 15), ist(2026, 11, 26, 6, 0)),
    ],
)
def test_next_cutoff(broker, saved, expected) -> None:
    assert next_cutoff(broker, saved) == expected


def test_list_starts_missing(client) -> None:
    body = {t["broker"]: t for t in client.get("/api/tokens").json()}
    assert set(body) == {"kite", "upstox"}
    for t in body.values():
        assert t["has_token"] is False
        assert t["needs_attention"] is True


def test_save_verify_expire_delete(client, clean) -> None:
    r = client.put("/api/tokens/upstox", json={"access_token": "  tok-upstox-1234  "})
    assert r.status_code == 200, r.text
    t = r.json()
    assert t["has_token"] and t["source"] == "saved" and t["verified"] is True
    assert t["token_hint"] == "…1234" and "tok-upstox" not in r.text
    assert t["expired"] is False and t["needs_attention"] is False
    assert clean["calls"] == [("upstox", "tok-upstox-1234")]
    assert svc.current_token("upstox") == "tok-upstox-1234"

    # Past its cutoff: reported expired and no longer handed to the feed.
    from orca.db import session_scope
    from orca.models import BrokerToken

    with session_scope() as db:
        # UTC, as the service writes it - SQLite drops the offset.
        db.get(BrokerToken, "upstox").expires_at = datetime.now(UTC) - timedelta(minutes=1)
    svc._forget_cache()
    t = {x["broker"]: x for x in client.get("/api/tokens").json()}["upstox"]
    assert t["expired"] is True and t["needs_attention"] is True
    assert svc.current_token("upstox") == ""

    assert client.delete("/api/tokens/upstox").json()["has_token"] is False


def test_rejected_token_not_saved(client, clean) -> None:
    clean["result"]["value"] = (False, "Upstox rejected the token (HTTP 401)")
    r = client.put("/api/tokens/upstox", json={"access_token": "bad-token-0000"})
    assert r.status_code == 422
    assert "rejected" in r.json()["detail"]
    assert client.get("/api/tokens").json()[1]["has_token"] is False


def test_unreachable_broker_still_saves(client, clean) -> None:
    clean["result"]["value"] = (None, "could not reach Zerodha Kite")
    t = client.put("/api/tokens/kite", json={"access_token": "kite-token-9999"}).json()
    assert t["has_token"] is True and t["verified"] is None


def test_broker_rejection_marks_token(client, clean) -> None:
    client.put("/api/tokens/kite", json={"access_token": "kite-token-9999"})
    svc.mark_rejected("kite", "some-other-token")  # stale rejection: ignored
    assert svc.current_token("kite") == "kite-token-9999"
    svc.mark_rejected("kite", "kite-token-9999")
    t = {x["broker"]: x for x in client.get("/api/tokens").json()}["kite"]
    assert t["rejected"] is True and t["expired"] is True
    assert svc.current_token("kite") == ""


def test_env_token_used_when_nothing_saved(clean, monkeypatch) -> None:
    monkeypatch.setattr(clean["settings"], "upstox_access_token", "env-token-5678")
    svc._forget_cache()
    assert svc.current_token("upstox") == "env-token-5678"
    s = svc.status("upstox")
    assert s["source"] == "env" and s["expires_at"] is None and s["needs_attention"]


def test_tokens_need_login() -> None:
    from orca.web.app import create_app

    with TestClient(create_app()) as anon:
        assert anon.get("/api/tokens").status_code == 401
        assert anon.put("/api/tokens/kite", json={"access_token": "x"}).status_code == 401


def test_gateway_picks_up_saved_token(clean, monkeypatch) -> None:
    from orca.execution import get_gateway, reset_gateway
    from orca.execution.kite import KiteGateway
    from orca.execution.paper import PaperGateway

    monkeypatch.setattr(clean["settings"], "execution_gateway", "kite")
    monkeypatch.setattr(clean["settings"], "kite_api_key", "key")
    reset_gateway()
    try:
        assert isinstance(get_gateway(), PaperGateway)  # no token yet: safe fallback
        svc.save_token("kite", "kite-token-abcd", verify=False)
        gw = get_gateway()
        assert isinstance(gw, KiteGateway) and gw.access_token == "kite-token-abcd"
        svc.delete_token("kite")
        assert isinstance(get_gateway(), PaperGateway)
    finally:
        reset_gateway()
